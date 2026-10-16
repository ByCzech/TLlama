#!/usr/bin/env python3
"""Measure native FIM before implementing public /api/generate suffix support.

Resolves registered TLlama models, but calls only the owned native worker.
See fim-smoke.md for evidence, deliberate launch overrides and interpretation.
"""

import argparse
import ast
import asyncio
from contextlib import asynccontextmanager
from dataclasses import replace
import json
import logging
import math
from pathlib import Path
import re
import signal

from tllama.backend import ModelManager
from tllama.config import load_backend_config_from_env
from tllama.worker import NativeWorker, WorkerError, WorkerRequestError


MODELS = ("codegemma-2b-Q8_0", "starcoder2-3b-Q8_0", "qwen2.5-coder-7b.Q4_K_M",
          "DeepSeek-Coder-V2-Lite-Instruct-Q4_K_L", "Codestral-22B-v0.1-Q4_K_M")
NEGATIVE = "unsloth/Llama-3.2-3B-Instruct-GGUF/Llama-3.2-3B-Instruct-Q8_0"
PREFIX = "def combine(a, b):\n    return "
SUFFIXES = {"add": "\n\nassert combine(7, 2) == 9\n",
            "multiply": "\n\nassert combine(7, 2) == 14\n"}
# Direct lexical probes complement arithmetic; they do not execute output.
PROBES = {case: {"prefix": PREFIX, "suffix": suffix, "checker": "python_ast"}
          for case, suffix in SUFFIXES.items()}
for language in ("python", "typescript"):
    for setting in ("violet", "amber"):
        PROBES[f"{language}-{setting}"] = {
            "prefix": ('def read_config(config):\n    return config["' if language == "python" else
                       'export function readConfig(config: { violet: string; amber: string }): string {\n  return config.'),
            "suffix": ('"]\n\n# Read the "' + setting + '" setting.\n' if language == "python" else
                       ';\n}\n\n// Read the ' + setting + ' setting.\n'),
            "checker": "exact_identifier", "expected": setting}
TOKEN_LINE = re.compile(r"FIM (PRE|SUF|MID|PAD|REP|SEP) token\s*=\s*(\d+) '([^'\n]*)'")


def require(condition, message):
    if not condition:
        raise WorkerError(message)


def token_inventory(log):
    # Diagnostic only: these are the loaded vocabulary's selected IDs, not a
    # production capability heuristic. /infill independently confirms support.
    result = {}
    for kind, token_id, piece in TOKEN_LINE.findall(log):
        value = {"id": int(token_id), "piece": piece}
        require(kind not in result or result[kind] == value, f"Conflicting FIM {kind} diagnostics")
        result[kind] = value
    return result


def semantic_check(content, case):
    """Inspect AST or exact identifier; never execute generated output."""
    probe = PROBES[case]
    if probe["checker"] == "exact_identifier":
        return content.strip() == probe["expected"]
    try:
        tree = ast.parse(probe["prefix"] + content + probe["suffix"])
    except (SyntaxError, ValueError, RecursionError):
        return False
    if len(tree.body) != 2 or not isinstance(tree.body[0], ast.FunctionDef):
        return False
    body = tree.body[0].body
    if len(body) != 1 or not isinstance(body[0], ast.Return):
        return False
    expr = body[0].value
    operator = ast.Add if case == "add" else ast.Mult
    return (isinstance(expr, ast.BinOp) and isinstance(expr.op, operator)
            and isinstance(expr.left, ast.Name) and isinstance(expr.right, ast.Name)
            and {expr.left.id, expr.right.id} == {"a", "b"})


async def post(worker, path, body):
    async with worker._response(path, {**body, "stream": False}) as response:
        return await worker._json_body(response)


@asynccontextmanager
async def events(worker, path, body):
    # Reuse the supervisor's authentication, slot, bounds and cancellation.
    async with worker._response(path, {**body, "stream": True}) as response:
        iterator = worker._events(response, chat=False)
        try:
            yield iterator
        finally:
            await iterator.aclose()


async def tokenize(worker, text):
    response = await post(worker, "/tokenize", {"content": text, "add_special": False,
                                                "parse_special": False})
    tokens = response.get("tokens")
    require(isinstance(tokens, list) and all(type(t) is int and t >= 0 for t in tokens),
            "Invalid native text tokenization")
    return tokens


async def framing(worker, inventory):
    for kind, value in inventory.items():
        actual = await worker.tokenize(value["piece"], add_special=False)
        require(actual == [value["id"]], f"FIM {kind} diagnostic and live tokenizer disagree")
    ordinary = await tokenize(worker, "x")
    special = await worker.tokenize("x", add_special=True)
    bos_piece = worker.props.get("bos_token") or ""
    bos_ids = await worker.tokenize(bos_piece, add_special=False) if bos_piece else []
    bos = bos_ids if len(bos_ids) == 1 and special[:1] == bos_ids and special != ordinary else []
    # Do not carry an automatically appended EOS into explicit infill input.
    require(special[:len(bos) + len(ordinary)] == bos + ordinary,
            "Unknown tokenizer special-token framing; inspect this model manually")
    return bos


async def prompt_tokens(worker, inventory, bos, prefix, suffix, order, *, native, batch):
    prefix_ids, suffix_ids = await tokenize(worker, prefix), await tokenize(worker, suffix)
    counts = {"prefix": len(prefix_ids), "suffix": len(suffix_ids)}
    extra = []
    if native:
        # Reproduce the inspected b11552 formatter as a comparison hypothesis.
        # The full reference below intentionally does NOT truncate or add this
        # synthetic repository/file header. Differences are evidence, not hidden.
        prefix_ids = prefix_ids[-3 * (batch // 4):]
        suffix_ids = suffix_ids[:max(0, batch // 4 - 2)]
        for kind, text in (("REP", "myproject\n"), ("SEP", "filename\n")):
            if kind in inventory:
                extra += [inventory[kind]["id"]] + await tokenize(worker, text)
    tokens = extra + bos + layout_tokens(inventory, prefix_ids, suffix_ids, order)
    counts.update(used_prefix=len(prefix_ids), used_suffix=len(suffix_ids),
                  extra=len(extra), total=len(tokens))
    return tokens, counts


def layout_tokens(inventory, prefix, suffix, order):
    pre, suf, mid = (inventory[key]["id"] for key in ("PRE", "SUF", "MID"))
    if order == "training-spm":
        # Independent training reference; NOT what --spm-infill constructs.
        return [pre, suf] + suffix + [mid] + prefix
    if order == "psm":
        return [pre] + prefix + [suf] + suffix + [mid]
    require(order == "spm", "Unknown FIM order")
    return [suf] + suffix + [pre] + prefix + [mid]


async def verified_piece(worker, piece):
    ids = await worker.tokenize(piece, add_special=False)
    require(len(ids) == 1, f"Expected one live token for the documented delimiter: {piece}")
    return {"id": ids[0], "piece": piece}


async def stop_policy(worker, inventory):
    # Explicit diagnostic candidates, not a production capability detector.
    stops = dict(inventory)
    pre = inventory["PRE"]["piece"]
    extra = {"<|fim_prefix|>": ("<|file_separator|>",),
             "<fim_prefix>": ("<file_sep>",)}.get(pre, ())
    for index, piece in enumerate(extra):
        stops[f"FILE{index}"] = await verified_piece(worker, piece)
    pieces = list(dict.fromkeys(value["piece"] for value in stops.values()))
    # Native process_token must SEE the special token before text stops work.
    return {"stop": pieces, "preserved_tokens": pieces}, stops


async def verify_stops(worker, sampling, policy, stops, prompt, emit):
    # Force a delimiter independently of model quality. In particular, file_sep
    # may be hidden by default, and a plain stop string would then do nothing.
    for key in ("PRE", *(key for key in stops if key.startswith("FILE"))):
        token = stops[key]
        for stream in (False, True):
            response = await collect(worker, "/completion", {
                **sampling, **policy, "prompt": prompt, "ignore_eos": True,
                "n_predict": 4, "logit_bias": [[token["id"], 1000]]}, stream)
            final = response["final"]
            ok = (response["content"] == "" and final.get("stop_type") == "word"
                  and final.get("stopping_word") == token["piece"])
            emit("stop-check", token=token, stream=stream, verified=ok, **response)
            require(ok, "Native delimiter stop was not honored; inspect stop-check evidence")


async def probe_refusal(worker, sampling, infill, emit, stage, *, allow_supported=False):
    for stream in (False, True):
        try:
            response = await collect(worker, "/infill", {**sampling, **infill}, stream)
        except WorkerRequestError as exc:
            message = str(exc).lower()
            require((400 <= exc.http_status < 500 or exc.http_status == 501)
                    and "infill is not supported by this model" in message
                    and "token is missing" in message, "Unexpected missing-token FIM error")
            emit(stage, stream=stream, http_status=exc.http_status, error=exc.payload)
        else:
            require(allow_supported, "Vocabulary without native FIM assignments unexpectedly accepted /infill")
            emit("native-fim-accepted", stream=stream, **response)


async def codestral_reference(worker, sampling, infill, emit):
    # Mistral InstructTokenizerV2.encode_fim, inherited by V3:
    # BOS SUFFIX encode_infilling(suffix) PREFIX encode(prefix); NO MIDDLE.
    inventory = {key: await verified_piece(worker, piece) for key, piece in
                 (("PRE", "[PREFIX]"), ("SUF", "[SUFFIX]"), ("MID", "[MIDDLE]"))}
    emit("codestral-vocabulary", tokens=inventory)
    bos = await framing(worker, inventory)
    require(len(bos) == 1, "Codestral reference requires the verified native BOS")
    # Official _encode_infilling strips the SentencePiece dummy leading space.
    sentinel = await tokenize(worker, "☺")
    emit("codestral-framing", bos_tokens=bos, suffix_sentinel_tokens=sentinel)
    require(len(sentinel) == 2, "Codestral suffix sentinel framing differs from mistral-common")
    policy, stops = await stop_policy(worker, inventory)
    prompt = bos + [inventory["PRE"]["id"]] + await tokenize(worker, PREFIX)
    await verify_stops(worker, sampling, policy, stops, prompt, emit)
    await probe_refusal(worker, sampling, infill, emit, "native-fim-unassigned", allow_supported=True)
    for case, probe in PROBES.items():
        suffix = await tokenize(worker, "☺" + probe["suffix"])
        require(suffix[:2] == sentinel, "Codestral suffix sentinel does not occupy the first two tokens")
        prompt = (bos + [inventory["SUF"]["id"]] + suffix[2:]
                  + [inventory["PRE"]["id"]] + await tokenize(worker, probe["prefix"]))
        emit("prompts", case=case, reference="codestral-official", full_tokens=prompt,
             prefix=probe["prefix"], suffix=probe["suffix"], checker=probe["checker"])
        outputs = []
        for stream in (False, True):
            response = await collect(worker, "/completion", {**sampling, **policy, "prompt": prompt}, stream)
            outputs.append(response["content"])
            emit("completion", case=case, path="/completion", reference="codestral-official", stream=stream,
                 semantic_match=semantic_check(response["content"], case), **response)
        emit("reference-stream-comparison", case=case, reference="codestral-official", matches=outputs[0] == outputs[1])
    async with events(worker, "/completion", {**sampling, "prompt": prompt, "ignore_eos": True,
                                              "n_predict": 512}) as iterator:
        async for value in iterator:
            if value.get("content") or value.get("tokens"):
                require(value.get("stop") is not True, "Codestral stream finished before cancellation")
                emit("stream-cancel", first=value)
                break
        else:
            raise WorkerError("Codestral cancel stream produced no tokens")
    emit("after-cancel", **await collect(worker, "/completion", {**sampling, **policy, "prompt": prompt}, False))


async def collect(worker, path, body, stream):
    if stream:
        chunks, final = [], None
        async with events(worker, path, body) as iterator:
            async for value in iterator:
                require(isinstance(value.get("content", ""), str), "Invalid FIM stream content")
                chunks.append(value.get("content", ""))
                if value.get("stop") is True:
                    final = value
        content = "".join(chunks)
    else:
        final = await post(worker, path, body)
        content = final.get("content")
    require(final is not None and final.get("stop") is True and isinstance(content, str),
            "Missing native completion content/terminal")
    return {"content": content, "final": final}


async def exercise_worker(worker, args, order, negative, name, emit):
    inventory = token_inventory(worker.load_log)
    emit("vocabulary", tokens=inventory)
    if "missing pre-tokenizer type" in worker.load_log:
        emit("tokenizer-warning", reason="missing_pre_tokenizer", quality_claims="unreliable")
    sampling = {"temperature": 0, "seed": 0, "n_predict": args.predict,
                "repeat_penalty": 1, "top_k": 0, "top_p": 1, "min_p": 0,
                "cache_prompt": False, "return_tokens": True}
    infill = {"input_prefix": PREFIX, "input_suffix": SUFFIXES["add"], "prompt": ""}
    if negative:
        require(not all(key in inventory for key in ("PRE", "SUF", "MID")),
                "Negative candidate has all FIM tokens; select another model")
        await probe_refusal(worker, sampling, infill, emit, "negative-rejected")
        response = await collect(worker, "/completion", {**sampling, "prompt": "Once upon a time"}, False)
        require(bool(response["content"]), "Plain completion after negative FIM produced no text")
        emit("after-negative", **response)
        return

    if "codestral" in name.lower():
        await codestral_reference(worker, sampling, infill, emit)
        return
    if not all(key in inventory for key in ("PRE", "SUF", "MID")):
        await probe_refusal(worker, sampling, infill, emit, "positive-native-rejected")
        raise WorkerError("Positive FIM diagnostics incomplete; native refusal recorded")
    bos = await framing(worker, inventory)
    policy, stops = await stop_policy(worker, inventory)
    emit("stop-policy", tokens=stops, request=policy)
    prompt, _ = await prompt_tokens(worker, inventory, bos, PREFIX, SUFFIXES["add"], order,
                                    native=False, batch=args.batch)
    await verify_stops(worker, sampling, policy, stops, prompt, emit)
    # Retain a raw baseline so delimiter handling is observable, not post-hoc
    # stripping that could disguise generation beyond the first terminator.
    emit("unstopped-baseline", **await collect(worker, "/completion", {**sampling, "prompt": prompt}, False))
    sampling = {**sampling, **policy}
    outputs = {}
    for case, probe in PROBES.items():
        prefix, suffix = probe["prefix"], probe["suffix"]
        full, full_counts = await prompt_tokens(worker, inventory, bos, prefix, suffix, order,
                                               native=False, batch=args.batch)
        equivalent, equivalent_counts = await prompt_tokens(worker, inventory, bos, prefix, suffix, order,
                                                            native=True, batch=args.batch)
        emit("prompts", case=case, prefix=prefix, suffix=suffix, checker=probe["checker"],
             full_tokens=full, native_equivalent_tokens=equivalent,
             full_counts=full_counts, native_equivalent_counts=equivalent_counts)
        paths = [("/infill", "infill", {"input_prefix": prefix, "input_suffix": suffix, "prompt": ""}),
                 ("/completion", "full", {"prompt": full}),
                 ("/completion", "native-equivalent", {"prompt": equivalent})]
        if order == "spm":
            training, _ = await prompt_tokens(worker, inventory, bos, prefix, suffix, "training-spm",
                                              native=False, batch=args.batch)
            emit("training-spm-prompt", case=case, full_tokens=training)
            paths.append(("/completion", "training-spm", {"prompt": training}))
        for path, label, body in paths:
            for stream in ((False,) if label == "native-equivalent" else (False, True)):
                response = await collect(worker, path, {**sampling, **body}, stream)
                outputs[case, label, stream] = response["content"]
                emit("completion", case=case, path=path, reference=label, stream=stream,
                     semantic_match=semantic_check(response["content"], case), **response)
            if label == "training-spm":
                emit("reference-stream-comparison", case=case, reference=label,
                     matches=outputs[case, label, False] == outputs[case, label, True])
        emit("comparison", case=case,
             infill_matches_native_equivalent=outputs[case, "infill", False] == outputs[case, "native-equivalent", False],
             infill_matches_full=outputs[case, "infill", False] == outputs[case, "full", False],
             infill_stream_matches=outputs[case, "infill", False] == outputs[case, "infill", True],
             full_stream_matches=outputs[case, "full", False] == outputs[case, "full", True])
    for first, second in (("add", "multiply"), ("python-violet", "python-amber"),
                          ("typescript-violet", "typescript-amber")):
        emit("suffix-influence", pair=[first, second], changed={
            label: outputs[first, label, False] != outputs[second, label, False]
            for label in ("infill", "full", "native-equivalent")})

    # Exceed BOTH native batch-derived limits, while keeping the full prompt
    # within the real context. One output token suffices to measure admission.
    prefix_unit, suffix_unit = "# prefix padding\n", "# suffix padding\n"
    prefix_n = math.ceil(args.batch / len(await tokenize(worker, prefix_unit)))
    suffix_n = math.ceil(args.batch / 2 / len(await tokenize(worker, suffix_unit)))
    long_prefix = "PREFIX_START = 123\n" + prefix_unit * prefix_n + PREFIX
    long_suffix = SUFFIXES["add"] + suffix_unit * suffix_n + "SUFFIX_END = 456\n"
    full, full_counts = await prompt_tokens(worker, inventory, bos, long_prefix, long_suffix, order,
                                           native=False, batch=args.batch)
    equivalent, equivalent_counts = await prompt_tokens(worker, inventory, bos, long_prefix, long_suffix, order,
                                                        native=True, batch=args.batch)
    context = worker.props["default_generation_settings"]["n_ctx"]
    require(len(full) + 1 < context, "Long full FIM reference exceeds live context; increase --ctx")
    require(full_counts["prefix"] > equivalent_counts["used_prefix"]
            and full_counts["suffix"] > equivalent_counts["used_suffix"], "Long probe did not exceed both native limits")
    emit("long-prompts", prefix=long_prefix, suffix=long_suffix, full_tokens=full,
         native_equivalent_tokens=equivalent, full_counts=full_counts, native_equivalent_counts=equivalent_counts)
    for path, label, body in (("/infill", "infill", {"input_prefix": long_prefix, "input_suffix": long_suffix, "prompt": ""}),
                              ("/completion", "full", {"prompt": full}),
                              ("/completion", "native-equivalent", {"prompt": equivalent})):
        response = await collect(worker, path, {**sampling, **body, "n_predict": 1}, False)
        expected = full_counts["total"] if label == "full" else equivalent_counts["total"]
        emit("long-completion", reference=label, expected_tokens=expected,
             token_count_matches=response["final"].get("tokens_evaluated") == expected, **response)

    cancel_sampling = {key: value for key, value in sampling.items() if key not in policy}
    async with events(worker, "/infill", {**cancel_sampling, **infill, "ignore_eos": True, "n_predict": 512}) as iterator:
        async for value in iterator:
            if value.get("content") or value.get("tokens"):
                require(value.get("stop") is not True, "FIM cancel stream finished before cancellation")
                emit("stream-cancel", first=value)
                break
        else:
            raise WorkerError("FIM cancel stream produced no text")
    emit("after-cancel", **await collect(worker, "/infill", {**sampling, **infill}, False))


async def exercise(args, emit):
    config = load_backend_config_from_env()
    if args.models_dir:
        config = replace(config, models_dir=str(args.models_dir.expanduser().resolve()))
    selected = ([(name, False) for name in (args.model or ())]
                + [(name, True) for name in (args.negative_model or ())])
    if not selected:
        selected = [(name, False) for name in MODELS] + [(NEGATIVE, True)]
    orders = ("psm", "spm") if args.order == "both" else (args.order,)
    emit("diagnostic-version", version=2, probes=list(PROBES),
         native_formatter="b11552 hypothesis", stopping="live-verified delimiters with preserved_tokens")
    all_ok = True
    semantic_matches, semantic_total, review_profiles = 0, 0, 0
    for index, (name, negative) in enumerate(selected):
        for order in (("psm",) if negative or "codestral" in name.lower() else orders):
            label = f"{index:02d}-{order}"
            workers = []
            def factory(spec):
                options = {**spec.runtime_options, "spm_infill": order == "spm",
                           "n_batch": args.batch, "n_ubatch": min(args.batch, spec.runtime_options.get("n_ubatch", args.batch))}
                worker = NativeWorker(replace(spec, runtime_options=options))
                workers.append(worker)
                return worker
            manager = ModelManager(config, native_worker_factory=factory, native_launch_options={
                "executable": args.executable.expanduser().resolve() if args.executable else None,
                "load_timeout": args.load_timeout, "request_timeout": args.request_timeout})
            observations = []
            def record(stage, **values):
                nonlocal semantic_matches, semantic_total
                if stage == "completion":
                    semantic_total += 1
                    semantic_matches += values["semantic_match"]
                    if not values["semantic_match"]:
                        observations.append("semantic_mismatch")
                if stage == "comparison" and not all(values[key] for key in (
                    "infill_matches_native_equivalent", "infill_matches_full",
                    "infill_stream_matches", "full_stream_matches")):
                    observations.append("reference_difference")
                if stage == "long-completion" and not values["token_count_matches"]:
                    observations.append("native_formatter_hypothesis_differs")
                if stage == "tokenizer-warning":
                    observations.append("missing_pre_tokenizer")
                if stage == "native-fim-unassigned":
                    observations.append("native_fim_unassigned")
                if stage == "reference-stream-comparison" and not values["matches"]:
                    observations.append("stream_difference")
                if stage == "suffix-influence" and not all(values["changed"].values()):
                    observations.append("suffix_did_not_change_output")
                emit(stage, profile=label, model=name, order=order, **values)
            ok = True
            try:
                async with manager.acquire_native_model(name, num_ctx=args.ctx, keep_alive=0) as lease:
                    worker = lease.worker
                    record("ready", generation=lease.generation, pid=worker.pid, argv=worker.command,
                           fingerprint=worker.runtime_fingerprint, context=worker.props["default_generation_settings"]["n_ctx"],
                           expectation="unsupported" if negative else "supported")
                    await exercise_worker(worker, args, order, negative, name, record)
            except Exception as exc:
                ok = False
                record("error", error=f"{type(exc).__name__}: {exc}")
            finally:
                await manager.shutdown()
                for worker in workers:
                    # startup may fail before spawning; a failed profile still
                    # has to release every resource which was actually created.
                    returncode = worker.process.returncode if worker.process else None
                    removed = worker.runtime_directory is None or not worker.runtime_directory.exists()
                    closed = worker.state == "closed" and removed and returncode in (None, 0, -signal.SIGTERM)
                    ok &= closed
                    record("worker-closed", returncode=returncode, directory_removed=removed, clean=closed)
                    (args.output / f"{label}-load.log").write_text(worker.load_log, encoding="utf-8")
                require(not manager.native_snapshot(), "Manager retained a generation after shutdown")
                record("profile-finished", transport_ok=ok, residency=manager.native_snapshot(),
                       review_required=bool(observations) or not ok, observations=sorted(set(observations)))
            review_profiles += bool(observations) or not ok
            all_ok &= ok
    emit("finished", transport_ok=all_ok, public_suffix="not_implemented",
         semantic_matches=semantic_matches, semantic_total=semantic_total, review_profiles=review_profiles,
         review_required=bool(review_profiles),
         note="Semantic matches and reference differences require review; exit status covers harness/protocol/cleanup only.")
    return all_ok


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", action="append", help="Registered positive model; repeat to select several")
    parser.add_argument("--negative-model", action="append", help="Registered model expected to lack FIM tokens")
    parser.add_argument("--models-dir", type=Path)
    parser.add_argument("--executable", type=Path)
    parser.add_argument("--output", type=Path, required=True, help="New evidence directory; never overwrites previous results")
    parser.add_argument("--order", choices=("both", "psm", "spm"), default="both")
    parser.add_argument("--ctx", type=int, default=4096)
    parser.add_argument("--batch", type=int, default=512)
    parser.add_argument("--predict", type=int, default=64)
    parser.add_argument("--load-timeout", type=float, default=600)
    parser.add_argument("--request-timeout", type=float, default=120)
    args = parser.parse_args(argv)
    if args.batch < 16 or args.predict <= 0 or args.ctx < 2 * args.batch + args.predict + 64:
        parser.error("Need --batch >= 16, --predict > 0 and --ctx >= 2*batch + predict + 64")
    if any(not math.isfinite(value) or value <= 0 for value in (args.load_timeout, args.request_timeout)):
        parser.error("Timeouts must be finite and positive")
    args.output = args.output.expanduser().resolve()
    try:
        args.output.mkdir(parents=True, exist_ok=False)
    except FileExistsError:
        parser.error("--output already exists; select a new evidence directory")
    logging.basicConfig(level=logging.DEBUG, filename=args.output / "worker.log", encoding="utf-8")
    with (args.output / "raw.jsonl").open("w", encoding="utf-8") as raw, \
         (args.output / "summary.jsonl").open("w", encoding="utf-8") as summary:
        def emit(stage, **values):
            record = {"stage": stage, **values}
            raw.write(json.dumps(record, ensure_ascii=False) + "\n")
            raw.flush()
            concise = {key: value for key, value in record.items()
                       if key not in {"full_tokens", "native_equivalent_tokens", "prefix", "suffix", "final", "argv"}}
            if "final" in record:
                concise["terminal"] = {key: record["final"].get(key) for key in
                                       ("stop", "stop_type", "stopping_word", "tokens_evaluated", "tokens_predicted")}
            line = json.dumps(concise, ensure_ascii=False)
            summary.write(line + "\n")
            summary.flush()
            print(line, flush=True)
        try:
            ok = asyncio.run(exercise(args, emit))
        except Exception as exc:
            emit("fatal", error=f"{type(exc).__name__}: {exc}")
            ok = False
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
