# Documentation review before 1.0.0 — 0128

Reviewed 27 October 2026 against TLlama after 0127, source commit
`cea7f297fefc5232e788d4010b164f9467998771`, with native compatibility default
`b11850-0e9433bfd3`. The user confirmed **4086 passed, 2 warnings** after
applying 0127. Warning details were not supplied. This is the last full code
suite result, not a new pytest run for the documentation-only 0128 patch.

## Reviewed areas and corrections

| Area | Source/evidence checked | Result |
| --- | --- | --- |
| Installation and entry points | pyproject.toml, client/daemon parsers, README | Python application and separate native runtime; optional HF/SDK dependencies; source commands and local/remote client selection |
| Server configuration | daemon_config, daemon_options, config, examples/server.toml | File discovery, CLI > env > TOML, model/request overrides, actual defaults and validation; corrected minimum model count and sampling precedence |
| Model TOML | model_toml, native_params, native_config | Added missing `[fim]`; clarified F16 K/V baseline, no ignored quantized V with FA off, model/backend compatibility and native device indices |
| Model management | backend pull resolver, Ollama router, CLI/import documentation | Exact HF reference syntax, optional suffix, file-only pull versus grouped import, TOML registration, shared weights and service permissions |
| Public API | main, Ollama/OpenAI routers, native routes/adapters | Complete application route/method table; explicit absent Responses/Completions/audio; current streaming and validation contracts |
| SDK/GUI/IDE usage | public adapter contracts and accepted SDK captures | Concrete SDK examples; generic GUI settings without untested client-version certification; optional SDK imports and provider extensions |
| Memory | native_reporting, native_telemetry, native memory contracts | Payload versus buffer/locality/residency; full-offload display exclusions; selected-source rules, unknowns, projector fallback and tensor diagnostics |
| Worker lifecycle | backend, residency, worker and launch contracts | Removed obsolete dual-engine/binding migration claims; kept immutable generations, leases, retirement and cleanup |
| Features and qualification | FIM/tools/vision/embedding/Decision/MTP contracts and reviewed audit | Current public contracts, strict diagnostic pins, accepted scope versus quality; corrected stale pending-gate statements and default Decision pin |
| Systemd | supplied unit, service guide and daemon schema | Executable/config paths, service account, runtime/state/cache directories, restart behavior, GPU access and mlock instructions retained |
| Historical records | native API checkpoint text and diagnostic recipes | Current guides separated from old builds/test counts; accepted captures remain accepted without repeat requests |

K/V behavior was checked in the pinned native `src/llama-context.cpp`: quantized
V with FA explicitly disabled is rejected, while automatic native FA can enable
it. TLlama's public boolean FA setting resolves to on/off. K-only quantization
must not be described as a blanket silent no-op. This correction does not change
runtime defaults, model files or sampling behavior.

## Validation

- Checked repository-local Markdown links and heading targets throughout README,
  docs and adjacent diagnostic guides.
- Parsed fenced Python, JSON and TOML examples and checked shell snippets with
  `bash -n`. This is syntax validation, not execution of model-generated code or
  downloads/system administration commands.
- Exercised client/daemon/subcommand help and both version entry points.
- Ran the shipped server TOML through actual `--check-config` and `--print-config`;
  checked documented schema/CLI mappings, source precedence and K/V expansion.
- Checked complete model TOML examples with the actual model parser, and checked
  fragment keys/types against the owned runtime/sampling contract.
- Compared the route table with source decorators, and checked documented script
  and test paths. Verified whitespace and clean application after 0127.

SDK examples require installed SDKs and a running configured model. They were
syntax checked here; the existing reviewed hardware/SDK captures remain their
runtime evidence. No new GPU qualification, native rebuild or service restart is
required for this documentation-only patch. Real service installation/access
checks remain host-specific; this review does not install or start systemd units,
perform downloads or certify untested GUI clients/backends.

All checks above passed: **43 Markdown files, 167 local links/anchors,
151 fenced examples** (8 Python, 6 JSON, 13 TOML, 124 shell), **17** help/version/
inspection commands, **29** daemon mapping rows, **23** public route/method pairs
and **67** referenced script/test paths. All 13 TOML examples/fragments passed
the model parser and owned key/value checks; 6 were complete model definitions.

## Remaining release decisions

Documentation corrections are prepared in 0128; apply that patch before treating
the source review as integrated. It does not set a release version or create a tag.
The following still need concrete disposition before 1.0.0:

- The open Pull errno 13 case needs the user's actual service-account/path result.
- Original sparse mmap accounting inflation was not reproduced by simplified
  captures; it must not be declared fixed from their zero capacity reduction.
- Legacy gfx900/gfx906/gfx90c HIP mmap investigation and the no-mmap workaround
  remain separate from accepted MI50/Vulkan and MI100/HIP configurations.
- FIM reference-model selection and subsequent independent TLlama quality
  qualification continue separately. Protocol parity does not settle the gate.
- Final merge/version/package/tag checks follow those decisions; no publication
  or tag is authorized by this review record alone.

The [runtime audit](runtime-audit-070.md) remains authoritative for accepted
feature/hardware scope. Personal evidence archives and work plans stay outside
Git; this document is the repository-facing review record.
