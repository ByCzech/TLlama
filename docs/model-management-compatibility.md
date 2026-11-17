# Model-management compatibility

TLlama shares several HTTP paths with Ollama. `create` and `pull` have the
following source contracts; sharing a path does not make model definitions or
registries interchangeable.

| Operation | TLlama daemon | Ollama daemon |
| --- | --- | --- |
| `POST /api/create` | TLlama TOML text in `definition`, optional `uploads` | Ollama structured create request; not TLlama TOML/import preflight |
| `POST /api/pull` | Hugging Face repository and GGUF file reference | Ollama registry model reference |
| `show --modelfile` | Original TLlama TOML | Ollama model representation; not an importable TLlama definition |

## Create a TLlama definition

```toml
# model.toml
[llm]
from = "model.gguf"
```

```sh
tllama --host 127.0.0.1:54800 create my-model --file model.toml
```

`from` names client-side files, resolved from the client's current working
directory. The client checks the definition and destination first, then hashes
and uploads complete GGUF groups. The daemon publishes verified data before the
definition. It never opens an arbitrary server path from `from`.

To reuse data already managed by that daemon, send an existing repository path:

```json
{
  "model": "my-model",
  "definition": "[llm]\nmodel = \"data/GROUP_SHA256/weights.gguf\"\n"
}
```

Replace `GROUP_SHA256` with the actual group identity. `uploads`, when needed,
maps component names (`llm`, `mmproj`, `draft`) to ordered arrays of
`{"digest":"sha256:HEX","filename":"source.gguf"}`. `force` replaces a
definition; `dry_run` checks it without publication and accepts no uploads.
Success is a single JSON object with `status: "success"`; preflight returns
`status: "ready"`. This create endpoint does not stream progress.

Ollama `from`/`files`/`adapters`/`parameters`/`messages` and Modelfile requests
are not translated. Unknown create fields remain errors, including when a
valid TLlama `definition` is also present. Sending Modelfile text inside
`definition` does not make it TOML. Malformed requests return HTTP 400 with a
string `error` naming the specific problem and the TLlama create contract.
Existing conflicts remain 409; missing repository files remain 404.

The TLlama client's create/import/preflight is specific to a TLlama daemon.
Against a different server, failures retain the server's original error and
explain the required contract. A failed preflight stops file scanning, hashing
and upload. This is not a new server-identification probe: no additional
requests, registry conversion or retry is introduced. A successful response
from an arbitrary server is not proof that it implemented TLlama semantics.

## Pull a Hugging Face GGUF

```sh
tllama --host 127.0.0.1:54800 pull namespace/repository/path/to/model.gguf
```

```json
{"model":"namespace/repository/path/to/model.gguf","stream":false}
```

The `.gguf` suffix may be omitted. `model` takes precedence over legacy `name`.
Omitted/true/null `stream` returns NDJSON progress ending in `status: "success"`;
false returns a single success JSON object with the managed `path`. Failures
after streaming begins use an NDJSON `error` and do not emit success.

For example, `qwen3:8b` is an Ollama registry tag, not a TLlama HF file
reference. A full `https://huggingface.co/.../blob/main/...` or `resolve/main`
browser/download URL is not accepted as a file reference either. Use
`namespace/repository/path/to/model.gguf`, removing the URL host and revision
segments. No URL or registry autodetection is performed. A literal repository
subdirectory named `blob` or `resolve` remains a literal path, not a URL rule.

Invalid references return HTTP 400 before remote lookup/download; a confirmed
missing HF file returns 404. These errors explain the HF/registry distinction.
An absent/private/unreachable repository is not necessarily a confirmed missing
file; existing lookup and transfer behavior is preserved. Authentication uses
the optional `password` as an HF token; `username` is not registry login.
The daemon needs the optional `huggingface` dependency.

Repeated pulls retain the origin-owned definition's manually selected role
when it still references the tracked data. They do not create a second alias
merely to restore automatic role detection. Repointing a definition to different
weights preserves it and allows a new alias, as documented in the repository
contract. This behavior already exists; compatibility error guidance does not
change import/registration policy.

The TLlama client sends pull references to the selected daemon without
rewriting them. With an Ollama daemon, use its registry model name; an HF
reference does not become an Ollama import. Pull errors retain their original
message and explain both source conventions without guessing the daemon from
an error string. Shared progress shape alone does not prove registry parity.

These boundaries are covered by local ASGI/HTTPX tests and the optional Ollama
Python SDK against TLlama. Simulated foreign-server failures verify client
diagnostics and upload ordering; they are not a live Ollama registry or HF
download qualification. See [model-repositories.md](model-repositories.md) for
storage/publication and [cli.md](cli.md#create-and-import-models) for import.
