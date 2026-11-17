"""Guidance for the existing, deliberately different model-management contracts."""

CREATE_HINT = (
    "TLlama /api/create requires a TLlama TOML string in 'definition' and an "
    "optional 'uploads' map; Ollama Modelfile and structured create payloads "
    "are not supported. Use tllama create MODEL --file model.toml with a TLlama daemon."
)

PULL_HINT = (
    "TLlama /api/pull downloads Hugging Face GGUF files using "
    "'namespace/repository/path/to/file[.gguf]'; it does not pull Ollama registry "
    "tags or accept Hugging Face browser URLs. When using an Ollama daemon, "
    "use an Ollama registry model name instead."
)


def with_contract_hint(error, hint):
    message = str(error)
    return message if hint in message else f'{message} {hint}'


def with_contract_hint(message, hint):
    """Retain the specific failure and avoid repeating server-side guidance."""
    message = str(message)
    return message if hint in message else f'{message} {hint}'
