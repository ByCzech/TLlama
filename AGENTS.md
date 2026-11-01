# Maintainer requirements

- Do not expand the public HTTP API without explicit prior agreement with the
  maintainer. This includes new routes/methods and additional request/response
  fields or provider-specific behavior, including on existing Ollama/OpenAI
  paths. An instruction to add a maintenance command is not approval to expose
  it through HTTP.
- Repository maintenance commands run locally through `tllamad`, against the
  local model repository. They must not start/contact the HTTP daemon or add
  corresponding commands to the `tllama` HTTP client without prior agreement.
