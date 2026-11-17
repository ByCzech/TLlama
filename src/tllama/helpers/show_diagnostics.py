"""Offline default-chat configuration, never a report of an active worker."""

from tllama.helpers.thinking import thinking_profile


def chat_diagnostics(metadata):
    result = {
        'renderer': 'llama.cpp',
        'parser': 'native',
        'template_source': 'unknown',
    }
    if metadata.get('template'):
        source = metadata.get('_template_source')
        if source in {'gguf_chat_template', 'tllama_toml'}:
            result['template_source'] = source

    profile = thinking_profile(metadata)
    tool_metadata = {**metadata, 'template': metadata.get('tool_template') or metadata.get('template')}
    if profile is not None and 'tool_template' in metadata and profile == thinking_profile(tool_metadata):
        result['thinking'] = {'values': list(profile.values), 'default': profile.default}
    return result
