"""Daemon argument vocabulary; help must not import server configuration."""

import argparse


# flag, canonical setting, environment variable, help/default description.
OPTIONS = (
    ('host', 'server.host', 'TLLAMA_HOST', 'Listener address, optionally HOST:PORT; default 127.0.0.1'),
    ('port', 'server.port', 'TLLAMA_PORT', 'Listener port; default 54800'),
    ('models-dir', 'storage.models_dir', 'TLLAMA_MODELS', 'Model store; default /var/lib/tllama/models'),
    ('runtime-path', 'engine.executable', 'TLLAMA_RUNTIME_EXECUTABLE', 'Absolute native executable; default /usr/lib/tllama/server/tllama-server'),
    ('runtime-version-policy', 'engine.version_policy', 'TLLAMA_RUNTIME_VERSION_POLICY', 'warn (default) or strict'),
    ('backend-priority', 'engine.backend_priority', 'TLLAMA_BACKEND_PRIORITY', 'Comma-separated backend names; default CUDA,ROCm,SYCL,Vulkan; empty = CPU only'),
    ('device', 'runtime.device', 'TLLAMA_RUNTIME_DEVICE', 'Global native device default; unset = automatic backend priority'),
    ('context-length', 'runtime.n_ctx', 'TLLAMA_CONTEXT_LENGTH', 'Global context default; 0 (default) = trained maximum'),
    ('gpu-layers', 'runtime.n_gpu_layers', 'TLLAMA_RUNTIME_N_GPU_LAYERS', 'GPU layer count; -1 = all'),
    ('threads', 'runtime.n_threads', 'TLLAMA_RUNTIME_N_THREADS', 'Evaluation threads; unset = native automatic'),
    ('threads-batch', 'runtime.n_threads_batch', 'TLLAMA_RUNTIME_N_THREADS_BATCH', 'Prompt evaluation threads; unset = native automatic'),
    ('batch-size', 'runtime.n_batch', 'TLLAMA_RUNTIME_N_BATCH', 'Logical batch size; worker default 512'),
    ('ubatch-size', 'runtime.n_ubatch', 'TLLAMA_RUNTIME_N_UBATCH', 'Physical batch size; worker default 512'),
    ('kv-cache-type', 'runtime.type_kv', 'TLLAMA_KV_CACHE_TYPE', 'Global K/V cache type; worker default f16'),
    ('k-cache-type', 'runtime.type_k', 'TLLAMA_K_CACHE_TYPE', 'Global K cache type; overrides same-layer shorthand'),
    ('v-cache-type', 'runtime.type_v', 'TLLAMA_V_CACHE_TYPE', 'Global V cache type; overrides same-layer shorthand'),
    ('keep-alive', 'lifecycle.keep_alive', 'TLLAMA_KEEP_ALIVE', 'Default idle duration; default 5m; negative = retain indefinitely'),
    ('max-loaded-models', 'lifecycle.max_loaded_models', 'TLLAMA_MAX_LOADED_MODELS', 'Resident model limit; default 1'),
    ('janitor-interval', 'lifecycle.janitor_interval', 'TLLAMA_JANITOR_INTERVAL', 'Janitor interval in seconds; default 10'),
    ('metadata-cache-ttl', 'cache.metadata_ttl', 'TLLAMA_METADATA_CACHE_TTL', 'Memory metadata TTL in seconds; default 300'),
    ('log-level', 'server.log_level', 'TLLAMA_LOG_LEVEL', 'critical/error/warning/info/debug/trace; default info'),
    ('origins', 'server.origins', 'TLLAMA_ORIGINS', 'Comma-separated browser origins added to built-in local origins'),
    ('memory-source', 'telemetry.memory_source', 'TLLAMA_MEMORY_SOURCE', 'auto (default), native or log'),
)
BOOL_OPTIONS = (
    ('reload', 'server.reload', 'TLLAMA_APP_RELOAD', 'Development reload; default false'),
    ('flash-attention', 'runtime.flash_attn', 'TLLAMA_RUNTIME_FLASH_ATTN', 'Global flash attention default; worker default false'),
    ('mmap', 'runtime.use_mmap', 'TLLAMA_RUNTIME_USE_MMAP', 'Global mmap default; worker default true'),
    ('mlock', 'runtime.use_mlock', 'TLLAMA_RUNTIME_USE_MLOCK', 'Global mlock default; worker default false'),
    ('memory-compare', 'telemetry.memory_compare', 'TLLAMA_MEMORY_COMPARE', 'Compare memory telemetry; default false'),
    ('ignore-unsupported-runner', 'engine.ignore_unsupported_runner', 'TLLAMA_IGNORE_UNSUPPORTED_RUNNER', 'Ignore unsupported model runner; default false'),
)


def add_daemon_options(parser):
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument('--config', metavar='PATH', help='Server TOML file (overrides TLLAMA_CONFIG).')
    selection.add_argument('--no-config', action='store_true', help='Ignore server TOML selection and discovery.')
    inspection = parser.add_mutually_exclusive_group()
    inspection.add_argument('--check-config', action='store_true', help='Validate configuration and exit without serving.')
    inspection.add_argument('--print-config', action='store_true', help='Print effective values and sources as JSON and exit.')
    inspection.add_argument('--list-devices', action='store_true', help='Print native runtime devices and exit without serving or loading a model.')
    for flag, key, env, help_text in OPTIONS:
        parser.add_argument('--' + flag, dest=key, default=argparse.SUPPRESS,
                            metavar=flag.replace('-', '_').upper(), help=f'{help_text} ({env}).')
    for flag, key, env, help_text in BOOL_OPTIONS:
        parser.add_argument('--' + flag, dest=key, default=argparse.SUPPRESS,
                            action=argparse.BooleanOptionalAction, help=f'{help_text} ({env}).')
