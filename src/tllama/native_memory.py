"""Initial allocations recognized from native runtime load diagnostics.

These figures describe backend buffers, not RSS/PSS or physical VRAM residency.
The parser keeps bounded structured observations independently of raw log retention.
"""

from __future__ import annotations

from decimal import Decimal, InvalidOperation
import re

from tllama.worker_config import RUNTIME_FINGERPRINT

# Log grammar and the input/staging-buffer placement rule are verified
# independently of the worker's broader runtime compatibility policy.
_LOG_RUNTIME_FINGERPRINTS = frozenset({"b11552-ec2797836f", "b11552-d1570abd3d",
                                    "b11552-3144c167ef", "b11850-0e9433bfd3"})

BACKEND_PREFIX = r"[A-Za-z][A-Za-z0-9]*(?:_[A-Za-z0-9]+)*"

PROJECTOR_BACKEND = re.compile(
    rf"clip_ctx:\s*CLIP using\s+({BACKEND_PREFIX})\s+backend",
    re.IGNORECASE,
)

PROJECTOR_SIZE = re.compile(
    r"load_hparams:\s*model size:\s*([\d.]+)\s*MiB",
    re.IGNORECASE,
)



_BUFFER = re.compile(
    rf"^(load_tensors|llama_kv_cache|llama_memory_recurrent|llama_context|sched_reserve|reserve_compute_meta):"
    rf"\s+({BACKEND_PREFIX})\s+(model|KV|RS|output|compute) buffer size\s*=\s*(\S+)\s+MiB\s*$")
# SRV_INF prints at most 12 characters of __func__: load_model_impl -> load_model_i.
_END = re.compile(r"^srv\s+load_model(?:_i)?:\s+initializing, n_slots\s*=\s*\d+")
_OFFLOAD = re.compile(r"^load_tensors:\s+offloaded ([0-9]{1,10})/([0-9]{1,10}) layers to GPU$")
_OFFLOAD_START = re.compile(r"^load_tensors:\s+offloaded\b")
_DEVICE = re.compile(r"^(CUDA|ROCm|HIP|Vulkan|SYCL|Metal|CANN|MUSA|OpenCL|WebGPU)\d*$", re.I)
_KINDS = {'load_tensors':'model', 'llama_kv_cache':'kv', 'llama_memory_recurrent':'rs',
          'llama_context':'output', 'sched_reserve':'compute', 'reserve_compute_meta':'compute'}
_MAX_BUFFERS = 256


def buffer_location(name: str) -> str:
    upper = name.upper()
    if upper == 'CPU' or upper.startswith('CPU_') or upper in {'BLAS', 'OPENBLAS'}:
        return 'host'
    if upper.endswith('_HOST'):
        return 'host'
    return 'device' if _DEVICE.fullmatch(name) else 'unknown'


def _bytes(value: str) -> int:
    try:
        number = Decimal(value)
        if not number.is_finite() or number < 0 or number > Decimal(2**63 - 1) / (1024**2):
            raise ValueError('Invalid allocation size')
        return int((number * (1024**2)).to_integral_value())
    except InvalidOperation as exc:
        raise ValueError('Invalid allocation size') from exc


class NativeMemory:
    """Observe one initial load, close it on a known marker, and never recount it."""

    def __init__(self, runtime_fingerprint=RUNTIME_FINGERPRINT, *, projector=False):
        self.runtime_fingerprint = runtime_fingerprint
        self.projector = projector
        self._buffers: dict[tuple[str, str], int] = {}
        self._ended = False
        self._invalid = None
        self._projector_backend = None
        self._projector_contexts = 0
        self._projector_weights = {}
        self._seen = set()
        self._layer_offload = None
        self._offload_seen = False

    def invalidate(self, reason):
        if not self._ended:
            self._invalid = self._invalid or reason

    def identify_runtime(self, fingerprint: str | None):
        """Attach observed identity without discarding already drained load lines."""
        self.runtime_fingerprint = fingerprint

    def observe(self, line: str):
        if self._ended or self._invalid:
            return
        line = line.strip()
        if _END.match(line):
            self._ended = True
            return
        if _OFFLOAD_START.match(line):
            match = _OFFLOAD.fullmatch(line)
            # A second summary could belong to another model/context. Keep
            # allocations usable, but do not infer placement from ambiguous logs.
            if match and not self._offload_seen and 0 <= int(match[1]) <= int(match[2]) and int(match[2]) > 0:
                self._layer_offload = {'offloaded_layers':int(match[1]), 'total_layers':int(match[2])}
            else:
                self._layer_offload = None
            self._offload_seen = True
            return
        backend = PROJECTOR_BACKEND.fullmatch(line)
        if backend:
            self._projector_contexts += 1
            if self._projector_contexts > 1:
                # Each modality's model-size line may describe the same GGUF.
                # Do not sum ambiguous multi-context projector summaries.
                self.invalidate('multiple_projector_contexts')
            self._projector_backend = backend[1]
            return
        size = PROJECTOR_SIZE.fullmatch(line)
        if size:
            if self._projector_backend is None:
                self.invalidate('unpaired_projector_size')
                return
            try:
                self._projector_weights[self._projector_backend] = _bytes(size[1])
            except ValueError:
                self.invalidate('invalid_projector_size')
            self._projector_backend = None
            return
        match = _BUFFER.fullmatch(line)
        if not match:
            if ('buffer size' in line and (line.split(':', 1)[0] in _KINDS or line.startswith('llama_'))
                    or line.startswith('load_hparams:') and 'model size:' in line):
                self.invalidate('unrecognized_buffer_report')
            return
        origin, backend, kind, size = match.groups()
        kind = kind.lower()
        if _KINDS[origin] != kind:
            self.invalidate('unexpected_buffer_kind')
            return
        if origin == 'reserve_compute_meta':
            kind = 'projector_compute'
        self._add(backend, kind, size, replace=kind in {'compute', 'projector_compute'})

    def _add(self, backend, kind, value, *, replace=False):
        key = (backend, kind)
        if key not in self._buffers and len(self._buffers) >= _MAX_BUFFERS:
            self.invalidate('buffer_limit')
            return
        try:
            size = _bytes(value)
        except ValueError:
            self.invalidate('invalid_buffer_size')
            return
        # Multiple model buffers of the same type are real allocations (e.g.
        # split GGUFs). Compute reservations instead replace the previous size.
        self._buffers[key] = size if replace else self._buffers.get(key, 0) + size
        self._seen.add(kind)

    def snapshot(self) -> dict:
        reason = self._invalid
        if reason is None:
            if not self._ended:
                reason = 'load_log_pending'
            elif not {'model', 'compute'} <= self._seen or not self._seen & {'kv', 'rs'}:
                reason = 'missing_buffer_reports'
            elif self.projector:
                # Pinned CLIP prints GGUF tensor sizes, not its allocated
                # weight buffer (which includes backend alignment/padding).
                # Keep the summary visible without calling it an allocation.
                reason = 'projector_allocation_unreported'
        buffers = {}
        for (backend, kind), size in self._buffers.items():
            buffers.setdefault(backend, {})[kind] = size
        allocated = sum(self._buffers.values()) if reason is None else None
        classified = all(buffer_location(backend) != 'unknown' for backend in buffers)
        device = (sum(size for (backend, _), size in self._buffers.items() if buffer_location(backend) == 'device')
                  if allocated is not None and classified else None)
        host = allocated - device if device is not None else None
        if reason is None and not classified:
            reason = 'unknown_buffer_location'
        return {'source':'native_load_log', 'runtime_fingerprint':self.runtime_fingerprint,
                'runtime_verified':self.runtime_fingerprint in _LOG_RUNTIME_FINGERPRINTS,
                'scope':'initial_load_allocations', 'precision':'rounded_mib', 'units':'bytes',
                'status':'known' if reason is None else 'partial' if buffers else 'unavailable',
                'reason':reason, 'buffers':buffers, 'allocated_bytes':allocated,
                'device_bytes':device, 'host_bytes':host, 'physical_residency_bytes':None,
                'layer_offload':dict(self._layer_offload) if self._layer_offload is not None else None,
                'projector_weights':dict(self._projector_weights),
                'projector_weights_source':'native_model_size_summary' if self._projector_weights else None}
