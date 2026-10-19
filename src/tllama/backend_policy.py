"""Ordered selector policy, without physical GPU identity or hardware rankings."""

import logging
import re


logger = logging.getLogger(__name__)
DEFAULT_BACKEND_PRIORITY = ('CUDA', 'ROCm', 'SYCL', 'Vulkan')
_NAMES = ('CPU', 'CUDA', 'ROCm', 'SYCL', 'Vulkan', 'Metal', 'OpenCL', 'WebGPU',
          'CANN', 'MUSA', 'Hexagon', 'OpenVINO', 'zDNN', 'ET', 'RPC', 'VirtGPU')
_CANONICAL = {name.casefold(): name for name in _NAMES}
_CANONICAL.update(hip='ROCm', mtl='Metal', htp='Hexagon')


def backend_priority(value):
    """Empty is CPU-only; unknown names stay usable by newer runtimes."""
    if isinstance(value, str):
        values = value.split(',') if value.strip() else ()
    elif isinstance(value, (tuple, list)):
        values = value
    else:
        raise ValueError('backend_priority must be a comma-separated string or a sequence of names')
    result, seen = [], set()
    for item in values:
        if not isinstance(item, str):
            logger.warning('Ignoring non-string backend priority entry %r', item)
            continue
        name = item.strip()
        if not re.fullmatch(r'[A-Za-z][A-Za-z0-9_-]*', name):
            logger.warning('Ignoring invalid backend priority entry %r', item)
            continue
        key = name.casefold()
        if key in {'blas', 'openblas', 'blis', 'mkl', 'zendnn'}:
            logger.warning('Ignoring CPU helper %s in backend priority; use CPU', name)
            continue
        name = _CANONICAL.get(key, name)
        key = name.casefold()
        if key not in seen:
            result.append(name)
            seen.add(key)
    return tuple(result)


def matches_backend(name, selector):
    """Known native exceptions, otherwise exact name or name plus ASCII digits."""
    key, device = name.casefold(), selector.casefold()
    if key == 'cpu':
        return False  # CPU is a terminal policy entry, not a listed GPU.
    if key == 'metal':
        return re.fullmatch(r'mtl[0-9]+', device) is not None
    if key == 'hexagon':
        return re.fullmatch(r'htp[0-9]+(?::[0-9]+)?', device) is not None
    if key == 'opencl':
        return device == 'gpuopencl'
    if key == 'virtgpu':
        return device.startswith('[virtgpu] ') and len(device) > len('[virtgpu] ')
    return re.fullmatch(re.escape(key) + r'(?:[0-9]+)?', device) is not None


def device_candidates(priority, selectors):
    """One backend per attempt; preserve native spelling and discovery order."""
    for name in priority:
        if name == 'CPU':
            break
        devices = tuple(selector for selector in selectors
                        if ',' not in selector and selector.strip() == selector
                        and not any(ord(c) < 32 for c in selector)
                        and matches_backend(name, selector))
        if devices:
            if name.casefold() not in _CANONICAL:
                logger.warning('Unknown backend %s matched native selector names; registry identity is unverified', name)
            yield name, ','.join(devices)
        else:
            log = logger.info if name.casefold() in _CANONICAL else logger.warning
            log('Backend priority entry %s has no matching discovered devices; skipping', name)
    yield 'CPU', 'none'
