"""Device capacity configuration and conservative worker selector sets."""

from dataclasses import dataclass
import json


@dataclass(frozen=True)
class DeviceLimit:
    name: str | None
    max_loaded_models: int

    def document(self):
        return {**({'name': self.name} if self.name is not None else {}),
                'max_loaded_models': self.max_loaded_models}


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f'duplicate JSON key {key!r}')
        result[key] = value
    return result


def device_limits(value, *, text=False):
    if text:
        value = json.loads(value, object_pairs_hook=_unique_object)
    if not isinstance(value, (list, tuple)):
        raise ValueError('lifecycle.device must be an array of device tables')
    result, names = [], set()
    for index, item in enumerate(value):
        if isinstance(item, DeviceLimit):
            item = item.document()
        if not isinstance(item, dict) or set(item) - {'name', 'max_loaded_models'}:
            raise ValueError(f'lifecycle.device[{index}] expects only name and max_loaded_models')
        limit = item.get('max_loaded_models')
        if type(limit) is not int or limit < 0:
            raise ValueError(f'lifecycle.device[{index}].max_loaded_models must be a nonnegative integer')
        name = item.get('name')
        if 'name' in item and (not isinstance(name, str) or not name or name.strip() != name
                              or ',' in name or any(ord(c) < 32 or ord(c) == 127 for c in name)):
            raise ValueError(f'lifecycle.device[{index}].name must be one nonempty native selector or CPU')
        if name == 'none':
            raise ValueError('Use CPU, not none, for a lifecycle.device name')
        if name in names:
            raise ValueError(f'duplicate lifecycle.device name {name!r} (only one unnamed default is allowed)')
        names.add(name)
        result.append(DeviceLimit(name, limit))
    return tuple(result)


def capacity_config(max_loaded_models, limits):
    limits = device_limits(limits)
    if max_loaded_models is None:
        max_loaded_models = 0 if limits else 1
    if type(max_loaded_models) is not int or max_loaded_models < 0:
        raise ValueError('max_loaded_models must be a nonnegative integer')
    if max_loaded_models > 0 and limits:
        raise ValueError('Conflicting lifecycle.max_loaded_models and lifecycle.device: '
                         'set the global limit to 0 or remove the device list')
    return max_loaded_models, limits


def worker_devices(spec):
    """Reserve each selected device once, including an explicitly placed helper.

    A CSV/tensor-split launch conservatively occupies every selected GPU.
    Host spill buffers do not turn a GPU worker into another CPU worker.
    """
    if spec.device is None:
        raise ValueError('Resolve an automatic candidate before reserving device capacity')
    names = set(spec.device.split(','))
    if spec.draft is not None and spec.runtime_options.get('draft_device'):
        names.update(spec.runtime_options['draft_device'].split(','))
    return frozenset('CPU' if name == 'none' else name for name in names)
