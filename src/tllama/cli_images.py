"""Client-local image files; only their validated bytes cross the API boundary."""

import base64
from dataclasses import dataclass
import os
from pathlib import Path
import stat

from tllama.client import ClientError
from tllama.client_config import DEFAULT_IMAGE_MAX_FILE_MIB, validate_size_limit_mib
from tllama.helpers.vision import to_image_data_url


@dataclass(frozen=True)
class PendingImage:
    path: str
    data: str


def validate_image(data, *, max_file_mib=DEFAULT_IMAGE_MAX_FILE_MIB):
    max_bytes = validate_size_limit_mib(max_file_mib) * 1024 * 1024
    if not isinstance(data, str):
        raise ValueError('Image must be base64 text.')
    if len(data) > 4 * ((max_bytes + 2) // 3):
        raise ValueError(f'Image exceeds the {max_file_mib} MiB decoded limit.')
    # Reuse server-side sniffing; the native worker remains the image decoder.
    payload = to_image_data_url(data).split(',', 1)[1]
    size = len(payload) // 4 * 3 - (len(payload) - len(payload.rstrip('=')))
    if size > max_bytes:
        raise ValueError(f'Image exceeds the {max_file_mib} MiB decoded limit.')


def read_image(path, *, max_file_mib=DEFAULT_IMAGE_MAX_FILE_MIB):
    source = path
    try:
        max_bytes = validate_size_limit_mib(max_file_mib) * 1024 * 1024
        source = Path(path).expanduser()
        fd = os.open(source, os.O_RDONLY | os.O_NONBLOCK)
        with os.fdopen(fd, 'rb') as stream:
            info = os.fstat(stream.fileno())
            if not stat.S_ISREG(info.st_mode):
                raise ValueError('Image must be a regular file.')
            if info.st_size > max_bytes:
                raise ValueError(f'Image exceeds the {max_file_mib} MiB file limit.')
            raw = stream.read(max_bytes + 1)
        if len(raw) > max_bytes:
            raise ValueError(f'Image exceeds the {max_file_mib} MiB file limit.')
        encoded = base64.b64encode(raw).decode('ascii')
        validate_image(encoded, max_file_mib=max_file_mib)
    except (OSError, ValueError, RuntimeError) as exc:
        raise ClientError(f'Cannot attach image {str(source)!r}: {exc}') from None
    return PendingImage(str(source), encoded)
