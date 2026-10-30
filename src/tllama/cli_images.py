"""Client-local image files; only their validated bytes cross the API boundary."""

import base64
from dataclasses import dataclass
import os
from pathlib import Path
import stat

from tllama.client import ClientError
from tllama.helpers.vision import to_image_data_url


MAX_IMAGE_BYTES = 16 * 1024 * 1024


@dataclass(frozen=True)
class PendingImage:
    path: str
    data: str


def validate_image(data):
    if not isinstance(data, str) or len(data) > 4 * ((MAX_IMAGE_BYTES + 2) // 3):
        raise ValueError('Image exceeds the 16 MiB decoded limit or is not base64 text.')
    # Reuse server-side sniffing; the native worker remains the image decoder.
    payload = to_image_data_url(data).split(',', 1)[1]
    size = len(payload) // 4 * 3 - (len(payload) - len(payload.rstrip('=')))
    if size > MAX_IMAGE_BYTES:
        raise ValueError('Image exceeds the 16 MiB decoded limit.')


def read_image(path):
    source = path
    try:
        source = Path(path).expanduser()
        fd = os.open(source, os.O_RDONLY | os.O_NONBLOCK)
        with os.fdopen(fd, 'rb') as stream:
            if not stat.S_ISREG(os.fstat(stream.fileno()).st_mode):
                raise ValueError('Image must be a regular file.')
            raw = stream.read(MAX_IMAGE_BYTES + 1)
        if len(raw) > MAX_IMAGE_BYTES:
            raise ValueError('Image exceeds the 16 MiB file limit.')
        encoded = base64.b64encode(raw).decode('ascii')
        validate_image(encoded)
    except (OSError, ValueError, RuntimeError) as exc:
        raise ClientError(f'Cannot attach image {str(source)!r}: {exc}') from None
    return PendingImage(str(source), encoded)
