"""Deterministic decodable PNGs for native image transport tests."""

import base64
import struct
import zlib


def png(left=(255, 0, 0), right=(0, 0, 255)):
    width, height = 32, 16
    def chunk(kind, payload):
        return struct.pack('>I', len(payload)) + kind + payload + struct.pack('>I', zlib.crc32(kind + payload))
    rows = (b'\0' + bytes(left) * (width//2) + bytes(right) * (width//2)) * height
    data = b'\x89PNG\r\n\x1a\n'
    data += chunk(b'IHDR', struct.pack('>IIBBBBB', width, height, 8, 2, 0, 0, 0))
    data += chunk(b'IDAT', zlib.compress(rows)) + chunk(b'IEND', b'')
    return base64.b64encode(data).decode()


RED_BLUE = png()
BLUE_RED = png((0, 0, 255), (255, 0, 0))
RED_BLUE_URL = 'data:image/png;base64,' + RED_BLUE
BLUE_RED_URL = 'data:image/png;base64,' + BLUE_RED
