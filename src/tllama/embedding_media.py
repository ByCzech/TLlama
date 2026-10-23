"""Strict inline image inputs shared by the two public embedding adapters."""

from tllama.helpers.vision import to_image_data_url


MAX_IMAGE_BYTES = 4 * 1024 * 1024
MAX_MEDIA_BYTES = 16 * 1024 * 1024
MAX_IMAGES = 32
MAX_MEDIA_ITEMS = 32


def content_entry(value, *, ollama=False):
    """Validate the whole item; never discard unsupported media or fields."""
    if ollama:
        if not isinstance(value, dict) or not value or set(value) - {'text', 'image'}:
            raise ValueError('Embedding objects support only text and image; audio/video are not integrated')
        parts = []
        if 'text' in value:
            if not isinstance(value['text'], str):
                raise ValueError('Embedding text must be a string')
            parts.append({'type': 'text', 'text': value['text']})
        if 'image' in value:
            parts.append({'type': 'image_url', 'image_url': {'url': value['image']}})
    else:
        if not isinstance(value, dict) or set(value) != {'content'}:
            raise ValueError('Embedding content objects support only content')
        parts = value['content']
    if not isinstance(parts, list) or not 1 <= len(parts) <= 64:
        raise ValueError('Embedding content requires 1..64 text/image parts')
    normalized = []
    for part in parts:
        if not isinstance(part, dict):
            raise ValueError('Embedding content parts must be objects')
        if part.get('type') == 'text' and set(part) == {'type', 'text'} and isinstance(part['text'], str):
            normalized.append(dict(part))
        elif part.get('type') == 'image_url' and set(part) == {'type', 'image_url'}:
            image = part['image_url']
            if not isinstance(image, dict) or set(image) != {'url'} or not isinstance(image['url'], str):
                raise ValueError('Embedding image_url supports only a string url')
            # Refuse before base64 decoding; the header allowance is bounded too.
            if len(image['url']) > 4 * ((MAX_IMAGE_BYTES + 2) // 3) + 256:
                raise ValueError('Embedding image exceeds the 4 MiB decoded limit')
            url = to_image_data_url(image['url'])
            data = url.split(',', 1)[1]
            size = len(data) // 4 * 3 - (len(data) - len(data.rstrip('=')))
            if size > MAX_IMAGE_BYTES:
                raise ValueError('Embedding image exceeds the 4 MiB decoded limit')
            normalized.append({'type': 'image_url', 'image_url': {'url': url}})
            media_budget([{'content': normalized}])
        else:
            raise ValueError('Embedding content supports only literal text and inline image_url parts')
    if not any(p['type'] == 'image_url' for p in normalized):
        text = ''.join(p['text'] for p in normalized)
        if not text:
            raise ValueError('Embedding content must not be empty')
        return text
    return {'content': normalized}


def media_budget(inputs):
    """A separate request bound, independent of image feature/token guesses."""
    images, size = 0, 0
    for entry in inputs:
        if isinstance(entry, dict):
            for part in entry['content']:
                if part['type'] == 'image_url':
                    images += 1
                    data = part['image_url']['url'].split(',', 1)[1]
                    size += len(data) // 4 * 3 - (len(data) - len(data.rstrip('=')))
    if images and (len(inputs) > MAX_MEDIA_ITEMS or images > MAX_IMAGES or size > MAX_MEDIA_BYTES):
        raise ValueError('Image embedding requests support at most 32 items, 32 images and 16 MiB decoded media')
    return bool(images)
