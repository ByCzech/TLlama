"""Compact header evidence for helper roles, not a second native loader."""

import re


NEXTN_ARCHITECTURES = frozenset({'qwen3next', 'qwen35', 'qwen35moe', 'qwen4exp',
                                'nemotron_h_moe', 'deepseek2', 'deepseek4'})
DRAFT_ARCHITECTURES = NEXTN_ARCHITECTURES | {'gemma4-assistant', 'dflash'}
_BLOCK = re.compile(r'blk\.(\d+)\.')
_PROJECTION = re.compile(r'blk\.(\d+)\.nextn\.eh_proj\.weight$')


def tensor_summary(arch, blocks, nextn, shared, names):
    """Consume names from the existing GGUF reader, never tensor values."""
    seen, projections = set(), set()
    markov = False
    for name in names:
        if match := _BLOCK.match(name):
            seen.add(int(match[1]))
        if match := _PROJECTION.fullmatch(name):
            projections.add(int(match[1]))
        markov |= name == 'markov_w1.weight'
    return {'version': 1, 'arch': arch, 'blocks': blocks, 'nextn': nextn,
            'shared_target': shared, 'first_block': min(seen) if seen else None,
            'last_block': max(seen) if seen else None,
            'projections': sorted(projections), 'markov': markov}


def group_draft_kind(metadata):
    """Classify only complete, already validated groups, including all shards."""
    if not metadata:
        return None
    arch = metadata[0].get('arch')
    if any(item.get('arch') != arch for item in metadata):
        return None
    summaries = [item.get('_draft') for item in metadata]
    if arch == 'dflash':
        return 'dspark' if any((item or {}).get('markov') for item in summaries) else 'dflash'
    if arch == 'gemma4-assistant':
        return 'mtp'
    if arch not in NEXTN_ARCHITECTURES or any(not isinstance(item, dict) or item.get('version') != 1 for item in summaries):
        return None
    first = summaries[0]
    blocks, nextn = first.get('blocks'), first.get('nextn')
    if type(blocks) is not int or type(nextn) is not int or not 0 < nextn < blocks:
        return None
    if any((item.get('blocks'), item.get('nextn')) != (blocks, nextn) for item in summaries):
        return None
    start = blocks - nextn
    # A single shard at the tail of a full model must never select its role.
    if any(item.get('first_block') is not None and item['first_block'] < start
           or item.get('last_block') is not None and item['last_block'] >= blocks for item in summaries):
        return None
    projections = {index for item in summaries for index in item.get('projections', ())}
    if len(projections) != nextn or min(projections, default=-1) != start or max(projections, default=-1) != blocks - 1:
        return None
    return 'mtp'


def active_draft(strategy, path, metadata):
    """DFlash stays dormant under MTP; an explicit unknown draft reaches native."""
    if strategy not in {'draft-mtp', 'draft-dflash'} or path is None:
        return None
    metadata = metadata or {}
    arch = metadata.get('arch')
    if strategy == 'draft-mtp' and arch == 'dflash':
        return None
    if group_draft_kind([metadata]) == 'dspark':
        raise ValueError('DSpark draft weights are not integrated; select a supported MTP or DFlash helper')
    if strategy == 'draft-dflash' and arch not in (None, '', 'unknown', 'dflash'):
        raise ValueError(f'draft-dflash requires dflash weights; got {arch!r}')
    if strategy == 'draft-mtp':
        if metadata.get('is_projector'):
            raise ValueError('draft-mtp requires MTP weights, not a projector')
        if (metadata.get('_draft') or {}).get('shared_target') is True:
            raise ValueError('Shared-target external MTP is unsupported by the audited native contract; use a self-contained MTP export')
    return path
