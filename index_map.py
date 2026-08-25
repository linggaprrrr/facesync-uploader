"""Map server-returned photo_ids back to the photos they belong to.

Own module so the mapping can be tested without importing the uploader's
insightface/aiohttp dependency chain.
"""
from typing import Dict, List, Optional


def build_index_map(start_idx: int, n_items: int, photo_ids: List[str],
                    photo_indexes: List[int]) -> Optional[Dict[int, str]]:
    """Map each returned photo_id back to the photo it belongs to.

    The server skips photos whose unit/outlet/type codes don't resolve, so
    photo_ids is dense while our batch is not. photo_indexes says which photo
    each id is for; without it one skipped photo shifts every id after it and
    files get uploaded into another photo's row.

    Returns None when the ids cannot be mapped safely — the caller must drop
    the batch rather than guess.
    """
    if len(photo_indexes) == len(photo_ids):
        return {start_idx + i: pid for i, pid in zip(photo_indexes, photo_ids)}
    if len(photo_ids) == n_items:
        # Old server, or nothing skipped: position is safe.
        return {start_idx + i: pid for i, pid in enumerate(photo_ids)}
    return None
