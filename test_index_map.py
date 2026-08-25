"""Check that returned photo_ids map back to the right photo.

Regression: the server skips photos with unresolvable entity codes, so a
positional map attached each file to the NEXT photo's row — wrong faces,
wrong outlet, wrong price. Run: python test_index_map.py
"""
from index_map import build_index_map

# Photo 1 of 4 was skipped by the server; ids are for photos 0, 2, 3.
assert build_index_map(0, 4, ["a", "c", "d"], [0, 2, 3]) == {0: "a", 2: "c", 3: "d"}

# Same, offset by the batch's start index.
assert build_index_map(10, 4, ["a", "c", "d"], [0, 2, 3]) == {10: "a", 12: "c", 13: "d"}

# Nothing skipped, old server with no photo_indexes: position is safe.
assert build_index_map(0, 3, ["a", "b", "c"], []) == {0: "a", 1: "b", 2: "c"}

# Short list and no photo_indexes: unmappable, must refuse rather than guess.
assert build_index_map(0, 4, ["a", "c", "d"], []) is None

print("OK")
