"""Check that derivatives are skipped and real photos are not.

Regression: the watcher ingested 'thumb_*.jpg' and '*.mp4.jpg' next to the
real photo, so one shot became 3-4 rows, each embedded and priced.
Run: python test_file_filters.py
"""
from file_filters import is_derived_file

for skip in ("thumb_IMG_5789_20250607_103131_3600.jpg",
             "THUMB_IMG_5789.JPG",
             "thumbnail_x.png",
             "20250607_103210_094.mp4.jpg",
             "20250607_103210_094.MOV.JPG",
             r"C:\photos\unit\outlet\thumb_a.jpg"):
    assert is_derived_file(skip), skip

for keep in ("IMG_5789_20250607_103131.JPG",
             "20250607_103210_094.jpg",
             "thumbs_up_guest.jpg",     # 'thumb' as a word, not the prefix
             "mp4_party_photo.jpg",
             "photo.jpeg"):
    assert not is_derived_file(keep), keep

print("OK")
