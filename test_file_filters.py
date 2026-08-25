"""Check that derivatives are skipped and real photos are not.

Regression: the watcher ingested 'thumb_*.jpg' and '*.mp4.jpg' next to the
real photo, so one shot became 3-4 rows, each embedded and priced.
Run: python test_file_filters.py
"""
import os
import tempfile

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

# The camera's wifi transfer app writes IMG_5790_..._3600.JPG beside the
# original. Skip it only when that original is really there.
with tempfile.TemporaryDirectory() as d:
    def touch(name):
        open(os.path.join(d, name), "w").close()
        return os.path.join(d, name)

    touch("IMG_5790.JPG")
    assert is_derived_file(os.path.join(d, "IMG_5790_3600.JPG"))
    # Extension case differs between the original and the copy.
    assert is_derived_file(os.path.join(d, "IMG_5790_3600.jpg"))
    # No original beside it: this is the only copy of the shot, keep it.
    assert not is_derived_file(os.path.join(d, "IMG_6001_3600.JPG"))
    # Bare name, no folder to check against: keep.
    assert not is_derived_file("IMG_5790_3600.JPG")

print("OK")
