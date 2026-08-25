"""Which files in a watched folder are real photos.

Own module (not core/) so it can be tested without importing core/__init__,
which pulls in insightface.
"""
import os

# Derivatives that sit next to the real photo in camera and phone folders.
# They contain the same face, so they get embedded and sold as blank or
# duplicate items: 'thumb_IMG_5789.jpg' next to 'IMG_5789.JPG', and
# 'clip.mp4.jpg' poster frames whose original is a video we never store.
DERIVED_PREFIXES = ('thumb_', 'thumbnail_')
VIDEO_EXTENSIONS = ('.mp4', '.mov', '.avi', '.mkv', '.3gp', '.m4v')


def is_derived_file(filename: str) -> bool:
    """True for thumbnails and video poster frames — skip, don't upload."""
    # Split on both separators: the uploader runs on Windows, the test doesn't.
    name = filename.replace("\\", "/").rsplit("/", 1)[-1].lower()
    if name.startswith(DERIVED_PREFIXES):
        return True
    # '.mp4.jpg' → the stem still carries a video extension
    stem = os.path.splitext(name)[0]
    return os.path.splitext(stem)[1] in VIDEO_EXTENSIONS
