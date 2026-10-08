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

# The camera's wifi transfer app writes a downscaled copy beside the original:
# IMG_5790_20250607_103144_3600.JPG next to IMG_5790_20250607_103144.JPG. Only
# skipped when that original is actually there — a lone _3600 is the only copy
# of that shot and must still be sold.
# ponytail: resolved against the folder as it is right now. If the transfer app
# writes the derivative first and the original lands seconds later, both get
# ingested. Compare on file stem inside the uploader's batch if that shows up.
RESIZED_SUFFIXES = ('_3600',)


def is_derived_file(filename: str) -> bool:
    """True for thumbnails and video poster frames — skip, don't upload."""
    # Split on both separators: the uploader runs on Windows, the test doesn't.
    name = filename.replace("\\", "/").rsplit("/", 1)[-1].lower()
    if name.startswith(DERIVED_PREFIXES):
        return True
    # '.mp4.jpg' → the stem still carries a video extension
    stem = os.path.splitext(name)[0]
    if os.path.splitext(stem)[1] in VIDEO_EXTENSIONS:
        return True

    return any(
        stem.endswith(suffix) and _original_exists(filename, suffix)
        for suffix in RESIZED_SUFFIXES
    )


def _original_exists(path: str, suffix: str) -> bool:
    """Is the un-suffixed original next to this file?

    Returns False for a bare filename with no directory — nothing to check
    against, so the file is kept.
    """
    folder = os.path.dirname(path)
    if not folder:
        return False
    stem, ext = os.path.splitext(os.path.basename(path))
    original = stem[: -len(suffix)]
    # The transfer app is inconsistent about .JPG vs .jpg.
    return any(
        os.path.exists(os.path.join(folder, original + e))
        for e in {ext, ext.lower(), ext.upper()}
    )


# Byte trailers a finished file ends with. The photobooth app writes big framed
# JPEGs slowly on the Windows boxes; ingesting mid-write uploaded the top of
# the photo and a smeared / blank remainder.
_TRAILERS = {'.jpg': b'\xff\xd9', '.jpeg': b'\xff\xd9', '.png': b'IEND\xaeB`\x82'}


def is_fully_written(path: str) -> bool:
    """True once a JPEG/PNG ends with its end marker. Other formats: True."""
    trailer = _TRAILERS.get(os.path.splitext(path)[1].lower())
    if not trailer:
        return True
    try:
        with open(path, 'rb') as f:
            f.seek(0, os.SEEK_END)
            if f.tell() < len(trailer):
                return False
            f.seek(-len(trailer), os.SEEK_END)
            return f.read() == trailer
    except OSError:
        return False  # Windows: still locked by the writer
