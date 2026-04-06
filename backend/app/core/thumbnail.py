"""
RECALIX Thumbnail Utils
Generates and retrieves thumbnails. Never called during search — only during indexing.
"""

import hashlib
from pathlib import Path
from typing import Optional

from PIL import Image, ImageOps

from app.config import THUMBNAILS_DIR, THUMBNAIL_SIZE, THUMBNAIL_QUALITY


def _thumb_filename(file_path: str) -> str:
    """Generate a stable filename from the file path hash."""
    h = hashlib.sha256(file_path.encode()).hexdigest()[:16]
    return f"{h}.jpg"


def get_thumbnail_path(file_path: str) -> Path:
    return THUMBNAILS_DIR / _thumb_filename(file_path)


def generate_thumbnail(file_path: str) -> Optional[Path]:
    """
    Generate a thumbnail for an image file.
    Returns the thumbnail Path or None on failure.
    """
    thumb_path = get_thumbnail_path(file_path)

    # Already exists — skip
    if thumb_path.exists():
        return thumb_path

    try:
        img = Image.open(file_path)
        img = ImageOps.exif_transpose(img)  # respect EXIF rotation
        img = img.convert("RGB")
        img.thumbnail(THUMBNAIL_SIZE, Image.LANCZOS)
        img.save(str(thumb_path), "JPEG", quality=THUMBNAIL_QUALITY, optimize=True)
        return thumb_path
    except Exception as e:
        print(f"[Thumbnail] Failed for {file_path}: {e}")
        return None


def open_image_for_clip(file_path: str) -> Optional[Image.Image]:
    """Open and return a PIL Image for CLIP embedding, or None on error."""
    try:
        img = Image.open(file_path)
        img = ImageOps.exif_transpose(img)
        return img.convert("RGB")
    except Exception as e:
        print(f"[Utils] Cannot open {file_path}: {e}")
        return None
