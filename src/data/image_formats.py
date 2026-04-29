from __future__ import annotations

from pathlib import Path

ARCHIVE_IMAGE_EXTS = {
    '.jpg', '.jpeg', '.jpe', '.jfif',
    '.png',
    '.tif', '.tiff',
    '.bmp', '.dib',
    '.gif',
    '.webp',
    '.heic', '.heif',
    '.avif',
}

RAW_IMAGE_EXTS = {'.orf'}
IMPORT_IMAGE_EXTS = ARCHIVE_IMAGE_EXTS | RAW_IMAGE_EXTS


def normalized_suffix(path: str | Path) -> str:
    return Path(path).suffix.lower()


def is_archive_image(path: str | Path) -> bool:
    return normalized_suffix(path) in ARCHIVE_IMAGE_EXTS


def is_raw_image(path: str | Path) -> bool:
    return normalized_suffix(path) in RAW_IMAGE_EXTS


def is_importable_image(path: str | Path) -> bool:
    return normalized_suffix(path) in IMPORT_IMAGE_EXTS


def image_file_dialog_filter() -> str:
    patterns = []
    for ext in sorted(IMPORT_IMAGE_EXTS):
        patterns.append(f'*{ext}')
        patterns.append(f'*{ext.upper()}')
    return f"Images ({' '.join(patterns)});;All Files (*)"
