from __future__ import annotations

from pathlib import Path
from tempfile import NamedTemporaryFile
from io import BytesIO

from PIL import Image


class RawConversionError(RuntimeError):
    pass


def convert_raw_to_jpeg(src: Path, dest: Path) -> Path:
    """Convert a camera RAW file to a display-ready JPEG using camera white balance."""
    src = Path(src)
    dest = Path(dest)
    rgb = _read_raw_rgb(src)
    try:
        dest.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(rgb).save(dest, format='JPEG', quality=95)
    except Exception as exc:
        raise RawConversionError(f'Failed to save converted RAW image {dest.name}: {exc}') from exc
    return dest


def convert_raw_bytes_to_jpeg_bytes(payload: bytes, suffix: str = '.orf') -> bytes:
    if not payload:
        raise RawConversionError('empty_raw_upload')
    with NamedTemporaryFile(suffix=suffix) as temp:
        temp.write(payload)
        temp.flush()
        rgb = _read_raw_rgb(Path(temp.name))
    buf = BytesIO()
    Image.fromarray(rgb).save(buf, format='JPEG', quality=95)
    return buf.getvalue()


def _read_raw_rgb(src: Path):
    try:
        import rawpy  # type: ignore[import-not-found]
    except ImportError as exc:
        raise RawConversionError(
            'Olympus RAW conversion requires rawpy in the starBoard Python environment.'
        ) from exc

    try:
        with rawpy.imread(str(src)) as raw:
            return raw.postprocess(use_camera_wb=True, output_bps=8)
    except Exception as exc:
        raise RawConversionError(f'Failed to convert RAW image {src.name}: {exc}') from exc
