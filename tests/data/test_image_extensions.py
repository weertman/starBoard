from src.data.image_formats import ARCHIVE_IMAGE_EXTS, IMPORT_IMAGE_EXTS, RAW_IMAGE_EXTS
from src.data.ingest import IMAGE_EXTS, image_file_dialog_filter


def test_supported_image_extensions_cover_common_upload_formats_case_insensitively():
    expected = {
        '.jpg', '.jpeg', '.jpe', '.jfif',
        '.png',
        '.tif', '.tiff',
        '.bmp', '.dib',
        '.gif',
        '.webp',
        '.heic', '.heif',
        '.avif',
    }

    assert expected.issubset(IMAGE_EXTS)
    assert IMAGE_EXTS == IMPORT_IMAGE_EXTS
    assert expected.issubset(ARCHIVE_IMAGE_EXTS)
    assert RAW_IMAGE_EXTS == {'.orf'}
    assert '.orf' in IMPORT_IMAGE_EXTS
    assert '.orf' not in ARCHIVE_IMAGE_EXTS
    assert {ext.lower() for ext in IMAGE_EXTS} == IMAGE_EXTS


def test_image_file_dialog_filter_includes_uppercase_raw_and_all_files_fallback():
    dialog_filter = image_file_dialog_filter()

    for pattern in ['*.jpg', '*.JPG', '*.jpeg', '*.JPEG', '*.heic', '*.HEIC', '*.webp', '*.WEBP', '*.orf', '*.ORF']:
        assert pattern in dialog_filter
    assert 'All Files (*)' in dialog_filter
