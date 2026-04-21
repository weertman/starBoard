from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

REQUIRED = [
    'star_browser/app',
    'star_browser/app/models',
    'star_browser/app/routes',
    'star_browser/app/services',
    'star_browser/app/adapters',
    'star_browser/frontend/src',
    'star_browser/tests',
    'star_browser/deploy',
]


def test_star_browser_scaffold_exists():
    missing = [p for p in REQUIRED if not (ROOT / p).exists()]
    assert missing == []


def test_frontend_placeholder_files_exist():
    root = ROOT / 'star_browser/frontend/src'
    assert (root / 'App.tsx').exists()
    assert (root / 'pages/BatchUploadPage.tsx').exists()
