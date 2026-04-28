import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication, QComboBox

from src.ui.tab_first_order import TabFirstOrder


class _Card:
    def __init__(self, gallery_id: str):
        self.gallery_id = gallery_id


def _app():
    return QApplication.instance() or QApplication([])


def test_gallery_jump_combo_uses_full_ranked_ids_not_only_visible_cards():
    _app()
    tab = TabFirstOrder.__new__(TabFirstOrder)
    tab.cmb_gallery_search = QComboBox()
    tab._cards = [_Card("rank_001_visible")]
    tab._gallery_jump_ids = [
        "rank_001_visible",
        "rank_002_with_many_underscores",
        "rank_003_tail",
    ]

    TabFirstOrder._update_gallery_search_combo(tab)

    assert [tab.cmb_gallery_search.itemText(i) for i in range(tab.cmb_gallery_search.count())] == [
        "rank_001_visible",
        "rank_002_with_many_underscores",
        "rank_003_tail",
    ]
