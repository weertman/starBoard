from types import SimpleNamespace

import numpy as np

from src.search.field_sets import ALL_FIELDS, COLOR_FIELDS, ORDINAL_FIELDS, TEXT_FIELDS
from star_browser.app.services import first_order_service


class _DummyEngine:
    def __init__(self):
        self.calls = []

    def rank(self, query_id, top_k=10, include_fields=None):
        self.calls.append(
            {
                'query_id': query_id,
                'top_k': top_k,
                'include_fields': include_fields,
            }
        )
        return [
            SimpleNamespace(
                gallery_id='candidate_001',
                score=0.9,
                k_contrib=2,
                field_breakdown={'field_a': 0.9},
            )
        ]


def test_first_order_search_all_preset_uses_all_fields(monkeypatch):
    engine = _DummyEngine()
    monkeypatch.setattr(first_order_service, '_get_engine', lambda: engine)

    response = first_order_service.run_first_order_search('query_001', top_k=7, preset='all')

    assert response.preset == 'all'
    assert engine.calls[0]['include_fields'] == set(ALL_FIELDS)


def test_first_order_search_colors_preset_uses_color_fields(monkeypatch):
    engine = _DummyEngine()
    monkeypatch.setattr(first_order_service, '_get_engine', lambda: engine)

    response = first_order_service.run_first_order_search('query_001', preset='colors')

    assert response.preset == 'colors'
    assert engine.calls[0]['include_fields'] == set(COLOR_FIELDS)


def test_first_order_search_text_preset_uses_text_fields(monkeypatch):
    engine = _DummyEngine()
    monkeypatch.setattr(first_order_service, '_get_engine', lambda: engine)

    response = first_order_service.run_first_order_search('query_001', preset='text')

    assert response.preset == 'text'
    assert engine.calls[0]['include_fields'] == set(TEXT_FIELDS)


def test_first_order_search_arms_patterns_preset_uses_expected_fields(monkeypatch):
    engine = _DummyEngine()
    monkeypatch.setattr(first_order_service, '_get_engine', lambda: engine)

    response = first_order_service.run_first_order_search('query_001', preset='arms_patterns')

    assert response.preset == 'arms_patterns'
    assert engine.calls[0]['include_fields'] == {
        'num_apparent_arms',
        'num_total_arms',
        'short_arm_code',
        *ORDINAL_FIELDS,
    }


def test_first_order_search_megastar_preset_uses_precomputed_numeric_scores(monkeypatch):
    engine = _DummyEngine()
    monkeypatch.setattr(first_order_service, '_get_engine', lambda: engine)
    monkeypatch.setattr(
        first_order_service,
        '_load_megastar_score_store',
        lambda: first_order_service._MegaStarScoreStore(
            query_ids=('query_001',),
            gallery_ids=('gallery_low', 'gallery_high'),
            scores=np.asarray([[0.25, 0.91]], dtype=np.float32),
        ),
    )

    response = first_order_service.run_first_order_search('query_001', top_k=1, preset='megastar')

    assert response.preset == 'megastar'
    assert engine.calls == []
    assert len(response.candidates) == 1
    assert response.candidates[0].entity_id == 'gallery_high'
    assert response.candidates[0].score == 0.91
    assert response.candidates[0].k_contrib == 1
    assert response.candidates[0].field_breakdown == {'megastar': 0.91}
