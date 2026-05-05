import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.annotation_schema import (
    AnnotationType,
    FIELD_BY_NAME,
    GALLERY_HEADER_V2,
    HEALTH_CODE_BY_CODE,
    HEALTH_CODE_DEFINITIONS,
    parse_health_codes,
    serialize_health_codes,
)


def test_health_codes_schema_field_is_additive_and_typed():
    field = FIELD_BY_NAME["health_codes"]

    assert field.display_name == "Health coding"
    assert field.annotation_type == AnnotationType.HEALTH_CODE
    assert field.group == "health"
    assert "health_codes" in GALLERY_HEADER_V2


def test_health_code_vocabulary_marks_counted_and_plus_codes():
    assert HEALTH_CODE_BY_CODE["L"].requires_count is True
    assert HEALTH_CODE_BY_CODE["L"].allows_plus is True
    assert HEALTH_CODE_BY_CODE["C-"].requires_count is True
    assert HEALTH_CODE_BY_CODE["BT"].requires_count is False
    assert HEALTH_CODE_BY_CODE["X"].exclusive is True
    assert HEALTH_CODE_BY_CODE["UNK"].exclusive is True


def test_health_code_vocabulary_is_rank_ordered_least_to_most_severe():
    category_rank = {
        "normal": 0,
        "feeding": 1,
        "status": 2,
        "mild": 3,
        "minor": 4,
        "major": 5,
    }
    ranks = [6 if d.terminal else category_rank[d.category] for d in HEALTH_CODE_DEFINITIONS]

    assert ranks == sorted(ranks)
    assert [d.code for d in HEALTH_CODE_DEFINITIONS[:3]] == ["X", "NA", "UNK"]
    assert [d.code for d in HEALTH_CODE_DEFINITIONS[-3:]] == ["IN", "DEAD", "RELEASED"]


def test_parse_and_serialize_health_codes_canonicalizes_tracker_style_values():
    entries = parse_health_codes("l(1+), c-(2), bt, Spawn")

    assert [(entry.code, entry.count, entry.plus) for entry in entries] == [
        ("L", 1, True),
        ("C-", 2, False),
        ("BT", None, False),
        ("SPAWN", None, False),
    ]
    assert serialize_health_codes(entries) == "L(1)+, C-(2), BT, SPAWN"
