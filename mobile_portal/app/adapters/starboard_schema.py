from __future__ import annotations

from src.data.annotation_schema import FIELD_DEFINITIONS, GROUP_BY_NAME, AnnotationType
from src.data.vocabulary_store import get_vocabulary_store


COLOR_FIELDS = {
    'stripe_color', 'arm_color', 'central_disc_color', 'papillae_central_disc_color',
    'rosette_color', 'papillae_stripe_color', 'madreporite_color', 'overall_color'
}


def _mobile_widget(field) -> str:
    t = field.annotation_type
    if t in {AnnotationType.NUMERIC_INT, AnnotationType.NUMERIC_FLOAT}:
        return 'number'
    if t == AnnotationType.MORPHOMETRIC_CODE:
        return 'text'
    if t == AnnotationType.TEXT_FREE:
        return 'textarea'
    if t == AnnotationType.TEXT_HISTORY:
        return 'text'
    if field.options:
        return 'select'
    return 'text'


def project_schema() -> list[dict]:
    vocab = get_vocabulary_store()
    fields = []
    for field in FIELD_DEFINITIONS:
        group = GROUP_BY_NAME[field.group]
        vocabulary = []
        if field.name in COLOR_FIELDS:
            vocabulary = vocab.get_colors(field.name)
        elif field.name == 'location':
            vocabulary = vocab.get_locations()
        fields.append({
            'name': field.name,
            'display_name': field.display_name,
            'field_type': field.annotation_type.value,
            'group': field.group,
            'group_display_name': group.display_name,
            'required': not field.nullable,
            'tooltip': field.tooltip,
            'min_value': field.min_value,
            'max_value': field.max_value,
            'options': [{'label': opt.label, 'value': opt.value} for opt in field.options],
            'vocabulary': vocabulary,
            'mobile_widget': _mobile_widget(field),
        })
    return fields
