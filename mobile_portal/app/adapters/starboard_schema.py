from __future__ import annotations

from src.data.annotation_schema import FIELD_DEFINITIONS, GROUP_BY_NAME, AnnotationType, HEALTH_CODE_DEFINITIONS
from src.data.location_sites import list_location_site_names
from src.data.vocabulary_store import get_vocabulary_store


COLOR_FIELDS = {
    'stripe_color', 'arm_color', 'central_disc_color', 'papillae_central_disc_color',
    'rosette_color', 'papillae_stripe_color', 'madreporite_color', 'overall_color'
}

# Groups that are auto-populated by the desktop app and not meaningful on mobile
_MOBILE_EXCLUDED_GROUPS = {'morphometric_auto', 'sync'}


def _mobile_widget(field) -> str:
    t = field.annotation_type
    if t in {AnnotationType.NUMERIC_INT, AnnotationType.NUMERIC_FLOAT}:
        return 'number'
    if t == AnnotationType.MORPHOMETRIC_CODE:
        return 'short_arm_code'
    if t == AnnotationType.HEALTH_CODE:
        return 'health_code'
    if t == AnnotationType.TEXT_FREE:
        return 'textarea'
    if field.name in COLOR_FIELDS:
        return 'color_select'
    if field.name == 'location':
        return 'location'
    if field.options:
        return 'select'
    if t == AnnotationType.TEXT_HISTORY:
        return 'text'
    return 'text'


def project_schema() -> list[dict]:
    vocab = get_vocabulary_store()
    fields = []
    for field in FIELD_DEFINITIONS:
        if field.group in _MOBILE_EXCLUDED_GROUPS:
            continue
        group = GROUP_BY_NAME[field.group]
        vocabulary = []
        if field.name in COLOR_FIELDS:
            vocabulary = sorted(set(vocab.get_colors(field.name)))
        elif field.name == 'location':
            vocabulary = list_location_site_names()
        options = [{'label': opt.label, 'value': opt.value} for opt in field.options]
        if field.annotation_type == AnnotationType.HEALTH_CODE:
            options = [
                {
                    'label': definition.label,
                    'value': definition.code,
                    'definition': definition.definition,
                    'category': definition.category,
                    'requires_count': definition.requires_count,
                    'allows_plus': definition.allows_plus,
                    'exclusive': definition.exclusive,
                    'terminal': definition.terminal,
                }
                for definition in HEALTH_CODE_DEFINITIONS
            ]
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
            'options': options,
            'vocabulary': vocabulary,
            'mobile_widget': _mobile_widget(field),
        })
    return fields
