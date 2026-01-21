# src/data/fields_config.py
"""
Configuration loader for first-order comparison field settings.

Loads fields_config.yaml from the archive folder and provides:
- Default enabled fields
- Per-field weights (for future weighted averaging)
- Named presets for quick field selection
- Scorer parameters

Usage:
    from src.data.fields_config import get_fields_config, get_enabled_fields, get_preset_fields
    
    # Get full config object
    config = get_fields_config()
    
    # Get set of enabled field names
    enabled = get_enabled_fields()  # {'num_apparent_arms', 'stripe_color', ...}
    
    # Get fields for a named preset
    preset_fields = get_preset_fields("colors_only")  # {'stripe_color', 'arm_color', ...}
"""
from __future__ import annotations
from typing import Dict, Set, List, Any, Optional
from dataclasses import dataclass, field
from pathlib import Path
import logging

_log = logging.getLogger("starBoard.data.fields_config")

# Lazy-loaded module state
_config_cache: Optional["FieldsConfig"] = None
_config_path: Optional[Path] = None


@dataclass
class FieldDef:
    """Definition for a single field."""
    name: str
    enabled: bool = True
    weight: float = 1.0
    description: str = ""


@dataclass
class PresetDef:
    """Definition for a named preset."""
    name: str
    description: str = ""
    fields: List[str] = field(default_factory=list)


@dataclass
class ScorerParams:
    """Parameters for scorer algorithms."""
    numeric_k: float = 2.0
    numeric_eps: float = 1e-6
    color_threshold: float = 50.0
    color_fallback_to_exact: bool = True
    short_arm_sigma: float = 1.0
    short_arm_default_n_arms: int = 5
    text_model_id: str = "BAAI/bge-small-en-v1.5"


@dataclass
class FieldsConfig:
    """Complete fields configuration."""
    numeric_fields: Dict[str, FieldDef] = field(default_factory=dict)
    ordinal_fields: Dict[str, FieldDef] = field(default_factory=dict)
    color_fields: Dict[str, FieldDef] = field(default_factory=dict)
    set_fields: Dict[str, FieldDef] = field(default_factory=dict)
    text_fields: Dict[str, FieldDef] = field(default_factory=dict)
    presets: Dict[str, PresetDef] = field(default_factory=dict)
    equalize_weights: bool = True
    default_preset: str = "all"
    default_top_k: int = 50
    scorer_params: ScorerParams = field(default_factory=ScorerParams)
    
    def all_fields(self) -> Dict[str, FieldDef]:
        """Return all field definitions merged into one dict."""
        result = {}
        for fields_dict in [
            self.numeric_fields,
            self.ordinal_fields,
            self.color_fields,
            self.set_fields,
            self.text_fields,
        ]:
            result.update(fields_dict)
        return result
    
    def enabled_field_names(self) -> Set[str]:
        """Return names of all enabled fields."""
        return {name for name, fd in self.all_fields().items() if fd.enabled}
    
    def get_weight(self, field_name: str) -> float:
        """Get the configured weight for a field (1.0 if not found)."""
        all_f = self.all_fields()
        if field_name in all_f:
            return all_f[field_name].weight
        return 1.0
    
    def get_weights(self) -> Dict[str, float]:
        """Return dict of field_name -> weight for all fields."""
        return {name: fd.weight for name, fd in self.all_fields().items()}
    
    def preset_field_names(self, preset_name: str) -> Set[str]:
        """
        Return field names for a preset.
        
        If the preset has an empty fields list, returns all enabled fields.
        If the preset is not found, returns all enabled fields.
        """
        preset = self.presets.get(preset_name)
        if preset is None:
            _log.warning("Preset '%s' not found, using all enabled fields", preset_name)
            return self.enabled_field_names()
        
        if not preset.fields:
            # Empty list means "all enabled fields"
            return self.enabled_field_names()
        
        # Return only fields that are in the preset AND enabled
        enabled = self.enabled_field_names()
        return {f for f in preset.fields if f in enabled}


def _get_config_path() -> Path:
    """Get path to fields_config.yaml."""
    global _config_path
    if _config_path is not None:
        return _config_path
    
    from src.data.archive_paths import archive_root
    _config_path = archive_root() / "fields_config.yaml"
    return _config_path


def _load_yaml(path: Path) -> Dict[str, Any]:
    """Load YAML file, return empty dict if not found or error."""
    if not path.exists():
        _log.warning("Config file not found: %s", path)
        return {}
    
    try:
        import yaml
        with path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
            return data if isinstance(data, dict) else {}
    except ImportError:
        _log.warning("PyYAML not installed, cannot load fields_config.yaml")
        return {}
    except Exception as e:
        _log.error("Failed to load %s: %s", path, e)
        return {}


def _parse_field_group(data: Dict[str, Any], group_name: str) -> Dict[str, FieldDef]:
    """Parse a field group (numeric_fields, color_fields, etc.) from YAML data."""
    group = data.get(group_name, {})
    if not isinstance(group, dict):
        return {}
    
    result = {}
    for name, cfg in group.items():
        if not isinstance(cfg, dict):
            # Simple boolean or missing - treat as enabled=True
            result[name] = FieldDef(name=name, enabled=bool(cfg) if cfg is not None else True)
            continue
        
        result[name] = FieldDef(
            name=name,
            enabled=cfg.get("enabled", True),
            weight=float(cfg.get("weight", 1.0)),
            description=str(cfg.get("description", "")),
        )
    
    return result


def _parse_presets(data: Dict[str, Any]) -> Dict[str, PresetDef]:
    """Parse presets section from YAML data."""
    presets_data = data.get("presets", {})
    if not isinstance(presets_data, dict):
        return {}
    
    result = {}
    for name, cfg in presets_data.items():
        if not isinstance(cfg, dict):
            result[name] = PresetDef(name=name)
            continue
        
        fields_list = cfg.get("fields", [])
        if not isinstance(fields_list, list):
            fields_list = []
        
        result[name] = PresetDef(
            name=name,
            description=str(cfg.get("description", "")),
            fields=list(fields_list),
        )
    
    return result


def _parse_scorer_params(data: Dict[str, Any]) -> ScorerParams:
    """Parse scorer_params section from YAML data."""
    params = data.get("scorer_params", {})
    if not isinstance(params, dict):
        return ScorerParams()
    
    numeric = params.get("numeric", {}) or {}
    color = params.get("color", {}) or {}
    short_arm = params.get("short_arm_code", {}) or {}
    text = params.get("text_embedding", {}) or {}
    
    return ScorerParams(
        numeric_k=float(numeric.get("k", 2.0)),
        numeric_eps=float(numeric.get("eps", 1e-6)),
        color_threshold=float(color.get("threshold", 50.0)),
        color_fallback_to_exact=bool(color.get("fallback_to_exact", True)),
        short_arm_sigma=float(short_arm.get("sigma", 1.0)),
        short_arm_default_n_arms=int(short_arm.get("default_n_arms", 5)),
        text_model_id=str(text.get("model_id", "BAAI/bge-small-en-v1.5")),
    )


def _parse_config(data: Dict[str, Any]) -> FieldsConfig:
    """Parse full configuration from YAML data."""
    settings = data.get("settings", {}) or {}
    
    return FieldsConfig(
        numeric_fields=_parse_field_group(data, "numeric_fields"),
        ordinal_fields=_parse_field_group(data, "ordinal_fields"),
        color_fields=_parse_field_group(data, "color_fields"),
        set_fields=_parse_field_group(data, "set_fields"),
        text_fields=_parse_field_group(data, "text_fields"),
        presets=_parse_presets(data),
        equalize_weights=bool(settings.get("equalize_weights", True)),
        default_preset=str(settings.get("default_preset", "all")),
        default_top_k=int(settings.get("default_top_k", 50)),
        scorer_params=_parse_scorer_params(data),
    )


def _get_default_config() -> FieldsConfig:
    """
    Return default configuration matching current code behavior.
    
    Used when fields_config.yaml doesn't exist or can't be loaded.
    """
    # Import field lists from engine to stay in sync
    from src.search.engine import (
        NUMERIC_FIELDS, ORDINAL_FIELDS, COLOR_FIELDS, SET_FIELDS, TEXT_FIELDS
    )
    
    def _make_fields(names: List[str]) -> Dict[str, FieldDef]:
        return {n: FieldDef(name=n, enabled=True, weight=1.0) for n in names}
    
    return FieldsConfig(
        numeric_fields=_make_fields(NUMERIC_FIELDS),
        ordinal_fields=_make_fields(ORDINAL_FIELDS),
        color_fields=_make_fields(COLOR_FIELDS),
        set_fields=_make_fields(SET_FIELDS),
        text_fields=_make_fields(TEXT_FIELDS),
        presets={
            "all": PresetDef(name="all", description="All fields", fields=[]),
        },
        equalize_weights=True,
        default_preset="all",
        default_top_k=50,
        scorer_params=ScorerParams(),
    )


def get_fields_config(*, reload: bool = False) -> FieldsConfig:
    """
    Load and return the fields configuration.
    
    Args:
        reload: If True, reload from disk even if cached.
        
    Returns:
        FieldsConfig object with all settings.
    """
    global _config_cache
    
    if _config_cache is not None and not reload:
        return _config_cache
    
    path = _get_config_path()
    data = _load_yaml(path)
    
    if not data:
        _log.info("Using default field configuration (no fields_config.yaml)")
        _config_cache = _get_default_config()
    else:
        _log.info("Loaded field configuration from %s", path)
        _config_cache = _parse_config(data)
    
    return _config_cache


def get_enabled_fields(*, reload: bool = False) -> Set[str]:
    """
    Get the set of field names that are enabled by default.
    
    Convenience function that returns config.enabled_field_names().
    """
    return get_fields_config(reload=reload).enabled_field_names()


def get_preset_fields(preset_name: str, *, reload: bool = False) -> Set[str]:
    """
    Get the set of field names for a named preset.
    
    If the preset has empty fields list, returns all enabled fields.
    If preset not found, returns all enabled fields with a warning.
    """
    return get_fields_config(reload=reload).preset_field_names(preset_name)


def get_field_weights(*, reload: bool = False) -> Dict[str, float]:
    """
    Get weights for all fields.
    
    Returns dict mapping field_name -> weight.
    """
    return get_fields_config(reload=reload).get_weights()


def get_scorer_params(*, reload: bool = False) -> ScorerParams:
    """Get scorer algorithm parameters."""
    return get_fields_config(reload=reload).scorer_params



