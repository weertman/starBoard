# src/data/annotation_schema.py
"""
Annotation schema definition for sunflower star metadata.

This module defines the typed annotation system with 5 annotation classes:
- NUMERIC: Integer or float measurements
- MORPHOMETRIC_CODE: Structured short arm coding
- COLOR_CATEGORICAL: Extensible color vocabulary
- MORPH_CATEGORICAL: Fixed ordinal categorical options
- TEXT: Free-form or history-backed text fields
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Tuple, Optional, Union


class AnnotationType(Enum):
    """The five annotation type classes."""
    NUMERIC_INT = "numeric_int"
    NUMERIC_FLOAT = "numeric_float"
    MORPHOMETRIC_CODE = "morphometric_code"
    HEALTH_CODE = "health_code"
    COLOR_CATEGORICAL = "color_categorical"
    MORPH_CATEGORICAL = "morph_categorical"
    TEXT_HISTORY = "text_history"
    TEXT_FREE = "text_free"


@dataclass
class CategoricalOption:
    """A single option in a categorical field."""
    label: str
    value: Union[int, float]

    def __str__(self) -> str:
        return self.label


@dataclass
class FieldDefinition:
    """Definition of a single annotation field."""
    name: str
    display_name: str
    annotation_type: AnnotationType
    group: str
    options: List[CategoricalOption] = field(default_factory=list)
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    nullable: bool = True
    tooltip: str = ""


@dataclass
class FieldGroup:
    """A group of related fields for UI organization."""
    name: str
    display_name: str
    fields: List[str] = field(default_factory=list)
    start_expanded: bool = True


# =============================================================================
# CATEGORICAL OPTIONS DEFINITIONS
# =============================================================================

STRIPE_ORDER_OPTIONS = [
    CategoricalOption("None", 0),
    CategoricalOption("Mixed", 1),
    CategoricalOption("Irregular", 2),
    CategoricalOption("Regular", 3),
]

STRIPE_PROMINENCE_OPTIONS = [
    CategoricalOption("None", 0),
    CategoricalOption("Weak", 1),
    CategoricalOption("Medium", 2),
    CategoricalOption("Strong", 3),
    CategoricalOption("Strongest", 4),
]

STRIPE_EXTENT_OPTIONS = [
    CategoricalOption("None", 0.0),
    CategoricalOption("Quarter", 0.25),
    CategoricalOption("Halfway", 0.5),
    CategoricalOption("Three quarters", 0.75),
    CategoricalOption("Full", 1.0),
]

STRIPE_THICKNESS_OPTIONS = [
    CategoricalOption("None", 0),
    CategoricalOption("Thin", 1),
    CategoricalOption("Medium", 2),
    CategoricalOption("Thick", 3),
]

RETICULATION_ORDER_OPTIONS = [
    CategoricalOption("None", 0),
    CategoricalOption("Mixed", 1),
    CategoricalOption("Meandering", 2),
    CategoricalOption("Train tracks", 3),
]

ROSETTE_PROMINENCE_OPTIONS = [
    CategoricalOption("Weak", 0),
    CategoricalOption("Medium", 1),
    CategoricalOption("Strong", 2),
]

ARM_THICKNESS_OPTIONS = [
    CategoricalOption("Thin", 0),
    CategoricalOption("Medium", 1),
    CategoricalOption("Thick", 2),
]

SHORT_ARM_SEVERITY_OPTIONS = [
    CategoricalOption("Short", "short"),
    CategoricalOption("Small short", "small"),
    CategoricalOption("Tiny short", "tiny"),
    CategoricalOption("Very tiny short", "very_tiny"),
]


@dataclass(frozen=True)
class HealthCodeDefinition:
    """A selectable lab health/symptom code."""
    code: str
    label: str
    definition: str
    category: str
    requires_count: bool = False
    allows_plus: bool = False
    exclusive: bool = False
    terminal: bool = False


HEALTH_CODE_DEFINITIONS = [
    # Least severe / non-symptom statuses first, then mild, minor, major, terminal.
    HealthCodeDefinition("X", "No symptoms present", "No symptoms present at check.", "normal", exclusive=True),
    HealthCodeDefinition("NA", "No appetite", "No interest in food on feeding days.", "feeding"),
    HealthCodeDefinition("UNK", "Unknown / not visible", "Star is not visible.", "status", exclusive=True),
    HealthCodeDefinition("SPAWN", "Actively spawning", "Individual is actively spawning.", "status"),
    HealthCodeDefinition("1-MA", "1-MA injection", "1-MA was injected to induce spawning.", "status"),
    HealthCodeDefinition("WV", "Wavy arms", "Sine-like shape to some or all arms.", "mild"),
    HealthCodeDefinition("WV-", "Wavy arms minor", "Minor wavy arms, typically one arm or barely wavy.", "mild"),
    HealthCodeDefinition("DL", "Depressed central disk", "Lumpy central disk, depressed but not completely flat.", "mild"),
    HealthCodeDefinition("DL-", "Depressed central disk minor", "Minor depressed or lumpy central disk.", "mild"),
    HealthCodeDefinition("BT", "Bent arms", "Middle ground between bent arm tips and extreme bent arms.", "mild"),
    HealthCodeDefinition("BT-", "Bent arm", "Arm tips bent at a 90 degree angle relative to arm length.", "mild"),
    HealthCodeDefinition("TH", "Thin", "Flat overall appearance with very thin arms.", "mild"),
    HealthCodeDefinition("IN-", "Inflated minor", "Minor inflated/pumping position.", "minor"),
    HealthCodeDefinition("S-", "Stretched minor", "Minor stretched appearance.", "minor"),
    HealthCodeDefinition("F-", "Flat minor", "Minor flat appearance.", "minor"),
    HealthCodeDefinition("DR-", "Droopy minor", "Slightly droopy tissue of the central disk or arms.", "minor"),
    HealthCodeDefinition("TW-", "Twisty arms minor", "Number of arms with twisted tips; not true curling.", "minor", requires_count=True),
    HealthCodeDefinition("C-", "Curling arms minor", "Number of arms crossed over another arm.", "minor", requires_count=True),
    HealthCodeDefinition("Y", "Cyst", "Boil-like cyst near base of arm.", "minor"),
    HealthCodeDefinition("WV+", "Extra wavy", "Extra wavy, but not extreme wavy.", "minor"),
    HealthCodeDefinition("BT+", "Spiral/extreme bent arms", "All arms bent like a spiral.", "minor"),
    HealthCodeDefinition("LOSS", "Dropped arms", "Number of arms completely removed from the body.", "major", requires_count=True),
    HealthCodeDefinition("S", "Stretched appearance", "Body is extended so the webbing between arms is under tension.", "major"),
    HealthCodeDefinition("L", "Lesions", "Open wounds; + means more lesions are assumed to be present.", "major", requires_count=True, allows_plus=True),
    HealthCodeDefinition("F", "Flat appearance", "Depressed entire body, with no or little inflation in central disk or arms.", "major"),
    HealthCodeDefinition("DR", "Droopy position", "Tissue of the central disk or arms looks sagging.", "major"),
    HealthCodeDefinition("C", "Curling arms", "Number of arms curled back on the central disk or twisted into a ball.", "major", requires_count=True),
    HealthCodeDefinition("TW", "Twisty arms", "Number of twisted arms with tube feet visible.", "major", requires_count=True),
    HealthCodeDefinition("XWV", "Extreme wavy arms", "Extreme sine-like arm shape.", "major"),
    HealthCodeDefinition("VS", "Viscera in tank", "Viscera present in the bottom of the tank.", "major"),
    HealthCodeDefinition("IN", "Inflated", "Over-inflated arms with deflated or lumpy central disk.", "major"),
    HealthCodeDefinition("DEAD", "Dead", "Individual was deemed unsavable and frozen at -80 degrees.", "status", exclusive=True, terminal=True),
    HealthCodeDefinition("RELEASED", "Released", "Individual was released back into the wild.", "status", exclusive=True, terminal=True),
]

HEALTH_CODE_BY_CODE = {d.code: d for d in HEALTH_CODE_DEFINITIONS}

# --- Image Sequence Quality Options ---
MARKER_VISIBILITY_OPTIONS = [
    CategoricalOption("Not visible", 0),
    CategoricalOption("Barely visible", 1),
    CategoricalOption("Adequately visible", 2),
    CategoricalOption("Excellently visible", 3),
]

POSTURAL_VISIBILITY_OPTIONS = [
    CategoricalOption("Very poor", 0),
    CategoricalOption("Poor", 1),
    CategoricalOption("Adequate", 2),
    CategoricalOption("Good", 3),
    CategoricalOption("Excellent", 4),
]


# =============================================================================
# FIELD DEFINITIONS (ordered by annotation workflow)
# =============================================================================

FIELD_DEFINITIONS: List[FieldDefinition] = [
    # --- Group 1: Numeric Measurements ---
    FieldDefinition(
        name="num_apparent_arms",
        display_name="Number of apparent arms",
        annotation_type=AnnotationType.NUMERIC_INT,
        group="numeric",
        min_value=0,
        max_value=30,
        tooltip="Number of visually apparent arms",
    ),
    FieldDefinition(
        name="num_total_arms",
        display_name="Total number of arms",
        annotation_type=AnnotationType.NUMERIC_INT,
        group="numeric",
        min_value=0,
        max_value=30,
        tooltip="Total number of arms including small hidden arms",
    ),
    FieldDefinition(
        name="tip_to_tip_size_cm",
        display_name="Tip-to-tip size (cm)",
        annotation_type=AnnotationType.NUMERIC_FLOAT,
        group="numeric",
        min_value=0.0,
        max_value=150.0,
        tooltip="Size of the star measured from furthest tip to tip",
    ),

    # --- Group 2: Short Arm Coding ---
    FieldDefinition(
        name="short_arm_code",
        display_name="Short arm coding",
        annotation_type=AnnotationType.MORPHOMETRIC_CODE,
        group="short_arm",
        tooltip="Arm positions and severity of short arms",
    ),

    # --- Group 2b: Health Coding ---
    FieldDefinition(
        name="health_codes",
        display_name="Health coding",
        annotation_type=AnnotationType.HEALTH_CODE,
        group="health",
        tooltip="Lab symptom/status health codes; supports multiple non-exclusive entries",
    ),

    # --- Group 3a: Stripe Morphology ---
    FieldDefinition(
        name="stripe_color",
        display_name="Stripe color",
        annotation_type=AnnotationType.COLOR_CATEGORICAL,
        group="stripe",
        tooltip="General color of arm stripes",
    ),
    FieldDefinition(
        name="stripe_order",
        display_name="Stripe order",
        annotation_type=AnnotationType.MORPH_CATEGORICAL,
        group="stripe",
        options=STRIPE_ORDER_OPTIONS,
        tooltip="Pattern regularity of stripes",
    ),
    FieldDefinition(
        name="stripe_prominence",
        display_name="Stripe prominence",
        annotation_type=AnnotationType.MORPH_CATEGORICAL,
        group="stripe",
        options=STRIPE_PROMINENCE_OPTIONS,
        tooltip="How prominent/visible the stripes are",
    ),
    FieldDefinition(
        name="stripe_extent",
        display_name="Stripe extent along arm",
        annotation_type=AnnotationType.MORPH_CATEGORICAL,
        group="stripe",
        options=STRIPE_EXTENT_OPTIONS,
        tooltip="How far stripes extend along the arm",
    ),
    FieldDefinition(
        name="stripe_thickness",
        display_name="Stripe thickness",
        annotation_type=AnnotationType.MORPH_CATEGORICAL,
        group="stripe",
        options=STRIPE_THICKNESS_OPTIONS,
        tooltip="Thickness of the radial stripes",
    ),

    # --- Group 3b: Arm Morphology ---
    FieldDefinition(
        name="arm_color",
        display_name="Arm color",
        annotation_type=AnnotationType.COLOR_CATEGORICAL,
        group="arm",
        tooltip="General color of the arms",
    ),
    FieldDefinition(
        name="arm_thickness",
        display_name="Arm thickness",
        annotation_type=AnnotationType.MORPH_CATEGORICAL,
        group="arm",
        options=ARM_THICKNESS_OPTIONS,
        tooltip="Relative thickness of arms",
    ),

    # --- Group 3c: Central Disc ---
    FieldDefinition(
        name="central_disc_color",
        display_name="Central disc color",
        annotation_type=AnnotationType.COLOR_CATEGORICAL,
        group="central_disc",
        tooltip="Color of the central disc",
    ),
    FieldDefinition(
        name="papillae_central_disc_color",
        display_name="Papillae central disc color",
        annotation_type=AnnotationType.COLOR_CATEGORICAL,
        group="central_disc",
        tooltip="Color of papillae on the central disc",
    ),

    # --- Group 3d: Rosettes ---
    FieldDefinition(
        name="rosette_color",
        display_name="Rosette color",
        annotation_type=AnnotationType.COLOR_CATEGORICAL,
        group="rosette",
        tooltip="General color of rosettes",
    ),
    FieldDefinition(
        name="rosette_prominence",
        display_name="Rosette prominence",
        annotation_type=AnnotationType.MORPH_CATEGORICAL,
        group="rosette",
        options=ROSETTE_PROMINENCE_OPTIONS,
        tooltip="How prominent/visible the rosettes are",
    ),

    # --- Group 3e: Papillae Stripes ---
    FieldDefinition(
        name="papillae_stripe_color",
        display_name="Papillae stripe color",
        annotation_type=AnnotationType.COLOR_CATEGORICAL,
        group="papillae_stripe",
        tooltip="Color of papillae in stripe regions",
    ),

    # --- Group 3f: Madreporite ---
    FieldDefinition(
        name="madreporite_color",
        display_name="Madreporite color",
        annotation_type=AnnotationType.COLOR_CATEGORICAL,
        group="madreporite",
        tooltip="Color of the madreporite",
    ),

    # --- Group 3g: Reticulation ---
    FieldDefinition(
        name="reticulation_order",
        display_name="Reticulation order",
        annotation_type=AnnotationType.MORPH_CATEGORICAL,
        group="reticulation",
        options=RETICULATION_ORDER_OPTIONS,
        tooltip="Pattern type of reticulation",
    ),

    # --- Group 3h: Overall ---
    FieldDefinition(
        name="overall_color",
        display_name="Overall color",
        annotation_type=AnnotationType.COLOR_CATEGORICAL,
        group="overall",
        tooltip="Overall color impression of the individual",
    ),

    # --- Group 4: Image Sequence Quality ---
    FieldDefinition(
        name="madreporite_visibility",
        display_name="Madreporite visibility",
        annotation_type=AnnotationType.MORPH_CATEGORICAL,
        group="image_quality",
        options=MARKER_VISIBILITY_OPTIONS,
        tooltip="Visibility of the madreporite marker for bilateral symmetry identification",
    ),
    FieldDefinition(
        name="anus_visibility",
        display_name="Anus visibility",
        annotation_type=AnnotationType.MORPH_CATEGORICAL,
        group="image_quality",
        options=MARKER_VISIBILITY_OPTIONS,
        tooltip="Visibility of the anus marker for bilateral symmetry identification",
    ),
    FieldDefinition(
        name="postural_visibility",
        display_name="Postural visibility",
        annotation_type=AnnotationType.MORPH_CATEGORICAL,
        group="image_quality",
        options=POSTURAL_VISIBILITY_OPTIONS,
        tooltip="Quality of the star's posture for re-identification purposes",
    ),

    # --- Group 5a: Location ---
    FieldDefinition(
        name="location",
        display_name="Location",
        annotation_type=AnnotationType.TEXT_HISTORY,
        group="location",
        tooltip="Written description of the star's location",
    ),
    FieldDefinition(
        name="latitude",
        display_name="Latitude",
        annotation_type=AnnotationType.NUMERIC_FLOAT,
        group="location",
        min_value=-90.0,
        max_value=90.0,
        tooltip="Latitude in decimal degrees (WGS84)",
    ),
    FieldDefinition(
        name="longitude",
        display_name="Longitude",
        annotation_type=AnnotationType.NUMERIC_FLOAT,
        group="location",
        min_value=-180.0,
        max_value=180.0,
        tooltip="Longitude in decimal degrees (WGS84)",
    ),
    # --- Group 5b: Text Annotations ---
    FieldDefinition(
        name="unusual_observation",
        display_name="Unusual observation",
        annotation_type=AnnotationType.TEXT_FREE,
        group="notes",
        tooltip="Any unusual observations about the star",
    ),
    FieldDefinition(
        name="health_observation",
        display_name="Health observation",
        annotation_type=AnnotationType.TEXT_FREE,
        group="notes",
        tooltip="Observations about the star's health",
    ),

    # --- Group 6: Morphometric Tool Measurements (Auto-populated) ---
    FieldDefinition(
        name="morph_num_arms",
        display_name="Arms detected",
        annotation_type=AnnotationType.NUMERIC_INT,
        group="morphometric_auto",
        min_value=0,
        max_value=30,
        tooltip="Number of arms detected by YOLO model (auto-populated from morphometric tool)",
    ),
    FieldDefinition(
        name="morph_area_mm2",
        display_name="Area (mm²)",
        annotation_type=AnnotationType.NUMERIC_FLOAT,
        group="morphometric_auto",
        min_value=0.0,
        max_value=100000.0,
        tooltip="Calibrated surface area in square millimeters (auto-populated)",
    ),
    FieldDefinition(
        name="morph_major_axis_mm",
        display_name="Major axis (mm)",
        annotation_type=AnnotationType.NUMERIC_FLOAT,
        group="morphometric_auto",
        min_value=0.0,
        max_value=1500.0,
        tooltip="Fitted ellipse major axis length in mm (auto-populated)",
    ),
    FieldDefinition(
        name="morph_minor_axis_mm",
        display_name="Minor axis (mm)",
        annotation_type=AnnotationType.NUMERIC_FLOAT,
        group="morphometric_auto",
        min_value=0.0,
        max_value=1500.0,
        tooltip="Fitted ellipse minor axis length in mm (auto-populated)",
    ),
    FieldDefinition(
        name="morph_mean_arm_length_mm",
        display_name="Mean arm length (mm)",
        annotation_type=AnnotationType.NUMERIC_FLOAT,
        group="morphometric_auto",
        min_value=0.0,
        max_value=750.0,
        tooltip="Average length of all detected arms in mm (auto-populated)",
    ),
    FieldDefinition(
        name="morph_max_arm_length_mm",
        display_name="Max arm length (mm)",
        annotation_type=AnnotationType.NUMERIC_FLOAT,
        group="morphometric_auto",
        min_value=0.0,
        max_value=750.0,
        tooltip="Length of the longest detected arm in mm (auto-populated)",
    ),
    FieldDefinition(
        name="morph_tip_to_tip_mm",
        display_name="Tip-to-tip diameter (mm)",
        annotation_type=AnnotationType.NUMERIC_FLOAT,
        group="morphometric_auto",
        min_value=0.0,
        max_value=1500.0,
        tooltip="Maximum diameter between opposing arm tips in mm (auto-populated)",
    ),
    FieldDefinition(
        name="morph_source_folder",
        display_name="Source mFolder",
        annotation_type=AnnotationType.TEXT_FREE,
        group="morphometric_auto",
        tooltip="Path to the source mFolder containing full morphometric data",
    ),

    # --- Group 7: Sync Metadata (Auto-populated) ---
    FieldDefinition(
        name="last_modified_utc",
        display_name="Last modified (UTC)",
        annotation_type=AnnotationType.TEXT_FREE,
        group="sync",
        tooltip="ISO 8601 timestamp of the last metadata save (auto-populated)",
    ),
    FieldDefinition(
        name="modified_by_lab",
        display_name="Modified by lab",
        annotation_type=AnnotationType.TEXT_FREE,
        group="sync",
        tooltip="Lab/machine identifier that last modified this record (auto-populated)",
    ),
    FieldDefinition(
        name="source_lab",
        display_name="Source lab",
        annotation_type=AnnotationType.TEXT_FREE,
        group="sync",
        tooltip="Lab/machine that originally created this record (auto-populated, set once)",
    ),
]

# Build lookup dict for quick access
FIELD_BY_NAME = {f.name: f for f in FIELD_DEFINITIONS}


# =============================================================================
# FIELD GROUPS (for UI layout)
# =============================================================================

FIELD_GROUPS: List[FieldGroup] = [
    FieldGroup(
        name="location",
        display_name="Location",
        fields=["location", "latitude", "longitude"],
        start_expanded=True,
    ),
    FieldGroup(
        name="numeric",
        display_name="Numeric Measurements",
        fields=["num_apparent_arms", "num_total_arms", "tip_to_tip_size_cm"],
        start_expanded=True,
    ),
    FieldGroup(
        name="short_arm",
        display_name="Short Arm Coding",
        fields=["short_arm_code"],
        start_expanded=True,
    ),
    FieldGroup(
        name="health",
        display_name="Health Coding",
        fields=["health_codes"],
        start_expanded=True,
    ),
    FieldGroup(
        name="stripe",
        display_name="Stripe Morphology",
        fields=["stripe_color", "stripe_order", "stripe_prominence", "stripe_extent", "stripe_thickness"],
        start_expanded=True,
    ),
    FieldGroup(
        name="arm",
        display_name="Arm Morphology",
        fields=["arm_color", "arm_thickness"],
        start_expanded=True,
    ),
    FieldGroup(
        name="central_disc",
        display_name="Central Disc",
        fields=["central_disc_color", "papillae_central_disc_color"],
        start_expanded=True,
    ),
    FieldGroup(
        name="rosette",
        display_name="Rosettes",
        fields=["rosette_color", "rosette_prominence"],
        start_expanded=True,
    ),
    FieldGroup(
        name="papillae_stripe",
        display_name="Papillae Stripes",
        fields=["papillae_stripe_color"],
        start_expanded=False,
    ),
    FieldGroup(
        name="madreporite",
        display_name="Madreporite",
        fields=["madreporite_color"],
        start_expanded=False,
    ),
    FieldGroup(
        name="reticulation",
        display_name="Reticulation",
        fields=["reticulation_order"],
        start_expanded=True,
    ),
    FieldGroup(
        name="overall",
        display_name="Overall Appearance",
        fields=["overall_color"],
        start_expanded=True,
    ),
    FieldGroup(
        name="image_quality",
        display_name="Image Sequence Quality",
        fields=["madreporite_visibility", "anus_visibility", "postural_visibility"],
        start_expanded=True,
    ),
    FieldGroup(
        name="notes",
        display_name="Notes",
        fields=["unusual_observation", "health_observation"],
        start_expanded=True,
    ),
    FieldGroup(
        name="morphometric_auto",
        display_name="Morphometric Measurements (Auto)",
        fields=[
            "morph_num_arms",
            "morph_area_mm2",
            "morph_major_axis_mm",
            "morph_minor_axis_mm",
            "morph_mean_arm_length_mm",
            "morph_max_arm_length_mm",
            "morph_tip_to_tip_mm",
            "morph_source_folder",
        ],
        start_expanded=False,  # Collapsed by default since auto-populated
    ),
    FieldGroup(
        name="sync",
        display_name="Sync Metadata (Auto)",
        fields=["last_modified_utc", "modified_by_lab", "source_lab"],
        start_expanded=False,  # Collapsed by default since auto-populated
    ),
]

GROUP_BY_NAME = {g.name: g for g in FIELD_GROUPS}


# =============================================================================
# CSV HEADER (for new schema)
# =============================================================================

def get_csv_header(include_id: bool = True, id_column: str = "gallery_id") -> List[str]:
    """Get the CSV header for the new annotation schema."""
    header = []
    if include_id:
        header.append(id_column)
    header.extend(f.name for f in FIELD_DEFINITIONS)
    return header


GALLERY_HEADER_V2 = get_csv_header(include_id=True, id_column="gallery_id")
QUERIES_HEADER_V2 = get_csv_header(include_id=True, id_column="query_id")


# =============================================================================
# SHORT ARM CODE PARSING/SERIALIZATION
# =============================================================================

@dataclass
class ShortArmEntry:
    """A single short arm entry."""
    position: int
    severity: str  # "tiny", "small", or "short"

    def to_string(self) -> str:
        """Serialize to string format like 'tiny(2)' or 'short(3)'."""
        return f"{self.severity}({self.position})"

    @classmethod
    def from_string(cls, s: str) -> Optional["ShortArmEntry"]:
        """Parse from string format like 'tiny(2)', '2**', '(3)', etc."""
        s = s.strip()
        if not s:
            return None

        # New format: severity(position)
        import re
        new_fmt = re.match(r"(very_tiny|tiny|small|short)\((\d+)\)", s, re.IGNORECASE)
        if new_fmt:
            return cls(position=int(new_fmt.group(2)), severity=new_fmt.group(1).lower())

        # Legacy format: position with asterisks or parentheses
        # Examples: 2**, 3, 10*, 11**, (3), 2***, 6(r)
        legacy = re.match(r"\(?(\d+)\)?(\*{0,3})(?:\(r\))?", s)
        if legacy:
            pos = int(legacy.group(1))
            stars = legacy.group(2)
            # Map asterisks to severity
            if len(stars) >= 2:
                severity = "tiny"
            elif len(stars) == 1:
                severity = "small"
            else:
                severity = "short"
            # Parentheses indicate very small/borderline - treat as tiny
            if s.startswith("(") and s.endswith(")"):
                severity = "tiny"
            return cls(position=pos, severity=severity)

        return None


def parse_short_arm_code(code_str: str) -> List[ShortArmEntry]:
    """Parse a short arm code string into list of entries."""
    if not code_str or not code_str.strip():
        return []

    entries = []
    # Split by comma, handling both old and new formats
    parts = [p.strip() for p in code_str.split(",")]
    for part in parts:
        entry = ShortArmEntry.from_string(part)
        if entry:
            entries.append(entry)

    return entries


def serialize_short_arm_code(entries: List[ShortArmEntry]) -> str:
    """Serialize list of short arm entries to string."""
    if not entries:
        return ""
    # Sort by position for consistent output
    sorted_entries = sorted(entries, key=lambda e: e.position)
    return ", ".join(e.to_string() for e in sorted_entries)


@dataclass
class HealthCodeEntry:
    """A single selected health/symptom code."""
    code: str
    count: Optional[int] = None
    plus: bool = False

    def to_string(self) -> str:
        code = self.code.upper().strip()
        out = code
        if self.count is not None:
            out += f"({self.count})"
        if self.plus:
            out += "+"
        return out


_HEALTH_CODES_BY_LENGTH = sorted(HEALTH_CODE_BY_CODE, key=len, reverse=True)


def _parse_health_code_part(part: str) -> Optional[HealthCodeEntry]:
    raw = part.strip().strip(".;")
    if not raw:
        return None
    compact = raw.upper().replace(" ", "")
    for code in _HEALTH_CODES_BY_LENGTH:
        if not compact.startswith(code):
            continue
        rest = compact[len(code):]
        count: Optional[int] = None
        plus = False
        if rest.startswith("("):
            close = rest.find(")")
            if close >= 0:
                inner = rest[1:close]
                if inner.endswith("+"):
                    plus = True
                    inner = inner[:-1]
                if inner.isdigit():
                    count = int(inner)
                rest = rest[close + 1:]
        if rest.startswith("+"):
            plus = True
        definition = HEALTH_CODE_BY_CODE[code]
        if not definition.requires_count:
            count = None
        if not definition.allows_plus:
            plus = False
        return HealthCodeEntry(code=code, count=count, plus=plus)
    return None


def parse_health_codes(code_str: str) -> List[HealthCodeEntry]:
    """Parse comma-separated lab health codes into canonical entries."""
    if not code_str or not code_str.strip():
        return []
    entries: List[HealthCodeEntry] = []
    for part in code_str.split(","):
        entry = _parse_health_code_part(part)
        if entry:
            entries.append(entry)
    return entries


def serialize_health_codes(entries: List[HealthCodeEntry]) -> str:
    """Serialize health code entries for CSV storage."""
    return ", ".join(entry.to_string() for entry in entries if entry.code.strip())


# =============================================================================
# DEFAULT COLOR VOCABULARIES
# =============================================================================

DEFAULT_COLORS = [
    # Basic colors
    "white",
    "yellow",
    "orange",
    "peach",
    "pink",
    "red",
    "maroon",
    "burgundy",
    "purple",
    "mauve",
    "brown",
    "tan",
    # Compound colors
    "light-purple",
    "light-brown",
    "light-orange",
    "dark-purple",
    "dark-brown",
    "dark-orange",
    "burnt-orange",
    "bright-orange",
    "rusty-orange",
    "white-yellow",
    "yellow-white",
    "brown-orange",
    "brown-mauve",
    "purple-maroon",
    "pink-maroon",
    "burgundy-mauve",
]



