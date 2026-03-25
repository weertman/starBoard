from __future__ import annotations

"""Shared field group constants for first-order search.

This module stays dependency-light so config/bootstrap code can access the
canonical field lists without importing the full search engine and its optional
embedding dependencies.
"""

# Numeric fields - continuous measurements
NUMERIC_FIELDS = [
    "num_apparent_arms",
    "num_total_arms",
    "tip_to_tip_size_cm",
]

# Ordinal categorical fields - stored as numeric values (0, 1, 2, 3...)
ORDINAL_FIELDS = [
    "stripe_order",
    "stripe_prominence",
    "stripe_extent",
    "stripe_thickness",
    "arm_thickness",
    "rosette_prominence",
    "reticulation_order",
]

# Color categorical fields
COLOR_FIELDS = [
    "stripe_color",
    "arm_color",
    "central_disc_color",
    "papillae_central_disc_color",
    "rosette_color",
    "papillae_stripe_color",
    "madreporite_color",
    "overall_color",
]

# Set/code fields
SET_FIELDS = [
    "short_arm_code",
]

# Free text fields
TEXT_FIELDS = [
    "location",
    "unusual_observation",
    "health_observation",
]

# All fields combined for iteration
ALL_FIELDS = NUMERIC_FIELDS + ORDINAL_FIELDS + COLOR_FIELDS + SET_FIELDS + TEXT_FIELDS
