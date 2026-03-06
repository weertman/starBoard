# starBoard Interaction Logging Gaps

This document tracks current known gaps in UI interaction logging.

It is intentionally scoped to the highest-signal gaps. It does not attempt a
full widget inventory, because that approach drifts quickly as the UI evolves.

Implementation truth lives in:

- `_ilog.log(...)` call sites under `src/ui/`
- `src/utils/interaction_logger.py`

Reviewed on 2026-03-06.

## Current Coverage Snapshot

The following areas already have substantial visible logging coverage and do
not need a separate widget-by-widget map:

- `src/ui/main_window.py`
- `src/ui/tab_first_order.py`
- `src/ui/tab_setup.py`
- `src/ui/tab_second_order.py`
- `src/ui/tab_gallery_review.py`
- `src/ui/tab_past_matches.py`
- `src/ui/tab_dl.py`
- `src/ui/lineup_card.py`

## Highest-Priority Gaps

### `src/ui/tab_morphometric.py`

Current visible logging is minimal. Only the camera configuration button has an
obvious `_ilog.log(...)` call.

Focus next on:

- capture and import actions
- measurement and analysis actions
- save and export actions
- key workflow selectors and toggles

### `src/ui/annotator_view_second.py`

No visible interaction logging was found for the shared annotation viewer.

Focus next on:

- tool selection
- zoom and pan interactions
- stack navigation
- point and vertex drag operations

### `src/ui/image_strip.py`

No visible interaction logging was found for image-strip interactions.

Focus next on:

- image navigation
- image toggle actions such as best or closest selection

### `src/ui/image_quality_panel.py`

The save action is logged, but per-field combo changes are not visibly logged.

Focus next on:

- `madreporite_visibility`
- `anus_visibility`
- `postural_visibility`

### `src/ui/color_picker.py`

No visible interaction logging was found for the color picker workflow.

Focus next on:

- color wheel changes
- value slider changes
- RGB spinbox changes
- hex input edits
- eyedropper use

## Partial-Coverage Areas

### `src/ui/fields_config_dialog.py`

Save and reset actions are logged, but fine-grained configuration edits are not
fully visible from logging calls.

Review whether individual field toggles and weight changes should be logged, or
whether save/reset events are sufficient for analytics.

## Refresh Process

When this document needs an update:

1. Search for `_ilog.log(...)` usage under `src/ui/`.
2. Compare shared widgets and newer tabs against their interactive controls.
3. Update only this gap list and the priority notes.
4. Avoid rebuilding a full static widget map unless it can be generated from
   code.
