# starBoard Interaction Widget Map

This document maps all interactive widgets in the starBoard UI that should be logged for user analytics.

**Legend:**
- ✅ = Currently logged
- ❌ = NOT logged (needs implementation)
- 🔶 = Partially logged (logged via signal in parent)

---

## MainWindow (`src/ui/main_window.py`)

| Widget | Type | Event | Handler | Status |
|--------|------|-------|---------|--------|
| `_tabs` | QTabWidget | currentChanged | `_on_tab_changed` | ✅ |

---

## TabFirstOrder (`src/ui/tab_first_order.py`)

### Controls Row 1

| Widget | Type | Event | Handler | Status |
|--------|------|-------|---------|--------|
| `cmb_query` | QComboBox | currentIndexChanged | `_on_query_changed` | ✅ |
| `btn_prev_query` | QPushButton | clicked | `_on_prev_query_clicked` | ❌ |
| `btn_next_query` | QPushButton | clicked | `_on_next_query_clicked` | ❌ |
| `cmb_preset` | QComboBox | currentIndexChanged | `_apply_preset` | ✅ |
| `spin_topk` | QSpinBox | valueChanged | (no handler) | ❌ |
| `btn_rebuild` | QPushButton | clicked | `_on_rebuild` | ✅ |
| `btn_refresh` | QPushButton | clicked | `_refresh_results` | ❌ |
| `btn_exclude` | QPushButton | clicked | `_open_exclude_dialog` | ✅ |
| `btn_config` | QPushButton | clicked | `_open_config_dialog` | ✅ |

### Controls Row 2

| Widget | Type | Event | Handler | Status |
|--------|------|-------|---------|--------|
| `chk_date` | QCheckBox | toggled | `_on_date_filter_changed` | ✅ |
| `date_from` | QDateEdit | dateChanged | `_on_date_filter_changed` | ✅ |
| `date_to` | QDateEdit | dateChanged | `_on_date_filter_changed` | ✅ |
| `chk_include_nodate` | QCheckBox | toggled | `_on_date_filter_changed` | ✅ |
| `chk_visual` | QCheckBox | toggled | `_on_visual_toggled` | ✅ |
| `cmb_model` | QComboBox | currentIndexChanged | `_on_model_changed` | ✅ |
| `cmb_visual_mode` | QComboBox | currentIndexChanged | `_on_visual_mode_changed` | ✅ |
| `btn_refresh_visual` | QPushButton | clicked | `_on_refresh_visual` | ❌ |
| `chk_roll_to_closest` | QCheckBox | toggled | `_on_roll_to_closest_toggled` | ❌ |
| `spin_roll_limit` | QSpinBox | valueChanged | `_on_roll_to_closest_toggled` | ❌ |
| `slider_fusion` | QSlider | valueChanged | `_on_fusion_changed` | ✅ |

### Query Panel

| Widget | Type | Event | Handler | Status |
|--------|------|-------|---------|--------|
| `btn_open_query` | QPushButton | clicked | `_open_query_folder` | ✅ |
| `btn_best_query` | QPushButton | clicked | `_on_set_best_query` | ✅ |
| `btn_meta_query` | QPushButton | clicked | `_show_query_metadata` | ❌ |
| `query_quality_panel` | ImageQualityPanel | saved | `_on_query_quality_saved` | ❌ |

### Gallery Navigation

| Widget | Type | Event | Handler | Status |
|--------|------|-------|---------|--------|
| `cmb_gallery_search` | QComboBox | currentIndexChanged | `_on_gallery_search_changed` | ❌ |
| `btn_gallery_prev` | QPushButton | clicked | `_on_gallery_prev_clicked` | ❌ |
| `btn_gallery_next` | QPushButton | clicked | `_on_gallery_next_clicked` | ❌ |

### Fields Panel (Checkboxes)

| Widget | Type | Event | Handler | Status |
|--------|------|-------|---------|--------|
| `chk_by_name[*]` | Dict[str, QCheckBox] | stateChanged | (no direct handler) | ❌ |

### Numeric Offsets Panel

| Widget | Type | Event | Handler | Status |
|--------|------|-------|---------|--------|
| `_offset_widgets[*]` | Dict[str, QSpinBox/QDoubleSpinBox] | valueChanged | `_on_offsets_changed` | ❌ |
| `btn_reset_offsets` | QPushButton | clicked | `_reset_offsets` | ❌ |

### Collapsible Sections

| Widget | Type | Event | Handler | Status |
|--------|------|-------|---------|--------|
| `filters_section` | CollapsibleSection | toggled | `_on_filters_toggled` | ❌ |

---

## LineupCard (`src/ui/lineup_card.py`)

| Widget | Type | Event | Handler | Status |
|--------|------|-------|---------|--------|
| `btn_pin` | QPushButton | clicked | (via parent signal) | 🔶 |
| `cmb_verdict` | QComboBox | (no direct handler) | - | ❌ |
| `btn_save_decision` | QPushButton | clicked | `_on_save_decision` | 🔶 |
| `btn_open` | QPushButton | clicked | `_open_folder` | ✅ |
| `btn_best` | QPushButton | clicked | `_on_set_best_gallery` | ✅ |
| `btn_meta` | QPushButton | clicked | (emits signal) | ❌ |

---

## TabSecondOrder (`src/ui/tab_second_order.py`)

### Controls Row 1

| Widget | Type | Event | Handler | Status |
|--------|------|-------|---------|--------|
| `cmb_query` | QComboBox | currentIndexChanged | `_on_query_changed` | ✅ |
| `btn_prev_query` | QPushButton | clicked | `_on_prev_query_clicked` | ❌ |
| `btn_next_query` | QPushButton | clicked | `_on_next_query_clicked` | ❌ |
| `cmb_gallery` | QComboBox | currentIndexChanged | `_on_gallery_changed` | ✅ |
| `cmb_recommended` | QComboBox | currentIndexChanged | `_on_recommended_changed` | ❌ |

### Query Panel

| Widget | Type | Event | Handler | Status |
|--------|------|-------|---------|--------|
| `btn_best_q` | QPushButton | clicked | `_on_set_best_query` | ❌ |
| `btn_open_q` | QPushButton | clicked | (opens folder) | ❌ |
| `query_quality_panel` | ImageQualityPanel | saved | `_on_query_quality_saved` | ❌ |

### Gallery Panel

| Widget | Type | Event | Handler | Status |
|--------|------|-------|---------|--------|
| `btn_best_g` | QPushButton | clicked | `_on_set_best_gallery` | ❌ |
| `btn_open_g` | QPushButton | clicked | (opens folder) | ❌ |
| `gallery_quality_panel` | ImageQualityPanel | saved | `_on_gallery_quality_saved` | ❌ |

### Decision Controls

| Widget | Type | Event | Handler | Status |
|--------|------|-------|---------|--------|
| `cmb_verdict` | QComboBox | (no direct handler) | - | ❌ |
| `edit_notes` | QLineEdit | (no direct handler) | - | ❌ |
| `btn_save` | QPushButton | clicked | `_on_save_decision` | ✅ |

---

## TabSetup (`src/ui/tab_setup.py`)

### Single Upload Mode

| Widget | Type | Event | Handler | Status |
|--------|------|-------|---------|--------|
| `btn_choose_files` | QPushButton | clicked | `_on_choose_files` | ❌ |
| `chk_move` | QCheckBox | (no handler) | - | ❌ |
| `chk_metadata_only` | QCheckBox | toggled | `_on_metadata_only_toggled` | ❌ |
| `cmb_target` | QComboBox | currentIndexChanged | `_refresh_id_list_single` | ❌ |
| `cmb_id` | QComboBox | currentIndexChanged | `_on_id_selection_changed_single` | ❌ |
| `edit_new_id` | QLineEdit | (no direct handler) | - | ❌ |
| `date_encounter` | QDateEdit | dateChanged | `_update_encounter_preview` | ❌ |
| `edit_suffix` | QLineEdit | textChanged | `_update_encounter_preview` | ❌ |
| `btn_save_single` | QPushButton | clicked | `_on_save_single` | ✅ |

### Batch Upload Mode

| Widget | Type | Event | Handler | Status |
|--------|------|-------|---------|--------|
| `cmb_target_batch` | QComboBox | (no handler) | - | ❌ |
| `btn_discover` | QPushButton | clicked | `_on_discover` | ❌ |
| `date_batch` | QDateEdit | (no handler) | - | ❌ |
| `edit_suffix_batch` | QLineEdit | (no handler) | - | ❌ |
| `btn_start_batch` | QPushButton | clicked | `_on_start_batch` | ✅ |

### Metadata Edit Mode

| Widget | Type | Event | Handler | Status |
|--------|------|-------|---------|--------|
| `cmb_target_edit` | QComboBox | currentIndexChanged | `_refresh_id_list_edit` | ❌ |
| `cmb_id_edit` | QComboBox | currentIndexChanged | `_on_edit_id_changed` | ❌ |
| `btn_prev_id_edit` | QPushButton | clicked | `_on_prev_id_edit_clicked` | ❌ |
| `btn_next_id_edit` | QPushButton | clicked | `_on_next_id_edit_clicked` | ❌ |
| `btn_save_only` | QPushButton | clicked | `_on_save_only` | ❌ |
| `btn_save_edit` | QPushButton | clicked | `_on_save_edits` | ✅ |
| `btn_set_best_edit` | QPushButton | clicked | `_on_set_best_edit` | ❌ |

### ImageViewer (nested)

| Widget | Type | Event | Handler | Status |
|--------|------|-------|---------|--------|
| `btn_best` | QPushButton | clicked | `_on_best_clicked` | ❌ |

---

## TabDeepLearning (`src/ui/tab_dl.py`)

### Model Management

| Widget | Type | Event | Handler | Status |
|--------|------|-------|---------|--------|
| `list_models` | QListWidget | currentRowChanged | `_on_model_selection_changed` | ❌ |
| `btn_set_active` | QPushButton | clicked | `_on_set_active` | ❌ |
| `btn_set_default` | QPushButton | clicked | `_on_set_default_model` | ❌ |
| `btn_import` | QPushButton | clicked | `_on_import_model` | ❌ |
| `btn_remove` | QPushButton | clicked | `_on_remove_model` | ❌ |

### Precomputation Controls

| Widget | Type | Event | Handler | Status |
|--------|------|-------|---------|--------|
| `chk_gallery` | QCheckBox | (no handler) | - | ❌ |
| `chk_queries` | QCheckBox | (no handler) | - | ❌ |
| `chk_tta` | QCheckBox | (no handler) | - | ❌ |
| `chk_reranking` | QCheckBox | (no handler) | - | ❌ |
| `spin_batch` | QSpinBox | (no handler) | - | ❌ |
| `cmb_speed_mode` | QComboBox | currentIndexChanged | `_on_speed_mode_changed` | ❌ |
| `btn_precompute` | QPushButton | clicked | `_on_precompute_selected` | ❌ |
| `btn_full_precompute` | QPushButton | clicked | `_on_run_full_precompute` | ❌ |
| `btn_update_pending` | QPushButton | clicked | `_on_update_pending` | ❌ |
| `btn_cancel` | QPushButton | clicked | `_on_cancel` | ✅ |

### Visualization

| Widget | Type | Event | Handler | Status |
|--------|------|-------|---------|--------|
| `btn_viz_identity` | QPushButton | clicked | `_on_viz_identity` | ❌ |
| `btn_viz_image` | QPushButton | clicked | `_on_viz_image` | ❌ |
| `btn_configure_tsne` | QPushButton | clicked | `_on_configure_tsne` | ❌ |

### Training (Advanced)

| Widget | Type | Event | Handler | Status |
|--------|------|-------|---------|--------|
| `btn_start_training` | QPushButton | clicked | `_on_start_training` | ❌ |

---

## Analytics & History Tab - TabPastMatches (`src/ui/tab_past_matches.py`)

### Utility Bar

| Widget | Type | Event | Handler | Status |
|--------|------|-------|---------|--------|
| `btn_refresh` | QPushButton | clicked | `_refresh` | ❌ |
| `btn_export_master` | QPushButton | clicked | `_export_master` | ❌ |
| `btn_export_summaries` | QPushButton | clicked | `_export_summaries` | ❌ |
| `btn_open_reports` | QPushButton | clicked | `_open_reports` | ❌ |

### Visualizations

| Widget | Type | Event | Handler | Status |
|--------|------|-------|---------|--------|
| `btn_totals` | QPushButton | clicked | `_show_totals` | ❌ |
| `btn_timeline` | QPushButton | clicked | `_show_timeline` | ❌ |
| `btn_by_query` | QPushButton | clicked | `_show_by_query` | ❌ |
| `btn_by_gallery` | QPushButton | clicked | `_show_by_gallery` | ❌ |
| `btn_matrix` | QPushButton | clicked | `_show_matrix` | ❌ |
| `btn_export_tidy` | QPushButton | clicked | `_export_tidy` | ❌ |

### Merge Controls

| Widget | Type | Event | Handler | Status |
|--------|------|-------|---------|--------|
| `cmb_gallery_merge` | QComboBox | currentIndexChanged | (refresh merge) | ❌ |
| `btn_merge` | QPushButton | clicked | `_on_merge` | ✅ |

### Revert Controls

| Widget | Type | Event | Handler | Status |
|--------|------|-------|---------|--------|
| `cmb_revert_gallery` | QComboBox | currentIndexChanged | `_on_revert_gallery_changed` | ❌ |
| `cmb_revert_batch` | QComboBox | (no handler) | - | ❌ |
| `btn_revert_batch` | QPushButton | clicked | `_on_revert_selected_batch` | ❌ |
| `btn_open_history` | QPushButton | clicked | `_on_open_history_rev` | ❌ |

---

## ImageStrip (`src/ui/image_strip.py`)

| Widget | Type | Event | Handler | Status |
|--------|------|-------|---------|--------|
| (image navigation) | QGraphicsView | wheelEvent / click | - | ❌ |
| (image toggle best/closest) | - | click | `_toggle_image` | ❌ |

---

## ImageQualityPanel (`src/ui/image_quality_panel.py`)

| Widget | Type | Event | Handler | Status |
|--------|------|-------|---------|--------|
| (quality combos) | QComboBox | currentIndexChanged | `_on_value_changed` | ❌ |
| `btn_save` | QPushButton | clicked | `_on_save_clicked` | ❌ |

---

## AnnotatorViewSecond (`src/ui/annotator_view_second.py`)

| Widget | Type | Event | Handler | Status |
|--------|------|-------|---------|--------|
| (tool selection) | QToolButton | clicked | - | ❌ |
| (zoom/pan) | - | - | - | ❌ |
| (adjustment sliders) | QSlider | valueChanged | `_on_adjust_changed` | ❌ |
| (stack navigation) | - | - | `_on_stack_changed` | ❌ |
| (point/vertex drag) | - | - | `_on_point_drag` / `_on_vertex_drag` | ❌ |

---

## FieldsConfigDialog (`src/ui/fields_config_dialog.py`)

| Widget | Type | Event | Handler | Status |
|--------|------|-------|---------|--------|
| (field enable checkboxes) | QCheckBox | toggled | `_on_toggle` | ❌ |
| (weight spinboxes) | QDoubleSpinBox | valueChanged | - | ❌ |
| `btn_save` | QPushButton | clicked | `_on_save` | ❌ |
| `btn_reset` | QPushButton | clicked | `_on_reset` | ❌ |

---

## ColorPickerDialog (`src/ui/color_picker.py`)

| Widget | Type | Event | Handler | Status |
|--------|------|-------|---------|--------|
| (color wheel) | custom | colorChanged | `_on_wheel_changed` | ❌ |
| (value slider) | QSlider | valueChanged | `_on_value_changed` | ❌ |
| (RGB spinboxes) | QSpinBox | valueChanged | `_on_rgb_changed` | ❌ |
| (hex input) | QLineEdit | textEdited | `_on_hex_edited` | ❌ |
| (eyedropper) | QPushButton | clicked | `_on_eyedropper_clicked` | ❌ |

---

## MetadataFormV2 / AnnotationWidgets (`src/ui/metadata_form_v2.py`, `src/ui/annotation_widgets.py`)

| Widget | Type | Event | Handler | Status |
|--------|------|-------|---------|--------|
| (all annotation inputs) | various | valueChanged | `_on_value_changed` | ❌ |
| (color picker buttons) | QPushButton | clicked | `_on_picker_button_clicked` | ❌ |
| (selection combos) | QComboBox | activated | `_on_selection_changed` | ❌ |

---

## Analytics & History Dialogs - VisPastMatches (`src/ui/vis_past_matches.py`)

| Widget | Type | Event | Handler | Status |
|--------|------|-------|---------|--------|
| `btn_refresh` | QPushButton | clicked | `_on_refresh` | ❌ |
| `btn_export_png` | QPushButton | clicked | `_on_export_png` | ❌ |
| `btn_export_csv` | QPushButton | clicked | `_on_export_csv` | ❌ |

---

## MatrixMatchesDialog (`src/ui/matrix_matches_dialog.py`)

| Widget | Type | Event | Handler | Status |
|--------|------|-------|---------|--------|
| (cell clicks) | QTableWidget | cellClicked | - | ❌ |
| (export buttons) | QPushButton | clicked | - | ❌ |

---

# Summary Statistics

| Category | Total Widgets | Currently Logged | Not Logged |
|----------|---------------|------------------|------------|
| **Buttons** | ~65 | 15 | ~50 |
| **ComboBoxes** | ~25 | 8 | ~17 |
| **CheckBoxes** | ~15 | 3 | ~12 |
| **Sliders** | ~5 | 1 | ~4 |
| **SpinBoxes** | ~10 | 0 | ~10 |
| **DateEdits** | ~5 | 1 | ~4 |
| **Other** | ~20 | 0 | ~20 |
| **TOTAL** | ~145 | ~28 | ~117 |

---

# Priority for Implementation

## High Priority (Core Workflow)
1. All navigation buttons (prev/next query, prev/next gallery)
2. All "Refresh" buttons
3. Checkbox toggles that affect ranking/filtering
4. ID selection combos in Setup tab
5. Visualization dialog opens

## Medium Priority (Secondary Workflow)
1. Offset spinbox changes
2. Field checkbox toggles
3. Quality panel interactions
4. Collapsible section toggles

## Low Priority (Auxiliary)
1. Color picker interactions
2. Annotation widget changes
3. Internal dialog interactions
4. Toolbar tool selections

---

*Generated for starBoard interaction logging system*









