# src/ui/fields_config_dialog.py
"""
Dialog for editing fields_config.yaml settings.

Provides a UI for:
- Enabling/disabling fields per type
- Setting per-field weights
- Editing global settings (equalize_weights, default_preset, default_top_k)
- Editing scorer parameters (advanced)
"""
from __future__ import annotations

from typing import Dict, List, Optional, Any
from pathlib import Path

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QTabWidget, QWidget,
    QTableWidget, QTableWidgetItem, QHeaderView, QCheckBox,
    QDoubleSpinBox, QSpinBox, QComboBox, QLabel, QPushButton,
    QGroupBox, QFormLayout, QLineEdit, QDialogButtonBox,
    QMessageBox, QScrollArea, QFrame, QToolButton, QSizePolicy,
)
from PySide6.QtGui import QFont

from src.data.fields_config import (
    get_fields_config, FieldsConfig, FieldDef, ScorerParams,
)


class _CollapsibleSection(QWidget):
    """Simple collapsible section widget."""
    
    def __init__(self, title: str, parent: QWidget | None = None):
        super().__init__(parent)
        self._expanded = False
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)
        
        # Header button
        self.toggle_btn = QToolButton()
        self.toggle_btn.setStyleSheet("QToolButton { border: none; font-weight: bold; }")
        self.toggle_btn.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        self.toggle_btn.setArrowType(Qt.RightArrow)
        self.toggle_btn.setText(title)
        self.toggle_btn.setCheckable(True)
        self.toggle_btn.toggled.connect(self._on_toggle)
        layout.addWidget(self.toggle_btn)
        
        # Content area
        self.content = QWidget()
        self.content.setVisible(False)
        self.content_layout = QVBoxLayout(self.content)
        self.content_layout.setContentsMargins(20, 4, 0, 4)
        layout.addWidget(self.content)
    
    def _on_toggle(self, checked: bool):
        self._expanded = checked
        self.toggle_btn.setArrowType(Qt.DownArrow if checked else Qt.RightArrow)
        self.content.setVisible(checked)
    
    def addWidget(self, widget: QWidget):
        self.content_layout.addWidget(widget)
    
    def addLayout(self, layout):
        self.content_layout.addLayout(layout)


class FieldsConfigDialog(QDialog):
    """
    Dialog for editing fields_config.yaml.
    
    Signals:
        configSaved: Emitted when configuration is saved successfully.
    """
    configSaved = Signal()
    
    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.setWindowTitle("Fields Configuration")
        self.setMinimumSize(700, 600)
        self.resize(800, 700)
        
        # Load current config
        self._config = get_fields_config(reload=True)
        
        # Track widgets for data extraction
        self._field_tables: Dict[str, QTableWidget] = {}
        self._global_widgets: Dict[str, QWidget] = {}
        self._scorer_widgets: Dict[str, QWidget] = {}
        
        self._setup_ui()
        self._populate_from_config()
    
    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(12)
        
        # === Field Settings (tabbed by type) ===
        fields_group = QGroupBox("Field Settings")
        fields_layout = QVBoxLayout(fields_group)
        
        self.tabs = QTabWidget()
        
        # Create a table for each field type
        field_groups = [
            ("Numeric", "numeric_fields"),
            ("Ordinal", "ordinal_fields"),
            ("Colors", "color_fields"),
            ("Codes", "set_fields"),
            ("Text", "text_fields"),
        ]
        
        for tab_name, attr_name in field_groups:
            table = self._create_field_table()
            self._field_tables[attr_name] = table
            
            # Wrap in scroll area
            scroll = QScrollArea()
            scroll.setWidgetResizable(True)
            scroll.setWidget(table)
            scroll.setFrameShape(QFrame.NoFrame)
            
            self.tabs.addTab(scroll, tab_name)
        
        fields_layout.addWidget(self.tabs)
        layout.addWidget(fields_group, 1)
        
        # === Global Settings ===
        global_group = QGroupBox("Global Settings")
        global_layout = QFormLayout(global_group)
        global_layout.setSpacing(8)
        
        # Equalize weights checkbox
        self.chk_equalize = QCheckBox("Equalize weights (ignore per-field weights, use 1.0 for all)")
        self._global_widgets["equalize_weights"] = self.chk_equalize
        global_layout.addRow(self.chk_equalize)
        
        # Default preset combo
        self.cmb_preset = QComboBox()
        self.cmb_preset.setMinimumWidth(200)
        self._global_widgets["default_preset"] = self.cmb_preset
        global_layout.addRow("Default preset:", self.cmb_preset)
        
        # Default top-K
        self.spin_topk = QSpinBox()
        self.spin_topk.setRange(1, 1000)
        self.spin_topk.setValue(50)
        self._global_widgets["default_top_k"] = self.spin_topk
        global_layout.addRow("Default top-K:", self.spin_topk)
        
        layout.addWidget(global_group)
        
        # === Scorer Parameters (collapsible) ===
        scorer_section = _CollapsibleSection("Scorer Parameters (Advanced)")
        
        scorer_form = QFormLayout()
        scorer_form.setSpacing(6)
        
        # Numeric params
        scorer_form.addRow(QLabel("<b>Numeric fields:</b>"))
        
        self.spin_numeric_k = QDoubleSpinBox()
        self.spin_numeric_k.setRange(0.1, 10.0)
        self.spin_numeric_k.setDecimals(2)
        self.spin_numeric_k.setSingleStep(0.1)
        self._scorer_widgets["numeric_k"] = self.spin_numeric_k
        scorer_form.addRow("  k (decay rate):", self.spin_numeric_k)
        
        # Color params
        scorer_form.addRow(QLabel("<b>Color fields:</b>"))
        
        self.spin_color_threshold = QDoubleSpinBox()
        self.spin_color_threshold.setRange(1.0, 200.0)
        self.spin_color_threshold.setDecimals(1)
        self.spin_color_threshold.setSingleStep(5.0)
        self._scorer_widgets["color_threshold"] = self.spin_color_threshold
        scorer_form.addRow("  ΔE threshold:", self.spin_color_threshold)
        
        self.chk_color_fallback = QCheckBox("Fallback to exact match for unknown colors")
        self._scorer_widgets["color_fallback_to_exact"] = self.chk_color_fallback
        scorer_form.addRow("", self.chk_color_fallback)
        
        # Short arm code params
        scorer_form.addRow(QLabel("<b>Short arm codes:</b>"))
        
        self.spin_arm_sigma = QDoubleSpinBox()
        self.spin_arm_sigma.setRange(0.1, 5.0)
        self.spin_arm_sigma.setDecimals(2)
        self.spin_arm_sigma.setSingleStep(0.1)
        self._scorer_widgets["short_arm_sigma"] = self.spin_arm_sigma
        scorer_form.addRow("  σ (position tolerance):", self.spin_arm_sigma)
        
        self.spin_arm_default_n = QSpinBox()
        self.spin_arm_default_n.setRange(3, 20)
        self._scorer_widgets["short_arm_default_n_arms"] = self.spin_arm_default_n
        scorer_form.addRow("  Default arm count:", self.spin_arm_default_n)
        
        # Text embedding params
        scorer_form.addRow(QLabel("<b>Text embeddings:</b>"))
        
        self.txt_model_id = QLineEdit()
        self.txt_model_id.setMinimumWidth(300)
        self._scorer_widgets["text_model_id"] = self.txt_model_id
        scorer_form.addRow("  Model ID:", self.txt_model_id)
        
        scorer_section.addLayout(scorer_form)
        layout.addWidget(scorer_section)
        
        # === Buttons ===
        btn_layout = QHBoxLayout()
        btn_layout.addStretch(1)
        
        self.btn_reset = QPushButton("Reset to Defaults")
        self.btn_reset.clicked.connect(self._on_reset)
        btn_layout.addWidget(self.btn_reset)
        
        btn_layout.addSpacing(20)
        
        self.btn_cancel = QPushButton("Cancel")
        self.btn_cancel.clicked.connect(self.reject)
        btn_layout.addWidget(self.btn_cancel)
        
        self.btn_save = QPushButton("Save && Apply")
        self.btn_save.setDefault(True)
        self.btn_save.clicked.connect(self._on_save)
        btn_layout.addWidget(self.btn_save)
        
        layout.addLayout(btn_layout)
    
    def _create_field_table(self) -> QTableWidget:
        """Create a table widget for field configuration."""
        table = QTableWidget()
        table.setColumnCount(4)
        table.setHorizontalHeaderLabels(["Field Name", "Enabled", "Weight", "Description"])
        
        header = table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.Fixed)
        header.setSectionResizeMode(2, QHeaderView.Fixed)
        header.setSectionResizeMode(3, QHeaderView.Stretch)
        
        table.setColumnWidth(1, 70)
        table.setColumnWidth(2, 80)
        
        table.verticalHeader().setVisible(False)
        table.setAlternatingRowColors(True)
        table.setSelectionMode(QTableWidget.NoSelection)
        
        return table
    
    def _populate_field_table(self, table: QTableWidget, fields: Dict[str, FieldDef]):
        """Populate a field table with field definitions."""
        table.setRowCount(len(fields))
        
        for row, (name, field_def) in enumerate(sorted(fields.items())):
            # Field name (read-only)
            name_item = QTableWidgetItem(name)
            name_item.setFlags(name_item.flags() & ~Qt.ItemIsEditable)
            font = name_item.font()
            font.setFamily("Consolas, Monaco, monospace")
            name_item.setFont(font)
            table.setItem(row, 0, name_item)
            
            # Enabled checkbox
            chk_widget = QWidget()
            chk_layout = QHBoxLayout(chk_widget)
            chk_layout.setContentsMargins(0, 0, 0, 0)
            chk_layout.setAlignment(Qt.AlignCenter)
            chk = QCheckBox()
            chk.setChecked(field_def.enabled)
            chk.setProperty("field_name", name)
            chk_layout.addWidget(chk)
            table.setCellWidget(row, 1, chk_widget)
            
            # Weight spinbox
            spin_widget = QWidget()
            spin_layout = QHBoxLayout(spin_widget)
            spin_layout.setContentsMargins(2, 0, 2, 0)
            spin = QDoubleSpinBox()
            spin.setRange(0.0, 10.0)
            spin.setDecimals(2)
            spin.setSingleStep(0.1)
            spin.setValue(field_def.weight)
            spin.setProperty("field_name", name)
            spin.setFixedWidth(70)
            spin_layout.addWidget(spin)
            table.setCellWidget(row, 2, spin_widget)
            
            # Description (read-only)
            desc_item = QTableWidgetItem(field_def.description)
            desc_item.setFlags(desc_item.flags() & ~Qt.ItemIsEditable)
            desc_item.setToolTip(field_def.description)
            table.setItem(row, 3, desc_item)
        
        table.resizeRowsToContents()
    
    def _populate_from_config(self):
        """Load current config values into all widgets."""
        config = self._config
        
        # Populate field tables
        field_groups = [
            ("numeric_fields", config.numeric_fields),
            ("ordinal_fields", config.ordinal_fields),
            ("color_fields", config.color_fields),
            ("set_fields", config.set_fields),
            ("text_fields", config.text_fields),
        ]
        
        for attr_name, fields in field_groups:
            table = self._field_tables.get(attr_name)
            if table and fields:
                self._populate_field_table(table, fields)
        
        # Global settings
        self.chk_equalize.setChecked(config.equalize_weights)
        
        # Populate preset combo
        self.cmb_preset.clear()
        preset_names = sorted(config.presets.keys())
        self.cmb_preset.addItems(preset_names)
        idx = self.cmb_preset.findText(config.default_preset)
        if idx >= 0:
            self.cmb_preset.setCurrentIndex(idx)
        
        self.spin_topk.setValue(config.default_top_k)
        
        # Scorer params
        sp = config.scorer_params
        self.spin_numeric_k.setValue(sp.numeric_k)
        self.spin_color_threshold.setValue(sp.color_threshold)
        self.chk_color_fallback.setChecked(sp.color_fallback_to_exact)
        self.spin_arm_sigma.setValue(sp.short_arm_sigma)
        self.spin_arm_default_n.setValue(sp.short_arm_default_n_arms)
        self.txt_model_id.setText(sp.text_model_id)
    
    def _extract_field_table(self, table: QTableWidget) -> Dict[str, Dict[str, Any]]:
        """Extract field settings from a table widget."""
        result = {}
        
        for row in range(table.rowCount()):
            name_item = table.item(row, 0)
            if not name_item:
                continue
            name = name_item.text()
            
            # Get checkbox
            chk_widget = table.cellWidget(row, 1)
            chk = chk_widget.findChild(QCheckBox) if chk_widget else None
            enabled = chk.isChecked() if chk else True
            
            # Get weight spinbox
            spin_widget = table.cellWidget(row, 2)
            spin = spin_widget.findChild(QDoubleSpinBox) if spin_widget else None
            weight = spin.value() if spin else 1.0
            
            # Get description
            desc_item = table.item(row, 3)
            description = desc_item.text() if desc_item else ""
            
            result[name] = {
                "enabled": enabled,
                "weight": weight,
                "description": description,
            }
        
        return result
    
    def _build_yaml_dict(self) -> Dict[str, Any]:
        """Build the YAML dictionary from current widget state."""
        data: Dict[str, Any] = {}
        
        # Field groups
        field_group_names = [
            "numeric_fields",
            "ordinal_fields", 
            "color_fields",
            "set_fields",
            "text_fields",
        ]
        
        for group_name in field_group_names:
            table = self._field_tables.get(group_name)
            if table:
                data[group_name] = self._extract_field_table(table)
        
        # Presets (preserve from original config)
        presets_data = {}
        for name, preset in self._config.presets.items():
            presets_data[name] = {
                "description": preset.description,
                "fields": preset.fields,
            }
        data["presets"] = presets_data
        
        # Global settings
        data["settings"] = {
            "equalize_weights": self.chk_equalize.isChecked(),
            "default_preset": self.cmb_preset.currentText(),
            "default_top_k": self.spin_topk.value(),
        }
        
        # Scorer params
        data["scorer_params"] = {
            "numeric": {
                "k": self.spin_numeric_k.value(),
                "eps": self._config.scorer_params.numeric_eps,  # preserve
            },
            "color": {
                "threshold": self.spin_color_threshold.value(),
                "fallback_to_exact": self.chk_color_fallback.isChecked(),
            },
            "short_arm_code": {
                "sigma": self.spin_arm_sigma.value(),
                "default_n_arms": self.spin_arm_default_n.value(),
            },
            "text_embedding": {
                "model_id": self.txt_model_id.text().strip() or "BAAI/bge-small-en-v1.5",
            },
        }
        
        return data
    
    def _generate_yaml_with_comments(self, data: Dict[str, Any]) -> str:
        """Generate YAML string with helpful comments."""
        lines = [
            "# fields_config.yaml",
            "# Configuration for first-order comparison ranking in starBoard",
            "#",
            "# This file controls:",
            "#   1. Which fields are enabled by default in the ranking",
            "#   2. Per-field weights for weighted averaging",
            "#   3. Scorer parameters for similarity algorithms",
            "#",
            "# === How First-Order Ranking Works ===",
            "#",
            "# For each query, the engine compares it against all gallery members by:",
            "#   1. Computing per-field similarity scores (each in [0, 1])",
            "#   2. Averaging scores across all active fields with data on both sides",
            "#   3. Ranking gallery items by descending combined score",
            "#",
            "# The final score formula is:",
            "#   score = Σ(weight_i × score_i) / Σ(weight_i)",
            "#",
            "# where only fields that are:",
            "#   - enabled (enabled: true below)",
            "#   - have a non-empty value in the query",
            "#   - have a non-empty value in the gallery member",
            "# contribute to the sum.",
            "",
        ]
        
        # Use PyYAML for the actual data
        try:
            import yaml
            
            def _section_header(title: str, scorer_info: str = "") -> List[str]:
                sep = "# " + "=" * 77
                header = [
                    "",
                    sep,
                    f"# {title}",
                    sep,
                ]
                if scorer_info:
                    header.append(f"# {scorer_info}")
                header.append("")
                return header
            
            # Numeric fields
            lines.extend(_section_header(
                "NUMERIC FIELDS",
                "Scorer: NumericGaussianScorer - exp(-|gallery - query| / (k × MAD))"
            ))
            lines.append(yaml.dump({"numeric_fields": data.get("numeric_fields", {})}, 
                                   default_flow_style=False, sort_keys=False, allow_unicode=True))
            
            # Ordinal fields
            lines.extend(_section_header(
                "ORDINAL FIELDS", 
                "Scorer: NumericGaussianScorer (order matters)"
            ))
            lines.append(yaml.dump({"ordinal_fields": data.get("ordinal_fields", {})},
                                   default_flow_style=False, sort_keys=False, allow_unicode=True))
            
            # Color fields
            lines.extend(_section_header(
                "COLOR FIELDS",
                "Scorer: ColorSpaceScorer - exp(-ΔE / threshold) in LAB space"
            ))
            lines.append(yaml.dump({"color_fields": data.get("color_fields", {})},
                                   default_flow_style=False, sort_keys=False, allow_unicode=True))
            
            # Set fields
            lines.extend(_section_header(
                "SET/CODE FIELDS",
                "Scorer: ShortArmCodeScorer - fuzzy position-aware matching"
            ))
            lines.append(yaml.dump({"set_fields": data.get("set_fields", {})},
                                   default_flow_style=False, sort_keys=False, allow_unicode=True))
            
            # Text fields
            lines.extend(_section_header(
                "TEXT FIELDS",
                "Scorer: TextEmbeddingBGEScorer or TextNgramScorer"
            ))
            lines.append(yaml.dump({"text_fields": data.get("text_fields", {})},
                                   default_flow_style=False, sort_keys=False, allow_unicode=True))
            
            # Presets
            lines.extend(_section_header("PRESETS"))
            lines.append(yaml.dump({"presets": data.get("presets", {})},
                                   default_flow_style=False, sort_keys=False, allow_unicode=True))
            
            # Global settings
            lines.extend(_section_header("GLOBAL SETTINGS"))
            lines.append(yaml.dump({"settings": data.get("settings", {})},
                                   default_flow_style=False, sort_keys=False, allow_unicode=True))
            
            # Scorer params
            lines.extend(_section_header("SCORER PARAMETERS"))
            lines.append(yaml.dump({"scorer_params": data.get("scorer_params", {})},
                                   default_flow_style=False, sort_keys=False, allow_unicode=True))
            
            return "\n".join(lines)
            
        except ImportError:
            # Fallback: just dump everything
            import yaml
            return yaml.dump(data, default_flow_style=False, sort_keys=False, allow_unicode=True)
    
    def _get_config_path(self) -> Path:
        """Get path to fields_config.yaml."""
        from src.data.archive_paths import archive_root
        return archive_root() / "fields_config.yaml"
    
    def _on_save(self):
        """Save configuration to YAML file."""
        try:
            data = self._build_yaml_dict()
            yaml_content = self._generate_yaml_with_comments(data)
            
            path = self._get_config_path()
            path.write_text(yaml_content, encoding="utf-8")
            
            # Reload config to update cache
            get_fields_config(reload=True)
            
            self.configSaved.emit()
            self.accept()
            
        except Exception as e:
            QMessageBox.critical(
                self,
                "Save Failed",
                f"Failed to save configuration:\n\n{e}"
            )
    
    def _on_reset(self):
        """Reset all settings to defaults."""
        reply = QMessageBox.question(
            self,
            "Reset Configuration",
            "Reset all settings to defaults?\n\nThis will not save until you click 'Save & Apply'.",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        
        if reply != QMessageBox.Yes:
            return
        
        # Reset to default config
        from src.data.fields_config import _get_default_config
        self._config = _get_default_config()
        self._populate_from_config()



