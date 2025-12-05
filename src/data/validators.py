from __future__ import annotations

import re
from dataclasses import dataclass

ID_PATTERN = re.compile(r"^[A-Za-z0-9_.\-]+$")

@dataclass
class ValidationResult:
    ok: bool
    message: str = ""

def validate_id(id_str: str) -> ValidationResult:
    s = (id_str or "").strip()
    if not s:
        return ValidationResult(False, "ID cannot be empty.")
    if "/" in s or "\\" in s:
        return ValidationResult(False, "ID cannot contain path separators.")
    if not ID_PATTERN.match(s):
        return ValidationResult(False, "ID may contain letters, numbers, hyphen, underscore, and dot only.")
    return ValidationResult(True, "")

def validate_mmddyy_string(s: str) -> ValidationResult:
    import re as _re
    if not _re.match(r"^\d{2}_\d{2}_\d{2}($|_)", s):
        return ValidationResult(False, "Encounter name must start with MM_DD_YY.")
    return ValidationResult(True, "")
