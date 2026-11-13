"""Canonical display metadata for upstream and built-in model catalogues."""

from __future__ import annotations

import math
import re
from typing import Any, Dict, Optional

_NUMBER = re.compile(r"[+-]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][+-]?[0-9]+)?")
_ANNOTATION = re.compile(
    r"(?:\(\s*[x×]\s*([0-9]+(?:\.[0-9]+)?)\s*\)|"
    r"（\s*[x×]\s*([0-9]+(?:\.[0-9]+)?)\s*）)", re.I)


def _number(value: Any) -> Optional[float]:
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        return None
    if isinstance(value, str) and not _NUMBER.fullmatch(value.strip()):
        return None
    try:
        number = float(value)
    except (ValueError, OverflowError):
        return None
    return number if math.isfinite(number) and number >= 0 else None


def normalize_model_metadata(model: Dict[str, Any]) -> Dict[str, Any]:
    """Copy a model, preserving valid nonnegative rates (including zero).

    Numeric strings become numbers. Missing/invalid rates may be inferred only
    from an explicit parenthesized x/× annotation; model versions are ignored.
    Unknown rates remain null, or absent if the upstream field was absent.
    Other fields and the original upstream record are left unchanged.
    """
    result = dict(model)
    multiplier = _number(model.get("multiplier"))
    if multiplier is None:
        match = _ANNOTATION.search(str(model.get("name") or ""))
        if match:
            multiplier = _number(match.group(1) or match.group(2))
    if multiplier is not None or "multiplier" in model:
        result["multiplier"] = multiplier
    return result
