"""
CPT code interface — opaque user-supplied IDs.

CPT codes are AMA-copyrighted. We do NOT ship any CPT codes, descriptions,
or frequency data. This module provides the interface for users to supply
their own CPT code lists as opaque identifiers.

Usage:
    from claims_synth.vocab.cpt import register_cpt_codes
    register_cpt_codes(["99213", "27130", ...])  # user-supplied
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class CPTCode:
    """Opaque CPT code — user-supplied ID, no description shipped."""
    code: str
    category: str = ""  # user-supplied category label (optional)


# ── User-supplied registry ─────────────────────────────────────────────────

_cpt_registry: dict[str, CPTCode] = {}

def register_cpt_codes(codes: list[str], category: str = "") -> None:
    """
    Register user-supplied CPT codes as opaque IDs.

    These are NOT shipped with the package — the user provides them.
    No descriptions, no frequency data, no AMA-copyrighted content.
    """
    for code in codes:
        _cpt_registry[code] = CPTCode(code=code, category=category)

def lookup(code: str) -> Optional[CPTCode]:
    """Look up a user-registered CPT code."""
    return _cpt_registry.get(code)

def all_codes() -> list[str]:
    """Return all registered CPT code strings."""
    return list(_cpt_registry.keys())

def clear() -> None:
    """Clear the CPT registry."""
    _cpt_registry.clear()
