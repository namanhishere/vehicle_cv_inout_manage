"""Vietnamese motorcycle plate normalization and validation.

Pure Python (stdlib ``re`` only). No dependencies on other gate modules.

Motorcycle plates are 2-line, 190x140 mm. This module normalizes raw assembled
OCR text into a canonical form and classifies it into one of four kinds:

- NEW:      ``29HA-002.33``   (2-letter series, 5-digit number)
- LEGACY5:  ``29A1-678.90``   (letter+digit series, 5-digit number)
- ELECTRIC: ``29MD1-002.12``  (M[Dd] series, 5-digit number)
- LEGACY4:  ``29F1-1234``     (letter+digit series, 4-digit number)

Notes:
- Blue government plates (letter+digit series) are covered structurally by
  LEGACY5; no separate pattern is needed.
- Car plates (e.g. ``51F-123.45`` -> alnum ``51F12345``) may normalize as
  LEGACY4 (``51F1-2345``): accepted behavior - they REJECT as UNREGISTERED
  at the database lookup, which is the intended policy for a motorcycle gate.
- OCR character fixes (O->0, I->1, ...) are applied ONLY in digit positions
  of the pattern being tried. A wrong *letter* must not silently produce a
  wrong plate; letter positions are never substituted (prefer REJECT).
"""

from __future__ import annotations

import enum
import re
from dataclasses import dataclass

__all__ = ["PlateKind", "Plate", "normalize", "format_for_display"]


class PlateKind(enum.Enum):
    NEW = "new"
    LEGACY5 = "legacy5"
    ELECTRIC = "electric"
    LEGACY4 = "legacy4"


@dataclass(frozen=True)
class Plate:
    text: str  # raw assembled OCR text, cleaned
    canonical: str  # e.g. "29HA-002.33", "29A1-678.90", "29MD1-002.12", "29F1-1234"
    kind: PlateKind


# Reserved NEW series pairs (TT 79/2024).
_FORBIDDEN_SERIES = frozenset({"CD", "CT", "DA", "HC", "LB", "LD", "MK"})

# OCR misread fixes, applied ONLY at digit positions. The PP-OCRv3 en
# recognizer emits lowercase glyphs (its dictionary is lowercase), so
# lowercase l for digit 1 is routine; raw text is uppercased first, making
# L->1 the same rule as I->1. Never applied to letter positions.
_FIX = {"O": "0", "I": "1", "L": "1", "B": "8", "D": "0",
        "S": "5", "Z": "2", "G": "6", "Q": "0"}

_NUM5 = r"\d{3}\.?\d{2}"  # dot already stripped by _clean(); kept for spec fidelity

# Pattern table: (kind, regex, digit-slot map by cleaned-string length).
# Digit slots = the positions of the pattern's `\d` tokens; OCR fixes may be
# applied there and nowhere else.
_PATTERNS = (
    (
        PlateKind.NEW,
        re.compile(rf"^(?P<prov>[1-9]\d)(?P<ser>[A-HKLMNPSTUVXYZ]{{2}})(?P<num>{_NUM5})$"),
        {9: (0, 1, 4, 5, 6, 7, 8)},
    ),
    (
        PlateKind.LEGACY5,
        re.compile(rf"^(?P<prov>[1-9]\d)(?P<ser>[A-HJ-NP-Z]\d)(?P<num>{_NUM5})$"),
        {9: (0, 1, 3, 4, 5, 6, 7, 8)},
    ),
    (
        PlateKind.ELECTRIC,
        re.compile(rf"^(?P<prov>[1-9]\d)(?P<ser>M[DĐ]\d?)(?P<num>{_NUM5})$"),
        {9: (0, 1, 4, 5, 6, 7, 8), 10: (0, 1, 4, 5, 6, 7, 8, 9)},
    ),
    (
        PlateKind.LEGACY4,
        re.compile(r"^(?P<prov>[1-9]\d)(?P<ser>[A-HJ-NP-Z]\d)(?P<num>\d{4})$"),
        {8: (0, 1, 3, 4, 5, 6, 7)},
    ),
)

_REMOVE = str.maketrans("", "", " \t\n\r\f\v.-_")


def _clean(raw: str) -> str:
    return raw.upper().translate(_REMOVE)


def _try_pattern(cleaned: str, pattern, slots_by_len):
    """Try ``pattern`` on ``cleaned``; on failure apply digit-slot OCR fixes.

    Fixes substitute ONLY at the pattern's digit slots and only for chars in
    the misread set; letter slots are never touched.
    """
    m = pattern.match(cleaned)
    if m is not None:
        return cleaned, m
    slots = slots_by_len.get(len(cleaned))
    if not slots:
        return None
    chars = list(cleaned)
    changed = False
    for i in slots:
        ch = cleaned[i]
        if not ch.isdigit() and ch in _FIX:
            chars[i] = _FIX[ch]
            changed = True
    if not changed:
        return None
    fixed = "".join(chars)
    m = pattern.match(fixed)
    if m is None:
        return None
    return fixed, m


def _canonical_from(kind: PlateKind, prov: str, ser: str, num: str) -> str:
    num = num.replace(".", "")
    if kind is PlateKind.ELECTRIC:
        # canonical series always uses MĐ, keeping the optional trailing digit
        digit = ser[2:] if len(ser) == 3 else ""
        return f"{prov}MĐ{digit}-{num[:3]}.{num[3:]}"
    if kind is PlateKind.LEGACY4:
        return f"{prov}{ser}-{num}"
    return f"{prov}{ser}-{num[:3]}.{num[3:]}"


def normalize(raw: str) -> Plate | None:
    """Normalize raw OCR text to a canonical :class:`Plate` or ``None``.

    Algorithm order (exact):
    1. Clean: strip whitespace, ``.``, ``-``, ``_``; uppercase (keep ``D``).
    2. Try NEW, LEGACY5, ELECTRIC, LEGACY4 in that order; first match wins.
       - NEW rejects reserved series pairs (CD/CT/DA/HC/LB/LD/MK).
       - Province codes below 11 (e.g. 00/01/10) never match.
    3. OCR char fixes applied ONLY in digit positions of the pattern being
       tried (single greedy pass per pattern); letter positions untouched.
    4. ``MD`` + digit in series position normalizes to ``MD`` + digit
       (ELECTRIC). Bare 2-char ``MD`` stays ``MD`` (valid NEW pair); the
       MD-vs-MD disambiguation happens in the decision layer (dual DB
       lookup), not here.
    5. No match -> ``None``.
    """
    cleaned = _clean(raw)
    if not cleaned:
        return None
    for kind, pattern, slots_by_len in _PATTERNS:
        result = _try_pattern(cleaned, pattern, slots_by_len)
        if result is None:
            continue
        _matched, m = result
        prov = m.group("prov")
        ser = m.group("ser")
        if int(prov) < 11:
            return None  # province codes run 11..99
        if kind is PlateKind.NEW and ser in _FORBIDDEN_SERIES:
            return None
        canonical = _canonical_from(kind, prov, ser, m.group("num"))
        return Plate(text=cleaned, canonical=canonical, kind=kind)
    return None


def format_for_display(canonical: str) -> str:
    """Return the canonical string for display (identity: already canonical)."""
    return canonical
