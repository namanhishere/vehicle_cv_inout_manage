"""Tests for gate/plate.py - normalization and validation."""

import pytest

from gate.plate import PlateKind, format_for_display, normalize

ACCEPT = [
    # user examples
    ("29-AB 123.45", "29AB-123.45", PlateKind.NEW),
    ("29-A1 678.90", "29A1-678.90", PlateKind.LEGACY5),
    ("29-MĐ1 002.12", "29MĐ1-002.12", PlateKind.ELECTRIC),  # literal Đ
    ("29-MD1 002.12", "29MĐ1-002.12", PlateKind.ELECTRIC),  # OCR form of Đ
    ("29HA 002.33", "29HA-002.33", PlateKind.NEW),  # official sample
    # dot-less, lowercase, messy separators
    ("29HA00233", "29HA-002.33", PlateKind.NEW),
    ("29-ha 002.33", "29HA-002.33", PlateKind.NEW),
    ("29-HA__0 02..33", "29HA-002.33", PlateKind.NEW),
    (" 29AB-123.45\t", "29AB-123.45", PlateKind.NEW),
    # MĐ without digit is ELECTRIC; bare latin MD stays NEW
    ("29MĐ-002.12", "29MĐ-002.12", PlateKind.ELECTRIC),
    ("29MD-002.12", "29MD-002.12", PlateKind.NEW),
    # LEGACY4
    ("29F1-1234", "29F1-1234", PlateKind.LEGACY4),
    ("29-F1 1234", "29F1-1234", PlateKind.LEGACY4),
    # car plate may normalize as LEGACY4 (docstring: REJECTs at DB lookup)
    ("51F-123.45", "51F1-2345", PlateKind.LEGACY4),
    # digit-position OCR fixes (O->0, I->1, B->8 ...)
    ("29AB-1B3.45", "29AB-183.45", PlateKind.NEW),
    ("29A1-6O8.90", "29A1-608.90", PlateKind.LEGACY5),
    ("29-AI 678.90", "29A1-678.90", PlateKind.LEGACY5),  # I in ser-digit slot
    ("29F1-I234", "29F1-1234", PlateKind.LEGACY4),
    ("29HA-0O2.33", "29HA-002.33", PlateKind.NEW),
    # 'B' at LEGACY4 ser-digit slot (index 3) is a digit position -> fixed
    ("29AB-123.4", "29A8-1234", PlateKind.LEGACY4),
    # 4-digit tail of a 5-digit plate normalizes as LEGACY4
    ("29-A1 678.9", "29A1-6789", PlateKind.LEGACY4),
    # digit after MD series belongs to the number part (NEW match wins)
    ("29MD1-123.4", "29MD-112.34", PlateKind.NEW),
    # Q misread of 0 at LEGACY5 ser-digit slot (Q is in the fix set)
    ("29QQ-123.45", "29Q0-123.45", PlateKind.LEGACY5),
]


@pytest.mark.parametrize("raw,canonical,kind", ACCEPT)
def test_normalize_accepts(raw, canonical, kind):
    plate = normalize(raw)
    assert plate is not None, f"{raw!r} should normalize"
    assert plate.canonical == canonical
    assert plate.kind is kind
    assert plate.text == raw.upper().translate(
        str.maketrans("", "", " \t\n\r\f\v.-_")
    )


def test_letter_slot_never_substituted():
    # 'O' sits in a LETTER slot (LEGACY5 ser letter at index 2); O is excluded
    # from both letter domains, and letter positions are never fixed ->
    # no match -> None. (Digit slots are the pattern's \d positions only.)
    assert normalize("29O1-678.90") is None
    # '0' in a letter slot (index 2) is not substituted either.
    assert normalize("290A-123.45") is None
    # Unknown glyph in a digit slot is NOT guessed (not in fix set).
    assert normalize("29A1-6X8.90") is None


def test_forbidden_new_series_rejected():
    for pair in ("CD", "CT", "DA", "HC", "LB", "LD", "MK"):
        assert normalize(f"29{pair}-123.45") is None, pair


def test_new_series_letter_domain():
    for ch in "IOJQRW":
        # excluded letter at NEW series first slot; LEGACY5 needs a digit at
        # the second slot, so no rescue exists for this shape
        assert normalize(f"29{ch}A-123.45") is None, ch
    for ch in "IO":
        # I/O are invalid even as LEGACY5 first letter
        assert normalize(f"29{ch}1-123.45") is None, ch
    for ch in "JWR":
        # second slot holds a letter that is NOT in the OCR fix set ->
        # no LEGACY5 ser-digit rescue
        assert normalize(f"29A{ch}-123.45") is None, ch


def test_province_codes():
    for raw in ("00AB-123.45", "01A1-678.90", "10AB-123.45", "10A1-678.90",
                "100AB-123.45"):
        assert normalize(raw) is None, raw


@pytest.mark.parametrize(
    "raw",
    ["", "   ", "29", "29AB", "ABC", "29AB-123456", "29A1-123.45.67",
     "29A1/678.90", "29A1 678.90x", "29-AB 123,45"],
)
def test_normalize_rejects(raw):
    assert normalize(raw) is None, raw


def test_plate_is_frozen():
    p = normalize("29AB-123.45")
    assert p is not None
    with pytest.raises(Exception):
        p.canonical = "x"  # frozen dataclass


def test_format_for_display_identity():
    assert format_for_display("29AB-123.45") == "29AB-123.45"
