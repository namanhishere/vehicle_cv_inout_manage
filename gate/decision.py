"""Access-control decision engine.

Pure logic over the plate registry and inside/outside state machine.
Depends on ``plate.py`` and the storage layer only.

REJECT-reason ordering (policy): format -> confidence -> cooldown ->
MD/MD disambiguation -> registry -> state. Tests pin this order.
"""

from __future__ import annotations

import enum
import time
from dataclasses import dataclass
from typing import Callable

from gate.plate import PlateKind, normalize

__all__ = ["Result", "Reason", "Decision", "DecisionEngine"]


class Result(enum.Enum):
    ALLOW = "ALLOW"
    REJECT = "REJECT"


class Reason(enum.Enum):
    ALLOW = "ALLOW"
    UNREGISTERED = "UNREGISTERED"
    INVALID_FORMAT = "INVALID_FORMAT"
    ALREADY_INSIDE = "ALREADY_INSIDE"
    ALREADY_OUTSIDE = "ALREADY_OUTSIDE"
    LOW_CONF = "LOW_CONF"


@dataclass(frozen=True)
class Decision:
    result: Result
    reason: Reason
    plate: str  # canonical ("" only when the format is invalid)
    confidence: float
    direction: str  # 'IN' | 'OUT' - carried so apply() is self-contained
    duplicate: bool = False  # True = cooldown dedupe; caller must skip work


class DecisionEngine:
    """Classify a plate reading against registry + state; never writes events
    (the controller does) and only changes state via :meth:`apply`."""

    def __init__(
        self,
        db,
        min_confidence: float,
        cooldown_s: float,
        now: Callable[[], float] = time.time,
    ):
        self._db = db
        self.min_confidence = min_confidence
        self.cooldown_s = cooldown_s
        self._now = now
        # (canonical, direction) -> (last_ts, decision)
        self._cooldown: dict[tuple[str, str], tuple[float, Decision]] = {}

    # -- classify ----------------------------------------------------------

    def classify(
        self, *, raw: str, direction: str, confidence: float, camera: str
    ) -> Decision:
        """Classify one plate reading (exact rule order below)."""
        now = self._now()

        # 1. format
        plate = normalize(raw)
        if plate is None:
            return Decision(
                Result.REJECT, Reason.INVALID_FORMAT, "", confidence,
                direction,
            )
        canonical = plate.canonical

        # 2. confidence
        if confidence < self.min_confidence:
            return Decision(
                Result.REJECT, Reason.LOW_CONF, canonical, confidence,
                direction,
            )

        # 3. cooldown dedupe of motion re-triggers during one passage
        key = (canonical, direction)
        prev = self._cooldown.get(key)
        if prev is not None and now - prev[0] < self.cooldown_s:
            last = prev[1]
            return Decision(
                last.result, last.reason, last.plate, last.confidence,
                direction, duplicate=True,
            )

        # 4. MD/MD disambiguation (only NEW-kind plates whose series is MD):
        #    the raw reading may be either the NEW pair "MD" or the electric
        #    series "MD" (D misread of D). Resolve against the registry.
        if plate.kind is PlateKind.NEW and canonical[2:4] == "MD":
            elect = canonical[:2] + "MĐ" + canonical[4:]
            have_md = self._db.lookup(canonical) is not None
            have_elect = self._db.lookup(elect) is not None
            if have_md and have_elect:
                # ambiguous - the reader cannot tell which vehicle this is
                decision = Decision(
                    Result.REJECT, Reason.LOW_CONF, canonical, confidence,
                    direction,
                )
                self._cooldown[key] = (now, decision)
                return decision
            if have_elect:
                canonical = elect  # registered as electric MĐ...
            elif not have_md:
                canonical = elect  # neither registered: treat as electric

        # 5./6. registry
        vehicle = self._db.lookup(canonical)
        if vehicle is None or not vehicle["registered"]:
            decision = Decision(
                Result.REJECT, Reason.UNREGISTERED, canonical, confidence,
                direction,
            )
            self._cooldown[key] = (now, decision)
            return decision

        # 7. inside/outside state transition
        inside = bool(vehicle["inside"])
        if direction == "IN" and inside:
            decision = Decision(
                Result.REJECT, Reason.ALREADY_INSIDE, canonical, confidence,
                direction,
            )
            self._cooldown[key] = (now, decision)
            return decision
        if direction == "OUT" and not inside:
            decision = Decision(
                Result.REJECT, Reason.ALREADY_OUTSIDE, canonical, confidence,
                direction,
            )
            self._cooldown[key] = (now, decision)
            return decision

        # 8. allow
        decision = Decision(
            Result.ALLOW, Reason.ALLOW, canonical, confidence, direction,
        )
        self._cooldown[key] = (now, decision)
        return decision

    # -- apply -------------------------------------------------------------

    def apply(self, decision: Decision) -> None:
        """Commit the state transition of a non-duplicate ALLOW decision.
        REJECT decisions never change state; calling apply on them is a no-op.
        Only the controller calls this."""
        if decision.result is Result.ALLOW:
            self._db.set_inside(decision.plate, decision.direction == "IN")
