"""Shared angle conventions used by geometric retargeting components."""

from __future__ import annotations

import math


def wrap_to_pi(angle: float) -> float:
    """Return finite ``angle`` in the canonical half-open interval ``[-pi, pi)``."""
    value = float(angle)
    if not math.isfinite(value):
        raise ValueError("angle must be finite")
    wrapped = (value + math.pi) % (2.0 * math.pi) - math.pi
    return 0.0 if wrapped == 0.0 else wrapped


def angular_difference(first: float, second: float) -> float:
    """Return the signed canonical difference ``first - second`` in ``[-pi, pi)``."""
    return wrap_to_pi(float(first) - float(second))
