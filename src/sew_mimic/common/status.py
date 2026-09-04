"""Stable status values shared by all retargeting solvers."""

from __future__ import annotations

from enum import Enum


class SolverStatus(str, Enum):
    """Outcome of a retargeting solve.

    ``SUCCESS_EXACT`` means every constraint claimed by that particular
    solver passed its post-validation tolerances. It does not imply that the
    solver constrains quantities outside its declared constraint set.

    ``SUCCESS_APPROX`` is a usable least-squares or otherwise approximate
    result for which at least one exact criterion was not met.

    ``UNREACHABLE`` means no exact target exists for the solver and model.
    ``JOINT_LIMIT`` means kinematic candidates exist, but none satisfy the
    accepted mechanical limits. ``SEW_SINGULAR`` identifies a target at which
    the configured SEW parameterization is undefined.

    ``NO_VALID_BRANCH`` means candidate generation succeeded but every
    candidate failed post-validation. ``INVALID_INPUT`` covers malformed,
    non-finite, or otherwise invalid inputs. ``NUMERICAL_FAILURE`` identifies
    an algorithmic numerical failure distinct from mathematical
    unreachability. ``LEGACY_FAILURE`` is reserved for exceptions from the
    unchanged Method-0 implementation that cannot be classified safely.
    """

    SUCCESS_EXACT = "SUCCESS_EXACT"
    SUCCESS_APPROX = "SUCCESS_APPROX"
    UNREACHABLE = "UNREACHABLE"
    JOINT_LIMIT = "JOINT_LIMIT"
    SEW_SINGULAR = "SEW_SINGULAR"
    NO_VALID_BRANCH = "NO_VALID_BRANCH"
    INVALID_INPUT = "INVALID_INPUT"
    NUMERICAL_FAILURE = "NUMERICAL_FAILURE"
    LEGACY_FAILURE = "LEGACY_FAILURE"
