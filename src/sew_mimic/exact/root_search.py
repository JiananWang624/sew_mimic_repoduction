"""Deterministic fixed-slot scalar root refinement for Stereo-SEW IK."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import math

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.optimize import brentq, minimize_scalar


Vector = NDArray[np.float64]


class _EvaluationBudgetExhausted(RuntimeError):
    """Internal control flow for a deterministic callback budget."""


def sp3_feasibility_margin(p1: ArrayLike, p2: ArrayLike, axis: ArrayLike, distance: float) -> float:
    """Normalized signed SP3 discriminant ``1 - (C/(2 r1 r2))^2``.

    A positive value means two geometric intersections, zero a tangent, and
    negative no exact intersection. Degenerate zero-radius circles return
    zero only when their constant distance equation is satisfied; otherwise
    they return negative infinity rather than fabricating a margin.
    """
    first, second, direction = (np.asarray(value, dtype=float).copy() for value in (p1, p2, axis))
    if first.shape != (3,) or second.shape != (3,) or direction.shape != (3,) or not np.all(np.isfinite(np.concatenate((first, second, direction)))):
        raise ValueError("SP3 margin inputs must be finite vectors of shape (3,)")
    d = float(distance)
    norm = float(np.linalg.norm(direction))
    if not math.isfinite(d) or d < 0 or norm == 0:
        raise ValueError("SP3 margin requires nonzero axis and finite nonnegative distance")
    direction /= norm
    axial = float(direction @ (first - second))
    perpendicular_first = first - direction * float(direction @ first)
    perpendicular_second = second - direction * float(direction @ second)
    radius_first, radius_second = float(np.linalg.norm(perpendicular_first)), float(np.linalg.norm(perpendicular_second))
    c = radius_first**2 + radius_second**2 + axial**2 - d**2
    denominator = 2.0 * radius_first * radius_second
    if denominator <= np.finfo(float).eps * max(1.0, radius_first, radius_second, abs(c)):
        return 0.0 if abs(c) <= 1e-12 * max(1.0, d**2, axial**2) else -math.inf
    value = c / denominator
    return 1.0 - value * value


def sp4_feasibility_margin(p: ArrayLike, h: ArrayLike, axis: ArrayLike, distance: float) -> float:
    """Dimensionless signed SP4 discriminant ``1 - (b / ||a||)^2``."""
    vector, normal, direction = (np.asarray(value, dtype=float) for value in (p, h, axis))
    if any(value.shape != (3,) for value in (vector, normal, direction)) or not np.all(np.isfinite(np.concatenate((vector, normal, direction)))):
        raise ValueError("SP4 margin inputs must be finite vectors of shape (3,)")
    d = float(distance)
    if not math.isfinite(d):
        raise ValueError("SP4 distance must be finite")
    vector_norm, direction_norm, normal_norm = float(np.linalg.norm(vector)), float(np.linalg.norm(direction)), float(np.linalg.norm(normal))
    if vector_norm == 0 or direction_norm == 0 or normal_norm == 0:
        raise ValueError("SP4 margin requires nonzero vector, axis, and normal")
    direction, normal = direction / direction_norm, normal / normal_norm
    basis = np.column_stack((np.cross(direction, vector), -np.cross(direction, np.cross(direction, vector))))
    a = normal @ basis
    b = d - float(normal @ direction) * float(direction @ vector)
    norm_a = float(np.linalg.norm(a))
    if norm_a <= 1e-12 * vector_norm:
        return 0.0 if abs(b) <= 1e-12 * max(1.0, vector_norm) else -math.inf
    ratio = b / norm_a
    return 1.0 - ratio * ratio


def sp2_feasibility_margin(p1: ArrayLike, p2: ArrayLike, axis1: ArrayLike, axis2: ArrayLike) -> float:
    """Minimum normalized feasibility of the two SP4 reductions defining SP2."""
    first, second, first_axis, second_axis = (np.asarray(value, dtype=float) for value in (p1, p2, axis1, axis2))
    if any(value.shape != (3,) or not np.all(np.isfinite(value)) for value in (first, second, first_axis, second_axis)):
        raise ValueError("SP2 margin inputs must be finite vectors with shape (3,)")
    norms = [float(np.linalg.norm(value)) for value in (first, second, first_axis, second_axis)]
    if any(norm == 0.0 for norm in norms):
        raise ValueError("SP2 margin requires nonzero vectors and axes")
    first, second, first_axis, second_axis = (value / norm for value, norm in zip((first, second, first_axis, second_axis), norms))
    if np.linalg.norm(np.cross(first_axis, second_axis)) <= 1e-12:
        raise ValueError("SP2 margin axes must not be parallel")
    return min(
        sp4_feasibility_margin(first, second_axis, first_axis, float(second_axis @ second)),
        sp4_feasibility_margin(second, first_axis, second_axis, float(first_axis @ first)),
    )


@dataclass(frozen=True)
class RootSearchConfig:
    """Reference-compatible coarse search and conservative refinement rules."""

    minimum: float = -math.pi
    maximum: float = 0.0
    samples: int = 200
    crossing_threshold: float = 0.1
    root_xtol: float = 1e-12
    root_residual_tolerance: float = 1e-9
    dedup_tolerance: float = 1e-10

    def __post_init__(self) -> None:
        if not (math.isfinite(self.minimum) and math.isfinite(self.maximum) and self.minimum < self.maximum):
            raise ValueError("root interval must be finite with minimum < maximum")
        if self.samples < 2:
            raise ValueError("samples must be at least two")
        for name in ("crossing_threshold", "root_xtol", "root_residual_tolerance", "dedup_tolerance"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be positive and finite")


@dataclass(frozen=True)
class FixedSlotRoot:
    """One validated root associated with an unchanged partial-solution slot."""

    angle: float
    slot: int
    residual: float


@dataclass(frozen=True)
class FixedSlotRootSearchResult:
    roots: tuple[FixedSlotRoot, ...]
    slot_count: int
    sampled_exact_zeros: int
    inactive_samples: int
    rejected_crossings: int
    rejected_refinements: int


@dataclass(frozen=True)
class EventAwareSearchConfig:
    """Bounded deterministic event localization for narrow feasible branches."""

    minimum: float = -math.pi
    maximum: float = 0.0
    initial_partitions: int = 64
    max_subdivision_depth: int = 24
    minimum_interval_width_rad: float = 1e-12
    event_margin_tolerance: float = 0.0
    # Eight fixed Gen3 slots each refine 64 partitions; the cap covers every
    # vector callback, not merely coarse samples.
    maximum_event_evaluations: int = 50_000
    alignment_samples_per_interval: int = 5
    root_xtol: float = 1e-12
    alignment_root_xtol: float = 1e-15
    alignment_maximum_iterations: int = 128
    root_residual_tolerance: float = 1e-9
    alignment_residual_tolerance: float = 1e-8
    dedup_tolerance: float = 1e-10
    alignment_dedup_tolerance: float = 1e-8

    def __post_init__(self) -> None:
        if not (math.isfinite(self.minimum) and math.isfinite(self.maximum) and self.minimum < self.maximum):
            raise ValueError("event interval must be finite with minimum < maximum")
        if self.initial_partitions < 1 or self.max_subdivision_depth < 0 or self.maximum_event_evaluations < 1 or self.alignment_samples_per_interval < 2 or self.alignment_maximum_iterations < 1:
            raise ValueError("event search counts are invalid")
        for name in ("minimum_interval_width_rad", "root_xtol", "alignment_root_xtol", "root_residual_tolerance", "alignment_residual_tolerance", "dedup_tolerance", "alignment_dedup_tolerance"):
            if not math.isfinite(float(getattr(self, name))) or float(getattr(self, name)) <= 0:
                raise ValueError(f"{name} must be positive and finite")
        if not math.isfinite(self.event_margin_tolerance) or self.event_margin_tolerance < 0.0:
            raise ValueError("event_margin_tolerance must be finite and nonnegative")


@dataclass(frozen=True)
class FeasibleInterval:
    slot: int
    left: float
    right: float


@dataclass(frozen=True)
class EventAwareRootSearchResult:
    roots: tuple[FixedSlotRoot, ...]
    intervals: tuple[FeasibleInterval, ...]
    slot_count: int
    margin_evaluations: int
    alignment_evaluations: int
    maximum_refinements: int
    budget_exhausted: bool
    inactive_samples: int


def _values(function: Callable[[float], ArrayLike], angle: float) -> Vector:
    values = np.asarray(function(float(angle)), dtype=float)
    if values.ndim != 1:
        raise ValueError("fixed-slot root function must return a rank-one vector")
    return values


def search_fixed_slot_roots(
    function: Callable[[float], ArrayLike],
    config: RootSearchConfig = RootSearchConfig(),
) -> FixedSlotRootSearchResult:
    """Find finite, continuous sign changes independently in fixed branch slots.

    The inclusive 200-sample default matches the official coarse search. NaN
    values mark an inactive slot. A crossing is refined only when both
    endpoint magnitudes are below ``crossing_threshold``; this deliberately rejects
    angle-wrap/discontinuity crossings before bracketed ``brentq`` refinement.
    """
    angles = np.linspace(config.minimum, config.maximum, config.samples)
    samples = [_values(function, angle) for angle in angles]
    slot_count = int(samples[0].size)
    if any(values.shape != (slot_count,) for values in samples):
        raise ValueError("fixed-slot root function changed vector shape")

    roots: list[FixedSlotRoot] = []
    sampled_exact_zeros = inactive_samples = rejected_crossings = rejected_refinements = 0
    for slot in range(slot_count):
        for index, angle in enumerate(angles):
            value = float(samples[index][slot])
            if not math.isfinite(value):
                inactive_samples += 1
                continue
            if abs(value) <= config.root_residual_tolerance:
                roots.append(FixedSlotRoot(float(angle), slot, abs(value)))
                sampled_exact_zeros += 1
        for index in range(len(angles) - 1):
            left, right = float(samples[index][slot]), float(samples[index + 1][slot])
            if not (math.isfinite(left) and math.isfinite(right)) or left == 0.0 or right == 0.0:
                continue
            if left * right >= 0.0:
                continue
            if max(abs(left), abs(right)) >= config.crossing_threshold:
                rejected_crossings += 1
                continue
            try:
                root = float(brentq(lambda angle: float(_values(function, angle)[slot]), float(angles[index]), float(angles[index + 1]), xtol=config.root_xtol))
                residual = abs(float(_values(function, root)[slot]))
            except (ValueError, RuntimeError, FloatingPointError):
                rejected_refinements += 1
                continue
            if not math.isfinite(residual) or residual > config.root_residual_tolerance:
                rejected_refinements += 1
                continue
            roots.append(FixedSlotRoot(root, slot, residual))

    strict_slots = {root.slot for root in roots if root.residual <= config.root_residual_tolerance}
    roots = [root for root in roots if root.slot not in strict_slots or root.residual <= config.root_residual_tolerance]
    unique: list[FixedSlotRoot] = []
    for root in sorted(roots, key=lambda value: (value.angle, value.slot)):
        if not any(root.slot == old.slot and abs(root.angle - old.angle) <= config.dedup_tolerance for old in unique):
            unique.append(root)
    return FixedSlotRootSearchResult(tuple(unique), slot_count, sampled_exact_zeros, inactive_samples, rejected_crossings, rejected_refinements)


def search_event_aware_fixed_slot_roots(
    margins: Callable[[float], ArrayLike],
    alignments: Callable[[float], ArrayLike] | None,
    config: EventAwareSearchConfig = EventAwareSearchConfig(),
) -> EventAwareRootSearchResult:
    """Find alignment roots only inside bounded, recovered feasible intervals.

    Each initial partition is independently maximized with bounded scalar
    refinement per slot. This can reveal a positive parabola wholly between
    coarse samples; all callback vectors retain their original fixed slots.
    """
    evaluations = alignment_evaluations = inactive = refinements = 0
    exhausted = False
    slot_count: int | None = None

    def margin_values(angle: float) -> Vector:
        nonlocal evaluations, slot_count, exhausted
        if evaluations >= config.maximum_event_evaluations:
            exhausted = True
            raise RuntimeError("event evaluation budget exhausted")
        value = _values(margins, angle)
        evaluations += 1
        if slot_count is None:
            slot_count = int(value.size)
        elif value.shape != (slot_count,):
            raise ValueError("fixed-slot margin callback changed vector shape")
        return value

    knots = np.linspace(config.minimum, config.maximum, config.initial_partitions + 1)
    try:
        coarse = [margin_values(angle) for angle in knots]
    except RuntimeError:
        return EventAwareRootSearchResult((), (), slot_count or 0, evaluations, 0, 0, True, 0)
    if slot_count is None:
        return EventAwareRootSearchResult((), (), 0, evaluations, 0, 0, True, 0)
    intervals: list[FeasibleInterval] = []
    for slot in range(slot_count):
        for index in range(len(knots) - 1):
            if exhausted:
                break
            left, right = float(knots[index]), float(knots[index + 1])
            left_value, right_value = float(coarse[index][slot]), float(coarse[index + 1][slot])
            if not (math.isfinite(left_value) or math.isfinite(right_value)):
                inactive += 1
                continue
            def negative(angle: float) -> float:
                return -float(margin_values(angle)[slot])
            try:
                if config.max_subdivision_depth == 0 or right - left <= config.minimum_interval_width_rad:
                    peak = 0.5 * (left + right)
                    peak_value = -negative(peak)
                else:
                    maximum = minimize_scalar(negative, bounds=(left, right), method="bounded", options={"xatol": config.minimum_interval_width_rad, "maxiter": config.max_subdivision_depth})
                    refinements += int(maximum.nfev)
                    peak, peak_value = float(maximum.x), -float(maximum.fun)
            except (RuntimeError, ValueError, FloatingPointError):
                continue
            if not math.isfinite(peak_value) or peak_value < config.event_margin_tolerance:
                continue
            # Locate each feasibility boundary only when a finite negative end
            # exists. Boundary-free portions remain clipped to the search range.
            interval_left, interval_right = left, right
            try:
                if math.isfinite(left_value) and left_value < config.event_margin_tolerance:
                    interval_left = float(brentq(lambda angle: float(margin_values(angle)[slot]) - config.event_margin_tolerance, left, peak, xtol=config.root_xtol))
                if math.isfinite(right_value) and right_value < config.event_margin_tolerance:
                    interval_right = float(brentq(lambda angle: float(margin_values(angle)[slot]) - config.event_margin_tolerance, peak, right, xtol=config.root_xtol))
            except (RuntimeError, ValueError, FloatingPointError):
                continue
            if interval_right - interval_left >= config.minimum_interval_width_rad or abs(peak_value - config.event_margin_tolerance) <= config.root_residual_tolerance:
                intervals.append(FeasibleInterval(slot, interval_left, interval_right))
    # Deterministically merge overlapping fragments generated by neighboring partitions.
    merged: list[FeasibleInterval] = []
    for interval in sorted(intervals, key=lambda value: (value.slot, value.left, value.right)):
        if merged and interval.slot == merged[-1].slot and interval.left <= merged[-1].right + config.dedup_tolerance:
            previous = merged.pop()
            merged.append(FeasibleInterval(previous.slot, previous.left, max(previous.right, interval.right)))
        else:
            merged.append(interval)

    if alignments is None:
        return EventAwareRootSearchResult((), tuple(merged), slot_count, evaluations, 0, refinements, exhausted, inactive)

    alignment = solve_alignment_roots(tuple(merged), alignments, config)
    return EventAwareRootSearchResult(
        alignment.roots,
        tuple(merged),
        slot_count,
        evaluations,
        alignment.alignment_evaluations,
        refinements,
        exhausted or alignment.budget_exhausted,
        inactive,
    )


def discover_feasible_intervals(
    margins: Callable[[float], ArrayLike],
    config: EventAwareSearchConfig = EventAwareSearchConfig(),
) -> EventAwareRootSearchResult:
    """Recover fixed-slot feasible intervals without evaluating alignment."""
    return search_event_aware_fixed_slot_roots(margins, None, config)


def solve_alignment_roots(
    intervals: tuple[FeasibleInterval, ...],
    alignments: Callable[[float], ArrayLike],
    config: EventAwareSearchConfig = EventAwareSearchConfig(),
) -> EventAwareRootSearchResult:
    """Refine alignment roots solely inside previously certified intervals."""
    roots: list[FixedSlotRoot] = []
    evaluations = 0
    slots: int | None = None
    exhausted = False

    def alignment_values(angle: float) -> Vector:
        nonlocal evaluations, slots, exhausted
        if evaluations >= config.maximum_event_evaluations:
            exhausted = True
            raise _EvaluationBudgetExhausted
        value = _values(alignments, angle)
        evaluations += 1
        if slots is None:
            slots = int(value.size)
        if value.shape != (slots,):
            raise ValueError("fixed-slot alignment callback changed vector shape")
        return value

    def alignment_value(angle: float, slot: int) -> float:
        return float(alignment_values(angle)[slot])

    for interval in intervals:
        if exhausted:
            break
        points = np.linspace(
            interval.left, interval.right, config.alignment_samples_per_interval
        )
        try:
            values = [alignment_values(float(point)) for point in points]
        except _EvaluationBudgetExhausted:
            break
        for index in range(len(points) - 1):
            if exhausted:
                break
            left = float(values[index][interval.slot])
            right = float(values[index + 1][interval.slot])
            if not (math.isfinite(left) and math.isfinite(right)):
                continue
            candidates = (
                [float(points[index])]
                if abs(left) <= config.alignment_residual_tolerance
                else []
            )
            if left * right < 0.0:
                try:
                    candidates.append(
                        float(
                            brentq(
                                lambda angle: alignment_value(angle, interval.slot),
                                float(points[index]),
                                float(points[index + 1]),
                                xtol=config.alignment_root_xtol,
                            )
                        )
                    )
                except _EvaluationBudgetExhausted:
                    break
            if not candidates:
                try:
                    tangent = minimize_scalar(
                        lambda angle: abs(alignment_value(angle, interval.slot)),
                        bounds=(float(points[index]), float(points[index + 1])),
                        method="bounded",
                        options={
                            "xatol": config.alignment_root_xtol,
                            "maxiter": config.alignment_maximum_iterations,
                        },
                    )
                    if tangent.fun <= config.alignment_residual_tolerance:
                        candidates.append(float(tangent.x))
                except _EvaluationBudgetExhausted:
                    break
                except (ValueError, RuntimeError, FloatingPointError):
                    pass
            for angle in candidates:
                try:
                    residual = abs(alignment_value(angle, interval.slot))
                except _EvaluationBudgetExhausted:
                    break
                if residual <= config.alignment_residual_tolerance:
                    roots.append(FixedSlotRoot(angle, interval.slot, residual))
    unique_roots: list[FixedSlotRoot] = []
    for root in sorted(roots, key=lambda value: (value.angle, value.slot)):
        duplicate_index = next(
            (
                index
                for index, old in enumerate(unique_roots)
                if root.slot == old.slot
                and abs(root.angle - old.angle) <= config.alignment_dedup_tolerance
            ),
            None,
        )
        if duplicate_index is None:
            unique_roots.append(root)
        elif (root.residual, root.angle) < (
            unique_roots[duplicate_index].residual,
            unique_roots[duplicate_index].angle,
        ):
            unique_roots[duplicate_index] = root
    unique_roots.sort(key=lambda value: (value.angle, value.slot))
    return EventAwareRootSearchResult(
        tuple(unique_roots), intervals, slots or 0, 0, evaluations, 0, exhausted, 0
    )
