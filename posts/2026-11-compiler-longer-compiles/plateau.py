"""When does a longer compile pay? Read it off the incumbent stream of one compile.

A 4colors compile streams every better circuit as it finds it (`job.incumbents()`). From the times and two-qubit
counts of that stream, three numbers tell you whether a longer budget is worth trying:

* `last_improvement_s`: when the last better circuit arrived;
* `time_to_within(points, frac)`: the first time the count was within `frac` of the final count;
* `late_gain(points, budget_s)`: the relative gain in the second half of the budget.

`advice(points, budget_s)` turns them into the post's rule of thumb: if the second half of the budget still
bought more than `frac` (default 1 %), try a longer budget; otherwise the answer you have is the answer.
All functions take [(seconds, two-qubit gates), ...] in arrival order and work for any arm's best-so-far curve.
"""
from __future__ import annotations

from typing import Iterable, List, Optional, Sequence, Tuple

Point = Tuple[float, float]


def best_so_far(points: Iterable[Sequence[float]]) -> List[Point]:
    """The strictly improving steps of a stream, in time order: [(t, value), ...] with falling values."""
    out: List[Point] = []
    for t, v in sorted(((float(t), float(v)) for t, v in points if v is not None), key=lambda p: p[0]):
        if not out or v < out[-1][1]:
            out.append((t, v))
    return out


def value_at(points: Iterable[Sequence[float]], t: float) -> Optional[float]:
    """The best value found by time t (None if nothing had arrived yet)."""
    cur = None
    for tt, v in best_so_far(points):
        if tt > t:
            break
        cur = v
    return cur


def last_improvement_s(points: Iterable[Sequence[float]]) -> Optional[float]:
    steps = best_so_far(points)
    return steps[-1][0] if steps else None


def time_to_within(points: Iterable[Sequence[float]], frac: float = 0.01) -> Optional[float]:
    """The first time the best-so-far value was within `frac` of the final value (final x (1 + frac))."""
    steps = best_so_far(points)
    if not steps:
        return None
    goal = steps[-1][1] * (1 + frac)
    return next(t for t, v in steps if v <= goal)


def late_gain(points: Iterable[Sequence[float]], budget_s: float) -> Optional[float]:
    """Relative gain bought by the second half of the budget: 1 - value(T) / value(T/2). None if nothing had
    arrived by T/2."""
    pts = list(points)
    half, end = value_at(pts, budget_s / 2), value_at(pts, budget_s)
    if half is None or end is None or half == 0:
        return None
    return 1.0 - end / half


def advice(points: Iterable[Sequence[float]], budget_s: float, frac: float = 0.01) -> str:
    """"longer may pay" if the second half of the budget still gained more than `frac`, else "plateau"
    ("no answer" if nothing arrived by half the budget)."""
    g = late_gain(list(points), budget_s)
    if g is None:
        return "no answer"
    return "longer may pay" if g > frac else "plateau"
