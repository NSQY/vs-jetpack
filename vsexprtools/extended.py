from __future__ import annotations

from vstools import get_neutral_value, get_peak_value, vs

from .operators import ExprOperators

__all__ = ["ExprFunctions"]


class ExprFunctions:

    def __init__(self, clips: list[vs.VideoNode]) -> None:
        self.clips = clips
        self.op = ExprOperators()

        assert self.clip.format

        self.peak = get_peak_value(self.clip)
        self.neutral = get_neutral_value(self.clip)

    def invert(self, x: float) -> float:
        """Inverts a value."""
        ...