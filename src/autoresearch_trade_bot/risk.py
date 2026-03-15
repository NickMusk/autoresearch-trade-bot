from __future__ import annotations

from typing import Dict, Mapping

from .config import RiskLimits


class RiskManager:
    """Applies explicit portfolio constraints outside strategy code."""

    def __init__(self, limits: RiskLimits) -> None:
        self.limits = limits

    def apply(
        self,
        target_weights: Mapping[str, float],
        current_weights: Mapping[str, float],
    ) -> Dict[str, float]:
        clipped = {
            symbol: max(
                -self.limits.max_symbol_weight,
                min(self.limits.max_symbol_weight, weight),
            )
            for symbol, weight in target_weights.items()
        }

        gross = sum(abs(weight) for weight in clipped.values())
        if gross > self.limits.max_gross_leverage and gross > 0:
            scale = self.limits.max_gross_leverage / gross
            clipped = {
                symbol: weight * scale for symbol, weight in clipped.items()
            }

        turnover = self._turnover(current_weights, clipped)
        if turnover > self.limits.max_turnover_per_step and turnover > 0:
            factor = self.limits.max_turnover_per_step / turnover
            adjusted = {}
            all_symbols = set(current_weights) | set(clipped)
            for symbol in all_symbols:
                current = current_weights.get(symbol, 0.0)
                target = clipped.get(symbol, 0.0)
                adjusted[symbol] = current + (target - current) * factor
            clipped = adjusted

        return {
            symbol: weight
            for symbol, weight in clipped.items()
            if abs(weight) > 1e-12
        }

    @staticmethod
    def _turnover(
        current_weights: Mapping[str, float],
        target_weights: Mapping[str, float],
    ) -> float:
        all_symbols = set(current_weights) | set(target_weights)
        return sum(
            abs(target_weights.get(symbol, 0.0) - current_weights.get(symbol, 0.0))
            for symbol in all_symbols
        )
