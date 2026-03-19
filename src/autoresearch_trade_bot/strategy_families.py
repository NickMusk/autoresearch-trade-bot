from __future__ import annotations

import ast
import pprint
from dataclasses import dataclass
from typing import Any, Mapping, Sequence


FAMILY_MOMENTUM = "momentum"
FAMILY_MEAN_REVERSION = "mean_reversion"
FAMILY_EMA_TREND = "ema_trend"
FAMILY_VOLATILITY_BREAKOUT = "volatility_breakout"

WAVE1_FAMILIES = (
    FAMILY_MEAN_REVERSION,
    FAMILY_EMA_TREND,
    FAMILY_VOLATILITY_BREAKOUT,
)


@dataclass(frozen=True)
class StrategyFamilyProfile:
    family_id: str
    display_name: str
    strategy_name: str
    default_train_config: dict[str, Any]
    attempt_role_specs: tuple[dict[str, str], ...]
    prompt_directions: tuple[str, ...]


PROFILES: dict[str, StrategyFamilyProfile] = {
    FAMILY_MOMENTUM: StrategyFamilyProfile(
        family_id=FAMILY_MOMENTUM,
        display_name="Cross-Sectional Momentum",
        strategy_name="configurable-momentum",
        default_train_config={
            "gross_target": 0.5,
            "lookback_bars": 24,
            "min_signal_strength": 0.0,
            "ranking_mode": "risk_adjusted",
            "regime_lookback_bars": 36,
            "regime_threshold": 0.015,
            "top_k": 1,
            "use_regime_filter": False,
            "min_cross_sectional_spread": 0.0,
            "volatility_floor": 0.0,
            "reversal_bias_weight": 0.0,
            "funding_penalty_weight": 0.0,
        },
        attempt_role_specs=(
            {
                "name": "exploit_baseline",
                "objective": "Stay close to the current momentum winner and improve score with one or two meaningful knob changes.",
                "constraints": "Avoid stacking restrictive filters that kill trading activity.",
            },
            {
                "name": "selectivity_without_no_trade",
                "objective": "Improve selectivity without collapsing into a no-trade strategy.",
                "constraints": "If you tighten thresholds, do it mildly and keep the strategy active.",
            },
            {
                "name": "alternative_signal_family",
                "objective": "Explore funding, reversal, or volatility-aware momentum variations.",
                "constraints": "Do not only tweak lookback. Shift the signal expression while staying cheap to evaluate.",
            },
        ),
        prompt_directions=(
            "Add selectivity carefully with min_cross_sectional_spread or min_signal_strength.",
            "Use volatility_floor to stabilize risk-adjusted ranking.",
            "Use reversal_bias_weight or funding_penalty_weight to avoid crowded or overextended entries.",
        ),
    ),
    FAMILY_MEAN_REVERSION: StrategyFamilyProfile(
        family_id=FAMILY_MEAN_REVERSION,
        display_name="Mean Reversion",
        strategy_name="mean-reversion-bbands-rsi",
        default_train_config={
            "gross_target": 0.5,
            "lookback_bars": 24,
            "rsi_period": 14,
            "rsi_lower": 35.0,
            "rsi_upper": 65.0,
            "band_std_mult": 1.5,
            "top_k": 1,
            "min_reversion_score": 0.0,
            "volatility_floor": 0.0,
            "use_trend_filter": False,
            "trend_lookback_bars": 48,
        },
        attempt_role_specs=(
            {
                "name": "exploit_baseline",
                "objective": "Refine the current mean-reversion baseline with modest threshold or lookback adjustments.",
                "constraints": "Preserve trade frequency. Avoid combining extreme RSI thresholds with very wide bands.",
            },
            {
                "name": "entry_selectivity",
                "objective": "Improve entry quality by balancing RSI and band-based mean reversion signals.",
                "constraints": "Do not set RSI thresholds so strict that no symbols qualify.",
            },
            {
                "name": "regime_aware_reversion",
                "objective": "Explore trend-aware or volatility-aware mean reversion variants.",
                "constraints": "Keep the strategy simple and cheap; no multi-stage confirmation logic.",
            },
        ),
        prompt_directions=(
            "Balance RSI thresholds with band width instead of tightening both at once.",
            "Use a mild trend filter to avoid fading strong trends too aggressively.",
            "Use volatility_floor to stabilize z-score estimates rather than increasing band width too much.",
        ),
    ),
    FAMILY_EMA_TREND: StrategyFamilyProfile(
        family_id=FAMILY_EMA_TREND,
        display_name="EMA Trend",
        strategy_name="ema-trend-filter",
        default_train_config={
            "gross_target": 0.5,
            "fast_ema_bars": 12,
            "slow_ema_bars": 48,
            "trend_ema_bars": 96,
            "top_k": 1,
            "min_signal_strength": 0.0,
            "use_trend_filter": True,
            "volume_confirmation": 0.0,
            "volatility_floor": 0.0,
        },
        attempt_role_specs=(
            {
                "name": "exploit_baseline",
                "objective": "Refine the current EMA trend baseline with small period or filter adjustments.",
                "constraints": "Do not flip the family into a breakout or reversion system.",
            },
            {
                "name": "trend_filter_quality",
                "objective": "Improve signal quality with better trend filtering or volume confirmation.",
                "constraints": "Do not make confirmation so strict that the strategy stops trading.",
            },
            {
                "name": "faster_vs_slower_response",
                "objective": "Explore a meaningfully different EMA response profile.",
                "constraints": "Keep fast < slow < trend and avoid cosmetic changes.",
            },
        ),
        prompt_directions=(
            "Experiment with faster or slower EMA spacing without breaking the ordering fast < slow < trend.",
            "Use volume_confirmation mildly as a signal quality filter.",
            "Use volatility_floor to avoid unstable normalized scores on quiet assets.",
        ),
    ),
    FAMILY_VOLATILITY_BREAKOUT: StrategyFamilyProfile(
        family_id=FAMILY_VOLATILITY_BREAKOUT,
        display_name="Volatility Breakout",
        strategy_name="vol-breakout-donchian",
        default_train_config={
            "gross_target": 0.5,
            "channel_bars": 24,
            "atr_lookback_bars": 14,
            "atr_multiplier": 1.0,
            "breakout_buffer": 0.0,
            "top_k": 1,
            "min_breakout_score": 0.0,
            "use_trend_filter": False,
            "trend_lookback_bars": 72,
        },
        attempt_role_specs=(
            {
                "name": "exploit_baseline",
                "objective": "Refine the current breakout baseline with modest channel or ATR adjustments.",
                "constraints": "Preserve breakout behavior; do not drift into generic momentum or no-trade filters.",
            },
            {
                "name": "breakout_selectivity",
                "objective": "Tune breakout quality via ATR and breakout buffer without eliminating trades.",
                "constraints": "Do not combine a very long channel with a very large breakout buffer.",
            },
            {
                "name": "trend_confirmed_breakout",
                "objective": "Explore trend-confirmed breakout variants that still remain active.",
                "constraints": "Keep the confirmation lightweight and cheap to compute.",
            },
        ),
        prompt_directions=(
            "Balance channel length with breakout_buffer instead of increasing both aggressively.",
            "Use atr_multiplier to normalize breakout strength rather than just demanding wider channels.",
            "Use a mild trend filter to avoid false breakouts, but keep the strategy trading.",
        ),
    ),
}


def get_strategy_family_profile(strategy_family: str) -> StrategyFamilyProfile:
    return PROFILES.get(strategy_family, PROFILES[FAMILY_MOMENTUM])


def normalize_train_config(
    train_config: Mapping[str, Any],
    *,
    strategy_family: str = FAMILY_MOMENTUM,
) -> dict[str, Any]:
    profile = get_strategy_family_profile(strategy_family)
    normalized = dict(profile.default_train_config)
    normalized.update(dict(train_config))
    return normalized


def extract_strategy_family(train_text: str) -> str:
    try:
        tree = ast.parse(train_text)
    except SyntaxError:
        return FAMILY_MOMENTUM
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "STRATEGY_FAMILY":
                    try:
                        value = ast.literal_eval(node.value)
                    except Exception:
                        return FAMILY_MOMENTUM
                    if isinstance(value, str) and value in PROFILES:
                        return value
    return FAMILY_MOMENTUM


def family_attempt_role_specs(strategy_family: str) -> tuple[dict[str, str], ...]:
    return get_strategy_family_profile(strategy_family).attempt_role_specs


def family_prompt_directions(strategy_family: str) -> tuple[str, ...]:
    return get_strategy_family_profile(strategy_family).prompt_directions


def render_train_file(
    train_config: Mapping[str, Any],
    *,
    strategy_family: str = FAMILY_MOMENTUM,
) -> str:
    payload = pprint.pformat(
        normalize_train_config(train_config, strategy_family=strategy_family),
        sort_dicts=True,
        width=88,
    )
    profile = get_strategy_family_profile(strategy_family)
    if strategy_family == FAMILY_MEAN_REVERSION:
        return _render_mean_reversion_file(payload, profile.strategy_name)
    if strategy_family == FAMILY_EMA_TREND:
        return _render_ema_trend_file(payload, profile.strategy_name)
    if strategy_family == FAMILY_VOLATILITY_BREAKOUT:
        return _render_volatility_breakout_file(payload, profile.strategy_name)
    return _render_momentum_file(payload, profile.strategy_name)


def deterministic_mutation_specs(
    strategy_family: str,
    config: Mapping[str, Any],
) -> list[dict[str, Any]]:
    profile_config = normalize_train_config(config, strategy_family=strategy_family)
    proposals: list[dict[str, Any]] = []
    if strategy_family == FAMILY_MEAN_REVERSION:
        for key, values in (
            ("lookback_bars", (12, 24, 36)),
            ("rsi_period", (10, 14, 21)),
            ("rsi_lower", (30.0, 35.0, 40.0)),
            ("rsi_upper", (60.0, 65.0, 70.0)),
            ("band_std_mult", (1.0, 1.5, 2.0)),
            ("top_k", (1, 2)),
        ):
            proposals.extend(_mutation_specs_for_values(profile_config, key, values))
        return proposals
    if strategy_family == FAMILY_EMA_TREND:
        for key, values in (
            ("fast_ema_bars", (8, 12, 16)),
            ("slow_ema_bars", (32, 48, 64)),
            ("trend_ema_bars", (72, 96, 144)),
            ("top_k", (1, 2)),
            ("min_signal_strength", (0.0, 0.02, 0.05)),
        ):
            proposals.extend(_mutation_specs_for_values(profile_config, key, values))
        proposals.extend(_mutation_specs_for_values(profile_config, "use_trend_filter", (False, True)))
        return proposals
    if strategy_family == FAMILY_VOLATILITY_BREAKOUT:
        for key, values in (
            ("channel_bars", (12, 24, 36)),
            ("atr_lookback_bars", (10, 14, 21)),
            ("atr_multiplier", (0.5, 1.0, 1.5)),
            ("breakout_buffer", (0.0, 0.25, 0.5)),
            ("top_k", (1, 2)),
        ):
            proposals.extend(_mutation_specs_for_values(profile_config, key, values))
        proposals.extend(_mutation_specs_for_values(profile_config, "use_trend_filter", (False, True)))
        return proposals
    for key, values, label_key in (
        ("lookback_bars", (12, 24, 36), "lookback"),
        ("top_k", (1, 2), "top-k"),
        ("gross_target", (0.25, 0.5, 0.75, 1.0), "gross"),
        ("ranking_mode", ("raw_return", "risk_adjusted"), "ranking"),
        ("use_regime_filter", (False, True), "regime"),
        ("min_signal_strength", (0.0, 0.02, 0.05), "signal-floor"),
        ("min_cross_sectional_spread", (0.0, 0.05, 0.1), "spread-floor"),
    ):
        proposals.extend(_mutation_specs_for_values(profile_config, key, values, label_key=label_key))
    return proposals


def validate_train_candidate_semantics(
    strategy_family: str,
    *,
    candidate_config: Mapping[str, Any],
    current_config: Mapping[str, Any],
    symbol_count: int,
) -> tuple[bool, str]:
    candidate = normalize_train_config(candidate_config, strategy_family=strategy_family)
    current = normalize_train_config(current_config, strategy_family=strategy_family)
    if candidate == current:
        return False, "no_op_train_config"
    if int(candidate.get("top_k", 1)) < 1:
        return False, "invalid_top_k"
    if int(candidate.get("top_k", 1)) * 2 > symbol_count:
        return False, "top_k_exceeds_cross_sectional_capacity"
    if float(candidate.get("gross_target", 0.5)) <= 0.0 or float(candidate.get("gross_target", 0.5)) > 1.5:
        return False, "invalid_gross_target"
    if strategy_family == FAMILY_MEAN_REVERSION:
        if not (0.0 < float(candidate["rsi_lower"]) < 50.0 < float(candidate["rsi_upper"]) < 100.0):
            return False, "invalid_rsi_thresholds"
        if float(candidate["band_std_mult"]) <= 0.0 or float(candidate["band_std_mult"]) > 4.0:
            return False, "invalid_band_std_mult"
        if (
            float(candidate["band_std_mult"]) > 2.5
            and float(candidate["rsi_lower"]) < 25.0
            and float(candidate["rsi_upper"]) > 75.0
        ):
            return False, "likely_no_trade_reversion_stack"
        return True, ""
    if strategy_family == FAMILY_EMA_TREND:
        if not (
            int(candidate["fast_ema_bars"]) < int(candidate["slow_ema_bars"]) < int(candidate["trend_ema_bars"])
        ):
            return False, "invalid_ema_ordering"
        if float(candidate["volume_confirmation"]) > 3.0:
            return False, "likely_no_trade_volume_confirmation"
        if float(candidate["min_signal_strength"]) > 0.15:
            return False, "likely_no_trade_signal_threshold"
        return True, ""
    if strategy_family == FAMILY_VOLATILITY_BREAKOUT:
        if int(candidate["channel_bars"]) <= int(candidate["atr_lookback_bars"]):
            return False, "invalid_breakout_window_order"
        if float(candidate["atr_multiplier"]) <= 0.0 or float(candidate["atr_multiplier"]) > 4.0:
            return False, "invalid_atr_multiplier"
        if float(candidate["breakout_buffer"]) > 1.5:
            return False, "likely_no_trade_breakout_buffer"
        if int(candidate["channel_bars"]) > 60 and float(candidate["breakout_buffer"]) > 0.5:
            return False, "likely_no_trade_breakout_stack"
        return True, ""
    if str(candidate["ranking_mode"]) not in {"raw_return", "risk_adjusted"}:
        return False, "invalid_ranking_mode"
    if (
        float(candidate["min_signal_strength"]) > 0.10
        and float(candidate["min_cross_sectional_spread"]) > 0.10
    ):
        return False, "likely_no_trade_filter_stack"
    if float(candidate["min_signal_strength"]) > 0.12:
        return False, "likely_no_trade_signal_threshold"
    if float(candidate["min_cross_sectional_spread"]) > 0.15:
        return False, "likely_no_trade_spread_threshold"
    if bool(candidate["use_regime_filter"]) and float(candidate["regime_threshold"]) > 0.06:
        return False, "likely_no_trade_regime_threshold"
    return True, ""


def config_traits(strategy_family: str, config: Mapping[str, Any]) -> list[str]:
    if not config:
        return []
    traits = [f"family={strategy_family}", f"top_k={config.get('top_k', '')}"]
    if strategy_family == FAMILY_MEAN_REVERSION:
        traits.extend(
            [
                f"band_std_mult={_bucket(float(config.get('band_std_mult', 0.0)), (1.2, 2.0))}",
                f"rsi_width={_bucket(float(config.get('rsi_upper', 0.0)) - float(config.get('rsi_lower', 0.0)), (25.0, 40.0))}",
            ]
        )
        if bool(config.get("use_trend_filter", False)):
            traits.append("trend_filter=on")
    elif strategy_family == FAMILY_EMA_TREND:
        traits.extend(
            [
                f"fast_ema={config.get('fast_ema_bars', '')}",
                f"slow_ema={config.get('slow_ema_bars', '')}",
            ]
        )
        if bool(config.get("use_trend_filter", False)):
            traits.append("trend_filter=on")
    elif strategy_family == FAMILY_VOLATILITY_BREAKOUT:
        traits.extend(
            [
                f"channel={config.get('channel_bars', '')}",
                f"breakout_buffer={_bucket(float(config.get('breakout_buffer', 0.0)), (0.1, 0.4))}",
            ]
        )
        if bool(config.get("use_trend_filter", False)):
            traits.append("trend_filter=on")
    else:
        traits.extend(
            [
                f"ranking_mode={config.get('ranking_mode', '')}",
                f"lookback={config.get('lookback_bars', '')}",
            ]
        )
        if bool(config.get("use_regime_filter", False)):
            traits.append("regime_filter=on")
    return traits


def _mutation_specs_for_values(
    config: Mapping[str, Any],
    key: str,
    values: Sequence[Any],
    *,
    label_key: str | None = None,
) -> list[dict[str, Any]]:
    proposals = []
    for value in values:
        if value != config.get(key):
            proposals.append(
                {
                    "label": f"{label_key or key}-{value}",
                    "config_updates": {key: value},
                }
            )
    return proposals


def _bucket(value: float, cutoffs: tuple[float, float]) -> str:
    low, high = cutoffs
    if value <= low:
        return "low"
    if value <= high:
        return "mid"
    return "high"


def _render_momentum_file(payload: str, strategy_name: str) -> str:
    return "\n".join(
        [
            "from __future__ import annotations",
            "",
            "from dataclasses import dataclass",
            "import math",
            "import statistics",
            "from typing import Dict, Mapping, Sequence",
            "",
            "from autoresearch_trade_bot.models import Bar",
            "from autoresearch_trade_bot.strategy import Strategy",
            "",
            f"TRAIN_CONFIG = {payload}",
            f'STRATEGY_NAME = "{strategy_name}"',
            f'STRATEGY_FAMILY = "{FAMILY_MOMENTUM}"',
            "",
            "",
            "@dataclass(frozen=True)",
            "class ConfigurableMomentumStrategy:",
            "    lookback_bars: int",
            "    top_k: int",
            "    gross_target: float",
            "    ranking_mode: str",
            "    use_regime_filter: bool",
            "    regime_lookback_bars: int",
            "    regime_threshold: float",
            "    min_signal_strength: float",
            "    min_cross_sectional_spread: float",
            "    volatility_floor: float",
            "    reversal_bias_weight: float",
            "    funding_penalty_weight: float",
            "",
            "    def target_weights(self, history_by_symbol: Mapping[str, Sequence[Bar]]) -> Dict[str, float]:",
            "        if not history_by_symbol:",
            "            return {}",
            "        if self.use_regime_filter and not self._passes_regime_filter(history_by_symbol):",
            "            return {}",
            "        ready_scores = []",
            "        for symbol, history in history_by_symbol.items():",
            "            if len(history) <= self.lookback_bars:",
            "                continue",
            "            score = self._score(history)",
            "            if abs(score) < self.min_signal_strength:",
            "                continue",
            "            ready_scores.append((symbol, score))",
            "        if len(ready_scores) < self.top_k * 2:",
            "            return {}",
            "        ranked = sorted(ready_scores, key=lambda item: item[1], reverse=True)",
            "        cross_sectional_spread = ranked[0][1] - ranked[-1][1]",
            "        if cross_sectional_spread < self.min_cross_sectional_spread:",
            "            return {}",
            "        longs = ranked[: self.top_k]",
            "        shorts = ranked[-self.top_k:]",
            "        side_gross = self.gross_target / 2.0",
            "        long_weight = side_gross / len(longs)",
            "        short_weight = -side_gross / len(shorts)",
            "        weights: Dict[str, float] = {}",
            "        for symbol, _score in longs:",
            "            weights[symbol] = long_weight",
            "        for symbol, _score in shorts:",
            "            weights[symbol] = short_weight",
            "        return weights",
            "",
            "    def _score(self, history: Sequence[Bar]) -> float:",
            "        recent = history[-(self.lookback_bars + 1):]",
            "        start_close = recent[0].close",
            "        end_close = recent[-1].close",
            "        raw_return = (end_close / start_close) - 1.0",
            "        one_bar_returns = []",
            "        for index in range(1, len(recent)):",
            "            previous = recent[index - 1].close",
            "            current = recent[index].close",
            "            one_bar_returns.append((current / previous) - 1.0)",
            "        volatility = max(statistics.pstdev(one_bar_returns), self.volatility_floor)",
            "        if math.isclose(volatility, 0.0):",
            "            base_signal = raw_return",
            "        else:",
            "            base_signal = raw_return if self.ranking_mode == 'raw_return' else raw_return / volatility",
            "        last_bar_return = one_bar_returns[-1] if one_bar_returns else 0.0",
            "        funding_penalty = self.funding_penalty_weight * recent[-1].funding_rate",
            "        reversal_penalty = self.reversal_bias_weight * last_bar_return",
            "        return base_signal - funding_penalty - reversal_penalty",
            "",
            "    def _passes_regime_filter(self, history_by_symbol: Mapping[str, Sequence[Bar]]) -> bool:",
            "        anchor_symbol = 'BTCUSDT' if 'BTCUSDT' in history_by_symbol else next(iter(history_by_symbol))",
            "        history = history_by_symbol[anchor_symbol]",
            "        if len(history) <= self.regime_lookback_bars:",
            "            return True",
            "        recent = history[-(self.regime_lookback_bars + 1):]",
            "        regime_return = (recent[-1].close / recent[0].close) - 1.0",
            "        return abs(regime_return) >= self.regime_threshold",
            "",
            "",
            "def build_strategy(_dataset_spec=None) -> Strategy:",
            "    return ConfigurableMomentumStrategy(",
            "        lookback_bars=int(TRAIN_CONFIG['lookback_bars']),",
            "        top_k=int(TRAIN_CONFIG['top_k']),",
            "        gross_target=float(TRAIN_CONFIG['gross_target']),",
            "        ranking_mode=str(TRAIN_CONFIG['ranking_mode']),",
            "        use_regime_filter=bool(TRAIN_CONFIG['use_regime_filter']),",
            "        regime_lookback_bars=int(TRAIN_CONFIG['regime_lookback_bars']),",
            "        regime_threshold=float(TRAIN_CONFIG['regime_threshold']),",
            "        min_signal_strength=float(TRAIN_CONFIG['min_signal_strength']),",
            "        min_cross_sectional_spread=float(TRAIN_CONFIG['min_cross_sectional_spread']),",
            "        volatility_floor=float(TRAIN_CONFIG['volatility_floor']),",
            "        reversal_bias_weight=float(TRAIN_CONFIG['reversal_bias_weight']),",
            "        funding_penalty_weight=float(TRAIN_CONFIG['funding_penalty_weight']),",
            "    )",
            "",
        ]
    )


def _render_mean_reversion_file(payload: str, strategy_name: str) -> str:
    return "\n".join(
        [
            "from __future__ import annotations",
            "",
            "from dataclasses import dataclass",
            "import math",
            "import statistics",
            "from typing import Dict, Mapping, Sequence",
            "",
            "from autoresearch_trade_bot.models import Bar",
            "from autoresearch_trade_bot.strategy import Strategy",
            "",
            f"TRAIN_CONFIG = {payload}",
            f'STRATEGY_NAME = "{strategy_name}"',
            f'STRATEGY_FAMILY = "{FAMILY_MEAN_REVERSION}"',
            "",
            "",
            "def _rsi(history: Sequence[Bar], period: int) -> float:",
            "    if len(history) <= period:",
            "        return 50.0",
            "    gains = []",
            "    losses = []",
            "    recent = history[-(period + 1):]",
            "    for index in range(1, len(recent)):",
            "        delta = recent[index].close - recent[index - 1].close",
            "        gains.append(max(delta, 0.0))",
            "        losses.append(abs(min(delta, 0.0)))",
            "    avg_gain = statistics.fmean(gains) if gains else 0.0",
            "    avg_loss = statistics.fmean(losses) if losses else 0.0",
            "    if math.isclose(avg_loss, 0.0):",
            "        return 100.0 if avg_gain > 0.0 else 50.0",
            "    rs = avg_gain / avg_loss",
            "    return 100.0 - (100.0 / (1.0 + rs))",
            "",
            "",
            "@dataclass(frozen=True)",
            "class MeanReversionStrategy:",
            "    lookback_bars: int",
            "    rsi_period: int",
            "    rsi_lower: float",
            "    rsi_upper: float",
            "    band_std_mult: float",
            "    top_k: int",
            "    gross_target: float",
            "    min_reversion_score: float",
            "    volatility_floor: float",
            "    use_trend_filter: bool",
            "    trend_lookback_bars: int",
            "",
            "    def target_weights(self, history_by_symbol: Mapping[str, Sequence[Bar]]) -> Dict[str, float]:",
            "        if not history_by_symbol:",
            "            return {}",
            "        ready_scores = []",
            "        for symbol, history in history_by_symbol.items():",
            "            if len(history) <= max(self.lookback_bars, self.rsi_period, self.trend_lookback_bars):",
            "                continue",
            "            score = self._score(history)",
            "            if abs(score) < self.min_reversion_score:",
            "                continue",
            "            ready_scores.append((symbol, score))",
            "        positive = [item for item in ready_scores if item[1] > 0.0]",
            "        negative = [item for item in ready_scores if item[1] < 0.0]",
            "        if len(positive) < self.top_k or len(negative) < self.top_k:",
            "            return {}",
            "        longs = sorted(positive, key=lambda item: item[1], reverse=True)[: self.top_k]",
            "        shorts = sorted(negative, key=lambda item: item[1])[: self.top_k]",
            "        side_gross = self.gross_target / 2.0",
            "        weights: Dict[str, float] = {}",
            "        for symbol, _score in longs:",
            "            weights[symbol] = side_gross / len(longs)",
            "        for symbol, _score in shorts:",
            "            weights[symbol] = -(side_gross / len(shorts))",
            "        return weights",
            "",
            "    def _score(self, history: Sequence[Bar]) -> float:",
            "        closes = [bar.close for bar in history[-self.lookback_bars:]]",
            "        mean_close = statistics.fmean(closes)",
            "        std_close = max(statistics.pstdev(closes), self.volatility_floor)",
            "        if math.isclose(std_close, 0.0):",
            "            return 0.0",
            "        zscore = (history[-1].close - mean_close) / std_close",
            "        rsi_value = _rsi(history, self.rsi_period)",
            "        if self.use_trend_filter:",
            "            trend_start = history[-self.trend_lookback_bars].close",
            "            trend_now = history[-1].close",
            "            trend_return = (trend_now / trend_start) - 1.0",
            "            if zscore < 0.0 and trend_return < -0.03:",
            "                return 0.0",
            "            if zscore > 0.0 and trend_return > 0.03:",
            "                return 0.0",
            "        if zscore <= -self.band_std_mult and rsi_value <= self.rsi_lower:",
            "            return abs(zscore) + ((self.rsi_lower - rsi_value) / 100.0)",
            "        if zscore >= self.band_std_mult and rsi_value >= self.rsi_upper:",
            "            return -(abs(zscore) + ((rsi_value - self.rsi_upper) / 100.0))",
            "        return 0.0",
            "",
            "",
            "def build_strategy(_dataset_spec=None) -> Strategy:",
            "    return MeanReversionStrategy(",
            "        lookback_bars=int(TRAIN_CONFIG['lookback_bars']),",
            "        rsi_period=int(TRAIN_CONFIG['rsi_period']),",
            "        rsi_lower=float(TRAIN_CONFIG['rsi_lower']),",
            "        rsi_upper=float(TRAIN_CONFIG['rsi_upper']),",
            "        band_std_mult=float(TRAIN_CONFIG['band_std_mult']),",
            "        top_k=int(TRAIN_CONFIG['top_k']),",
            "        gross_target=float(TRAIN_CONFIG['gross_target']),",
            "        min_reversion_score=float(TRAIN_CONFIG['min_reversion_score']),",
            "        volatility_floor=float(TRAIN_CONFIG['volatility_floor']),",
            "        use_trend_filter=bool(TRAIN_CONFIG['use_trend_filter']),",
            "        trend_lookback_bars=int(TRAIN_CONFIG['trend_lookback_bars']),",
            "    )",
            "",
        ]
    )


def _render_ema_trend_file(payload: str, strategy_name: str) -> str:
    return "\n".join(
        [
            "from __future__ import annotations",
            "",
            "from dataclasses import dataclass",
            "import math",
            "import statistics",
            "from typing import Dict, Mapping, Sequence",
            "",
            "from autoresearch_trade_bot.models import Bar",
            "from autoresearch_trade_bot.strategy import Strategy",
            "",
            f"TRAIN_CONFIG = {payload}",
            f'STRATEGY_NAME = "{strategy_name}"',
            f'STRATEGY_FAMILY = "{FAMILY_EMA_TREND}"',
            "",
            "",
            "def _ema(closes: Sequence[float], period: int) -> float:",
            "    alpha = 2.0 / (period + 1.0)",
            "    value = closes[0]",
            "    for close in closes[1:]:",
            "        value = alpha * close + ((1.0 - alpha) * value)",
            "    return value",
            "",
            "",
            "@dataclass(frozen=True)",
            "class EMATrendStrategy:",
            "    fast_ema_bars: int",
            "    slow_ema_bars: int",
            "    trend_ema_bars: int",
            "    top_k: int",
            "    gross_target: float",
            "    min_signal_strength: float",
            "    use_trend_filter: bool",
            "    volume_confirmation: float",
            "    volatility_floor: float",
            "",
            "    def target_weights(self, history_by_symbol: Mapping[str, Sequence[Bar]]) -> Dict[str, float]:",
            "        if not history_by_symbol:",
            "            return {}",
            "        ready_scores = []",
            "        min_bars = max(self.fast_ema_bars, self.slow_ema_bars, self.trend_ema_bars) + 2",
            "        for symbol, history in history_by_symbol.items():",
            "            if len(history) < min_bars:",
            "                continue",
            "            score = self._score(history)",
            "            if abs(score) < self.min_signal_strength:",
            "                continue",
            "            ready_scores.append((symbol, score))",
            "        if len(ready_scores) < self.top_k * 2:",
            "            return {}",
            "        ranked = sorted(ready_scores, key=lambda item: item[1], reverse=True)",
            "        longs = ranked[: self.top_k]",
            "        shorts = ranked[-self.top_k:]",
            "        side_gross = self.gross_target / 2.0",
            "        weights: Dict[str, float] = {}",
            "        for symbol, _score in longs:",
            "            weights[symbol] = side_gross / len(longs)",
            "        for symbol, _score in shorts:",
            "            weights[symbol] = -(side_gross / len(shorts))",
            "        return weights",
            "",
            "    def _score(self, history: Sequence[Bar]) -> float:",
            "        closes = [bar.close for bar in history]",
            "        fast = _ema(closes[-self.fast_ema_bars * 3 :], self.fast_ema_bars)",
            "        slow = _ema(closes[-self.slow_ema_bars * 3 :], self.slow_ema_bars)",
            "        trend = _ema(closes[-self.trend_ema_bars * 3 :], self.trend_ema_bars)",
            "        signal = fast - slow",
            "        recent_returns = []",
            "        for index in range(max(1, len(closes) - self.slow_ema_bars), len(closes)):",
            "            previous = closes[index - 1]",
            "            current = closes[index]",
            "            recent_returns.append((current / previous) - 1.0)",
            "        volatility = max(statistics.pstdev(recent_returns), self.volatility_floor)",
            "        normalized_signal = signal if math.isclose(volatility, 0.0) else signal / volatility",
            "        if self.use_trend_filter:",
            "            if history[-1].close > trend and normalized_signal < 0.0:",
            "                return 0.0",
            "            if history[-1].close < trend and normalized_signal > 0.0:",
            "                return 0.0",
            "        if self.volume_confirmation > 0.0:",
            "            recent_volumes = [bar.volume for bar in history[-self.slow_ema_bars:]]",
            "            volume_ratio = history[-1].volume / max(statistics.fmean(recent_volumes), 1e-9)",
            "            if volume_ratio < self.volume_confirmation:",
            "                return 0.0",
            "        return normalized_signal",
            "",
            "",
            "def build_strategy(_dataset_spec=None) -> Strategy:",
            "    return EMATrendStrategy(",
            "        fast_ema_bars=int(TRAIN_CONFIG['fast_ema_bars']),",
            "        slow_ema_bars=int(TRAIN_CONFIG['slow_ema_bars']),",
            "        trend_ema_bars=int(TRAIN_CONFIG['trend_ema_bars']),",
            "        top_k=int(TRAIN_CONFIG['top_k']),",
            "        gross_target=float(TRAIN_CONFIG['gross_target']),",
            "        min_signal_strength=float(TRAIN_CONFIG['min_signal_strength']),",
            "        use_trend_filter=bool(TRAIN_CONFIG['use_trend_filter']),",
            "        volume_confirmation=float(TRAIN_CONFIG['volume_confirmation']),",
            "        volatility_floor=float(TRAIN_CONFIG['volatility_floor']),",
            "    )",
            "",
        ]
    )


def _render_volatility_breakout_file(payload: str, strategy_name: str) -> str:
    return "\n".join(
        [
            "from __future__ import annotations",
            "",
            "from dataclasses import dataclass",
            "import statistics",
            "from typing import Dict, Mapping, Sequence",
            "",
            "from autoresearch_trade_bot.models import Bar",
            "from autoresearch_trade_bot.strategy import Strategy",
            "",
            f"TRAIN_CONFIG = {payload}",
            f'STRATEGY_NAME = "{strategy_name}"',
            f'STRATEGY_FAMILY = "{FAMILY_VOLATILITY_BREAKOUT}"',
            "",
            "",
            "def _atr(history: Sequence[Bar], lookback: int) -> float:",
            "    if len(history) <= lookback:",
            "        return 0.0",
            "    true_ranges = []",
            "    recent = history[-(lookback + 1):]",
            "    for index in range(1, len(recent)):",
            "        current = recent[index]",
            "        previous = recent[index - 1]",
            "        true_ranges.append(max(current.high - current.low, abs(current.high - previous.close), abs(current.low - previous.close)))",
            "    return statistics.fmean(true_ranges) if true_ranges else 0.0",
            "",
            "",
            "@dataclass(frozen=True)",
            "class VolatilityBreakoutStrategy:",
            "    channel_bars: int",
            "    atr_lookback_bars: int",
            "    atr_multiplier: float",
            "    breakout_buffer: float",
            "    top_k: int",
            "    gross_target: float",
            "    min_breakout_score: float",
            "    use_trend_filter: bool",
            "    trend_lookback_bars: int",
            "",
            "    def target_weights(self, history_by_symbol: Mapping[str, Sequence[Bar]]) -> Dict[str, float]:",
            "        if not history_by_symbol:",
            "            return {}",
            "        ready_scores = []",
            "        min_bars = max(self.channel_bars, self.atr_lookback_bars, self.trend_lookback_bars) + 2",
            "        for symbol, history in history_by_symbol.items():",
            "            if len(history) < min_bars:",
            "                continue",
            "            score = self._score(history)",
            "            if abs(score) < self.min_breakout_score:",
            "                continue",
            "            ready_scores.append((symbol, score))",
            "        positive = [item for item in ready_scores if item[1] > 0.0]",
            "        negative = [item for item in ready_scores if item[1] < 0.0]",
            "        if len(positive) < self.top_k or len(negative) < self.top_k:",
            "            return {}",
            "        longs = sorted(positive, key=lambda item: item[1], reverse=True)[: self.top_k]",
            "        shorts = sorted(negative, key=lambda item: item[1])[: self.top_k]",
            "        side_gross = self.gross_target / 2.0",
            "        weights: Dict[str, float] = {}",
            "        for symbol, _score in longs:",
            "            weights[symbol] = side_gross / len(longs)",
            "        for symbol, _score in shorts:",
            "            weights[symbol] = -(side_gross / len(shorts))",
            "        return weights",
            "",
            "    def _score(self, history: Sequence[Bar]) -> float:",
            "        previous_window = history[-(self.channel_bars + 1):-1]",
            "        channel_high = max(bar.high for bar in previous_window)",
            "        channel_low = min(bar.low for bar in previous_window)",
            "        atr_value = max(_atr(history, self.atr_lookback_bars), 1e-9)",
            "        current_close = history[-1].close",
            "        upper_trigger = channel_high + (self.breakout_buffer * atr_value)",
            "        lower_trigger = channel_low - (self.breakout_buffer * atr_value)",
            "        if self.use_trend_filter:",
            "            trend_start = history[-self.trend_lookback_bars].close",
            "            trend_return = (current_close / trend_start) - 1.0",
            "            if current_close > upper_trigger and trend_return <= 0.0:",
            "                return 0.0",
            "            if current_close < lower_trigger and trend_return >= 0.0:",
            "                return 0.0",
            "        if current_close > upper_trigger:",
            "            return (current_close - channel_high) / (atr_value * self.atr_multiplier)",
            "        if current_close < lower_trigger:",
            "            return -((channel_low - current_close) / (atr_value * self.atr_multiplier))",
            "        return 0.0",
            "",
            "",
            "def build_strategy(_dataset_spec=None) -> Strategy:",
            "    return VolatilityBreakoutStrategy(",
            "        channel_bars=int(TRAIN_CONFIG['channel_bars']),",
            "        atr_lookback_bars=int(TRAIN_CONFIG['atr_lookback_bars']),",
            "        atr_multiplier=float(TRAIN_CONFIG['atr_multiplier']),",
            "        breakout_buffer=float(TRAIN_CONFIG['breakout_buffer']),",
            "        top_k=int(TRAIN_CONFIG['top_k']),",
            "        gross_target=float(TRAIN_CONFIG['gross_target']),",
            "        min_breakout_score=float(TRAIN_CONFIG['min_breakout_score']),",
            "        use_trend_filter=bool(TRAIN_CONFIG['use_trend_filter']),",
            "        trend_lookback_bars=int(TRAIN_CONFIG['trend_lookback_bars']),",
            "    )",
            "",
        ]
    )
