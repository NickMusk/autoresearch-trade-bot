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
    strategy_class_name: str
    required_config_keys: tuple[str, ...]
    default_train_config: dict[str, Any]
    attempt_role_specs: tuple[dict[str, str], ...]
    prompt_directions: tuple[str, ...]


PROFILES: dict[str, StrategyFamilyProfile] = {
    FAMILY_MOMENTUM: StrategyFamilyProfile(
        family_id=FAMILY_MOMENTUM,
        display_name="Cross-Sectional Momentum",
        strategy_name="configurable-momentum",
        strategy_class_name="ConfigurableMomentumStrategy",
        required_config_keys=(
            "lookback_bars",
            "top_k",
            "gross_target",
            "ranking_mode",
            "use_regime_filter",
            "regime_lookback_bars",
            "regime_threshold",
            "min_signal_strength",
            "min_cross_sectional_spread",
            "volatility_floor",
            "reversal_bias_weight",
            "funding_penalty_weight",
        ),
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
        display_name="IBS Reversion",
        strategy_name="ibs-reversion-short-horizon",
        strategy_class_name="IBSReversionStrategy",
        required_config_keys=(
            "lookback_bars",
            "reversion_horizon_bars",
            "ibs_threshold",
            "top_k",
            "gross_target",
            "reversion_strength_floor",
            "volatility_floor",
            "use_trend_filter",
            "trend_lookback_bars",
        ),
        default_train_config={
            "gross_target": 0.5,
            "lookback_bars": 24,
            "reversion_horizon_bars": 6,
            "ibs_threshold": 0.25,
            "top_k": 1,
            "reversion_strength_floor": 0.0,
            "volatility_floor": 0.0,
            "use_trend_filter": False,
            "trend_lookback_bars": 48,
        },
        attempt_role_specs=(
            {
                "name": "exploit_baseline",
                "objective": "Refine the current short-horizon reversion baseline with modest horizon, IBS, or floor adjustments.",
                "constraints": "Preserve trade activity. Avoid stacking strict floors with a long trend filter.",
            },
            {
                "name": "entry_balance",
                "objective": "Balance bar-position reversal and short-horizon return reversal without making entries too rare.",
                "constraints": "Do not make ibs_threshold, reversion_strength_floor, and volatility_floor strict at the same time.",
            },
            {
                "name": "trend_aware_reversion",
                "objective": "Explore mild trend-aware fading that avoids fighting the strongest moves.",
                "constraints": "Keep the trend filter light and cheap; do not build multi-stage confirmation.",
            },
        ),
        prompt_directions=(
            "Use ibs_threshold to shape entry timing, but keep it moderate enough that symbols still qualify.",
            "Use reversion_strength_floor and volatility_floor carefully; aggressive floors quickly create no-trade candidates.",
            "If you enable a trend filter, keep trend_lookback_bars moderate so the strategy still trades reversals.",
        ),
    ),
    FAMILY_EMA_TREND: StrategyFamilyProfile(
        family_id=FAMILY_EMA_TREND,
        display_name="Dual Momentum",
        strategy_name="dual-momentum-timeseries",
        strategy_class_name="DualMomentumStrategy",
        required_config_keys=(
            "gross_target",
            "fast_horizon_bars",
            "medium_horizon_bars",
            "slow_horizon_bars",
            "top_k",
            "min_signal_strength",
            "absolute_momentum_floor",
            "relative_strength_weight",
            "use_absolute_filter",
            "volatility_floor",
        ),
        default_train_config={
            "gross_target": 0.5,
            "fast_horizon_bars": 12,
            "medium_horizon_bars": 36,
            "slow_horizon_bars": 96,
            "top_k": 1,
            "min_signal_strength": 0.0,
            "absolute_momentum_floor": 0.0,
            "relative_strength_weight": 0.6,
            "use_absolute_filter": True,
            "volatility_floor": 0.0,
        },
        attempt_role_specs=(
            {
                "name": "exploit_baseline",
                "objective": "Refine the current dual-momentum baseline with bounded horizon or weighting changes.",
                "constraints": "Do not collapse the family into a pure cross-sectional ranking with no absolute trend component.",
            },
            {
                "name": "absolute_vs_relative_balance",
                "objective": "Improve the balance between time-series confirmation and cross-sectional ranking.",
                "constraints": "Avoid overly strict absolute_momentum_floor and high signal floors together.",
            },
            {
                "name": "faster_vs_slower_response",
                "objective": "Explore materially different horizon spacing while preserving fast < medium < slow.",
                "constraints": "Keep the ordering valid and avoid cosmetic changes only.",
            },
        ),
        prompt_directions=(
            "Keep fast_horizon_bars < medium_horizon_bars < slow_horizon_bars and use spacing to control responsiveness.",
            "Use relative_strength_weight to balance cross-sectional ranking against absolute trend confirmation.",
            "Keep absolute_momentum_floor and min_signal_strength mild unless recent evidence clearly supports more selectivity.",
        ),
    ),
    FAMILY_VOLATILITY_BREAKOUT: StrategyFamilyProfile(
        family_id=FAMILY_VOLATILITY_BREAKOUT,
        display_name="Donchian ATR Breakout",
        strategy_name="donchian-atr-breakout-trend",
        strategy_class_name="DonchianATRBreakoutStrategy",
        required_config_keys=(
            "gross_target",
            "channel_bars",
            "atr_lookback_bars",
            "atr_multiplier",
            "breakout_buffer",
            "top_k",
            "breakout_score_floor",
            "use_trend_filter",
            "trend_lookback_bars",
        ),
        default_train_config={
            "gross_target": 0.5,
            "channel_bars": 24,
            "atr_lookback_bars": 14,
            "atr_multiplier": 1.0,
            "breakout_buffer": 0.0,
            "top_k": 1,
            "breakout_score_floor": 0.0,
            "use_trend_filter": False,
            "trend_lookback_bars": 72,
        },
        attempt_role_specs=(
            {
                "name": "exploit_baseline",
                "objective": "Refine the current Donchian breakout baseline with modest channel or ATR adjustments.",
                "constraints": "Preserve breakout behavior; do not drift into generic momentum or dead filters.",
            },
            {
                "name": "breakout_selectivity",
                "objective": "Tune channel, ATR, and buffer thresholds without eliminating breakout activity.",
                "constraints": "Do not combine a long channel, large atr_multiplier, and large breakout_buffer.",
            },
            {
                "name": "trend_confirmed_breakout",
                "objective": "Explore lightweight trend-confirmed breakout variants that still remain active.",
                "constraints": "Keep trend confirmation simple and avoid over-constraining entries.",
            },
        ),
        prompt_directions=(
            "Balance channel_bars with breakout_buffer instead of increasing both aggressively.",
            "Use atr_multiplier for normalization, but keep it moderate so breakout entries remain attainable.",
            "If you add a trend filter, keep trend_lookback_bars moderate and avoid pairing it with already strict thresholds.",
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


def extract_strategy_name(train_text: str) -> str:
    try:
        tree = ast.parse(train_text)
    except SyntaxError:
        return ""
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "STRATEGY_NAME":
                    try:
                        value = ast.literal_eval(node.value)
                    except Exception:
                        return ""
                    return value if isinstance(value, str) else ""
    return ""


def family_attempt_role_specs(strategy_family: str) -> tuple[dict[str, str], ...]:
    return get_strategy_family_profile(strategy_family).attempt_role_specs


def family_prompt_directions(strategy_family: str) -> tuple[str, ...]:
    return get_strategy_family_profile(strategy_family).prompt_directions


def family_template_constraints(strategy_family: str) -> tuple[str, ...]:
    profile = get_strategy_family_profile(strategy_family)
    if strategy_family == FAMILY_MOMENTUM:
        return (
            f"Preserve the {profile.strategy_class_name} template and keep build_strategy returning {profile.strategy_class_name}.",
            "Keep the momentum family config surface explicit; do not replace TRAIN_CONFIG with a different family's keys.",
        )
    return (
        f"Preserve the {profile.strategy_class_name} template and keep build_strategy returning {profile.strategy_class_name}.",
        (
            "Keep the candidate inside the family-specific TRAIN_CONFIG surface with keys: "
            + ", ".join(profile.required_config_keys)
            + "."
        ),
        "Do not fall back to the generic configurable-momentum template in this family branch.",
    )


def family_mutation_bounds(
    strategy_family: str,
    *,
    symbol_count: int,
) -> dict[str, Any]:
    max_top_k = max(1, symbol_count // 2)
    bounds: dict[str, Any] = {
        "symbol_count": symbol_count,
        "top_k_min": 1,
        "top_k_max": max_top_k,
        "gross_target_min_exclusive": 0.0,
        "gross_target_max": 1.5,
        "family_rules": [],
    }
    family_rules: list[str] = bounds["family_rules"]
    if strategy_family == FAMILY_MEAN_REVERSION:
        family_rules.extend(
            [
                "reversion_horizon_bars must satisfy 2 <= value < trend_lookback_bars",
                "ibs_threshold must satisfy 0 < ibs_threshold < 0.5",
                "reversion_strength_floor must stay <= 2.5",
            ]
        )
    elif strategy_family == FAMILY_EMA_TREND:
        family_rules.extend(
            [
                "Horizon ordering must satisfy fast_horizon_bars < medium_horizon_bars < slow_horizon_bars",
                "relative_strength_weight must stay between 0.0 and 1.0 inclusive",
                "absolute_momentum_floor and min_signal_strength should stay <= 1.5",
            ]
        )
    elif strategy_family == FAMILY_VOLATILITY_BREAKOUT:
        family_rules.extend(
            [
                "channel_bars must be greater than atr_lookback_bars",
                "atr_multiplier must satisfy 0 < atr_multiplier <= 4.0",
                "breakout_buffer must stay <= 1.5",
            ]
        )
    else:
        family_rules.extend(
            [
                "ranking_mode must be one of raw_return or risk_adjusted",
                "min_signal_strength must stay <= 0.12",
                "min_cross_sectional_spread must stay <= 0.15",
            ]
        )
    return bounds


def render_family_mutation_bounds(
    strategy_family: str,
    *,
    symbol_count: int,
) -> str:
    bounds = family_mutation_bounds(strategy_family, symbol_count=symbol_count)
    lines = [
        f"- symbol_count={bounds['symbol_count']}",
        (
            f"- paired long/short engine requires top_k between "
            f"{bounds['top_k_min']} and {bounds['top_k_max']} inclusive"
        ),
        (
            f"- gross_target must be > {bounds['gross_target_min_exclusive']} "
            f"and <= {bounds['gross_target_max']}"
        ),
    ]
    lines.extend(f"- {rule}" for rule in bounds["family_rules"])
    return "\n".join(lines)


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
            ("lookback_bars", (16, 24, 36)),
            ("reversion_horizon_bars", (3, 6, 9)),
            ("ibs_threshold", (0.15, 0.25, 0.35)),
            ("reversion_strength_floor", (0.0, 0.2, 0.5)),
            ("volatility_floor", (0.0, 0.01, 0.02)),
            ("top_k", (1, 2)),
        ):
            proposals.extend(_mutation_specs_for_values(profile_config, key, values))
        proposals.extend(
            _mutation_specs_for_values(profile_config, "use_trend_filter", (False, True))
        )
        proposals.extend(
            _mutation_specs_for_values(profile_config, "trend_lookback_bars", (36, 48, 72))
        )
        return proposals
    if strategy_family == FAMILY_EMA_TREND:
        for key, values in (
            ("fast_horizon_bars", (8, 12, 16)),
            ("medium_horizon_bars", (24, 36, 48)),
            ("slow_horizon_bars", (72, 96, 144)),
            ("top_k", (1, 2)),
            ("min_signal_strength", (0.0, 0.1, 0.25)),
            ("absolute_momentum_floor", (0.0, 0.1, 0.25)),
            ("relative_strength_weight", (0.4, 0.6, 0.8)),
            ("volatility_floor", (0.0, 0.01, 0.02)),
        ):
            proposals.extend(_mutation_specs_for_values(profile_config, key, values))
        proposals.extend(
            _mutation_specs_for_values(profile_config, "use_absolute_filter", (False, True))
        )
        return proposals
    if strategy_family == FAMILY_VOLATILITY_BREAKOUT:
        for key, values in (
            ("channel_bars", (12, 24, 36)),
            ("atr_lookback_bars", (10, 14, 21)),
            ("atr_multiplier", (0.75, 1.0, 1.5)),
            ("breakout_buffer", (0.0, 0.25, 0.5)),
            ("breakout_score_floor", (0.0, 0.2, 0.5)),
            ("top_k", (1, 2)),
        ):
            proposals.extend(_mutation_specs_for_values(profile_config, key, values))
        proposals.extend(
            _mutation_specs_for_values(profile_config, "use_trend_filter", (False, True))
        )
        proposals.extend(
            _mutation_specs_for_values(profile_config, "trend_lookback_bars", (48, 72, 96))
        )
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
        proposals.extend(
            _mutation_specs_for_values(profile_config, key, values, label_key=label_key)
        )
    return proposals


def validate_train_candidate_semantics(
    strategy_family: str,
    *,
    candidate_text: str | None = None,
    candidate_config: Mapping[str, Any],
    current_config: Mapping[str, Any],
    symbol_count: int,
) -> tuple[bool, str]:
    if candidate_text is not None:
        template_ok, template_failure = validate_family_template_integrity(
            candidate_text,
            strategy_family=strategy_family,
        )
        if not template_ok:
            return False, template_failure
    candidate = normalize_train_config(candidate_config, strategy_family=strategy_family)
    current = normalize_train_config(current_config, strategy_family=strategy_family)
    bounds = family_mutation_bounds(strategy_family, symbol_count=symbol_count)
    if candidate == current:
        return False, "no_op_train_config"
    top_k = int(candidate.get("top_k", 1))
    if top_k < bounds["top_k_min"]:
        return False, "invalid_top_k"
    if top_k > bounds["top_k_max"]:
        return False, "top_k_exceeds_cross_sectional_capacity"
    gross_target = float(candidate.get("gross_target", 0.5))
    if (
        gross_target <= float(bounds["gross_target_min_exclusive"])
        or gross_target > float(bounds["gross_target_max"])
    ):
        return False, "invalid_gross_target"
    if strategy_family == FAMILY_MEAN_REVERSION:
        if int(candidate["reversion_horizon_bars"]) < 2:
            return False, "invalid_reversion_horizon"
        if int(candidate["reversion_horizon_bars"]) >= int(candidate["trend_lookback_bars"]):
            return False, "invalid_reversion_trend_order"
        if not (0.0 < float(candidate["ibs_threshold"]) < 0.5):
            return False, "invalid_ibs_threshold"
        if float(candidate.get("reversion_strength_floor", 0.0)) > 2.5:
            return False, "likely_no_trade_reversion_floor"
        if float(candidate.get("volatility_floor", 0.0)) > 0.05:
            return False, "likely_no_trade_volatility_floor"
        if (
            float(candidate.get("ibs_threshold", 0.0)) <= 0.12
            and float(candidate.get("reversion_strength_floor", 0.0)) >= 0.5
        ):
            return False, "likely_no_trade_ibs_floor_stack"
        if (
            bool(candidate.get("use_trend_filter", False))
            and int(candidate.get("trend_lookback_bars", 48)) >= 72
            and float(candidate.get("reversion_strength_floor", 0.0)) >= 0.2
        ):
            return False, "likely_no_trade_reversion_trend_stack"
        if (
            float(candidate.get("ibs_threshold", 0.0)) <= 0.15
            and float(candidate.get("volatility_floor", 0.0)) >= 0.02
            and int(candidate.get("reversion_horizon_bars", 0)) >= 9
        ):
            return False, "likely_no_trade_reversion_threshold_stack"
        return True, ""
    if strategy_family == FAMILY_EMA_TREND:
        if not (
            int(candidate["fast_horizon_bars"])
            < int(candidate["medium_horizon_bars"])
            < int(candidate["slow_horizon_bars"])
        ):
            return False, "invalid_horizon_ordering"
        if not (0.0 <= float(candidate.get("relative_strength_weight", 0.0)) <= 1.0):
            return False, "invalid_relative_strength_weight"
        if float(candidate.get("absolute_momentum_floor", 0.0)) > 1.5:
            return False, "likely_no_trade_absolute_floor"
        if float(candidate.get("min_signal_strength", 0.0)) > 1.5:
            return False, "likely_no_trade_signal_threshold"
        if (
            bool(candidate.get("use_absolute_filter", False))
            and float(candidate.get("absolute_momentum_floor", 0.0)) >= 0.25
            and float(candidate.get("min_signal_strength", 0.0)) >= 0.25
        ):
            return False, "likely_no_trade_confirmation_stack"
        if (
            int(candidate["fast_horizon_bars"]) >= 16
            and int(candidate["medium_horizon_bars"]) >= 48
            and int(candidate["slow_horizon_bars"]) >= 144
            and float(candidate.get("min_signal_strength", 0.0)) > 0.1
        ):
            return False, "likely_no_trade_wide_horizon_stack"
        if (
            float(candidate.get("volatility_floor", 0.0)) >= 0.02
            and float(candidate.get("absolute_momentum_floor", 0.0)) >= 0.25
        ):
            return False, "likely_no_trade_trend_filter_stack"
        return True, ""
    if strategy_family == FAMILY_VOLATILITY_BREAKOUT:
        if int(candidate["channel_bars"]) <= int(candidate["atr_lookback_bars"]):
            return False, "invalid_breakout_window_order"
        if float(candidate["atr_multiplier"]) <= 0.0 or float(candidate["atr_multiplier"]) > 4.0:
            return False, "invalid_atr_multiplier"
        if float(candidate["breakout_buffer"]) > 1.5:
            return False, "likely_no_trade_breakout_buffer"
        if float(candidate.get("breakout_score_floor", 0.0)) > 2.5:
            return False, "likely_no_trade_breakout_score"
        if (
            int(candidate["channel_bars"]) >= 36
            and float(candidate.get("breakout_buffer", 0.0)) >= 0.25
            and float(candidate.get("atr_multiplier", 0.0)) >= 1.5
        ):
            return False, "likely_no_trade_breakout_threshold_stack"
        if (
            bool(candidate.get("use_trend_filter", False))
            and int(candidate.get("trend_lookback_bars", 72)) >= 96
            and float(candidate.get("breakout_buffer", 0.0)) >= 0.25
        ):
            return False, "likely_no_trade_breakout_trend_stack"
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


def validate_family_template_integrity(
    train_text: str,
    *,
    strategy_family: str,
) -> tuple[bool, str]:
    profile = get_strategy_family_profile(strategy_family)
    try:
        tree = ast.parse(train_text)
    except SyntaxError:
        return False, "invalid_python_syntax"
    raw_config = _extract_raw_train_config(train_text)
    missing_keys = [key for key in profile.required_config_keys if key not in raw_config]
    if missing_keys:
        return False, f"missing_family_config_keys:{','.join(sorted(missing_keys))}"
    class_names = {
        node.name
        for node in tree.body
        if isinstance(node, ast.ClassDef)
    }
    if profile.strategy_class_name not in class_names:
        return False, f"missing_family_strategy_class:{profile.strategy_class_name}"
    build_strategy = next(
        (
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "build_strategy"
        ),
        None,
    )
    if build_strategy is None:
        return False, "missing_build_strategy"
    for statement in build_strategy.body:
        if isinstance(statement, ast.Return) and isinstance(statement.value, ast.Call):
            callee = statement.value.func
            if isinstance(callee, ast.Name) and callee.id == profile.strategy_class_name:
                return True, ""
    return False, f"build_strategy_must_return:{profile.strategy_class_name}"


def config_traits(strategy_family: str, config: Mapping[str, Any]) -> list[str]:
    if not config:
        return []
    traits = [f"family={strategy_family}", f"top_k={config.get('top_k', '')}"]
    if strategy_family == FAMILY_MEAN_REVERSION:
        traits.extend(
            [
                f"ibs_threshold={_bucket(float(config.get('ibs_threshold', 0.0)), (0.18, 0.3))}",
                f"reversion_horizon={_bucket(float(config.get('reversion_horizon_bars', 0.0)), (4.0, 7.0))}",
                f"reversion_floor={_bucket(float(config.get('reversion_strength_floor', 0.0)), (0.1, 0.25))}",
            ]
        )
        if bool(config.get("use_trend_filter", False)):
            traits.append("trend_filter=on")
        if float(config.get("volatility_floor", 0.0)) > 0.0:
            traits.append(
                f"volatility_floor={_bucket(float(config.get('volatility_floor', 0.0)), (0.01, 0.02))}"
            )
    elif strategy_family == FAMILY_EMA_TREND:
        fast_horizon = int(config.get("fast_horizon_bars", 0))
        medium_horizon = int(config.get("medium_horizon_bars", 0))
        slow_horizon = int(config.get("slow_horizon_bars", 0))
        horizon_stack = "balanced"
        if medium_horizon >= max(fast_horizon * 4, 1) and slow_horizon >= max(medium_horizon * 3, 1):
            horizon_stack = "wide"
        elif medium_horizon <= max(fast_horizon * 2, 1) or slow_horizon <= max(medium_horizon * 2, 1):
            horizon_stack = "compressed"
        traits.extend(
            [
                f"fast_horizon={fast_horizon}",
                f"signal_floor={_bucket(float(config.get('min_signal_strength', 0.0)), (0.1, 0.25))}",
                f"absolute_floor={_bucket(float(config.get('absolute_momentum_floor', 0.0)), (0.1, 0.25))}",
                f"relative_weight={_bucket(float(config.get('relative_strength_weight', 0.0)), (0.45, 0.7))}",
                f"dual_momentum_stack={horizon_stack}",
            ]
        )
        if bool(config.get("use_absolute_filter", False)):
            traits.append("absolute_filter=on")
        if float(config.get("volatility_floor", 0.0)) > 0.0:
            traits.append(
                f"volatility_floor={_bucket(float(config.get('volatility_floor', 0.0)), (0.01, 0.02))}"
            )
    elif strategy_family == FAMILY_VOLATILITY_BREAKOUT:
        traits.extend(
            [
                f"channel={_bucket(float(config.get('channel_bars', 0.0)), (18.0, 30.0))}",
                f"breakout_buffer={_bucket(float(config.get('breakout_buffer', 0.0)), (0.1, 0.4))}",
                f"atr_multiplier={_bucket(float(config.get('atr_multiplier', 0.0)), (0.75, 1.25))}",
                f"breakout_score_floor={_bucket(float(config.get('breakout_score_floor', 0.0)), (0.1, 0.25))}",
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


def _extract_raw_train_config(train_text: str) -> dict[str, Any]:
    try:
        tree = ast.parse(train_text)
    except SyntaxError:
        return {}
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "TRAIN_CONFIG":
                    try:
                        value = ast.literal_eval(node.value)
                    except Exception:
                        return {}
                    return dict(value) if isinstance(value, dict) else {}
    return {}


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
            "def _internal_bar_strength(bar: Bar) -> float:",
            "    bar_range = bar.high - bar.low",
            "    if math.isclose(bar_range, 0.0):",
            "        return 0.5",
            "    return min(max((bar.close - bar.low) / bar_range, 0.0), 1.0)",
            "",
            "",
            "def _recent_volatility(history: Sequence[Bar], lookback: int, volatility_floor: float) -> float:",
            "    recent = history[-(lookback + 1):]",
            "    returns = []",
            "    for index in range(1, len(recent)):",
            "        previous = recent[index - 1].close",
            "        current = recent[index].close",
            "        returns.append((current / previous) - 1.0)",
            "    if not returns:",
            "        return max(volatility_floor, 1e-9)",
            "    return max(statistics.pstdev(returns), volatility_floor, 1e-9)",
            "",
            "",
            "@dataclass(frozen=True)",
            "class IBSReversionStrategy:",
            "    lookback_bars: int",
            "    reversion_horizon_bars: int",
            "    ibs_threshold: float",
            "    top_k: int",
            "    gross_target: float",
            "    reversion_strength_floor: float",
            "    volatility_floor: float",
            "    use_trend_filter: bool",
            "    trend_lookback_bars: int",
            "",
            "    def target_weights(self, history_by_symbol: Mapping[str, Sequence[Bar]]) -> Dict[str, float]:",
            "        if not history_by_symbol:",
            "            return {}",
            "        ready_scores = []",
            "        min_bars = max(self.lookback_bars, self.reversion_horizon_bars + 1, self.trend_lookback_bars) + 2",
            "        for symbol, history in history_by_symbol.items():",
            "            if len(history) < min_bars:",
            "                continue",
            "            score = self._score(history)",
            "            if abs(score) < self.reversion_strength_floor:",
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
            "        volatility = _recent_volatility(history, self.lookback_bars, self.volatility_floor)",
            "        reference_close = history[-(self.reversion_horizon_bars + 1)].close",
            "        recent_return = (history[-1].close / reference_close) - 1.0",
            "        normalized_reversal = -(recent_return / volatility)",
            "        ibs_value = _internal_bar_strength(history[-1])",
            "        if ibs_value <= self.ibs_threshold:",
            "            ibs_component = (self.ibs_threshold - ibs_value) / max(self.ibs_threshold, 1e-9)",
            "        elif ibs_value >= 1.0 - self.ibs_threshold:",
            "            ibs_component = -((ibs_value - (1.0 - self.ibs_threshold)) / max(self.ibs_threshold, 1e-9))",
            "        else:",
            "            ibs_component = 0.0",
            "        combined = (0.65 * normalized_reversal) + (0.35 * ibs_component)",
            "        if self.use_trend_filter:",
            "            trend_start = history[-self.trend_lookback_bars].close",
            "            trend_return = (history[-1].close / trend_start) - 1.0",
            "            trend_threshold = max(0.015, self.reversion_strength_floor * 0.05)",
            "            if combined > 0.0 and trend_return < -trend_threshold:",
            "                return 0.0",
            "            if combined < 0.0 and trend_return > trend_threshold:",
            "                return 0.0",
            "        return combined",
            "",
            "",
            "def build_strategy(_dataset_spec=None) -> Strategy:",
            "    return IBSReversionStrategy(",
            "        lookback_bars=int(TRAIN_CONFIG['lookback_bars']),",
            "        reversion_horizon_bars=int(TRAIN_CONFIG['reversion_horizon_bars']),",
            "        ibs_threshold=float(TRAIN_CONFIG['ibs_threshold']),",
            "        top_k=int(TRAIN_CONFIG['top_k']),",
            "        gross_target=float(TRAIN_CONFIG['gross_target']),",
            "        reversion_strength_floor=float(TRAIN_CONFIG['reversion_strength_floor']),",
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
            "def _normalized_return(history: Sequence[Bar], horizon: int, volatility_floor: float) -> float:",
            "    if len(history) <= horizon:",
            "        return 0.0",
            "    anchor_close = history[-(horizon + 1)].close",
            "    raw_return = (history[-1].close / anchor_close) - 1.0",
            "    recent = history[-(horizon + 1):]",
            "    returns = []",
            "    for index in range(1, len(recent)):",
            "        previous = recent[index - 1].close",
            "        current = recent[index].close",
            "        returns.append((current / previous) - 1.0)",
            "    volatility = max(statistics.pstdev(returns) if returns else 0.0, volatility_floor, 1e-9)",
            "    return raw_return / volatility",
            "",
            "",
            "@dataclass(frozen=True)",
            "class DualMomentumStrategy:",
            "    fast_horizon_bars: int",
            "    medium_horizon_bars: int",
            "    slow_horizon_bars: int",
            "    top_k: int",
            "    gross_target: float",
            "    min_signal_strength: float",
            "    absolute_momentum_floor: float",
            "    relative_strength_weight: float",
            "    use_absolute_filter: bool",
            "    volatility_floor: float",
            "",
            "    def target_weights(self, history_by_symbol: Mapping[str, Sequence[Bar]]) -> Dict[str, float]:",
            "        if not history_by_symbol:",
            "            return {}",
            "        ready_scores = []",
            "        min_bars = self.slow_horizon_bars + 2",
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
            "        fast_signal = _normalized_return(history, self.fast_horizon_bars, self.volatility_floor)",
            "        medium_signal = _normalized_return(history, self.medium_horizon_bars, self.volatility_floor)",
            "        slow_signal = _normalized_return(history, self.slow_horizon_bars, self.volatility_floor)",
            "        relative_signal = (0.6 * fast_signal) + (0.4 * medium_signal)",
            "        combined = (self.relative_strength_weight * relative_signal) + ((1.0 - self.relative_strength_weight) * slow_signal)",
            "        if self.use_absolute_filter:",
            "            if abs(slow_signal) < self.absolute_momentum_floor:",
            "                return 0.0",
            "            if combined > 0.0 and slow_signal <= 0.0:",
            "                return 0.0",
            "            if combined < 0.0 and slow_signal >= 0.0:",
            "                return 0.0",
            "        return combined",
            "",
            "",
            "def build_strategy(_dataset_spec=None) -> Strategy:",
            "    return DualMomentumStrategy(",
            "        fast_horizon_bars=int(TRAIN_CONFIG['fast_horizon_bars']),",
            "        medium_horizon_bars=int(TRAIN_CONFIG['medium_horizon_bars']),",
            "        slow_horizon_bars=int(TRAIN_CONFIG['slow_horizon_bars']),",
            "        top_k=int(TRAIN_CONFIG['top_k']),",
            "        gross_target=float(TRAIN_CONFIG['gross_target']),",
            "        min_signal_strength=float(TRAIN_CONFIG['min_signal_strength']),",
            "        absolute_momentum_floor=float(TRAIN_CONFIG['absolute_momentum_floor']),",
            "        relative_strength_weight=float(TRAIN_CONFIG['relative_strength_weight']),",
            "        use_absolute_filter=bool(TRAIN_CONFIG['use_absolute_filter']),",
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
            "class DonchianATRBreakoutStrategy:",
            "    channel_bars: int",
            "    atr_lookback_bars: int",
            "    atr_multiplier: float",
            "    breakout_buffer: float",
            "    top_k: int",
            "    gross_target: float",
            "    breakout_score_floor: float",
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
            "            if abs(score) < self.breakout_score_floor:",
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
            "    return DonchianATRBreakoutStrategy(",
            "        channel_bars=int(TRAIN_CONFIG['channel_bars']),",
            "        atr_lookback_bars=int(TRAIN_CONFIG['atr_lookback_bars']),",
            "        atr_multiplier=float(TRAIN_CONFIG['atr_multiplier']),",
            "        breakout_buffer=float(TRAIN_CONFIG['breakout_buffer']),",
            "        top_k=int(TRAIN_CONFIG['top_k']),",
            "        gross_target=float(TRAIN_CONFIG['gross_target']),",
            "        breakout_score_floor=float(TRAIN_CONFIG['breakout_score_floor']),",
            "        use_trend_filter=bool(TRAIN_CONFIG['use_trend_filter']),",
            "        trend_lookback_bars=int(TRAIN_CONFIG['trend_lookback_bars']),",
            "    )",
            "",
        ]
    )
