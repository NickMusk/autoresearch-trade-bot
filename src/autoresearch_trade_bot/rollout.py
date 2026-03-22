from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Iterable

FAST_VALIDATION_STAGE = "fast_holdout"
ROLLOUT_VALIDATION_STAGE = "rollout_certification"


def _ensure_iso(value: datetime | str) -> str:
    if isinstance(value, str):
        return value
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc).isoformat()
    return value.astimezone(timezone.utc).isoformat()


def paper_gate_failures(gate_failures: Iterable[str]) -> list[str]:
    return [failure for failure in gate_failures if failure != "acceptance_rate_below_gate"]


def candidate_key(*, train_sha1: str, git_commit: str, strategy_name: str) -> str:
    if train_sha1:
        return train_sha1
    if git_commit:
        return git_commit
    return strategy_name


def build_candidate_record(
    report,
    *,
    observed_at: datetime | str,
    mutation_label: str,
    source_decision: str,
) -> dict[str, Any]:
    filtered_failures = paper_gate_failures(getattr(report, "gate_failures", []))
    return {
        "candidate_key": candidate_key(
            train_sha1=str(getattr(report, "train_sha1", "")),
            git_commit=str(getattr(report, "git_commit", "")),
            strategy_name=str(getattr(report, "strategy_name", "")),
        ),
        "strategy_name": str(getattr(report, "strategy_name", "")),
        "git_commit": str(getattr(report, "git_commit", "")),
        "train_sha1": str(getattr(report, "train_sha1", "")),
        "research_score": float(getattr(report, "research_score", 0.0)),
        "average_metrics": dict(getattr(report, "average_metrics", {})),
        "gate_failures": list(getattr(report, "gate_failures", [])),
        "paper_gate_failures": filtered_failures,
        "paper_gate_ready": not filtered_failures,
        "ready_for_paper": bool(getattr(report, "ready_for_paper", False)),
        "observed_at": _ensure_iso(observed_at),
        "last_mutation_label": mutation_label,
        "source_decision": source_decision,
        "fast_validation_results": {},
        "latest_fast_validation_summary": {},
        "rollout_validation_results": {},
        "latest_rollout_validation_summary": {},
    }


def _stage_results_key(stage: str) -> str:
    if stage == FAST_VALIDATION_STAGE:
        return "fast_validation_results"
    if stage == ROLLOUT_VALIDATION_STAGE:
        return "rollout_validation_results"
    raise ValueError(f"unknown validation stage: {stage}")


def _stage_summary_key(stage: str) -> str:
    if stage == FAST_VALIDATION_STAGE:
        return "latest_fast_validation_summary"
    if stage == ROLLOUT_VALIDATION_STAGE:
        return "latest_rollout_validation_summary"
    raise ValueError(f"unknown validation stage: {stage}")


def latest_validation_summary(entry: dict[str, Any], stage: str) -> dict[str, Any]:
    return dict(entry.get(_stage_summary_key(stage), {}))


def upsert_candidate_record(
    registry: dict[str, dict[str, Any]],
    report,
    *,
    observed_at: datetime | str,
    mutation_label: str,
    source_decision: str,
) -> tuple[dict[str, dict[str, Any]], str]:
    entry = build_candidate_record(
        report,
        observed_at=observed_at,
        mutation_label=mutation_label,
        source_decision=source_decision,
    )
    key = str(entry["candidate_key"])
    existing = dict(registry.get(key, {}))
    entry["fast_validation_results"] = dict(existing.get("fast_validation_results", {}))
    entry["latest_fast_validation_summary"] = dict(existing.get("latest_fast_validation_summary", {}))
    entry["rollout_validation_results"] = dict(existing.get("rollout_validation_results", {}))
    entry["latest_rollout_validation_summary"] = dict(existing.get("latest_rollout_validation_summary", {}))
    registry[key] = entry
    return registry, key


def record_candidate_validation(
    registry: dict[str, dict[str, Any]],
    *,
    candidate_id: str,
    stage: str,
    validation_summary: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    if candidate_id not in registry:
        return registry
    entry = dict(registry[candidate_id])
    results_key = _stage_results_key(stage)
    summary_key = _stage_summary_key(stage)
    validation_results = dict(entry.get(results_key, {}))
    validation_results[str(validation_summary.get("campaign_id", ""))] = dict(validation_summary)
    entry[results_key] = validation_results
    entry[summary_key] = dict(validation_summary)
    registry[candidate_id] = entry
    return registry


def shortlist_candidate_ids(
    registry: dict[str, dict[str, Any]],
    *,
    limit: int,
) -> list[str]:
    shortlistable = [
        entry
        for entry in registry.values()
        if entry.get("paper_gate_ready", False)
        or bool(latest_validation_summary(entry, FAST_VALIDATION_STAGE).get("passes_stage_gate", False))
        or bool(latest_validation_summary(entry, ROLLOUT_VALIDATION_STAGE).get("validated_for_rollout", False))
    ]
    ordered = sorted(
        shortlistable,
        key=lambda entry: (
            1 if latest_validation_summary(entry, ROLLOUT_VALIDATION_STAGE).get("validated_for_rollout", False) else 0,
            float(latest_validation_summary(entry, ROLLOUT_VALIDATION_STAGE).get("validation_pass_rate", 0.0)),
            1 if latest_validation_summary(entry, FAST_VALIDATION_STAGE).get("passes_stage_gate", False) else 0,
            float(latest_validation_summary(entry, FAST_VALIDATION_STAGE).get("validation_pass_rate", 0.0)),
            float(entry.get("research_score", 0.0)),
            str(entry.get("observed_at", "")),
        ),
        reverse=True,
    )
    return [str(entry["candidate_key"]) for entry in ordered[:limit]]


def select_rollout_champion(
    registry: dict[str, dict[str, Any]],
) -> dict[str, Any] | None:
    eligible = [
        entry
        for entry in registry.values()
        if bool(latest_validation_summary(entry, ROLLOUT_VALIDATION_STAGE).get("validated_for_rollout", False))
    ]
    if not eligible:
        return None
    ordered = sorted(
        eligible,
        key=lambda entry: (
            float(latest_validation_summary(entry, ROLLOUT_VALIDATION_STAGE).get("validation_pass_rate", 0.0)),
            float(entry.get("research_score", 0.0)),
            str(entry.get("observed_at", "")),
        ),
        reverse=True,
    )
    return dict(ordered[0])


def build_research_champion_summary(
    report,
    *,
    fast_validation_summary: dict[str, Any],
    rollout_validation_summary: dict[str, Any],
) -> dict[str, Any]:
    return {
        "strategy_name": str(getattr(report, "strategy_name", "")),
        "git_commit": str(getattr(report, "git_commit", "")),
        "train_sha1": str(getattr(report, "train_sha1", "")),
        "research_score": float(getattr(report, "research_score", 0.0)),
        "average_metrics": dict(getattr(report, "average_metrics", {})),
        "gate_failures": list(getattr(report, "gate_failures", [])),
        "paper_gate_failures": paper_gate_failures(getattr(report, "gate_failures", [])),
        "paper_gate_ready": not paper_gate_failures(getattr(report, "gate_failures", [])),
        "fast_validation_summary": dict(fast_validation_summary),
        "rollout_validation_summary": dict(rollout_validation_summary),
    }


def build_rollout_champion_summary(candidate: dict[str, Any] | None) -> dict[str, Any]:
    if candidate is None:
        return {}
    return {
        "strategy_name": str(candidate.get("strategy_name", "")),
        "git_commit": str(candidate.get("git_commit", "")),
        "train_sha1": str(candidate.get("train_sha1", "")),
        "research_score": float(candidate.get("research_score", 0.0)),
        "average_metrics": dict(candidate.get("average_metrics", {})),
        "paper_gate_failures": list(candidate.get("paper_gate_failures", [])),
        "paper_gate_ready": bool(candidate.get("paper_gate_ready", False)),
        "fast_validation_summary": latest_validation_summary(candidate, FAST_VALIDATION_STAGE),
        "rollout_validation_summary": latest_validation_summary(candidate, ROLLOUT_VALIDATION_STAGE),
    }


def build_rollout_shortlist_summary(
    registry: dict[str, dict[str, Any]],
    *,
    candidate_ids: list[str],
) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    for candidate_id in candidate_ids:
        entry = registry.get(candidate_id)
        if entry is None:
            continue
        summaries.append(
            {
                "strategy_name": str(entry.get("strategy_name", "")),
                "research_score": float(entry.get("research_score", 0.0)),
                "paper_gate_ready": bool(entry.get("paper_gate_ready", False)),
                "fast_validation_pass_rate": float(
                    latest_validation_summary(entry, FAST_VALIDATION_STAGE).get("validation_pass_rate", 0.0)
                ),
                "fast_holdout_passed": bool(
                    latest_validation_summary(entry, FAST_VALIDATION_STAGE).get("passes_stage_gate", False)
                ),
                "rollout_validation_pass_rate": float(
                    latest_validation_summary(entry, ROLLOUT_VALIDATION_STAGE).get("validation_pass_rate", 0.0)
                ),
                "validated_for_rollout": bool(
                    latest_validation_summary(entry, ROLLOUT_VALIDATION_STAGE).get("validated_for_rollout", False)
                ),
            }
        )
    return summaries
