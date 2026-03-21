from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Iterable


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
        "validation_results": {},
        "latest_validation_summary": {},
    }


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
    validations = dict(existing.get("validation_results", {}))
    latest_validation_summary = dict(existing.get("latest_validation_summary", {}))
    entry["validation_results"] = validations
    entry["latest_validation_summary"] = latest_validation_summary
    registry[key] = entry
    return registry, key


def record_candidate_validation(
    registry: dict[str, dict[str, Any]],
    *,
    candidate_id: str,
    validation_summary: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    if candidate_id not in registry:
        return registry
    entry = dict(registry[candidate_id])
    validation_results = dict(entry.get("validation_results", {}))
    validation_results[str(validation_summary.get("campaign_id", ""))] = dict(validation_summary)
    entry["validation_results"] = validation_results
    entry["latest_validation_summary"] = dict(validation_summary)
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
        or bool(entry.get("latest_validation_summary", {}).get("validated_for_rollout", False))
    ]
    ordered = sorted(
        shortlistable,
        key=lambda entry: (
            1 if entry.get("latest_validation_summary", {}).get("validated_for_rollout", False) else 0,
            float(entry.get("latest_validation_summary", {}).get("validation_pass_rate", 0.0)),
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
        if bool(entry.get("latest_validation_summary", {}).get("validated_for_rollout", False))
    ]
    if not eligible:
        return None
    ordered = sorted(
        eligible,
        key=lambda entry: (
            float(entry.get("latest_validation_summary", {}).get("validation_pass_rate", 0.0)),
            float(entry.get("research_score", 0.0)),
            str(entry.get("observed_at", "")),
        ),
        reverse=True,
    )
    return dict(ordered[0])


def build_research_champion_summary(
    report,
    *,
    validation_summary: dict[str, Any],
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
        "validation_summary": dict(validation_summary),
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
        "validation_summary": dict(candidate.get("latest_validation_summary", {})),
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
                "validation_pass_rate": float(
                    entry.get("latest_validation_summary", {}).get("validation_pass_rate", 0.0)
                ),
                "validated_for_rollout": bool(
                    entry.get("latest_validation_summary", {}).get("validated_for_rollout", False)
                ),
            }
        )
    return summaries
