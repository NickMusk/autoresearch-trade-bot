from __future__ import annotations

import base64
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


def _ensure_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


@dataclass(frozen=True)
class LeaderboardEntry:
    strategy_name: str
    params: dict[str, Any]
    accepted: bool
    score: float
    metrics: dict[str, Any]
    rejection_reasons: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "strategy_name": self.strategy_name,
            "params": self.params,
            "accepted": self.accepted,
            "score": self.score,
            "metrics": self.metrics,
            "rejection_reasons": self.rejection_reasons,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "LeaderboardEntry":
        return cls(
            strategy_name=str(payload["strategy_name"]),
            params=dict(payload["params"]),
            accepted=bool(payload["accepted"]),
            score=float(payload["score"]),
            metrics=dict(payload["metrics"]),
            rejection_reasons=[str(item) for item in payload.get("rejection_reasons", [])],
        )


@dataclass(frozen=True)
class CycleSummary:
    cycle_id: str
    dataset_id: str
    completed_at: datetime
    last_processed_bar: datetime
    accepted: bool
    score: float
    strategy_name: str
    params: dict[str, Any]
    metrics: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "cycle_id": self.cycle_id,
            "dataset_id": self.dataset_id,
            "completed_at": self.completed_at.isoformat(),
            "last_processed_bar": self.last_processed_bar.isoformat(),
            "accepted": self.accepted,
            "score": self.score,
            "strategy_name": self.strategy_name,
            "params": self.params,
            "metrics": self.metrics,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "CycleSummary":
        return cls(
            cycle_id=str(payload["cycle_id"]),
            dataset_id=str(payload["dataset_id"]),
            completed_at=_ensure_utc(datetime.fromisoformat(payload["completed_at"])),
            last_processed_bar=_ensure_utc(datetime.fromisoformat(payload["last_processed_bar"])),
            accepted=bool(payload["accepted"]),
            score=float(payload["score"]),
            strategy_name=str(payload["strategy_name"]),
            params=dict(payload["params"]),
            metrics=dict(payload["metrics"]),
        )


@dataclass(frozen=True)
class WorkerCheckpoint:
    last_processed_bar: datetime | None = None
    last_multi_window_bar: datetime | None = None
    consecutive_failures: int = 0
    last_cycle_completed_at: datetime | None = None

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "consecutive_failures": self.consecutive_failures,
        }
        if self.last_processed_bar is not None:
            payload["last_processed_bar"] = self.last_processed_bar.isoformat()
        if self.last_multi_window_bar is not None:
            payload["last_multi_window_bar"] = self.last_multi_window_bar.isoformat()
        if self.last_cycle_completed_at is not None:
            payload["last_cycle_completed_at"] = self.last_cycle_completed_at.isoformat()
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "WorkerCheckpoint":
        return cls(
            last_processed_bar=(
                _ensure_utc(datetime.fromisoformat(payload["last_processed_bar"]))
                if payload.get("last_processed_bar")
                else None
            ),
            last_multi_window_bar=(
                _ensure_utc(datetime.fromisoformat(payload["last_multi_window_bar"]))
                if payload.get("last_multi_window_bar")
                else None
            ),
            consecutive_failures=int(payload.get("consecutive_failures", 0)),
            last_cycle_completed_at=(
                _ensure_utc(datetime.fromisoformat(payload["last_cycle_completed_at"]))
                if payload.get("last_cycle_completed_at")
                else None
            ),
        )


@dataclass(frozen=True)
class ResearchStatusSnapshot:
    mission: str
    phase: str
    research_rollout_ready: bool
    research_blockers: list[str]
    baseline_strategy: str
    promotion_gate: dict[str, Any]
    baseline_metrics: dict[str, Any]
    accepted_for_paper: bool
    next_milestones: list[str]
    loop_state: str
    latest_dataset_id: str
    latest_cycle_completed_at: str | None
    last_processed_bar: str | None
    recent_acceptance_rate: float
    consecutive_failures: int
    multi_window_summary: dict[str, Any] = field(default_factory=dict)
    leaderboard: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "mission": self.mission,
            "phase": self.phase,
            "research_rollout_ready": self.research_rollout_ready,
            "research_blockers": list(self.research_blockers),
            "baseline_strategy": self.baseline_strategy,
            "promotion_gate": dict(self.promotion_gate),
            "baseline_metrics": dict(self.baseline_metrics),
            "accepted_for_paper": self.accepted_for_paper,
            "next_milestones": list(self.next_milestones),
            "loop_state": self.loop_state,
            "latest_dataset_id": self.latest_dataset_id,
            "latest_cycle_completed_at": self.latest_cycle_completed_at,
            "last_processed_bar": self.last_processed_bar,
            "recent_acceptance_rate": self.recent_acceptance_rate,
            "consecutive_failures": self.consecutive_failures,
            "multi_window_summary": dict(self.multi_window_summary),
            "leaderboard": list(self.leaderboard),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ResearchStatusSnapshot":
        return cls(
            mission=str(payload["mission"]),
            phase=str(payload["phase"]),
            research_rollout_ready=bool(payload["research_rollout_ready"]),
            research_blockers=[str(item) for item in payload["research_blockers"]],
            baseline_strategy=str(payload["baseline_strategy"]),
            promotion_gate=dict(payload["promotion_gate"]),
            baseline_metrics=dict(payload["baseline_metrics"]),
            accepted_for_paper=bool(payload["accepted_for_paper"]),
            next_milestones=[str(item) for item in payload["next_milestones"]],
            loop_state=str(payload.get("loop_state", "idle")),
            latest_dataset_id=str(payload.get("latest_dataset_id", "")),
            latest_cycle_completed_at=payload.get("latest_cycle_completed_at"),
            last_processed_bar=payload.get("last_processed_bar"),
            recent_acceptance_rate=float(payload.get("recent_acceptance_rate", 0.0)),
            consecutive_failures=int(payload.get("consecutive_failures", 0)),
            multi_window_summary=dict(payload.get("multi_window_summary", {})),
            leaderboard=[dict(item) for item in payload.get("leaderboard", [])],
        )


class FilesystemResearchStateStore:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)

    @property
    def latest_status_path(self) -> Path:
        return self.root / "latest_status.json"

    @property
    def leaderboard_path(self) -> Path:
        return self.root / "leaderboard.json"

    @property
    def history_path(self) -> Path:
        return self.root / "history.json"

    @property
    def checkpoint_path(self) -> Path:
        return self.root / "checkpoint.json"

    def save_snapshot(self, snapshot: ResearchStatusSnapshot) -> Path:
        return self._write_json(self.latest_status_path, snapshot.to_dict())

    def load_snapshot(self) -> ResearchStatusSnapshot | None:
        if not self.latest_status_path.exists():
            return None
        payload = json.loads(self.latest_status_path.read_text(encoding="utf-8"))
        return ResearchStatusSnapshot.from_dict(payload)

    def save_leaderboard(self, entries: list[LeaderboardEntry]) -> Path:
        payload = {"leaderboard": [entry.to_dict() for entry in entries]}
        return self._write_json(self.leaderboard_path, payload)

    def load_history(self) -> list[CycleSummary]:
        if not self.history_path.exists():
            return []
        payload = json.loads(self.history_path.read_text(encoding="utf-8"))
        return [CycleSummary.from_dict(item) for item in payload.get("cycles", [])]

    def save_history(self, cycles: list[CycleSummary]) -> Path:
        payload = {"cycles": [cycle.to_dict() for cycle in cycles]}
        return self._write_json(self.history_path, payload)

    def load_checkpoint(self) -> WorkerCheckpoint:
        if not self.checkpoint_path.exists():
            return WorkerCheckpoint()
        payload = json.loads(self.checkpoint_path.read_text(encoding="utf-8"))
        return WorkerCheckpoint.from_dict(payload)

    def save_checkpoint(self, checkpoint: WorkerCheckpoint) -> Path:
        return self._write_json(self.checkpoint_path, checkpoint.to_dict())

    def _write_json(self, path: Path, payload: dict[str, Any]) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return path


class GitHubStatusPublisher:
    def __init__(
        self,
        token: str,
        repo: str,
        branch: str = "render-state",
        base_path: str = "status",
        api_base_url: str = "https://api.github.com",
    ) -> None:
        self.token = token
        self.repo = repo
        self.branch = branch
        self.base_path = base_path.strip("/")
        self.api_base_url = api_base_url.rstrip("/")

    def publish_json(self, filename: str, payload: dict[str, Any], message: str) -> None:
        path = "/".join(part for part in (self.base_path, filename) if part)
        content = json.dumps(payload, indent=2).encode("utf-8")
        sha = self._lookup_existing_sha(path)
        body: dict[str, Any] = {
            "message": message,
            "content": base64.b64encode(content).decode("ascii"),
            "branch": self.branch,
        }
        if sha is not None:
            body["sha"] = sha
        request = Request(
            f"{self.api_base_url}/repos/{self.repo}/contents/{path}",
            data=json.dumps(body).encode("utf-8"),
            method="PUT",
            headers=self._headers(),
        )
        with urlopen(request, timeout=30):
            return

    def _lookup_existing_sha(self, path: str) -> str | None:
        request = Request(
            f"{self.api_base_url}/repos/{self.repo}/contents/{path}?ref={self.branch}",
            headers=self._headers(),
        )
        try:
            with urlopen(request, timeout=30) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except HTTPError as exc:
            if exc.code == 404:
                return None
            raise
        return str(payload["sha"])

    def _headers(self) -> dict[str, str]:
        return {
            "Accept": "application/vnd.github+json",
            "Authorization": f"Bearer {self.token}",
            "User-Agent": "autoresearch-trade-bot",
            "Content-Type": "application/json",
        }


def load_status_snapshot_from_path(path: str | Path) -> ResearchStatusSnapshot | None:
    file_path = Path(path)
    if not file_path.exists():
        return None
    payload = json.loads(file_path.read_text(encoding="utf-8"))
    return ResearchStatusSnapshot.from_dict(payload)


def load_status_snapshot_from_url(url: str, timeout_seconds: int = 10) -> ResearchStatusSnapshot | None:
    request = Request(url, headers={"User-Agent": "autoresearch-trade-bot"})
    try:
        with urlopen(request, timeout=timeout_seconds) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except (HTTPError, URLError, TimeoutError, json.JSONDecodeError):
        return None
    return ResearchStatusSnapshot.from_dict(payload)
