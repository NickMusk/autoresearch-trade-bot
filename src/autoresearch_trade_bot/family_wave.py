from __future__ import annotations

import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

from .autoresearch import GitAutoresearchDecision
from .mutations import run_llm_mutation_campaign
from .strategy_families import WAVE1_FAMILIES, extract_strategy_family, get_strategy_family_profile, render_train_file


@dataclass(frozen=True)
class FamilyWaveRunResult:
    strategy_family: str
    branch_name: str
    repo_root: str
    decisions: tuple[GitAutoresearchDecision, ...]


def default_family_branch_name(strategy_family: str) -> str:
    return f"codex/family-{strategy_family.replace('_', '-')}"


def prepare_family_repo(
    *,
    source_repo_root: str | Path,
    family_repo_root: str | Path,
    strategy_family: str,
    branch_name: str | None = None,
) -> tuple[Path, str]:
    source_repo = Path(source_repo_root).resolve()
    repo_root = Path(family_repo_root).resolve()
    repo_root.parent.mkdir(parents=True, exist_ok=True)
    if not repo_root.exists():
        _run_git(None, "clone", str(source_repo), str(repo_root))

    _run_git(repo_root, "fetch", "origin")
    _run_git(repo_root, "checkout", "main")
    _run_git(repo_root, "merge", "--ff-only", "origin/main")

    family_branch = branch_name or default_family_branch_name(strategy_family)
    existing_branch = _run_git(repo_root, "branch", "--list", family_branch).strip()
    if existing_branch:
        _run_git(repo_root, "checkout", family_branch)
    else:
        _run_git(repo_root, "checkout", "-b", family_branch, "main")

    train_path = repo_root / "train.py"
    profile = get_strategy_family_profile(strategy_family)
    current_family = (
        extract_strategy_family(train_path.read_text(encoding="utf-8"))
        if train_path.exists()
        else ""
    )
    if current_family != strategy_family:
        train_path.write_text(
            render_train_file(
                profile.default_train_config,
                strategy_family=strategy_family,
            ),
            encoding="utf-8",
        )
        _run_git(repo_root, "add", "train.py")
        _run_git(repo_root, "commit", "-m", f"Seed {strategy_family} baseline")

    _run_git(repo_root, "checkout", "main")
    return repo_root, family_branch


def run_llm_family_wave(
    *,
    campaign_path: str | Path,
    source_repo_root: str | Path,
    family_repos_root: str | Path = ".autoresearch/family_repos",
    model_name: str = "gpt-5-mini",
    max_mutations: int = 1,
    recent_results_limit: int = 5,
    strategy_families: Sequence[str] = WAVE1_FAMILIES,
    max_parallel: int | None = None,
    family_runner: Callable[..., list[GitAutoresearchDecision]] = run_llm_mutation_campaign,
) -> list[FamilyWaveRunResult]:
    family_specs: list[tuple[str, Path, str]] = []
    family_repos_dir = Path(family_repos_root)
    for strategy_family in strategy_families:
        family_repo_root, branch_name = prepare_family_repo(
            source_repo_root=source_repo_root,
            family_repo_root=family_repos_dir / strategy_family,
            strategy_family=strategy_family,
        )
        family_specs.append((strategy_family, family_repo_root, branch_name))

    parallelism = max_parallel or len(family_specs)
    results: list[FamilyWaveRunResult] = []
    with ThreadPoolExecutor(max_workers=max(1, parallelism)) as executor:
        futures = {
            executor.submit(
                family_runner,
                campaign_path=campaign_path,
                repo_root=family_repo_root,
                branch_name=branch_name,
                model_name=model_name,
                max_mutations=max_mutations,
                worktrees_root=family_repo_root / ".autoresearch" / "worktrees",
                recent_results_limit=recent_results_limit,
            ): (strategy_family, family_repo_root, branch_name)
            for strategy_family, family_repo_root, branch_name in family_specs
        }
        for future in as_completed(futures):
            strategy_family, family_repo_root, branch_name = futures[future]
            decisions = tuple(future.result())
            results.append(
                FamilyWaveRunResult(
                    strategy_family=strategy_family,
                    branch_name=branch_name,
                    repo_root=str(family_repo_root),
                    decisions=decisions,
                )
            )
    results.sort(key=lambda item: item.strategy_family)
    return results


def serialize_family_wave_results(results: Sequence[FamilyWaveRunResult]) -> list[dict[str, Any]]:
    payload: list[dict[str, Any]] = []
    for item in results:
        payload.append(
            {
                "strategy_family": item.strategy_family,
                "branch_name": item.branch_name,
                "repo_root": item.repo_root,
                "decisions": [
                    {
                        "decision": decision.decision,
                        "stage": decision.stage,
                        "mutation_label": decision.mutation_label,
                        "baseline_score": decision.baseline_score,
                        "candidate_score": decision.candidate_score,
                        "kept_commit": decision.kept_commit,
                        "failure_reason": decision.report.failure_reason,
                    }
                    for decision in item.decisions
                ],
            }
        )
    return payload


def _run_git(repo_root: Path | None, *args: str) -> str:
    completed = subprocess.run(
        ["git", *args],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()
