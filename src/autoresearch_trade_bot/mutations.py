from __future__ import annotations

import ast
import csv
import hashlib
import json
import os
import re
import socket
import subprocess
import time
import urllib.error
import urllib.request
import uuid
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence

from .autoresearch import (
    DEFAULT_WORKTREE_ROOT,
    DeterministicTrainMutator,
    GitAutoresearchDecision,
    GitAutoresearchRunner,
    MutationProposal,
    normalize_train_config,
    load_campaign,
    render_train_file,
)

ALLOWED_IMPORTS = {
    "__future__",
    "dataclasses",
    "math",
    "statistics",
    "typing",
    "autoresearch_trade_bot.models",
    "autoresearch_trade_bot.strategy",
}
FORBIDDEN_CALLS = {"open", "exec", "eval", "compile", "__import__", "input"}
MAX_CANDIDATE_BYTES = 24_000


@dataclass(frozen=True)
class MutationContext:
    campaign_id: str
    campaign_name: str
    branch_name: str
    parent_commit: str
    train_path: Path
    results_path: Path
    artifact_root: Path
    program_text: str
    current_train_text: str
    recent_results: tuple[dict[str, str], ...]
    experiment_memory_summary: str = ""


@dataclass(frozen=True)
class LLMCompletion:
    content: str
    model_name: str
    prompt_id: str
    raw_response: dict[str, Any]


class MutationProvider(Protocol):
    def generate(
        self,
        *,
        context: MutationContext,
        max_mutations: int,
    ) -> list[MutationProposal]:
        ...


class LLMClient(Protocol):
    def generate(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
    ) -> LLMCompletion:
        ...


class DeterministicMutationProvider:
    def generate(
        self,
        *,
        context: MutationContext,
        max_mutations: int,
    ) -> list[MutationProposal]:
        proposals = []
        current_config = extract_train_config(context.current_train_text)
        for item in DeterministicTrainMutator().generate(context.train_path)[:max_mutations]:
            candidate_config = dict(current_config)
            candidate_config.update(item.config_updates)
            proposals.append(
                MutationProposal(
                    label=item.label,
                    candidate_text=render_train_file(candidate_config),
                    commit_message=item.commit_message,
                    provider_name="deterministic",
                    notes=json.dumps(item.config_updates, sort_keys=True),
                )
            )
        return proposals


class OpenAIResponsesClient:
    def __init__(
        self,
        *,
        api_key: str | None = None,
        model_name: str = "gpt-5-mini",
        timeout_seconds: float = 60.0,
        max_retries: int = 2,
        retry_backoff_seconds: float = 0.0,
        base_url: str = "https://api.openai.com/v1/responses",
    ) -> None:
        self.api_key = api_key or os.getenv("OPENAI_API_KEY", "")
        self.model_name = model_name
        self.timeout_seconds = timeout_seconds
        self.max_retries = max(1, max_retries)
        self.retry_backoff_seconds = max(0.0, retry_backoff_seconds)
        self.base_url = base_url
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY is required for LLM mutation runs")

    def generate(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
    ) -> LLMCompletion:
        payload = {
            "model": self.model_name,
            "instructions": system_prompt,
            "input": user_prompt,
        }
        request = urllib.request.Request(
            self.base_url,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        raw_payload = self._execute_with_retry(request)

        content = extract_response_text(raw_payload)
        if not content.strip():
            raise RuntimeError("openai_empty_response")
        return LLMCompletion(
            content=content,
            model_name=str(raw_payload.get("model", self.model_name)),
            prompt_id=str(raw_payload.get("id", "")),
            raw_response=raw_payload,
        )

    def _execute_with_retry(self, request: urllib.request.Request) -> dict[str, Any]:
        attempts = self.max_retries
        for attempt in range(1, attempts + 1):
            try:
                with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                    return json.loads(response.read().decode("utf-8"))
            except urllib.error.HTTPError as exc:
                detail = exc.read().decode("utf-8", errors="replace")
                raise RuntimeError(f"openai_http_error:{exc.code}:{detail}") from exc
            except TimeoutError as exc:
                if attempt == attempts:
                    raise RuntimeError("openai_timeout") from exc
                time.sleep(self.retry_backoff_seconds)
            except socket.timeout as exc:
                if attempt == attempts:
                    raise RuntimeError("openai_timeout") from exc
                time.sleep(self.retry_backoff_seconds)
            except urllib.error.URLError as exc:
                reason = exc.reason
                if isinstance(reason, (TimeoutError, socket.timeout)):
                    if attempt == attempts:
                        raise RuntimeError("openai_timeout") from exc
                    time.sleep(self.retry_backoff_seconds)
                    continue
                raise RuntimeError(f"openai_transport_error:{reason}") from exc
        raise RuntimeError("openai_timeout")


class LLMMutationProvider:
    def __init__(
        self,
        *,
        client: LLMClient,
        mutation_label_prefix: str = "llm",
    ) -> None:
        self.client = client
        self.mutation_label_prefix = mutation_label_prefix

    def generate(
        self,
        *,
        context: MutationContext,
        max_mutations: int,
    ) -> list[MutationProposal]:
        system_prompt, user_prompt = build_llm_mutation_prompt(context=context, max_mutations=max_mutations)
        proposals: list[MutationProposal] = []
        seen_candidate_sha1s: set[str] = set()
        for attempt_index in range(max_mutations):
            attempt_user_prompt = "\n\n".join(
                [
                    user_prompt,
                    (
                        "Batch attempt:\n"
                        f"- attempt_index={attempt_index + 1}\n"
                        f"- total_attempts={max_mutations}\n"
                        "- Return a candidate meaningfully different from prior attempts in this same cycle."
                    ),
                ]
            )
            completion = self.client.generate(system_prompt=system_prompt, user_prompt=attempt_user_prompt)
            candidate_text = extract_python_candidate(completion.content)
            candidate_sha1 = hashlib.sha1(candidate_text.encode("utf-8")).hexdigest()
            if candidate_sha1 in seen_candidate_sha1s:
                continue
            seen_candidate_sha1s.add(candidate_sha1)
            label = f"{self.mutation_label_prefix}-{uuid.uuid4().hex[:8]}"
            proposal = MutationProposal(
                label=label,
                candidate_text=candidate_text,
                commit_message=f"Mutate train.py via {completion.model_name}",
                provider_name="llm",
                model_name=completion.model_name,
                prompt_id=completion.prompt_id,
                notes=f"llm-mutation-attempt-{attempt_index + 1}",
                candidate_sha1=candidate_sha1,
            )
            artifact_path = write_proposal_artifact(
                artifact_root=context.artifact_root,
                proposal=proposal,
                system_prompt=system_prompt,
                user_prompt=attempt_user_prompt,
                response_text=completion.content,
                raw_response=completion.raw_response,
            )
            proposals.append(
                MutationProposal(
                    label=f"{self.mutation_label_prefix}-{uuid.uuid4().hex[:8]}",
                    candidate_text=candidate_text,
                    commit_message=proposal.commit_message,
                    provider_name=proposal.provider_name,
                    model_name=proposal.model_name,
                    prompt_id=proposal.prompt_id,
                    notes=proposal.notes,
                    proposal_artifact_path=str(artifact_path),
                    candidate_sha1=candidate_sha1,
                )
            )
        return proposals


def validate_train_candidate_text(candidate_text: str) -> tuple[bool, str]:
    encoded = candidate_text.encode("utf-8")
    if len(encoded) > MAX_CANDIDATE_BYTES:
        return False, "candidate_too_large"
    try:
        tree = ast.parse(candidate_text)
    except SyntaxError:
        return False, "candidate_syntax_error"

    required_names = {"TRAIN_CONFIG", "STRATEGY_NAME", "build_strategy"}
    seen_names: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    seen_names.add(target.id)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            seen_names.add(node.name)

    missing = required_names - seen_names
    if missing:
        return False, f"missing_required_symbols:{','.join(sorted(missing))}"

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name not in ALLOWED_IMPORTS:
                    return False, f"forbidden_import:{alias.name}"
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            if module not in ALLOWED_IMPORTS:
                return False, f"forbidden_import:{module}"
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.func.id in FORBIDDEN_CALLS:
                return False, f"forbidden_call:{node.func.id}"
    return True, ""


def build_mutation_context(
    *,
    repo_root: str | Path,
    campaign_path: str | Path,
    branch_name: str,
    train_path: Path,
    results_path: Path,
    artifact_root: Path,
    recent_results_limit: int = 5,
) -> MutationContext:
    campaign = load_campaign(campaign_path)
    recent_results = tuple(load_recent_results(results_path, limit=max(recent_results_limit, 12)))
    return MutationContext(
        campaign_id=campaign.campaign_id,
        campaign_name=campaign.name,
        branch_name=branch_name,
        parent_commit=_safe_read_git_head(train_path.parent),
        train_path=train_path,
        results_path=results_path,
        artifact_root=artifact_root,
        program_text=(Path(repo_root) / "program.md").read_text(encoding="utf-8"),
        current_train_text=train_path.read_text(encoding="utf-8"),
        recent_results=recent_results,
        experiment_memory_summary=build_experiment_memory_summary(
            recent_results,
            current_train_text=train_path.read_text(encoding="utf-8"),
        ),
    )


def build_experiment_memory_summary(
    recent_results: Sequence[Mapping[str, str]],
    *,
    current_train_text: str,
) -> str:
    normalized_config = normalize_train_config(extract_train_config(current_train_text))
    if not recent_results:
        return (
            "Decision mix: no prior LLM evaluations yet.\n"
            f"Current baseline config: {json.dumps(normalized_config, sort_keys=True)}\n"
            "Promising directions:\n"
            "- Add selectivity with min_cross_sectional_spread or min_signal_strength.\n"
            "- Use volatility_floor to stabilize risk-adjusted ranking.\n"
            "- Use reversal_bias_weight or funding_penalty_weight to avoid crowded or overextended entries."
        )

    decisions = Counter(str(row.get("decision", "")) for row in recent_results)
    stages = Counter(str(row.get("stage", "")) for row in recent_results)
    failures = Counter(str(row.get("failure_reason", "")) for row in recent_results if row.get("failure_reason"))

    deltas: list[tuple[str, float, str, str]] = []
    for row in recent_results:
        raw_delta = (row.get("delta_score") or "").strip()
        if not raw_delta:
            continue
        try:
            delta_value = float(raw_delta)
        except ValueError:
            continue
        deltas.append(
            (
                str(row.get("mutation_label", "")),
                delta_value,
                str(row.get("decision", "")),
                str(row.get("stage", "")),
            )
        )

    best_delta_lines = [
        f"- {label}: delta_score={delta:.4f} decision={decision} stage={stage}"
        for label, delta, decision, stage in sorted(deltas, key=lambda item: item[1], reverse=True)[:3]
    ] or ["- no positive or recorded deltas yet"]

    common_failure_lines = [
        f"- {reason}: {count}"
        for reason, count in failures.most_common(3)
    ] or ["- no validation/runtime failures recorded"]

    search_directions = []
    discard_screen_count = decisions.get("discard_screen", 0)
    if discard_screen_count >= max(2, len(recent_results) // 2):
        search_directions.append(
            "- Most candidates fail at screen. Prefer materially more selective changes, not tiny parameter churn."
        )
        search_directions.append(
            "- Try raising min_signal_strength or min_cross_sectional_spread, or lowering gross_target when signal quality is weak."
        )
    search_directions.append(
        "- Consider volatility_floor > 0 when ranking_mode='risk_adjusted' to reduce unstable scores from tiny realized volatility."
    )
    search_directions.append(
        "- Consider reversal_bias_weight > 0 to avoid chasing one-bar overextension."
    )
    search_directions.append(
        "- Consider funding_penalty_weight > 0 to avoid crowded perp positioning."
    )
    if not normalized_config.get("use_regime_filter", False):
        search_directions.append(
            "- Explore use_regime_filter=True only if paired with a meaningful threshold change, not as a cosmetic toggle."
        )

    return "\n".join(
        [
            "Decision mix: "
            + ", ".join(f"{decision}={count}" for decision, count in sorted(decisions.items())),
            "Stage mix: " + ", ".join(f"{stage}={count}" for stage, count in sorted(stages.items())),
            f"Current baseline config: {json.dumps(normalized_config, sort_keys=True)}",
            "Best recent deltas:",
            *best_delta_lines,
            "Common failures:",
            *common_failure_lines,
            "Promising directions:",
            *search_directions,
        ]
    )


def build_llm_mutation_prompt(
    *,
    context: MutationContext,
    max_mutations: int,
) -> tuple[str, str]:
    recent_result_lines = "\n".join(
        [
            f"- mutation={row.get('mutation_label','')} stage={row.get('stage','')} "
            f"decision={row.get('decision','')} score={row.get('research_score','')} "
            f"delta={row.get('delta_score','')} failure={row.get('failure_reason','')}"
            for row in context.recent_results[-6:]
        ]
    ) or "- no prior results"
    system_prompt = "\n".join(
        [
            "You are mutating exactly one file: train.py.",
            "Return only valid Python source code for the full train.py file.",
            "Do not add markdown fences or commentary.",
            "Keep the editable surface within TRAIN_CONFIG, STRATEGY_NAME, class logic, and build_strategy.",
            "Do not import forbidden modules or perform I/O, subprocess, networking, or dynamic imports.",
            "Optimize research_score while respecting the hard gates described below.",
            "Prefer one substantial research hypothesis over cosmetic rewrites.",
            "Do not only rename identifiers or make no-op refactors.",
        ]
    )
    user_prompt = "\n\n".join(
        [
            f"Program rules:\n{context.program_text}",
            f"Research memory:\n{context.experiment_memory_summary}",
            "Recent raw results:\n" + recent_result_lines,
            "Current train.py to mutate:\n" + context.current_train_text,
            (
                "Mutation request:\n"
                "Return a single best full train.py candidate as raw Python only.\n"
                f"- The harness will evaluate up to {max_mutations} mutation attempt(s) this cycle, but this response should contain one candidate only.\n"
                "- Make a meaningful change to strategy behavior.\n"
                "- Prefer changes that improve selectivity, reduce weak-signal trades, or reduce crowding/overextension risk.\n"
                "- Keep compute cheap and stay within the one-file editable boundary."
            ),
        ]
    )
    return system_prompt, user_prompt


def extract_python_candidate(response_text: str) -> str:
    fenced_match = re.search(r"```(?:python)?\s*(.*?)```", response_text, flags=re.DOTALL)
    if fenced_match:
        return fenced_match.group(1).strip() + "\n"
    return response_text.strip() + "\n"


def extract_response_text(payload: Mapping[str, Any]) -> str:
    output_text = payload.get("output_text")
    if isinstance(output_text, str) and output_text.strip():
        return output_text
    outputs = payload.get("output", [])
    for item in outputs:
        for content in item.get("content", []):
            text = content.get("text")
            if isinstance(text, str) and text.strip():
                return text
            if isinstance(text, Mapping):
                value = text.get("value")
                if isinstance(value, str) and value.strip():
                    return value
    return ""


def write_proposal_artifact(
    *,
    artifact_root: Path,
    proposal: MutationProposal,
    system_prompt: str,
    user_prompt: str,
    response_text: str,
    raw_response: Mapping[str, Any],
) -> Path:
    proposals_root = artifact_root / "proposals"
    proposals_root.mkdir(parents=True, exist_ok=True)
    artifact_path = proposals_root / f"{proposal.label}.json"
    artifact_path.write_text(
        json.dumps(
            {
                "label": proposal.label,
                "provider_name": proposal.provider_name,
                "model_name": proposal.model_name,
                "prompt_id": proposal.prompt_id,
                "notes": proposal.notes,
                "commit_message": proposal.commit_message,
                "system_prompt": system_prompt,
                "user_prompt": user_prompt,
                "response_text": response_text,
                "candidate_text": proposal.candidate_text,
                "raw_response": raw_response,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return artifact_path


def load_recent_results(results_path: str | Path, *, limit: int) -> list[dict[str, str]]:
    path = Path(results_path)
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    return rows[-limit:]


def extract_train_config(train_text: str) -> dict[str, Any]:
    match = re.search(r"TRAIN_CONFIG = (.*?)\nSTRATEGY_NAME", train_text, flags=re.DOTALL)
    if match is None:
        raise ValueError("train.py is missing TRAIN_CONFIG block")
    return ast.literal_eval(match.group(1).strip())


def run_llm_mutation_campaign(
    *,
    campaign_path: str | Path,
    repo_root: str | Path,
    branch_name: str,
    model_name: str = "gpt-5-mini",
    max_mutations: int = 1,
    worktrees_root: str | Path = DEFAULT_WORKTREE_ROOT,
    recent_results_limit: int = 5,
    openai_timeout_seconds: float = 60.0,
    openai_max_retries: int = 2,
    openai_retry_backoff_seconds: float = 0.0,
    provider: MutationProvider | None = None,
    client: LLMClient | None = None,
) -> list[GitAutoresearchDecision]:
    runner = GitAutoresearchRunner(
        repo_root=repo_root,
        branch_name=branch_name,
        worktrees_root=worktrees_root,
    )
    runner.ensure_worktree()
    context = build_mutation_context(
        repo_root=repo_root,
        campaign_path=campaign_path,
        branch_name=branch_name,
        train_path=runner.train_path,
        results_path=runner.results_path,
        artifact_root=runner.artifact_root,
        recent_results_limit=recent_results_limit,
    )
    resolved_provider = provider or LLMMutationProvider(
        client=client
        or OpenAIResponsesClient(
            model_name=model_name,
            timeout_seconds=openai_timeout_seconds,
            max_retries=openai_max_retries,
            retry_backoff_seconds=openai_retry_backoff_seconds,
        )
    )
    decisions = []
    for proposal in resolved_provider.generate(context=context, max_mutations=max_mutations):
        decisions.append(
            runner.apply_mutation_proposal_staged(
                campaign_path=campaign_path,
                proposal=proposal,
                validator=validate_train_candidate_text,
            )
        )
    return decisions


def _safe_read_git_head(cwd: Path) -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=cwd,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except Exception:
        return ""
