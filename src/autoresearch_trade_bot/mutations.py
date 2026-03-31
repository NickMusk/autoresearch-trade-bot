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
    load_campaign,
)
from .strategy_families import (
    FAMILY_EMA_TREND,
    FAMILY_MEAN_REVERSION,
    FAMILY_MOMENTUM,
    FAMILY_VOLATILITY_BREAKOUT,
    config_traits as family_config_traits,
    extract_strategy_family,
    family_template_constraints,
    family_attempt_role_specs,
    family_mutation_bounds,
    family_prompt_directions,
    render_family_mutation_bounds,
    normalize_train_config as normalize_family_train_config,
    render_train_file as render_family_train_file,
    validate_train_candidate_semantics as validate_family_candidate_semantics,
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
MAX_CANDIDATE_BYTES = 48_000


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
    strategy_family: str
    recent_results: tuple[dict[str, str], ...]
    symbol_count: int
    experiment_memory_summary: str = ""
    experiment_memory_artifact: dict[str, Any] | None = None


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
                    candidate_text=render_family_train_file(
                        candidate_config,
                        strategy_family=context.strategy_family,
                    ),
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
                if self._is_retryable_http_status(exc.code) and attempt < attempts:
                    time.sleep(self._retry_delay_seconds(exc.headers, attempt))
                    continue
                raise RuntimeError(f"openai_http_error:{exc.code}:{detail}") from exc
            except TimeoutError as exc:
                if attempt == attempts:
                    raise RuntimeError("openai_timeout") from exc
                time.sleep(self._retry_delay_seconds(None, attempt))
            except socket.timeout as exc:
                if attempt == attempts:
                    raise RuntimeError("openai_timeout") from exc
                time.sleep(self._retry_delay_seconds(None, attempt))
            except urllib.error.URLError as exc:
                reason = exc.reason
                if isinstance(reason, (TimeoutError, socket.timeout)):
                    if attempt == attempts:
                        raise RuntimeError("openai_timeout") from exc
                    time.sleep(self._retry_delay_seconds(None, attempt))
                    continue
                raise RuntimeError(f"openai_transport_error:{reason}") from exc
        raise RuntimeError("openai_timeout")

    @staticmethod
    def _is_retryable_http_status(status_code: int) -> bool:
        return status_code in {408, 409, 429, 500, 502, 503, 504}

    def _retry_delay_seconds(self, headers: Any, attempt: int) -> float:
        retry_after_seconds = self._retry_after_seconds(headers)
        if retry_after_seconds is not None:
            return retry_after_seconds
        # Exponential backoff with a small floor, while staying simple and predictable.
        multiplier = max(1, attempt)
        return self.retry_backoff_seconds * multiplier

    @staticmethod
    def _retry_after_seconds(headers: Any) -> float | None:
        if headers is None:
            return None
        retry_after = None
        if hasattr(headers, "get"):
            retry_after = headers.get("Retry-After")
        elif isinstance(headers, Mapping):
            retry_after = headers.get("Retry-After")
        if retry_after is None:
            return None
        try:
            parsed = float(str(retry_after).strip())
        except ValueError:
            return None
        return max(0.0, parsed)


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
            role_name, attempt_user_prompt = build_attempt_prompt(
                base_user_prompt=user_prompt,
                attempt_index=attempt_index,
                max_mutations=max_mutations,
                strategy_family=context.strategy_family,
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
                notes=f"llm-mutation-attempt-{attempt_index + 1}:{role_name}",
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
                    label=label,
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

    required_names = {"TRAIN_CONFIG", "STRATEGY_NAME", "STRATEGY_FAMILY", "build_strategy"}
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
    return True, ""


def validate_train_candidate_semantics(
    candidate_text: str,
    *,
    current_train_text: str,
    strategy_family: str | None = None,
    symbol_count: int,
) -> tuple[bool, str]:
    current_family = strategy_family or extract_strategy_family(current_train_text)
    candidate_family = extract_strategy_family(candidate_text)
    if candidate_family != current_family:
        return False, "strategy_family_mismatch"
    candidate_config = extract_train_config(candidate_text)
    current_config = extract_train_config(current_train_text)
    return validate_family_candidate_semantics(
        current_family,
        candidate_text=candidate_text,
        candidate_config=candidate_config,
        current_config=current_config,
        symbol_count=symbol_count,
    )


def build_mutation_context(
    *,
    repo_root: str | Path,
    campaign_path: str | Path,
    branch_name: str,
    train_path: Path,
    results_path: Path,
    artifact_root: Path,
    strategy_family: str | None = None,
    recent_results_limit: int = 12,
) -> MutationContext:
    campaign = load_campaign(campaign_path)
    current_train_text = train_path.read_text(encoding="utf-8")
    resolved_strategy_family = strategy_family or extract_strategy_family(current_train_text)
    recent_results = tuple(load_recent_results(results_path, limit=max(recent_results_limit, 12)))
    memory_payload = build_experiment_memory_artifact(
        recent_results,
        current_train_text=current_train_text,
        strategy_family=resolved_strategy_family,
    )
    memory_summary = build_experiment_memory_summary(
        recent_results,
        current_train_text=current_train_text,
        strategy_family=resolved_strategy_family,
    )
    write_experiment_memory_artifact(
        artifact_root=artifact_root,
        memory_payload=memory_payload,
        summary_text=memory_summary,
    )
    return MutationContext(
        campaign_id=campaign.campaign_id,
        campaign_name=campaign.name,
        branch_name=branch_name,
        parent_commit=_safe_read_git_head(train_path.parent),
        train_path=train_path,
        results_path=results_path,
        artifact_root=artifact_root,
        program_text=(Path(repo_root) / "program.md").read_text(encoding="utf-8"),
        current_train_text=current_train_text,
        strategy_family=resolved_strategy_family,
        recent_results=recent_results,
        symbol_count=len(campaign.symbols),
        experiment_memory_summary=memory_summary,
        experiment_memory_artifact=memory_payload,
    )


def build_experiment_memory_artifact(
    recent_results: Sequence[Mapping[str, str]],
    *,
    current_train_text: str,
    strategy_family: str | None = None,
) -> dict[str, Any]:
    family = strategy_family or extract_strategy_family(current_train_text)
    normalized_config = normalize_family_train_config(
        extract_train_config(current_train_text),
        strategy_family=family,
    )
    if not recent_results:
        return {
            "strategy_family": family,
            "current_baseline_config": normalized_config,
            "decision_mix": {},
            "stage_mix": {},
            "common_failures": [],
            "common_gate_failures": [],
            "near_wins": [],
            "dead_zones": [],
            "promising_directions": list(family_prompt_directions(family)),
        }

    decisions = Counter(str(row.get("decision", "")) for row in recent_results)
    stages = Counter(str(row.get("stage", "")) for row in recent_results)
    failures = Counter(str(row.get("failure_reason", "")) for row in recent_results if row.get("failure_reason"))
    gate_failures = Counter()
    no_trade_traits = Counter()
    duplicate_traits = Counter()
    near_wins: list[dict[str, Any]] = []

    for row in recent_results:
        config = _load_train_config_from_result_row(row, strategy_family=family)
        traits = _config_traits(family, config)
        for failure in _parse_gate_failures(row):
            gate_failures[failure] += 1
            if failure == "no_trades_executed":
                no_trade_traits.update(traits)
        if str(row.get("failure_reason", "")) == "duplicate_candidate":
            duplicate_traits.update(traits)
        raw_delta = (row.get("delta_score") or "").strip()
        if raw_delta:
            try:
                delta = float(raw_delta)
            except ValueError:
                delta = None
            if (
                delta is not None
                and delta > -1.5
                and str(row.get("decision", "")) in {"discard_screen", "discard_full", "screen_pass"}
            ):
                near_wins.append(
                    {
                        "mutation_label": str(row.get("mutation_label", "")),
                        "delta_score": delta,
                        "decision": str(row.get("decision", "")),
                        "stage": str(row.get("stage", "")),
                        "config_focus": _format_config_focus(family, config),
                    }
                )

    dead_zones = [
        {"trait": trait, "count": count}
        for trait, count in no_trade_traits.most_common(8)
    ]
    if duplicate_traits:
        dead_zones.extend(
            {"trait": f"duplicate::{trait}", "count": count}
            for trait, count in duplicate_traits.most_common(2)
        )

    promising_directions = _promising_directions_from_memory(
        strategy_family=family,
        normalized_config=normalized_config,
        dead_zones=dead_zones,
        near_wins=near_wins,
        decisions=decisions,
    )

    return {
        "strategy_family": family,
        "current_baseline_config": normalized_config,
        "decision_mix": dict(sorted(decisions.items())),
        "stage_mix": dict(sorted(stages.items())),
        "common_failures": [
            {"failure_reason": reason, "count": count}
            for reason, count in failures.most_common(4)
        ],
        "common_gate_failures": [
            {"gate_failure": reason, "count": count}
            for reason, count in gate_failures.most_common(5)
        ],
        "near_wins": sorted(near_wins, key=lambda item: item["delta_score"], reverse=True)[:4],
        "dead_zones": dead_zones,
        "promising_directions": promising_directions,
    }


def build_experiment_memory_summary(
    recent_results: Sequence[Mapping[str, str]],
    *,
    current_train_text: str,
    strategy_family: str | None = None,
) -> str:
    memory = build_experiment_memory_artifact(
        recent_results,
        current_train_text=current_train_text,
        strategy_family=strategy_family,
    )
    decision_mix = memory["decision_mix"]
    stage_mix = memory["stage_mix"]
    common_failures = memory["common_failures"]
    common_gate_failures = memory["common_gate_failures"]
    near_wins = memory["near_wins"]
    dead_zones = memory["dead_zones"]
    return "\n".join(
        [
            "Decision mix: "
            + (
                ", ".join(f"{decision}={count}" for decision, count in decision_mix.items())
                if decision_mix
                else "no prior LLM evaluations yet"
            ),
            "Stage mix: "
            + (
                ", ".join(f"{stage}={count}" for stage, count in stage_mix.items())
                if stage_mix
                else "no prior stages"
            ),
            f"Current baseline config: {json.dumps(memory['current_baseline_config'], sort_keys=True)}",
            "Near wins:",
            *(
                [
                    f"- {item['mutation_label']}: delta_score={item['delta_score']:.4f} decision={item['decision']} stage={item['stage']} focus={item['config_focus']}"
                    for item in near_wins
                ]
                or ["- no near wins recorded yet"]
            ),
            "Common failures:",
            *(
                [f"- {item['failure_reason']}: {item['count']}" for item in common_failures]
                or ["- no validation/runtime failures recorded"]
            ),
            "Common gate failures:",
            *(
                [f"- {item['gate_failure']}: {item['count']}" for item in common_gate_failures]
                or ["- no gate failures recorded"]
            ),
            "Dead zones:",
            *(
                [f"- {item['trait']}: {item['count']}" for item in dead_zones]
                or ["- no repeated dead zones detected yet"]
            ),
            "Promising directions:",
            *(f"- {direction}" for direction in memory["promising_directions"]),
        ]
    )


def build_llm_mutation_prompt(
    *,
    context: MutationContext,
    max_mutations: int,
) -> tuple[str, str]:
    bounds = family_mutation_bounds(
        context.strategy_family,
        symbol_count=context.symbol_count,
    )
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
            f"Keep the candidate under {MAX_CANDIDATE_BYTES} bytes of UTF-8 source code.",
            "Keep the editable surface within TRAIN_CONFIG, STRATEGY_NAME, class logic, and build_strategy.",
            f"Preserve STRATEGY_FAMILY as {context.strategy_family}. Do not switch strategy families in this branch.",
            "Do not import forbidden modules or perform I/O, subprocess, networking, or dynamic imports.",
            "Optimize research_score while respecting the hard gates described below.",
            "Prefer one substantial research hypothesis over cosmetic rewrites.",
            "Do not only rename identifiers or make no-op refactors.",
            "Preserve the existing file structure whenever possible instead of expanding helpers or rewriting the whole template.",
            "Avoid adding new helper functions, verbose comments, or repeated logic unless absolutely necessary for the mutation.",
        ]
    )
    user_prompt = "\n\n".join(
        [
            f"Program rules:\n{context.program_text}",
            "Hard mutation bounds:\n"
            + render_family_mutation_bounds(
                context.strategy_family,
                symbol_count=context.symbol_count,
            ),
            "Family template constraints:\n"
            + "\n".join(
                f"- {item}" for item in family_template_constraints(context.strategy_family)
            ),
            f"Research memory:\n{context.experiment_memory_summary}",
            "Recent raw results:\n" + recent_result_lines,
            "Current train.py to mutate:\n" + context.current_train_text,
            (
                "Mutation request:\n"
                "Return a single best full train.py candidate as raw Python only.\n"
                f"- The harness will evaluate up to {max_mutations} mutation attempt(s) this cycle, but this response should contain one candidate only.\n"
                f"- Stay inside the {context.strategy_family} strategy family.\n"
                f"- Do not set top_k above {bounds['top_k_max']} for the current {bounds['symbol_count']}-symbol paired engine.\n"
                "- Make a meaningful change to strategy behavior.\n"
                "- Prefer changes that improve selectivity, reduce weak-signal trades, or reduce crowding/overextension risk.\n"
                "- Use the promising directions from research memory and avoid repeated dead zones.\n"
                "- Keep compute cheap and stay within the one-file editable boundary.\n"
                "- Make the smallest full-file edit that expresses the hypothesis; do not bloat the file.\n"
                "- Reuse existing helpers and structure instead of introducing large new code blocks.\n"
                "- Preserve the family-specific class template and explicit config surface; do not swap in another family's template."
            ),
        ]
    )
    return system_prompt, user_prompt


def build_attempt_prompt(
    *,
    base_user_prompt: str,
    attempt_index: int,
    max_mutations: int,
    strategy_family: str = FAMILY_MOMENTUM,
) -> tuple[str, str]:
    role_specs = family_attempt_role_specs(strategy_family)
    role = role_specs[attempt_index % len(role_specs)]
    return (
        role["name"],
        "\n\n".join(
            [
                base_user_prompt,
                (
                    "Batch attempt:\n"
                    f"- attempt_index={attempt_index + 1}\n"
                    f"- total_attempts={max_mutations}\n"
                    f"- role_name={role['name']}\n"
                    f"- role_objective={role['objective']}\n"
                    f"- role_constraints={role['constraints']}\n"
                    "- Return a candidate meaningfully different from prior attempts in this same cycle."
                ),
            ]
        ),
    )


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


def write_experiment_memory_artifact(
    *,
    artifact_root: Path,
    memory_payload: Mapping[str, Any],
    summary_text: str,
) -> Path:
    memory_root = artifact_root / "memory"
    memory_root.mkdir(parents=True, exist_ok=True)
    artifact_path = memory_root / "latest_memory_summary.json"
    artifact_path.write_text(
        json.dumps(
            {
                "summary_text": summary_text,
                "memory": memory_payload,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return artifact_path


def _parse_json_field(raw_value: str | None) -> Any:
    value = (raw_value or "").strip()
    if not value:
        return None
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return None


def _load_train_config_from_result_row(
    row: Mapping[str, str],
    *,
    strategy_family: str,
) -> dict[str, Any]:
    payload = _parse_json_field(row.get("train_config_json"))
    if isinstance(payload, Mapping):
        return normalize_family_train_config(payload, strategy_family=strategy_family)

    artifact_path = (row.get("proposal_artifact_path") or "").strip()
    if artifact_path:
        try:
            artifact_payload = json.loads(Path(artifact_path).read_text(encoding="utf-8"))
            candidate_text = artifact_payload.get("candidate_text")
            if isinstance(candidate_text, str) and candidate_text.strip():
                artifact_family = extract_strategy_family(candidate_text)
                return normalize_family_train_config(
                    extract_train_config(candidate_text),
                    strategy_family=artifact_family,
                )
        except Exception:
            return {}
    return {}


def _parse_gate_failures(row: Mapping[str, str]) -> list[str]:
    payload = _parse_json_field(row.get("gate_failures_json"))
    if isinstance(payload, list):
        return [str(item) for item in payload]
    return []


def _config_traits(strategy_family: str, config: Mapping[str, Any]) -> list[str]:
    return family_config_traits(strategy_family, config)


def _format_config_focus(strategy_family: str, config: Mapping[str, Any]) -> str:
    if not config:
        return "config unavailable"
    fields = [f"family={strategy_family}", f"top_k={config.get('top_k')}", f"gross={config.get('gross_target')}"]
    if strategy_family == "mean_reversion":
        fields.extend(
            [
                f"lookback={config.get('lookback_bars')}",
                f"reversion_horizon={config.get('reversion_horizon_bars')}",
                f"ibs_threshold={config.get('ibs_threshold')}",
                f"reversion_floor={config.get('reversion_strength_floor')}",
            ]
        )
        if bool(config.get("use_trend_filter", False)):
            fields.append(f"trend_lookback={config.get('trend_lookback_bars')}")
    elif strategy_family == "ema_trend":
        fields.extend(
            [
                f"horizons={config.get('fast_horizon_bars')}/{config.get('medium_horizon_bars')}/{config.get('slow_horizon_bars')}",
                f"signal_floor={config.get('min_signal_strength')}",
            ]
        )
        if bool(config.get("use_absolute_filter", False)):
            fields.append("absolute_filter=on")
        if float(config.get("absolute_momentum_floor", 0.0)) > 0.0:
            fields.append(f"absolute_floor={config.get('absolute_momentum_floor')}")
    elif strategy_family == "volatility_breakout":
        fields.extend(
            [
                f"channel={config.get('channel_bars')}",
                f"atr={config.get('atr_lookback_bars')}x{config.get('atr_multiplier')}",
                f"breakout_buffer={config.get('breakout_buffer')}",
            ]
        )
        if float(config.get("breakout_score_floor", 0.0)) > 0.0:
            fields.append(f"breakout_score_floor={config.get('breakout_score_floor')}")
        if bool(config.get("use_trend_filter", False)):
            fields.append(f"trend_lookback={config.get('trend_lookback_bars')}")
    else:
        fields.extend(
            [
                f"lookback={config.get('lookback_bars')}",
                f"ranking={config.get('ranking_mode')}",
            ]
        )
        if float(config.get("min_signal_strength", 0.0)) > 0.0:
            fields.append(f"signal_floor={config.get('min_signal_strength')}")
        if float(config.get("min_cross_sectional_spread", 0.0)) > 0.0:
            fields.append(f"spread_floor={config.get('min_cross_sectional_spread')}")
        if bool(config.get("use_regime_filter", False)):
            fields.append(f"regime_threshold={config.get('regime_threshold')}")
    return ", ".join(fields)


def _promising_directions_from_memory(
    *,
    strategy_family: str,
    normalized_config: Mapping[str, Any],
    dead_zones: Sequence[Mapping[str, Any]],
    near_wins: Sequence[Mapping[str, Any]],
    decisions: Counter,
) -> list[str]:
    directions: list[str] = list(family_prompt_directions(strategy_family))
    dead_zone_traits = {str(item["trait"]) for item in dead_zones}
    if decisions.get("discard_screen", 0) >= 2:
        directions.append(
            "Most candidates still fail at screen. Prefer substantial but bounded changes, not tiny churn."
        )
    if "signal_floor=high" in dead_zone_traits:
        directions.append(
            "Avoid high min_signal_strength. If you add a signal floor, keep it modest."
        )
    if "spread_floor=high" in dead_zone_traits:
        directions.append(
            "Avoid aggressive min_cross_sectional_spread. Tighten selectivity more gently."
        )
    if "regime_filter=on" in dead_zone_traits:
        directions.append(
            "Recent regime-filter variants look fragile. Only enable use_regime_filter with mild thresholds."
        )
    if any("funding_penalty" in str(item.get("config_focus", "")) for item in near_wins):
        directions.append(
            "Near-wins suggest funding-aware candidates may help. Explore modest funding_penalty_weight."
        )
    if any("reversal_bias" in str(item.get("config_focus", "")) for item in near_wins):
        directions.append(
            "Near-wins suggest reversal-aware candidates may help. Explore moderate reversal_bias_weight."
        )
    if strategy_family == FAMILY_MOMENTUM and not normalized_config.get("use_regime_filter", False):
        directions.append(
            "Only turn on use_regime_filter if paired with a clear reason and conservative threshold."
        )
    if strategy_family == FAMILY_MEAN_REVERSION:
        if "ibs_threshold=low" in dead_zone_traits:
            directions.append(
                "Avoid very low ibs_threshold values unless you also keep reversion floors light. Very selective bar-position entries have stopped trading."
            )
        if "reversion_horizon=high" in dead_zone_traits:
            directions.append(
                "Avoid long reversion_horizon_bars when other filters are already active. Short-horizon reversal has looked more reliable."
            )
        if "reversion_floor=high" in dead_zone_traits:
            directions.append(
                "Keep reversion_strength_floor modest. High reversal floors have recently killed trade frequency."
            )
        if "trend_filter=on" in dead_zone_traits:
            directions.append(
                "Trend-filtered IBS reversion has looked fragile. Only use a mild trend filter with a moderate lookback."
            )
    elif strategy_family == FAMILY_EMA_TREND:
        if "signal_floor=high" in dead_zone_traits:
            directions.append(
                "Avoid high min_signal_strength. Dual-momentum candidates need a low or modest signal floor to stay active."
            )
        if "absolute_floor=high" in dead_zone_traits:
            directions.append(
                "Keep absolute_momentum_floor mild. Recent strict absolute filters have stopped dual-momentum candidates from trading."
            )
        if "dual_momentum_stack=wide" in dead_zone_traits:
            directions.append(
                "Avoid extremely wide horizon spacing when using confirmation filters. Prefer tighter but still ordered momentum horizons."
            )
        if "volatility_floor=high" in dead_zone_traits:
            directions.append(
                "Use volatility_floor conservatively. High floors have recently suppressed too many trend entries."
            )
    elif strategy_family == FAMILY_VOLATILITY_BREAKOUT:
        if "channel=long" in dead_zone_traits:
            directions.append(
                "Avoid very long breakout channels unless the breakout buffer stays near zero."
            )
        if "breakout_buffer=high" in dead_zone_traits:
            directions.append(
                "Keep breakout_buffer modest. Large buffers have repeatedly created no-trade breakout candidates."
            )
        if "atr_multiplier=high" in dead_zone_traits:
            directions.append(
                "Keep atr_multiplier moderate. Large ATR hurdles have recently made breakout entries too rare."
            )
        if "breakout_score_floor=high" in dead_zone_traits:
            directions.append(
                "Keep breakout_score_floor low or modest until breakout candidates show reliable activity again."
            )
    deduped: list[str] = []
    for direction in directions:
        if direction not in deduped:
            deduped.append(direction)
    return deduped[:5]


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
    strategy_family: str | None = None,
    model_name: str = "gpt-5-mini",
    max_mutations: int = 1,
    worktrees_root: str | Path = DEFAULT_WORKTREE_ROOT,
    recent_results_limit: int = 12,
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
        strategy_family=strategy_family,
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
    def combined_validator(candidate_text: str) -> tuple[bool, str]:
        is_valid, failure_reason = validate_train_candidate_text(candidate_text)
        if not is_valid:
            return is_valid, failure_reason
        return validate_train_candidate_semantics(
            candidate_text,
            current_train_text=context.current_train_text,
            strategy_family=context.strategy_family,
            symbol_count=context.symbol_count,
        )
    decisions = []
    for proposal in resolved_provider.generate(context=context, max_mutations=max_mutations):
        decisions.append(
            runner.apply_mutation_proposal_staged(
                campaign_path=campaign_path,
                proposal=proposal,
                validator=combined_validator,
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
