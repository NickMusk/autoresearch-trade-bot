from __future__ import annotations

import ast
import csv
import json
import os
import re
import subprocess
import urllib.error
import urllib.request
import uuid
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
        base_url: str = "https://api.openai.com/v1/responses",
    ) -> None:
        self.api_key = api_key or os.getenv("OPENAI_API_KEY", "")
        self.model_name = model_name
        self.timeout_seconds = timeout_seconds
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
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                raw_payload = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"openai_http_error:{exc.code}:{detail}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"openai_transport_error:{exc.reason}") from exc

        content = extract_response_text(raw_payload)
        if not content.strip():
            raise RuntimeError("openai_empty_response")
        return LLMCompletion(
            content=content,
            model_name=str(raw_payload.get("model", self.model_name)),
            prompt_id=str(raw_payload.get("id", "")),
            raw_response=raw_payload,
        )


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
        completion = self.client.generate(system_prompt=system_prompt, user_prompt=user_prompt)
        candidate_text = extract_python_candidate(completion.content)
        proposal = MutationProposal(
            label=f"{self.mutation_label_prefix}-{uuid.uuid4().hex[:8]}",
            candidate_text=candidate_text,
            commit_message=f"Mutate train.py via {completion.model_name}",
            provider_name="llm",
            model_name=completion.model_name,
            prompt_id=completion.prompt_id,
            notes="llm-mutation",
        )
        artifact_path = write_proposal_artifact(
            artifact_root=context.artifact_root,
            proposal=proposal,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            response_text=completion.content,
            raw_response=completion.raw_response,
        )
        return [
            MutationProposal(
                label=proposal.label,
                candidate_text=proposal.candidate_text,
                commit_message=proposal.commit_message,
                provider_name=proposal.provider_name,
                model_name=proposal.model_name,
                prompt_id=proposal.prompt_id,
                notes=proposal.notes,
                proposal_artifact_path=str(artifact_path),
            )
        ]


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
        recent_results=tuple(load_recent_results(results_path, limit=recent_results_limit)),
    )


def build_llm_mutation_prompt(
    *,
    context: MutationContext,
    max_mutations: int,
) -> tuple[str, str]:
    results_summary = "\n".join(
        [
            f"- mutation={row.get('mutation_label','')} score={row.get('research_score','')} "
            f"decision={row.get('decision','')} failure={row.get('failure_reason','')}"
            for row in context.recent_results
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
        ]
    )
    user_prompt = "\n\n".join(
        [
            f"Program:\n{context.program_text}",
            "Recent results:\n" + results_summary,
            "Current train.py:\n" + context.current_train_text,
            (
                "Mutation request:\n"
                f"Produce {max_mutations} candidate if possible, but if you can only return one, return the single best "
                "full train.py candidate. The output must be raw Python only."
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
        client=client or OpenAIResponsesClient(model_name=model_name)
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
