# Repo Agent Contract

This file is the repository-level operating contract for human and automated agents.

It is intentionally static. Do not use it as a live status log, incident tracker, or experiment journal.

## File Roles

- `AGENTS.md`
  - Purpose: repository workflow rules, ownership boundaries, memory contract, and where truth lives.
  - Update when: operating model or repo structure changes.
  - Do not store: current blockers, temporary incidents, per-cycle findings, or deploy-specific notes.

- `program.md`
  - Purpose: research mission and immutable program rules for the autoresearch loop.
  - Source of truth for: what the loop is trying to optimize and what is editable.

- `docs/repo-map.md`
  - Purpose: compact map of major modules, boundaries, and where key logic lives.
  - Source of truth for: navigation and architectural orientation.

- `docs/research-memory.md`
  - Purpose: durable research learnings that should survive across many runs and future agent sessions.
  - Update when: a finding has repeated enough to be considered stable guidance.
  - Do not store: one-off failures, transient Render incidents, or stale production status.

- Runtime state and artifacts under worker `state_root` and `artifact_root`
  - Purpose: machine-updated operational and experiment memory.
  - Source of truth for: recent results, candidate registry, snapshots, validation artifacts, and prompt memory payloads.
  - Do not mirror this data into markdown docs unless it has become a stable lesson.

## Memory Contract

Use the smallest memory layer that matches the problem.

1. Static repository guidance belongs in `AGENTS.md`.
2. Stable research lessons belong in `docs/research-memory.md`.
3. Current operational truth belongs in status snapshots and runtime state, not repo docs.
4. Prompt-facing machine memory belongs in structured artifacts, not hand-written markdown.

If the same fact appears in multiple files, one file must be declared the source of truth and the others must link to it rather than rephrase it.

## Source Of Truth Rules

- Research mission and immutable loop rules: `program.md`
- Current production status: published status snapshots and dashboard API
- Runtime configuration defaults: `src/autoresearch_trade_bot/config.py`
- Production deployment intent: `render.yaml`
- Prompt memory and recent experiment summaries: generated artifacts from `src/autoresearch_trade_bot/mutations.py`

## Editing Guidance

- Prefer explicit changes over broad documentation rewrites.
- Keep memory files concise and curated.
- Do not add a new memory file unless an existing file has a clearly wrong role.
- When adding research learnings, write them as reusable guidance, not as a timeline entry.
- When a lesson stops being true, delete or rewrite it. Do not accumulate contradictory notes.

## Research Workflow Notes

- The editable research surface remains `train.py`.
- Infrastructure, simulator, risk, deployment, and dashboard code are not part of the strategy mutation surface.
- Changes to prompts, mutation bounds, validation topology, or rollout semantics are infrastructure changes and should be treated explicitly as such.

## Ownership Expectations

- `AGENTS.md`: repo maintainers
- `program.md`: research program owner
- `docs/repo-map.md`: engineers changing module boundaries
- `docs/research-memory.md`: researchers or agents only when a finding is durable
- Runtime artifacts: the code that produces them
