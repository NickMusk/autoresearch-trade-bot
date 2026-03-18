from __future__ import annotations

import html
import json
import os
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from .dashboard import build_dashboard_snapshot


class DashboardHandler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:  # noqa: N802
        if self.path == "/health":
            self._write_json(HTTPStatus.OK, {"status": "ok"})
            return
        if self.path == "/api/status":
            snapshot = build_dashboard_snapshot()
            self._write_json(HTTPStatus.OK, snapshot.__dict__)
            return
        if self.path == "/":
            snapshot = build_dashboard_snapshot()
            self._write_html(HTTPStatus.OK, render_dashboard(snapshot.__dict__))
            return
        self._write_json(HTTPStatus.NOT_FOUND, {"error": "not_found"})

    def log_message(self, format: str, *args) -> None:  # noqa: A003
        return

    def _write_json(self, status: HTTPStatus, payload) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _write_html(self, status: HTTPStatus, payload: str) -> None:
        body = payload.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def render_dashboard(snapshot: dict) -> str:
    blockers = "".join(
        f"<li>{html.escape(blocker)}</li>" for blocker in snapshot["research_blockers"]
    )
    milestones = "".join(
        f"<li>{html.escape(item)}</li>" for item in snapshot["next_milestones"]
    )
    multi_window = snapshot.get("multi_window_summary", {})
    multi_window_items = "".join(
        (
            f"<li><code>{html.escape(str(key))}</code>: {html.escape(str(value))}</li>"
            if key != "reports"
            else ""
        )
        for key, value in multi_window.items()
    )
    leaderboard_items = "".join(
        (
            "<li>"
            f"<strong>{html.escape(str(item['strategy_name']))}</strong> "
            f"(score {html.escape(str(round(float(item['score']), 4)))}, "
            f"accepted {html.escape('yes' if item['accepted'] else 'no')})"
            "</li>"
        )
        for item in snapshot.get("leaderboard", [])
    )
    metric_cards = "".join(
        (
            "<div class='metric'>"
            f"<span class='label'>{html.escape(str(key))}</span>"
            f"<strong>{html.escape(str(value))}</strong>"
            "</div>"
        )
        for key, value in snapshot["baseline_metrics"].items()
    )
    gate_cards = "".join(
        (
            "<div class='metric small'>"
            f"<span class='label'>{html.escape(str(key))}</span>"
            f"<strong>{html.escape(str(value))}</strong>"
            "</div>"
        )
        for key, value in snapshot["promotion_gate"].items()
    )

    accepted_label = "YES" if snapshot["accepted_for_paper"] else "NO"
    ready_label = "READY" if snapshot["research_rollout_ready"] else "NOT READY"
    cycle_completed_label = snapshot.get("latest_cycle_completed_at") or "n/a"
    processed_bar_label = snapshot.get("last_processed_bar") or "n/a"
    loop_state_label = snapshot.get("loop_state", "idle")
    acceptance_rate_label = snapshot.get("recent_acceptance_rate", 0.0)
    current_best_strategy_name = snapshot.get("current_best_strategy_name") or "n/a"
    latest_decision = snapshot.get("latest_decision") or {}
    latest_candidate_summary = snapshot.get("latest_candidate_summary") or {}
    latest_kept_summary = snapshot.get("latest_kept_summary") or {}
    latest_candidate_items = "".join(
        (
            f"<li><code>{html.escape(str(key))}</code>: {html.escape(str(value))}</li>"
            if key != "average_metrics"
            else ""
        )
        for key, value in latest_candidate_summary.items()
        if value not in ({}, [], "", None)
    )
    latest_kept_change_items = "".join(
        (
            f"<li><code>{html.escape(str(item.get('key')))}</code>: "
            f"{html.escape(str(item.get('before')))} -&gt; {html.escape(str(item.get('after')))}</li>"
        )
        for item in latest_kept_summary.get("config_diff", [])
    )

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Autoresearch Trade Bot</title>
  <style>
    :root {{
      --bg: #f4efe4;
      --card: #fff9f0;
      --ink: #1f2521;
      --muted: #596359;
      --accent: #0c7c59;
      --warn: #9d3c2c;
      --border: #d3c8b5;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: Georgia, "Times New Roman", serif;
      color: var(--ink);
      background:
        radial-gradient(circle at top left, rgba(12, 124, 89, 0.18), transparent 28%),
        linear-gradient(180deg, #f8f4eb 0%, var(--bg) 100%);
    }}
    main {{
      max-width: 1080px;
      margin: 0 auto;
      padding: 40px 20px 56px;
    }}
    h1, h2 {{ margin: 0 0 12px; }}
    p {{ color: var(--muted); line-height: 1.5; }}
    .hero {{
      padding: 28px;
      border: 1px solid var(--border);
      border-radius: 20px;
      background: rgba(255, 249, 240, 0.86);
      backdrop-filter: blur(6px);
      box-shadow: 0 16px 40px rgba(31, 37, 33, 0.08);
    }}
    .badge-row {{
      display: flex;
      gap: 12px;
      flex-wrap: wrap;
      margin: 18px 0 0;
    }}
    .badge {{
      padding: 8px 12px;
      border-radius: 999px;
      background: #ece4d6;
      color: var(--ink);
      font-size: 14px;
      letter-spacing: 0.04em;
    }}
    .grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
      gap: 16px;
      margin-top: 24px;
    }}
    .panel {{
      padding: 22px;
      border-radius: 18px;
      border: 1px solid var(--border);
      background: var(--card);
      box-shadow: 0 8px 24px rgba(31, 37, 33, 0.06);
    }}
    .metrics {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(120px, 1fr));
      gap: 12px;
    }}
    .metric {{
      padding: 14px;
      border-radius: 14px;
      background: #f7f0e2;
      border: 1px solid #e2d6c0;
      min-height: 90px;
    }}
    .metric.small {{
      min-height: 72px;
    }}
    .metric strong {{
      display: block;
      margin-top: 10px;
      font-size: 24px;
    }}
    .label {{
      color: var(--muted);
      text-transform: uppercase;
      font-size: 12px;
      letter-spacing: 0.08em;
    }}
    ul {{
      margin: 0;
      padding-left: 20px;
      color: var(--muted);
      line-height: 1.6;
    }}
    .warn strong {{ color: var(--warn); }}
    .footer {{
      margin-top: 24px;
      font-size: 14px;
      color: var(--muted);
    }}
    @media (max-width: 640px) {{
      main {{ padding: 24px 16px 40px; }}
      .hero, .panel {{ padding: 18px; }}
    }}
  </style>
</head>
<body>
  <main>
    <section class="hero">
      <h1>Autoresearch Trade Bot</h1>
      <p>{html.escape(snapshot["mission"])}</p>
      <div class="badge-row">
        <span class="badge">Phase: {html.escape(snapshot["phase"])}</span>
        <span class="badge">Paper Gate: {accepted_label}</span>
        <span class="badge">Research Rollout: {ready_label}</span>
        <span class="badge">Loop State: {html.escape(str(loop_state_label))}</span>
      </div>
    </section>

    <section class="grid">
      <article class="panel">
        <h2>Baseline Strategy</h2>
        <p>{html.escape(snapshot["baseline_strategy"])}</p>
        <ul>
          <li><code>current_best_strategy_name</code>: {html.escape(str(current_best_strategy_name))}</li>
        </ul>
      </article>
      <article class="panel">
        <h2>Promotion Gate</h2>
        <div class="metrics">{gate_cards}</div>
      </article>
    </section>

    <section class="grid">
      <article class="panel">
        <h2>Baseline Metrics</h2>
        <div class="metrics">{metric_cards}</div>
      </article>
      <article class="panel warn">
        <h2>Research Blockers</h2>
        <ul>{blockers}</ul>
      </article>
    </section>

    <section class="grid">
      <article class="panel">
        <h2>Next Milestones</h2>
        <ul>{milestones}</ul>
      </article>
      <article class="panel">
        <h2>Multi-Window</h2>
        <ul>{multi_window_items or "<li>No multi-window summary yet.</li>"}</ul>
      </article>
    </section>

    <section class="grid">
      <article class="panel">
        <h2>Worker Status</h2>
        <ul>
          <li><code>latest_cycle_completed_at</code>: {html.escape(str(cycle_completed_label))}</li>
          <li><code>last_processed_bar</code>: {html.escape(str(processed_bar_label))}</li>
          <li><code>recent_acceptance_rate</code>: {html.escape(str(acceptance_rate_label))}</li>
          <li><code>consecutive_failures</code>: {html.escape(str(snapshot.get("consecutive_failures", 0)))}</li>
          <li><code>latest_decision</code>: {html.escape(str(latest_decision.get("decision", "n/a")))}</li>
          <li><code>latest_candidate_score</code>: {html.escape(str(latest_decision.get("candidate_score", "n/a")))}</li>
          <li><code>baseline_score</code>: {html.escape(str(latest_decision.get("baseline_score", "n/a")))}</li>
        </ul>
      </article>
      <article class="panel">
        <h2>Leaderboard</h2>
        <ul>{leaderboard_items or "<li>No persisted leaderboard yet.</li>"}</ul>
      </article>
      <article class="panel">
        <h2>Latest Candidate</h2>
        <ul>{latest_candidate_items or "<li>No latest candidate summary yet.</li>"}</ul>
      </article>
      <article class="panel">
        <h2>Latest Kept Change</h2>
        <ul>
          <li><code>strategy_name</code>: {html.escape(str(latest_kept_summary.get("strategy_name") or "n/a"))}</li>
          <li><code>baseline_strategy_name</code>: {html.escape(str(latest_kept_summary.get("baseline_strategy_name") or "n/a"))}</li>
          <li><code>score_delta</code>: {html.escape(str(latest_kept_summary.get("score_delta") or "n/a"))}</li>
        </ul>
        <ul>{latest_kept_change_items or "<li>No kept config diff published yet.</li>"}</ul>
      </article>
      <article class="panel">
        <h2>Endpoints</h2>
        <ul>
          <li><code>/</code> dashboard</li>
          <li><code>/health</code> health probe</li>
          <li><code>/api/status</code> JSON snapshot</li>
        </ul>
      </article>
    </section>

    <p class="footer">This dashboard is intentionally read-only in v1. It exposes the current research kernel state, not a live trading control plane.</p>
  </main>
</body>
</html>"""


def main() -> None:
    host = os.environ.get("AUTORESEARCH_TRADE_BOT_HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", os.environ.get("AUTORESEARCH_TRADE_BOT_PORT", "10000")))
    server = ThreadingHTTPServer((host, port), DashboardHandler)
    print(f"Serving autoresearch dashboard on http://{host}:{port}")
    server.serve_forever()


if __name__ == "__main__":
    main()
