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
      </div>
    </section>

    <section class="grid">
      <article class="panel">
        <h2>Baseline Strategy</h2>
        <p>{html.escape(snapshot["baseline_strategy"])}</p>
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
