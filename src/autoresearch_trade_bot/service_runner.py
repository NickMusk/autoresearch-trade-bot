from __future__ import annotations

import atexit
import os
import signal
import subprocess
import sys
from typing import Sequence

from . import llm_worker


def _parse_bool(value: str | None, *, default: bool) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _terminate_process(process: subprocess.Popen[bytes] | None) -> None:
    if process is None or process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        process.kill()


def _spawn_history_process(mode: str) -> subprocess.Popen[bytes]:
    env = os.environ.copy()
    env["AUTORESEARCH_HISTORY_REFRESH_MODE"] = mode
    return subprocess.Popen(
        [sys.executable, "-m", "autoresearch_trade_bot.history_refresh"],
        env=env,
    )


def run_llm_service() -> None:
    enable_refresher = _parse_bool(
        os.environ.get("AUTORESEARCH_ENABLE_HISTORY_REFRESHER"),
        default=True,
    )
    daemon_process: subprocess.Popen[bytes] | None = None
    if enable_refresher:
        bootstrap_process = _spawn_history_process("bootstrap")
        bootstrap_code = bootstrap_process.wait()
        if bootstrap_code != 0:
            raise SystemExit(bootstrap_code)
        daemon_process = _spawn_history_process("daemon")
        atexit.register(_terminate_process, daemon_process)

        def _handle_signal(signum: int, _frame) -> None:
            _terminate_process(daemon_process)
            raise SystemExit(0)

        signal.signal(signal.SIGTERM, _handle_signal)
        signal.signal(signal.SIGINT, _handle_signal)

    llm_worker.main()


def main(argv: Sequence[str] | None = None) -> None:
    args = list(argv if argv is not None else sys.argv[1:])
    target = args[0] if args else "llm-worker"
    if target != "llm-worker":
        raise SystemExit(f"unsupported service runner target: {target}")
    run_llm_service()


if __name__ == "__main__":
    main()
