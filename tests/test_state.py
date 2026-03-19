from __future__ import annotations

import io
import json
import unittest
from unittest.mock import patch
from urllib.error import HTTPError

from autoresearch_trade_bot.state import GitHubStatusPublisher


class GitHubStatusPublisherTests(unittest.TestCase):
    def test_publish_json_retries_on_conflict_and_refreshes_sha(self) -> None:
        publisher = GitHubStatusPublisher(
            token="token",
            repo="NickMusk/autoresearch-trade-bot",
            branch="render-state",
            base_path="status/test",
            api_base_url="https://api.github.test",
        )
        lookup_calls: list[str] = []
        publish_attempts = {"count": 0}

        def fake_lookup(path: str) -> str:
            lookup_calls.append(path)
            return f"sha-{len(lookup_calls)}"

        def fake_urlopen(request):  # type: ignore[no-untyped-def]
            publish_attempts["count"] += 1
            if publish_attempts["count"] == 1:
                raise HTTPError(
                    request.full_url,
                    409,
                    "Conflict",
                    hdrs=None,
                    fp=io.BytesIO(b"{}"),
                )
            return object()

        with patch.object(publisher, "_lookup_existing_sha", side_effect=fake_lookup):
            with patch.object(publisher, "_urlopen_with_retry", side_effect=fake_urlopen):
                with patch("autoresearch_trade_bot.state.time.sleep", return_value=None):
                    publisher.publish_json(
                        "latest_status.json",
                        {"phase": "active"},
                        message="Publish status",
                    )

        self.assertEqual(publish_attempts["count"], 2)
        self.assertEqual(lookup_calls, ["status/test/latest_status.json", "status/test/latest_status.json"])

    def test_publish_json_raises_conflict_after_retry_budget_exhausted(self) -> None:
        publisher = GitHubStatusPublisher(
            token="token",
            repo="NickMusk/autoresearch-trade-bot",
            branch="render-state",
            base_path="status/test",
            api_base_url="https://api.github.test",
        )

        def always_conflict(request):  # type: ignore[no-untyped-def]
            raise HTTPError(
                request.full_url,
                409,
                "Conflict",
                hdrs=None,
                fp=io.BytesIO(b"{}"),
            )

        with patch.object(publisher, "_lookup_existing_sha", return_value="sha-1"):
            with patch.object(publisher, "_urlopen_with_retry", side_effect=always_conflict):
                with patch("autoresearch_trade_bot.state.time.sleep", return_value=None):
                    with self.assertRaises(HTTPError):
                        publisher.publish_json(
                            "latest_status.json",
                            {"phase": "active"},
                            message="Publish status",
                        )


if __name__ == "__main__":
    unittest.main()
