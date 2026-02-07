from __future__ import annotations

import json
import time
from typing import Any

import httpx


class RetryingJsonHttpClient:
    def __init__(
        self,
        endpoint: str,
        timeout_sec: float,
        max_retries: int,
        backoff_sec: float,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self.endpoint = endpoint.rstrip("/")
        self.timeout_sec = timeout_sec
        self.max_retries = max_retries
        self.backoff_sec = backoff_sec
        self.transport = transport

    def post_json(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        url = f"{self.endpoint}{path}"
        errors: list[str] = []

        for attempt in range(self.max_retries + 1):
            try:
                with httpx.Client(timeout=self.timeout_sec, transport=self.transport) as client:
                    response = client.post(url, json=payload)
                if response.status_code >= 500 and attempt < self.max_retries:
                    errors.append(f"status={response.status_code}")
                    time.sleep(self.backoff_sec * (attempt + 1))
                    continue
                response.raise_for_status()
                data = response.json()
                if not isinstance(data, dict):
                    raise RuntimeError(f"invalid JSON response type: {type(data)!r}")
                return data
            except Exception as exc:  # pragma: no cover
                errors.append(str(exc))
                if attempt < self.max_retries:
                    time.sleep(self.backoff_sec * (attempt + 1))
                    continue
                break

        raise RuntimeError(
            "service request failed "
            f"url={url}, payload_keys={list(payload.keys())}, "
            f"errors={json.dumps(errors, ensure_ascii=False)}"
        )
