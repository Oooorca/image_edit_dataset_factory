from __future__ import annotations

import asyncio
import logging
import os
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypeVar

import anyio
from fastapi import HTTPException

T = TypeVar("T")
LOGGER = logging.getLogger(__name__)


def env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    return int(raw)


def env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return default
    return float(raw)


def env_str(name: str, default: str) -> str:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw


@dataclass
class ServiceRuntime:
    cache_dir: Path
    preload: bool
    max_concurrency: int
    max_queue: int
    infer_timeout_sec: float


class RequestLimiter:
    def __init__(self, max_concurrency: int, max_queue: int) -> None:
        self._sem = asyncio.Semaphore(max_concurrency)
        self._pending = 0
        self._max_queue = max_queue
        self._lock = asyncio.Lock()

    async def _reserve(self) -> None:
        async with self._lock:
            if self._pending >= self._max_queue:
                raise HTTPException(
                    status_code=429,
                    detail={"code": "queue_full", "message": "request queue is full"},
                )
            self._pending += 1
        await self._sem.acquire()

    async def _release(self) -> None:
        self._sem.release()
        async with self._lock:
            self._pending = max(0, self._pending - 1)

    async def run(self, fn: Callable[[], T], timeout_sec: float) -> T:
        await self._reserve()
        try:
            with anyio.fail_after(timeout_sec):
                return await anyio.to_thread.run_sync(fn)
        except TimeoutError as exc:
            raise HTTPException(
                status_code=504,
                detail={"code": "infer_timeout", "message": "inference timeout"},
            ) from exc
        finally:
            await self._release()


class BackendState:
    def __init__(self, backend: Any) -> None:
        self.backend = backend
        self.last_error: str | None = None

    def is_ready(self) -> bool:
        if hasattr(self.backend, "_pipeline"):
            return self.backend._pipeline is not None
        return True

    def try_preload(self) -> None:
        if hasattr(self.backend, "_lazy_init"):
            try:
                self.backend._lazy_init()
            except Exception as exc:  # pragma: no cover
                self.last_error = str(exc)
                LOGGER.exception("service_preload_failed error=%s", exc)


def infer_runtime_name(backend: Any, default: str) -> str:
    return str(getattr(backend, "_runtime", default))
