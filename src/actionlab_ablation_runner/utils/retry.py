from __future__ import annotations

from collections.abc import Callable
from typing import Any, TypeVar

from tenacity import retry, stop_after_attempt, wait_exponential

T = TypeVar("T")


def with_backoff(func: Callable[..., T]) -> Callable[..., T]:
    decorated = retry(
        wait=wait_exponential(multiplier=1, min=1, max=8), stop=stop_after_attempt(3)
    )(func)
    return decorated


def redact_secrets(payload: dict[str, Any]) -> dict[str, Any]:
    redacted: dict[str, Any] = {}
    for key, value in payload.items():
        if any(token in key.lower() for token in ("key", "token", "secret", "password")):
            redacted[key] = "***REDACTED***"
        else:
            redacted[key] = value
    return redacted
