"""Structured, Flask-independent errors for optional on-demand science features."""

from __future__ import annotations

from typing import Any, Dict, Optional


class ComputationError(Exception):
    """A computation failure carrying stable API metadata for the frontend."""

    def __init__(
        self,
        code: str,
        message: str,
        *,
        status: int = 400,
        details: Optional[Dict[str, Any]] = None,
        retryable: bool = False,
        user_fixable: bool = True,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.status = status
        self.details = details or {}
        self.retryable = retryable
        self.user_fixable = user_fixable

    def to_dict(self) -> Dict[str, Any]:
        return {
            "error": self.message,
            "code": self.code,
            "details": self.details,
            "retryable": self.retryable,
            "user_fixable": self.user_fixable,
        }

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return (
            f"{type(self).__name__}({self.code!r}, {self.message!r}, "
            f"status={self.status})"
        )


class IpdError(ComputationError):
    """A structured failure from IPD capability, input, history, or computation."""


class DeltaMtpError(ComputationError):
    """A structured failure from Delta-MTP capability, input, or computation."""
