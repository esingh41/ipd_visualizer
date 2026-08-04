"""Structured API errors for the on-demand IPD workflow.

Its own module so ``dataset_store`` and ``ipd_compute`` can both raise these without
importing each other, and so nothing here needs Flask.

The existing routes answer with a bare ``{"error": "..."}``; this keeps that key so
``callJson()`` in the frontend still finds a message, and adds the fields the sidebar
needs to decide what to *show*: which columns are missing, whether the user can fix it,
and whether a retry is worth offering.
"""

from __future__ import annotations

from typing import Any, Dict, Optional


class IpdError(Exception):
    """An IPD failure the frontend can render as something more useful than "500".

    ``code`` is the stable machine-readable discriminator; ``message`` is the sentence a
    user reads. ``retryable`` says whether pressing the button again could plausibly work
    (a transient write conflict, yes; a missing column, no), and ``user_fixable`` whether
    the fix is theirs to make (install apnet_pt, supply the column) rather than a bug.
    """

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
        return f"IpdError({self.code!r}, {self.message!r}, status={self.status})"
