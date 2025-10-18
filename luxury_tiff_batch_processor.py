# file: luxury_tiff_batch_processor.py
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Backward-compatible shim for the legacy `luxury_tiff_batch_processor.py`.

Preserves README examples like:
    python luxury_tiff_batch_processor.py ...

The real CLI lives at `luxury_tiff_batch_processor.cli:main`.
"""

from __future__ import annotations

from typing import Any, Callable, Optional
import os
import sys
import traceback

__all__ = ["main"]

# Attempt to import the real entry point once at module import.
_REAL_MAIN: Optional[Callable[..., Any]] = None
_IMPORT_ERR: Optional[BaseException] = None
try:
    # mypy: function object is fine to assign to Optional[Callable]
    from luxury_tiff_batch_processor.cli import main as _REAL_MAIN  # type: ignore[assignment]
except Exception as _exc:  # avoid swallowing KeyboardInterrupt/SystemExit
    _IMPORT_ERR = _exc


def _print_exception_compat(exc: BaseException) -> None:
    """Python 3.8–3.12 compatible exception printer (why: consistent CI logs)."""
    traceback.print_exception(type(exc), exc, exc.__traceback__, file=sys.stderr)


def _print_deprecation_note() -> None:
    # Why: guide users toward supported entry points without breaking workflows.
    if os.environ.get("LUXURY_TIFF_SHIM_SILENCE"):
        return
    msg = (
        "[luxury_tiff_batch_processor] Note: this shim is deprecated.\n"
        "Use one of the supported forms instead:\n"
        "  - python -m luxury_tiff_batch_processor\n"
        "  - luxury-tiff-batch  (installed console script)"
    )
    print(msg, file=sys.stderr)


def main(*args: Any, **kwargs: Any) -> int:
    """
    Backward-compatible entry that forwards to `luxury_tiff_batch_processor.cli:main`.

    Returns a normalized integer exit code.
    """
    _print_deprecation_note()

    if _REAL_MAIN is None:
        print(
            "[luxury_tiff_batch_processor] ERROR: could not import "
            "`luxury_tiff_batch_processor.cli:main`. Is the package installed?\n"
            "Try: pip install -e .   (from repo root)  or  pip install luxury-tiff-batch\n",
            file=sys.stderr,
        )
        if _IMPORT_ERR is not None:
            print("Original import error:\n", file=sys.stderr)
            _print_exception_compat(_IMPORT_ERR)
        return 1

    try:
        result = _REAL_MAIN(*args, **kwargs)
    except SystemExit as exc:
        # Normalize SystemExit to exit code semantics:
        # - int code passes through
        # - None means success (0)
        # - non-int (e.g., message string) means failure (1)
        code = exc.code
        if isinstance(code, int):
            return code
        if code is None:
            return 0
        return 1
    except BaseException as exc:
        print(f"[luxury_tiff_batch_processor] Unhandled error: {exc}", file=sys.stderr)
        traceback.print_exc()
        return 1

    # Some CLIs return None; treat as success.
    return int(result) if isinstance(result, int) else 0


if __name__ == "__main__":  # pragma: no cover (exercised via integration)
    raise SystemExit(main())