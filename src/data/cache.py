"""Tiny on-disk cache for finalized-season tables.

Completed seasons are immutable, so we pull/compute them once and store them as
Parquet (typed, compact, fast). Subsequent reads come straight off disk; pass
``refresh=True`` to re-fetch (use it for the in-progress season, which still
changes). Cache files live under ``data/`` and are git-ignored.
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable

import pandas as pd


def cached_parquet(
    path: Path, build: Callable[[], pd.DataFrame], refresh: bool = False
) -> pd.DataFrame:
    """Return the cached table at ``path``, building + saving it if missing."""
    if path.exists() and not refresh:
        return pd.read_parquet(path)
    df = build()
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    return df
