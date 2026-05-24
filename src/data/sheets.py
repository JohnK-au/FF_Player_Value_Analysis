"""Read the league's contract Google Sheet as CSV — no authentication required.

This is "Option A": the sheet is shared as "Anyone with the link can view", so
Google's CSV-export endpoint returns the data without credentials. The
spreadsheet ID is read from the environment (``.env``) so it is never committed
to the public repository.
"""
from __future__ import annotations

import os
from io import StringIO
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv

load_dotenv()

_EXPORT_URL = (
    "https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=csv&gid={gid}"
)

# Repo root is two levels up from this file (src/data/sheets.py).
RAW_DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"


def _sheet_id(sheet_id: str | None = None) -> str:
    sheet_id = sheet_id or os.environ.get("CONTRACTS_SHEET_ID")
    if not sheet_id:
        raise RuntimeError(
            "CONTRACTS_SHEET_ID is not set. Copy .env.example to .env and fill it in."
        )
    return sheet_id


def fetch_tab(
    gid: int | str | None = None, *, sheet_id: str | None = None
) -> pd.DataFrame:
    """Fetch one tab (by ``gid``) of the contract sheet as a raw DataFrame.

    The sheet uses a visual, multi-team layout rather than a single tidy table,
    so the frame is returned un-parsed (no header row). Cleaning the layout into
    tidy per-player records happens downstream.
    """
    if gid is None:
        gid = os.environ.get("CONTRACTS_DEFAULT_GID", 0)
    url = _EXPORT_URL.format(sheet_id=_sheet_id(sheet_id), gid=gid)
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    return pd.read_csv(StringIO(resp.text), header=None)


def save_tab(gid: int | str | None = None, *, sheet_id: str | None = None) -> Path:
    """Fetch a tab and cache it under ``data/raw/``. Returns the saved path."""
    resolved_gid = gid if gid is not None else os.environ.get("CONTRACTS_DEFAULT_GID", 0)
    df = fetch_tab(resolved_gid, sheet_id=sheet_id)
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    out = RAW_DATA_DIR / f"contracts_gid{resolved_gid}.csv"
    df.to_csv(out, index=False, header=False)
    return out


if __name__ == "__main__":
    path = save_tab()
    print(f"Saved raw contract data to {path}")
