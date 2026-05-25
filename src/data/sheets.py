"""Read the league's contract Google Sheet — no authentication required.

This is "Option A": the sheet is shared as "Anyone with the link can view", so
Google's export endpoints return the data without credentials. The spreadsheet
ID is read from the environment (``.env``) so it is never committed to the
public repository.

The workbook has many tabs but only three matter — read them **by name** via the
xlsx export (``read_tab`` / ``cache_tabs``), which avoids needing per-tab gids.
``fetch_tab`` (CSV export by gid) is kept for ad-hoc/legacy use.
"""
from __future__ import annotations

import os
from io import BytesIO, StringIO
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv

from ..config import RAW_DATA_DIR

load_dotenv()

_CSV_URL = (
    "https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=csv&gid={gid}"
)
_XLSX_URL = "https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=xlsx"

# The only tabs that matter: friendly key -> exact tab name in the workbook.
TABS = {
    "master_cap": "Master Cap Sheet",
    "trade_log": "Trade Log",
    "contract_extensions": "Contract Extensions",
}


def _sheet_id(sheet_id: str | None = None) -> str:
    sheet_id = sheet_id or os.environ.get("CONTRACTS_SHEET_ID")
    if not sheet_id:
        raise RuntimeError(
            "CONTRACTS_SHEET_ID is not set. Copy .env.example to .env and fill it in."
        )
    return sheet_id


# --- Workbook-by-name access (preferred) ---------------------------------------

def fetch_workbook(sheet_id: str | None = None) -> bytes:
    """Download the entire workbook as xlsx bytes (all tabs in one request)."""
    url = _XLSX_URL.format(sheet_id=_sheet_id(sheet_id))
    resp = requests.get(url, timeout=60)
    resp.raise_for_status()
    return resp.content


def read_tab(
    tab: str,
    *,
    sheet_id: str | None = None,
    workbook_bytes: bytes | None = None,
) -> pd.DataFrame:
    """Read one tab (by friendly key or exact name) as a raw, header-less frame.

    Pass ``workbook_bytes`` to avoid re-downloading when reading several tabs.
    """
    name = TABS.get(tab, tab)
    if workbook_bytes is None:
        workbook_bytes = fetch_workbook(sheet_id)
    return pd.read_excel(
        BytesIO(workbook_bytes), sheet_name=name, header=None, dtype=str,
        engine="openpyxl",
    )


def cached_csv_path(key: str) -> Path:
    return RAW_DATA_DIR / f"contracts_{key}.csv"


def cache_tabs(
    keys: list[str] | None = None, *, sheet_id: str | None = None
) -> dict[str, Path]:
    """Download the workbook once and cache each relevant tab as a CSV."""
    keys = keys or list(TABS)
    workbook = fetch_workbook(sheet_id)
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    saved: dict[str, Path] = {}
    for key in keys:
        df = read_tab(key, workbook_bytes=workbook)
        out = cached_csv_path(key)
        df.to_csv(out, index=False, header=False)
        saved[key] = out
    return saved


def load_tab(
    key: str, *, use_cache: bool = True, sheet_id: str | None = None
) -> pd.DataFrame:
    """Load a tab as strings, preferring the cached CSV if present."""
    path = cached_csv_path(key)
    if use_cache and path.exists():
        return pd.read_csv(path, header=None, dtype=str)
    return read_tab(key, sheet_id=sheet_id)


# --- CSV-by-gid access (legacy / ad-hoc) ---------------------------------------

def fetch_tab(
    gid: int | str | None = None, *, sheet_id: str | None = None
) -> pd.DataFrame:
    """Fetch one tab by ``gid`` via CSV export (header-less raw frame)."""
    if gid is None:
        gid = os.environ.get("CONTRACTS_DEFAULT_GID", 0)
    url = _CSV_URL.format(sheet_id=_sheet_id(sheet_id), gid=gid)
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    return pd.read_csv(StringIO(resp.text), header=None)


if __name__ == "__main__":
    for key, path in cache_tabs().items():
        print(f"Cached {TABS[key]!r} -> {path}")
