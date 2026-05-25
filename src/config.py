"""Shared paths and league constants — the single source of truth.

Imported across the data / viz / (future) model layers so filesystem locations
and the league's fixed parameters live in exactly one place.
"""
from pathlib import Path

# --- Paths (repo-relative) -----------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"          # cached raw sheet/API pulls (git-ignored)
PROCESSED_DIR = DATA_DIR / "processed"   # tidy/derived tables (git-ignored)
FIGURES_DIR = REPO_ROOT / "figures"      # generated figures (git-ignored)

# --- League constants ----------------------------------------------------------
TEAMS = ["Nate", "Seeb", "Silv", "Kerr", "Will", "Drew", "Couc", "Haft"]

# The contract sheet's active "yrs remain" is measured as of CURRENT_SEASON;
# UPCOMING_SEASON is the season the analysis centers on.
CURRENT_SEASON = 2025
UPCOMING_SEASON = 2026

# Salary cap per team per season (confirmed: CAP USED + CAP SPACE = 1500).
CAP_TOTAL = 1500

# Dead-cap rate applied to existing cuts. The league rule is moving to 0.50 but
# current sheet cuts are still charged at the legacy 0.20 (see docs/rules.md).
DEAD_CAP_RATE = 0.20
