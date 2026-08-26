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
FIG_CONTRACTS = FIGURES_DIR / "contracts"  # contract timelines
FIG_CAP = FIGURES_DIR / "cap"              # cap distribution / projection / by-position
FIG_VALUE = FIGURES_DIR / "value"          # value scatters (static + interactive)

# --- League constants ----------------------------------------------------------
# "Couc" franchise changed ownership -> renamed "Paik" (2026-08 offseason).
TEAMS = ["Nate", "Seeb", "Silv", "Kerr", "Will", "Drew", "Paik", "Haft"]

# The contract sheet's active "yrs remain" is measured as of CURRENT_SEASON;
# UPCOMING_SEASON is the season the analysis centers on.
# 2026-08: the master sheet rolled forward to a 2026 base (yrs-remain now as of
# 2026, cap summary 2026-2030), so CURRENT_SEASON advanced 2025 -> 2026. It now
# coincides with UPCOMING_SEASON (the auction season). NOTE: the separate
# CURRENT_SEASON constants in team.py / production.py stay 2025 -- those track
# the last *completed* NFL season for stats, which is still 2025.
CURRENT_SEASON = 2026
UPCOMING_SEASON = 2026

# Salary cap per team per season (confirmed: CAP USED + CAP SPACE = 1500).
CAP_TOTAL = 1500

# Dead-cap rate applied to existing cuts. The league rule is moving to 0.50 but
# current sheet cuts are still charged at the legacy 0.20 (see docs/rules.md).
DEAD_CAP_RATE = 0.20
