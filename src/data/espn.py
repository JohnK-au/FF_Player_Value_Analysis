"""Read ESPN fantasy league data via authenticated API access.

The league is private, so this requires the user's ESPN auth cookies
(``ESPN_S2``, ``ESPN_SWID``) plus the league id — all read from the git-ignored
``.env``. Built on the ``espn-api`` library (https://github.com/cwendt94/espn-api),
which wraps ESPN's JSON API and returns structured League/Team/Player objects.
"""
from __future__ import annotations

import os

from dotenv import load_dotenv
from espn_api.football import League

load_dotenv()


def _require(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(
            f"{name} is not set. Copy .env.example to .env and fill it in."
        )
    return value


def get_league(year: int | None = None) -> League:
    """Return an authenticated :class:`League` for the given season.

    Defaults to ``ESPN_SEASON`` from the environment, or 2025 (the last
    completed season with full data) if unset.
    """
    if year is None:
        year = int(os.environ.get("ESPN_SEASON", 2025))
    return League(
        league_id=int(_require("ESPN_LEAGUE_ID")),
        year=year,
        espn_s2=_require("ESPN_S2"),
        swid=_require("ESPN_SWID"),
    )


if __name__ == "__main__":
    league = get_league()
    print(f"{league.year} season — {len(league.teams)} teams")
    for team in league.teams:
        print(f"  {team.team_id}: {team.team_name} ({team.wins}-{team.losses})")
