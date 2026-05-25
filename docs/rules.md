# League Rules

> **Living document.** Records this dynasty league's contract & salary-cap rules
> so the analysis encodes them correctly. Each rule is tagged with how sure we
> are. Please correct anything wrong and fill in the remaining ❓ items.

**Last updated:** 2026-05-25

### Confidence legend
- ✅ **Confirmed** by the manager/commissioner.
- 🟡 **Inferred** from the spreadsheet/ESPN — please verify.
- ❓ **Open question** — not yet known.

---

## 1. Overview

- 8-team **dynasty** league with an offline, NFL-style salary cap and multi-year
  contracts. ✅
- **Active / upcoming season: 2026.** Analysis is centered on 2026. ✅
- Teams (sheet nicknames): `Nate, Seeb, Silv, Kerr, Will, Drew, Couc, Haft`. 🟡

## 2. Source of truth — the Google Sheet

Only **three tabs** matter; everything else (yearly rookie-draft tabs, free-agent
tabs, "defunct", "Sheet10") is ignored. ✅

| Tab | Holds |
| --- | --- |
| **Master Cap Sheet** | Active rosters, salaries, cap summary, IR/practice squad, amnesty, cuts/dead cap, draft picks. |
| **Trade Log** | One row per trade: `Trade Date \| Trade Details \| Team 1 \| Team 2`. |
| **Contract Extensions** | Per-team extensions negotiated to take effect the following season. |

Best read by fetching the workbook as **xlsx and indexing by tab name** (no `gid`s
needed).

**When the active roster and the Contract Extensions tab disagree on which team
owns a player, the Extensions tab is current.** ✅ The active-roster section lags
trades, and the Trade Log is incomplete (1 entry), so traded-but-not-extended
players may still appear under their former team on the active roster — a known
data-quality caveat until trades are reconciled.

## 3. Salary cap

- **Cap = 1500 (units) per team, the same every season.** ✅
- Tracked per team per season as CAP USED / DEAD CAP / CAP SPACE (used + space =
  1500). ✅
- The cap summary's **DEAD CAP column is a trade cap-adjustment**, *not* dead cap
  from cuts: e.g. Nate −30 = 30 cap space **gained** in a trade, Silv +30 = 30
  **given up**. ✅

## 4. Roster

ESPN roster = **32 slots**: 14 starters + 14 bench + 4 IR. ✅

- **Starting lineup (14):** QB, 2×RB, 2×WR, WR/TE, TE, RB/WR/TE (flex), LB, DL,
  DB, K, P, **HC (head coach)**. ✅ (the HC slot is why "Pats HC" etc. appear)
- **28 non-IR spots** = exactly the **28 veteran contract slots**. ✅
- **Plus 1 practice-squad player** (offline; a given player may be stashed there
  only once, ever). ✅
- **Veteran contract-slot pool** (by length): **5×1yr, 5×2yr, 7×3yr, 6×4yr,
  5×5yr** = 28. A team can only sign a veteran into a slot length that is still
  available. ✅
- IR has **4 slots**. ✅

## 5. Contracts

- Max length **5 years**. A signing occupies one slot of the chosen length from
  the available pool above. ✅
- Sheet fields: `yrs left @ acquisition` = the slot/length the player was signed
  into; `season @ acquisition` = the year signed; `yrs remain` = years left now.
  ✅ (IR-replacement players can create slot/length discrepancies. ✅)
- **Rookie deals are tracked separately** (Rookies section) and do **not** consume
  a veteran slot until the player is extended (or tagged). 🟡

## 6. Free agency & in-season signings

- **Pre-season:** an **offline auction**. Bidding funds available depend on your
  cap space; if you overspend, you have **24 hours** to make trades/cuts to get
  back under the cap. ✅
- **In-season:** players are acquired by **bidding**; after the waiver period the
  highest bid wins the player. ✅
- The winning bid sets the player's **salary**; the manager places them in an
  available **length slot** (1–5 yrs). 🟡 (length-choice mechanism)

## 7. Rookie contracts

- Rookies sign **3-year deals**, paid by draft slot: ✅
  - **Round 1, picks 1–8:** 109, 64, 50, 45, 34, 30, 26, 24
  - **Round 2:** 20 · **Round 3:** 15 · **Round 4:** 10
- **4th-year option:** after the rookie's **2nd season**, the manager may pick up
  the option → a **25% raise in years 3 AND 4** (this raised figure is the
  **"true salary" ≈ drafted ×1.25**), making it a 4-year deal. ✅
- **Extension eligibility:** after year 3, **only if** the option was picked up,
  the rookie may be put up for a contract extension; once extended, the deal's
  length then counts against the veteran slot pool. ✅
- Rookies may also be **franchise-tagged** (see §8). ✅

## 8. Franchise tag

- **One player per year**, who is **entering free agency** (contract expiring that
  season — e.g. a deal expiring after 2025 can be tagged for 2026). ✅
- **Tag-year-1 salary** = the **higher of**:
  - 120% of the player's current salary, or
  - the **average of the top-5 salaries at his position**. ✅
- **Tag-year-2** (tagged again the next year): **+20% on top of the year-1 tag
  salary**. ✅
- **Tag-year-3** (3rd straight year): the **higher of** **+44% on top of the
  year-2 tag salary** or the **average of the top-5 salaries at the
  highest-paid position**. ✅
- **No 4-year tags.** ✅
- A player on a tag is **never eligible for a contract extension.** ✅

## 9. Contract extensions

- A player is eligible to be extended **before the start of the season in which
  he enters the final year of his contract**. ✅
- The extension is recorded on the **Contract Extensions** tab and **takes effect
  the following season** (fields: player, salary, years, goes-into-effect,
  yrs-until-end, contract-starts). 🟡
- Tagged players cannot be extended (§8); rookies must have had the option picked
  up to be eligible (§7). ✅
- ❓ How are the extension **salary and length** decided (manager-set, bid, or a
  formula)?

## 10. Cuts / dead cap

- A cut charges a **% of the player's salary per remaining contract year**, over
  the window `[season cut, original end year]`. The sheet's **"salary owed" = the
  player's full salary** when cut. ✅
- **Rule going forward: 50%/yr** (cut a $20 player with 4 yrs left → $10/yr × 4).
  **But every cut currently on the sheet is still charged at the legacy 20%/yr** —
  the 50% change hasn't been applied yet. ✅ (`cap.py` uses `DEAD_CAP_RATE = 0.20`
  to match the sheet; use 0.50 for new/projected cuts.)
- The **"CAP HITS (… divided by 5)" rows are admin-imposed cap penalties** — also
  20% of the listed value per year, treated like a cut. ✅
- On the sheet, `yrs left (1-5)` is the penalty-window length; the leading number
  is the **years of penalty remaining** as of 2025. ✅
- Reconstructing CAP USED from active + rookies + tags + dead cap (@20%) + the
  trade adjustment matches the sheet to within a few units (2025 near-exact); the
  small 2026 residuals are mostly **IR players returning** (not yet added). 🟡

## 11. Amnesty

- **Cut one player with zero cap penalty.** ✅
- Usable **once every 3 seasons** (e.g. Najee Harris amnestied 2024 → next
  available 2027). An empty amnesty slot means it hasn't been used. ✅

## 12. Injured Reserve (IR)

- A player placed on IR is **out for the rest of the season**. ✅
- The IR player's contract is **removed from the current-season cap**, and a
  **replacement player's contract counts in its place**. ✅
- Up to **4 IR slots**. ✅
- **At season's end, the manager chooses one:** ✅
  1. **Retain the replacement** and **cut the IR player** (incurs the 50% cut
     penalty), or
  2. **Reactivate the IR player** and **release the replacement** with **no
     penalty**.

## 13. Trades

- Logged in the **Trade Log** tab (date, free-text details, the two teams). ✅
- Tradeable assets include **players, draft picks, and the rights to future
  contract extensions**. ✅
- ❓ Trade deadline / cap-matching rules (ESPN has a `trade_deadline` setting we
  can read).

## 14. Draft picks

- Future **rookie-draft picks (R1–R4)** are tracked per team per year on the
  Master Cap Sheet, including pick-swap conditions. 🟡
- ❓ Draft order determination and any other rookie-draft rules.

## 15. Scoring & lineup (ESPN)

- Highly **customized** scoring: IDP (LB/DL/DB), special teams (K/P, returns),
  **head-coach** scoring (win-margin & win/loss bonuses), and unusual passing
  (e.g. +0.35 per completion, −1 per attempt, −0.66 per incompletion, 6 pts per
  passing TD). ✅ (read from ESPN)
- The **full scoring table and lineup** are pulled directly from the ESPN API
  (`league.settings`), so they don't need to be transcribed here.

---

## Remaining open questions

1. How are **contract-extension** salary and length decided? (§9)
2. **Draft order** determination and any other rookie-draft rules. (§14)
3. **Trade deadline** / cap-matching specifics. (§13)
4. Confirm rookie deals don't consume a veteran slot until extended. (§5)
