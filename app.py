# -*- coding: utf-8 -*-
"""
Datumly — Rolling 6-GW Transfer Planner (Streamlit)
========================================================================
v5.4 fixes:
  - Non-starters no longer over-inflated: xPts now multiplied by
    start_prob (team-relative minutes share) so rotation/bench players
    are properly discounted.
  - ep_next blend reduced from 30% → 15% to stop dragging regular
    starters down to 3-4 pts.
  - Captain projections use the same start_prob method (was previously
    using a generous 0.5-clipped mins_factor).

Corrected scoring per official 2025/26 FPL rules:
  - CS = 4pts for DEF/GKP (was incorrectly 6 in v5.1)
  - CS = 1pt for MID
  - DC = 2pts (probability × 2, not doubled again)
  - Captain projection uses cs × 4.0 for DEF/GKP (fixed from cs × 6.0)

Data sources (all free, no auth):
  - FPL API    → squad, prices, form, ep_next, GW history, DC stats
  - Understat  → xG, xA, npxG, shots, key passes per player
  - FBref      → xG, xA, npxG, shots, key passes (StatsBomb/Opta model)
                  Used as cross-validation against Understat's independent model.

Deploy: streamlit run app.py
"""

import streamlit as st
import requests
import pandas as pd
import numpy as np
import unicodedata
import re
import json
import time
import warnings
from io import StringIO
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor, as_completed

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────
#  CONSTANTS
# ─────────────────────────────────────────────────────────
MAX_TRANSFER_BANK = 5
BASE = "https://fantasy.premierleague.com/api"
DC_THREADS = 25

TEAM_MAP = {
    "Manchester City":         "Man City",
    "Manchester United":       "Man Utd",
    "Tottenham":               "Spurs",
    "Nottingham Forest":       "Nott'm Forest",
    "Newcastle United":        "Newcastle",
    "Wolverhampton Wanderers": "Wolves",
    "Leeds United":            "Leeds",
    "Sunderland":              "Sunderland",
}


# ══════════════════════════════════════════════════════════════════════
#  1.  FPL API HELPERS
# ══════════════════════════════════════════════════════════════════════

def fpl_get(endpoint):
    r = requests.get(f"{BASE}/{endpoint}", timeout=20)
    r.raise_for_status()
    return r.json()

def get_bootstrap():
    return fpl_get("bootstrap-static/")

def get_my_team(manager_id, boot):
    now    = datetime.now(timezone.utc)
    events = boot["events"]

    next_gw = None
    for event in events:
        deadline = datetime.fromisoformat(
            event["deadline_time"].replace("Z", "+00:00")
        )
        if deadline > now:
            next_gw = event["id"]
            break

    if next_gw is None:
        active_gw = events[-1]["id"]
        next_gw   = active_gw
    elif next_gw == 1:
        active_gw = 1
    else:
        active_gw = next_gw - 1

    picks = fpl_get(f"entry/{manager_id}/event/{active_gw}/picks/")
    return picks, active_gw, next_gw


def get_free_transfers(manager_id, active_gw):
    try:
        history    = fpl_get(f"entry/{manager_id}/history/")
        gw_history = history.get("current", [])
        transfers_made = {h["event"]: h["event_transfers"] for h in gw_history}

        bank = 1
        for gw in range(1, active_gw + 1):
            made = transfers_made.get(gw, 0)
            bank = min(bank + 1, MAX_TRANSFER_BANK)
            bank = max(bank - made, 0)

        free_transfers = max(1, min(bank, MAX_TRANSFER_BANK))

        chips      = history.get("chips", [])
        chips_used = [c["name"] for c in chips]
        chips_available = []
        if "wildcard" not in chips_used:
            chips_available.append("Wildcard")
        if "freehit" not in chips_used:
            chips_available.append("Free Hit")
        if "bboost" not in chips_used:
            chips_available.append("Bench Boost")
        if "3xc" not in chips_used:
            chips_available.append("Triple Captain")

        return free_transfers, chips_available
    except Exception:
        return 1, []


def compute_sell_prices(squad_ids, players_df, manager_id):
    """
    Calculate actual FPL sell prices for each player in the squad.

    FPL sell price formula:
        profit = now_cost - purchase_price
        sell_price = purchase_price + floor(profit / 2)

    Purchase price is determined by:
      - Transfer history: the element_in_cost of the most recent transfer IN
      - GW1 picks: now_cost - cost_change_start (i.e. start-of-season price)

    Returns dict: {fpl_id: sell_price_in_millions}
    """
    # Get transfer history (public endpoint, no auth needed)
    try:
        transfers = fpl_get(f"entry/{manager_id}/transfers/")
    except Exception:
        transfers = []

    # Build map: player_id -> most recent purchase price (in tenths, raw API)
    # Transfers are returned newest-first by the API
    purchase_map = {}  # fpl_id -> element_in_cost (tenths of £)
    for t in transfers:
        pid = int(t.get("element_in", 0))
        if pid in squad_ids and pid not in purchase_map:
            purchase_map[pid] = int(t.get("element_in_cost", 0))

    # For players not in transfer history (owned since GW1), calculate from
    # cost_change_start: start_price = now_cost - cost_change_start
    sell_prices = {}
    for pid in squad_ids:
        row = players_df[players_df["fpl_id"] == pid]
        if row.empty:
            continue
        row = row.iloc[0]
        now_cost_raw = int(row.get("now_cost", 0))

        if pid in purchase_map:
            purchase_price = purchase_map[pid]
        else:
            # Owned since start of season
            ccs = int(row.get("cost_change_start", 0))
            purchase_price = now_cost_raw - ccs

        # FPL formula: you keep half profit, rounded down
        profit = now_cost_raw - purchase_price
        if profit > 0:
            sell_raw = purchase_price + (profit // 2)
        else:
            # Price dropped: you lose the full drop
            sell_raw = now_cost_raw

        sell_prices[pid] = sell_raw / 10  # convert to £m

    return sell_prices


def get_fixtures():
    return fpl_get("fixtures/")

def build_fpl_table(boot):
    players = pd.DataFrame(boot["elements"]).copy()
    teams   = pd.DataFrame(boot["teams"])[["id","name","short_name"]].copy()
    teams   = teams.rename(columns={
        "id": "team_id", "name": "team_name", "short_name": "team_short"
    })

    players = players.rename(columns={"team": "team_ref"})
    players = players.merge(teams, left_on="team_ref", right_on="team_id", how="left")

    pos_map = {1:"GKP", 2:"DEF", 3:"MID", 4:"FWD"}
    players["position"] = players["element_type"].map(pos_map)
    players["price"]    = players["now_cost"] / 10
    players["full_name"]= players["first_name"] + " " + players["second_name"]
    players["fpl_id"]   = players["id"]

    num_cols = [
        "total_points","form","minutes","goals_scored","assists",
        "clean_sheets","bonus","ep_next","ep_this","bps",
        "selected_by_percent","saves","penalties_saved",
        "yellow_cards","red_cards","goals_conceded",
        "expected_goals","expected_assists",
        "expected_goal_involvements","expected_goals_conceded",
        "threat","creativity","influence","ict_index",
        "transfers_in_event","transfers_out_event",
        "chance_of_playing_next_round","chance_of_playing_this_round",
    ]
    for c in num_cols:
        if c in players.columns:
            players[c] = pd.to_numeric(players[c], errors="coerce").fillna(0)

    keep = ["fpl_id","full_name","web_name","team_name","team_short","team_id",
            "position","price","now_cost","cost_change_start","status"] + \
           [c for c in num_cols if c in players.columns]

    players = players.loc[:, ~players.columns.duplicated()]
    keep    = [c for c in keep if c in players.columns]
    return players[keep].reset_index(drop=True).copy()


# ══════════════════════════════════════════════════════════════════════
#  2.  DEFENSIVE CONTRIBUTION STATS (2025/26 rules)
# ══════════════════════════════════════════════════════════════════════
#  DEF:     2 pts if CBIT >= 10 in a match
#  MID/FWD: 2 pts if CBIRT >= 12 in a match (CBIT + ball recoveries)
#  GKP:     DC does not apply.
#  Cap:     Hard cap of 2 pts per match.
# ══════════════════════════════════════════════════════════════════════

def _fetch_one_dc(pid, gw_start, position_type):
    if position_type == 1:
        return {"fpl_id": pid}
    try:
        data    = fpl_get(f"element-summary/{pid}/")
        history = data.get("history", [])
        recent  = [
            h for h in history
            if h.get("round", 0) >= gw_start and h.get("minutes", 0) > 0
        ]
        if not recent:
            return {"fpl_id": pid}

        dc_pts_earned = 0.0
        total_mins    = 0.0
        cbit_total    = 0.0
        gw_count      = 0

        for gw in recent:
            total_mins    += float(gw.get("minutes", 0) or 0)
            tackles        = float(gw.get("tackles",       gw.get("attempted_tackles", 0)) or 0)
            interceptions  = float(gw.get("interceptions", 0) or 0)
            blocks         = float(gw.get("blocked_shots", gw.get("blocked", 0)) or 0)
            clearances     = float(gw.get("clearances",    gw.get("clearances_off_line", 0)) or 0)
            recoveries     = float(gw.get("recoveries",    0) or 0)

            cbit = tackles + interceptions + blocks + clearances

            if position_type == 2:        # DEF: threshold 10 CBIT
                threshold     = 10
                total_actions = cbit
            else:                          # MID/FWD: threshold 12 CBIRT
                threshold     = 12
                total_actions = cbit + recoveries

            dc_pts_earned += 2.0 if total_actions >= threshold else 0.0
            cbit_total    += cbit
            gw_count      += 1

        nineties = total_mins / 90 if total_mins > 0 else None
        return {
            "fpl_id":      pid,
            "dc_pts_p90":  dc_pts_earned / nineties if nineties else 0.0,
            "dc_hit_rate": (dc_pts_earned / 2.0) / gw_count if gw_count else 0.0,
            "cbit_p90":    cbit_total / nineties if nineties else 0.0,
        }
    except Exception:
        return {"fpl_id": pid}


def fetch_dc_stats(boot, current_gw, lookback=8, progress_bar=None):
    elements = pd.DataFrame(boot["elements"])
    targets  = elements[
        elements["element_type"].isin([2, 3, 4])
    ][["id", "element_type"]].values.tolist()

    gw_start  = max(1, current_gw - lookback + 1)
    total     = len(targets)

    rows, completed = [], 0
    with ThreadPoolExecutor(max_workers=DC_THREADS) as executor:
        futures = {
            executor.submit(_fetch_one_dc, int(pid), gw_start, int(pos)): pid
            for pid, pos in targets
        }
        for future in as_completed(futures):
            rows.append(future.result())
            completed += 1
            if progress_bar and (completed % 25 == 0 or completed == total):
                progress_bar.progress(completed / total,
                                      text=f"Fetching DC stats: {completed}/{total}")

    dc_df = pd.DataFrame(rows).fillna(0)
    for col in ["dc_pts_p90", "dc_hit_rate", "cbit_p90"]:
        if col not in dc_df.columns:
            dc_df[col] = 0.0
    return dc_df.reset_index(drop=True)


# ══════════════════════════════════════════════════════════════════════
#  3.  FIXTURE ANALYSIS
# ══════════════════════════════════════════════════════════════════════

def compute_fixture_difficulty(boot, fixtures, from_gw, horizon=3):
    future = [
        f for f in fixtures
        if f["event"]
        and from_gw < f["event"] <= from_gw + horizon
        and not f["finished_provisional"]
    ]
    team_diffs  = {t["id"]: [] for t in boot["teams"]}
    team_ngames = {t["id"]: 0  for t in boot["teams"]}

    for f in future:
        h, a = f["team_h"], f["team_a"]
        team_diffs[h].append(f["team_h_difficulty"])
        team_diffs[a].append(f["team_a_difficulty"])
        team_ngames[h] += 1
        team_ngames[a] += 1

    teams = pd.DataFrame(boot["teams"])[["id","name","short_name"]]
    rows  = []
    for tid in team_diffs:
        diffs = team_diffs[tid]
        rows.append({
            "team_id":        tid,
            "avg_difficulty": float(np.mean(diffs)) if diffs else 4.5,
            "num_fixtures":   team_ngames[tid],
            "has_dgw":        int(team_ngames[tid] > horizon),
        })
    return pd.DataFrame(rows).merge(teams, left_on="team_id", right_on="id", how="left")


# ══════════════════════════════════════════════════════════════════════
#  4.  CS PROBABILITY
# ══════════════════════════════════════════════════════════════════════

def get_team_defensive_stats(boot):
    """
    Compute team-level defensive stats including:
      - xgc_p90: expected goals conceded per 90 (season average)
      - cs_rate: observed clean sheet rate
      - cs_prob: base CS probability (Poisson from xGC)
      - xg_scored_p90: team's attacking xG per 90 (used to adjust opponent CS)
    """
    players = pd.DataFrame(boot["elements"]).copy()
    teams   = pd.DataFrame(boot["teams"])[["id","name"]].copy()

    for c in ["expected_goals_conceded","expected_goals","minutes","clean_sheets"]:
        players[c] = pd.to_numeric(players.get(c, 0), errors="coerce").fillna(0)

    gks = players[players["element_type"] == 1].copy()
    gk_agg = gks.groupby("team").agg(
        team_xgc     = ("expected_goals_conceded", "sum"),
        team_cs      = ("clean_sheets", "sum"),
        team_minutes = ("minutes", "sum"),
    ).reset_index().rename(columns={"team": "team_id"})

    gk_agg["games_played"] = (gk_agg["team_minutes"] / 90).replace(0, np.nan)
    gk_agg["xgc_p90"]      = gk_agg["team_xgc"] / gk_agg["games_played"]
    gk_agg["cs_rate"]      = gk_agg["team_cs"]  / gk_agg["games_played"]
    gk_agg["cs_prob"]      = np.exp(-gk_agg["xgc_p90"].fillna(1.2))

    # Also compute team attacking strength (xG scored per 90)
    # This is used to adjust opponent's CS probability per fixture
    all_outfield = players[players["element_type"].isin([2, 3, 4])].copy()
    team_atk = all_outfield.groupby("team").agg(
        team_xg_scored = ("expected_goals", "sum"),
        team_atk_mins  = ("minutes", "sum"),
    ).reset_index().rename(columns={"team": "team_id"})
    team_atk["atk_games"] = (team_atk["team_atk_mins"] / (90 * 10)).replace(0, np.nan)  # ~10 outfield players
    team_atk["xg_scored_p90"] = team_atk["team_xg_scored"] / team_atk["atk_games"]

    gk_agg = gk_agg.merge(teams, left_on="team_id", right_on="id", how="left")
    gk_agg = gk_agg.merge(
        team_atk[["team_id", "xg_scored_p90"]],
        on="team_id", how="left"
    )
    gk_agg["xg_scored_p90"] = gk_agg["xg_scored_p90"].fillna(1.3)

    return gk_agg[["team_id","name","xgc_p90","cs_rate","cs_prob","xg_scored_p90"]].reset_index(drop=True)


def compute_fixture_cs_prob(defending_team_id, attacking_team_id, team_def_df):
    """
    Compute fixture-specific CS probability for the defending team.

    Instead of using the defending team's flat season-average CS prob,
    we adjust based on how strong the opponent's attack is relative to
    the league average.

    Method:
      1. Get defending team's base xGC/90
      2. Get attacking team's xG scored/90
      3. League avg xG scored/90 ≈ 1.3
      4. Adjusted xGC = base_xgc × (opponent_xg / league_avg_xg)
      5. CS prob = exp(-adjusted_xgc)  (Poisson P(0 goals))

    This means: Arsenal defending vs Ipswich (low xG) → higher CS prob
                Arsenal defending vs Man City (high xG) → lower CS prob
    """
    league_avg_xg = team_def_df["xg_scored_p90"].mean()
    if league_avg_xg <= 0:
        league_avg_xg = 1.3

    def_row = team_def_df[team_def_df["team_id"] == defending_team_id]
    atk_row = team_def_df[team_def_df["team_id"] == attacking_team_id]

    if def_row.empty:
        base_xgc = 1.2
    else:
        base_xgc = float(def_row.iloc[0].get("xgc_p90", 1.2))

    if atk_row.empty:
        opp_xg = league_avg_xg
    else:
        opp_xg = float(atk_row.iloc[0].get("xg_scored_p90", league_avg_xg))

    # Adjust: scale xGC by opponent attacking strength relative to league average
    adjusted_xgc = base_xgc * (opp_xg / league_avg_xg)

    return float(np.exp(-adjusted_xgc))


# ══════════════════════════════════════════════════════════════════════
#  5.  UNDERSTAT
# ══════════════════════════════════════════════════════════════════════

def get_understat_stats():
    url = "https://understat.com/league/EPL/2025"
    headers = {"User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/122.0.0.0 Safari/537.36"
    )}
    try:
        r = requests.get(url, headers=headers, timeout=30)
        r.raise_for_status()
    except Exception:
        return pd.DataFrame()

    match = re.search(r"var\s+playersData\s*=\s*JSON\.parse\('(.+?)'\)", r.text)
    if not match:
        return pd.DataFrame()

    try:
        raw  = match.group(1).encode("utf-8").decode("unicode_escape")
        data = json.loads(raw)
    except Exception:
        return pd.DataFrame()

    rows = []
    for p in data:
        mins = float(p.get("time", 0) or 0)
        n90  = mins / 90 if mins >= 90 else np.nan
        rows.append({
            "player_name": p.get("player_name", ""),
            "team_title":  p.get("team_title",  ""),
            "minutes_us":  mins,
            "xg_p90":    float(p.get("xG",         0) or 0) / n90 if n90 else 0,
            "xa_p90":    float(p.get("xA",         0) or 0) / n90 if n90 else 0,
            "npxg_p90":  float(p.get("npxG",       0) or 0) / n90 if n90 else 0,
            "shots_p90": float(p.get("shots",      0) or 0) / n90 if n90 else 0,
            "kp_p90":    float(p.get("key_passes", 0) or 0) / n90 if n90 else 0,
        })

    df = pd.DataFrame(rows)
    df = df[df["minutes_us"] >= 90].reset_index(drop=True)
    return df


# ══════════════════════════════════════════════════════════════════════
#  5b. FBREF (StatsBomb/Opta xG — independent second source)
# ══════════════════════════════════════════════════════════════════════

FBREF_TEAM_MAP = {
    "Manchester Utd":  "Man Utd",
    "Manchester City": "Man City",
    "Tottenham":       "Spurs",
    "Nott'ham Forest": "Nott'm Forest",
    "Newcastle Utd":   "Newcastle",
    "Wolverhampton":   "Wolves",
    "West Ham":        "West Ham",
    "Leeds United":    "Leeds",
}

def get_fbref_stats():
    """
    Scrape FBref's Premier League player standard stats table.
    Uses StatsBomb/Opta xG model — independent from Understat.
    Returns per-90 stats with fbref_ prefix for cross-validation.

    FBref rate-limits to 1 request per 3 seconds. We scrape two pages
    (standard stats + shooting) with appropriate delays.
    """
    headers = {"User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/122.0.0.0 Safari/537.36"
    )}

    # ── Page 1: Standard stats (xG, xA, npxG, goals, assists, minutes) ──
    url_std = "https://fbref.com/en/comps/9/stats/Premier-League-Stats"
    try:
        r = requests.get(url_std, headers=headers, timeout=30)
        r.raise_for_status()
    except Exception:
        return pd.DataFrame()

    # FBref hides some tables inside HTML comments — strip them
    html = r.text.replace('<!--', '').replace('-->', '')

    try:
        tables = pd.read_html(StringIO(html))
    except Exception:
        return pd.DataFrame()

    # Find the main player stats table — it has 'Player' and 'xG' columns
    std_df = None
    for tbl in tables:
        # Handle multi-level columns
        if isinstance(tbl.columns, pd.MultiIndex):
            tbl.columns = [
                c[-1] if isinstance(c, tuple) else c
                for c in tbl.columns
            ]
        cols_lower = [str(c).lower() for c in tbl.columns]
        if 'player' in cols_lower and 'xg' in cols_lower and len(tbl) > 50:
            std_df = tbl.copy()
            break

    if std_df is None:
        return pd.DataFrame()

    # Flatten multi-index columns if needed
    if isinstance(std_df.columns, pd.MultiIndex):
        std_df.columns = [c[-1] for c in std_df.columns]

    # Remove header rows that appear inline (FBref quirk)
    std_df = std_df[std_df["Player"] != "Player"].copy()

    # Normalise column names
    col_map = {}
    for c in std_df.columns:
        cl = str(c).strip()
        col_map[c] = cl
    std_df = std_df.rename(columns=col_map)

    # Extract the columns we need
    needed = {
        "Player": "player_name",
        "Squad": "team_name",
        "Min": "minutes_fb",
        "90s": "nineties",
        "Gls": "goals",
        "Ast": "assists",
        "xG": "xg_total",
        "xAG": "xa_total",
        "npxG": "npxg_total",
        "Sh": "shots_total",
    }

    # Some columns might use slightly different names across seasons
    alt_names = {
        "xAG": ["xAG", "xA"],
        "Min": ["Min", "Playing Time Min"],
        "90s": ["90s", "Playing Time 90s"],
        "Sh": ["Sh", "Shooting Sh"],
    }

    available = {}
    for target, aliases in alt_names.items():
        for alias in aliases:
            if alias in std_df.columns:
                available[target] = alias
                break

    # Build rename dict with available columns
    rename = {}
    for orig, dest in needed.items():
        col = available.get(orig, orig)
        if col in std_df.columns:
            rename[col] = dest

    std_df = std_df.rename(columns=rename)

    # Keep only rows with data
    keep = [v for v in rename.values() if v in std_df.columns]
    if "player_name" not in keep:
        return pd.DataFrame()
    std_df = std_df[keep].copy()

    # Convert numeric columns
    for c in std_df.columns:
        if c not in ("player_name", "team_name"):
            std_df[c] = pd.to_numeric(std_df[c], errors="coerce").fillna(0)

    # Filter minimum minutes
    if "minutes_fb" in std_df.columns:
        std_df = std_df[std_df["minutes_fb"] >= 90].copy()
    elif "nineties" in std_df.columns:
        std_df = std_df[std_df["nineties"] >= 1.0].copy()

    # Compute per-90 stats
    if "nineties" in std_df.columns:
        n90 = std_df["nineties"].replace(0, np.nan)
    elif "minutes_fb" in std_df.columns:
        n90 = (std_df["minutes_fb"] / 90).replace(0, np.nan)
    else:
        return pd.DataFrame()

    std_df["fbref_xg_p90"]    = std_df.get("xg_total", 0) / n90
    std_df["fbref_xa_p90"]    = std_df.get("xa_total", 0) / n90
    std_df["fbref_npxg_p90"]  = std_df.get("npxg_total", 0) / n90
    std_df["fbref_shots_p90"] = std_df.get("shots_total", 0) / n90

    # Clean team names
    if "team_name" in std_df.columns:
        std_df["team_name"] = std_df["team_name"].apply(
            lambda t: FBREF_TEAM_MAP.get(str(t).strip(), str(t).strip())
        )

    result = std_df[["player_name", "team_name",
                      "fbref_xg_p90", "fbref_xa_p90",
                      "fbref_npxg_p90", "fbref_shots_p90"]].copy()
    result = result.dropna(subset=["player_name"]).reset_index(drop=True)

    return result


def match_fbref(fpl_df, fb_df, threshold=0.45):
    """Match FBref players to FPL players using fuzzy name + team matching."""
    fbref_cols = ["fbref_xg_p90", "fbref_xa_p90", "fbref_npxg_p90", "fbref_shots_p90"]
    for c in fbref_cols:
        fpl_df[c] = 0.0

    if fb_df.empty:
        return fpl_df.reset_index(drop=True)

    fb_names_norm = [_norm(n) for n in fb_df["player_name"].tolist()]
    fb_teams_norm = [_norm(FBREF_TEAM_MAP.get(t, t)) for t in fb_df["team_name"].tolist()]

    matched = 0
    for idx in fpl_df.index:
        fpl_tokens  = set(_norm(fpl_df.at[idx, "full_name"]).split())
        team_tokens = set(_norm(fpl_df.at[idx, "team_name"]).split())

        best_score, best_i = 0.0, -1
        for i, (fn, ft) in enumerate(zip(fb_names_norm, fb_teams_norm)):
            s = _token_match(fpl_tokens, fn, ft, team_tokens)
            if s > best_score:
                best_score, best_i = s, i

        if best_score >= threshold and best_i >= 0:
            row = fb_df.iloc[best_i]
            for c in fbref_cols:
                fpl_df.at[idx, c] = float(row.get(c, 0) or 0)
            matched += 1

    return fpl_df.reset_index(drop=True)


def merge_dual_source_xg(df):
    """
    Blend Understat and FBref xG/xA into a single consensus estimate.

    Strategy:
      - If both sources have data: average them (reduces model-specific noise)
      - If only one has data: use that source alone
      - Stores the blended values AND keeps originals for comparison

    Also computes discrepancy metrics for the UI.
    """
    for metric in ["xg_p90", "xa_p90", "npxg_p90", "shots_p90"]:
        us_col = metric                    # Understat column
        fb_col = f"fbref_{metric}"         # FBref column

        us = df[us_col].fillna(0) if us_col in df.columns else 0
        fb = df[fb_col].fillna(0) if fb_col in df.columns else 0

        # Store originals for comparison display
        df[f"us_{metric}"]    = us
        df[f"fb_{metric}"]    = fb

        # Blend: average when both have data, otherwise use whichever has it
        has_us = us > 0
        has_fb = fb > 0
        both   = has_us & has_fb

        blended = pd.Series(0.0, index=df.index)
        blended[both]          = (us[both] + fb[both]) / 2
        blended[has_us & ~both] = us[has_us & ~both]
        blended[has_fb & ~both] = fb[has_fb & ~both]

        df[metric] = blended  # overwrite with blended value

    # Discrepancy: absolute difference between sources (for flagging)
    df["xg_discrepancy"] = abs(df.get("us_xg_p90", 0) - df.get("fb_xg_p90", 0))
    df["xa_discrepancy"] = abs(df.get("us_xa_p90", 0) - df.get("fb_xa_p90", 0))

    return df


# ══════════════════════════════════════════════════════════════════════
#  6.  NAME MATCHING
# ══════════════════════════════════════════════════════════════════════

def _norm(name: str) -> str:
    name = unicodedata.normalize("NFD", str(name))
    name = "".join(c for c in name if unicodedata.category(c) != "Mn")
    return re.sub(r"[^a-z ]", "", name.lower().strip())

def _token_match(a_tokens, b_norm, b_team_norm, ref_team_tokens, squad_bonus=0.08):
    b_tokens = set(b_norm.split())
    if not b_tokens:
        return 0.0
    overlap = len(a_tokens & b_tokens) / max(len(a_tokens | b_tokens), 1)
    bonus   = squad_bonus if (ref_team_tokens & set(b_team_norm.split())) else 0
    return overlap + bonus

def match_understat(fpl_df, us_df, threshold=0.45):
    ext_cols = ["xg_p90","xa_p90","npxg_p90","shots_p90","kp_p90"]
    for c in ext_cols:
        fpl_df[c] = 0.0

    if us_df.empty:
        return fpl_df.reset_index(drop=True)

    us_names_norm = [_norm(n) for n in us_df["player_name"].tolist()]
    us_teams_norm = [_norm(TEAM_MAP.get(t, t)) for t in us_df["team_title"].tolist()]

    for idx in fpl_df.index:
        fpl_tokens  = set(_norm(fpl_df.at[idx, "full_name"]).split())
        team_tokens = set(_norm(fpl_df.at[idx, "team_name"]).split())

        best_score, best_i = 0.0, -1
        for i, (un, ut) in enumerate(zip(us_names_norm, us_teams_norm)):
            s = _token_match(fpl_tokens, un, ut, team_tokens)
            if s > best_score:
                best_score, best_i = s, i

        if best_score >= threshold and best_i >= 0:
            row = us_df.iloc[best_i]
            for c in ext_cols:
                fpl_df.at[idx, c] = float(row.get(c, 0) or 0)

    return fpl_df.reset_index(drop=True)


# ══════════════════════════════════════════════════════════════════════
#  7.  RISK FLAGS
# ══════════════════════════════════════════════════════════════════════

def add_risk_flags(df):
    df    = df.reset_index(drop=True)
    mins  = df["minutes"].values.astype(float)
    pts   = df["total_points"].values.astype(float)
    games = np.where(pts > 0, np.maximum(pts / 3.0, 1.0), 1.0)
    df["minutes_per_game"] = mins / games

    risk = []
    for s, c, m in zip(df["status"].values,
                        df["chance_of_playing_next_round"].values.astype(float),
                        mins):
        if   s == "i":  risk.append("Injured")
        elif s == "s":  risk.append("Suspended")
        elif c < 50:    risk.append("Doubt")
        elif m < 500:   risk.append("Low mins")
        else:           risk.append("Ok")
    df["rotation_risk"] = risk
    return df


# ══════════════════════════════════════════════════════════════════════
#  8.  EXPECTED FPL POINTS MODEL (v5.3 — replaces z-score weights)
# ══════════════════════════════════════════════════════════════════════
#
#  Converts per-90 stats into actual expected FPL points per match,
#  using the official 2025/26 scoring table. This allows meaningful
#  cross-position comparison (e.g. Salah vs Tarkowski).
#
#  Expected points per match =
#    appearance_pts (2 if >60min expected)
#  + position_goal_pts × xG/90
#  + assist_pts × xA/90
#  + cs_prob × cs_pts_for_position
#  + dc_hit_rate × 2  (DC points)
#  + saves_p90 / 3    (GKP only, 1pt per 3 saves)
#  + bonus_p90         (estimated from historical bonus)
#  - yellow_rate × 1   (yellow card deduction)
#  - goals_conceded_penalty (GKP/DEF: -0.5 per goal conceded expected)
#
#  Then adjusted for fixture difficulty.
# ══════════════════════════════════════════════════════════════════════

def _safe(df, col):
    if col in df.columns:
        return pd.to_numeric(df[col], errors="coerce").fillna(0)
    return pd.Series(np.zeros(len(df)), index=df.index)


def compute_expected_pts(df):
    """
    Compute expected FPL points per match for every player.
    Uses official 2025/26 scoring and per-90 underlying stats.

    v5.4 fixes:
      - Added start_prob: minutes-based probability the player actually
        starts / plays meaningful minutes. Non-regulars are properly
        discounted instead of receiving full per-90 credit.
      - Reduced ep_next blend from 30% → 15% so the richer underlying
        model dominates while still capturing FPL's injury signal.
      - Appearance points now scaled by start_prob (a bench player
        shouldn't get a flat 2.0 every week).

    Returns the dataframe with 'xPts' and 'fpl_value_score' columns.
    """
    df = df.copy().reset_index(drop=True)

    mins     = _safe(df, "minutes").replace(0, np.nan)
    total_gw = mins / 90  # approximate games played

    # ── Start probability (minutes-based) ──────────────────────
    # Estimate how likely this player is to play meaningful minutes
    # in a given GW, based on their share of available minutes.
    #
    # max_team_mins = the most minutes any single player on the same
    # team has played (proxy for "total GWs played by that team").
    # start_prob ≈ player_mins / max_team_mins, clamped [0.05, 1.0].
    #
    # Examples (assuming max_team_mins ≈ 2700 at GW30):
    #   Regular starter (2500 mins) → ~0.93
    #   Rotation player (1200 mins) → ~0.44
    #   Bench warmer    (300 mins)  → ~0.11
    team_max_mins = df.groupby("team_id")["minutes"].transform("max")
    team_max_mins = team_max_mins.replace(0, np.nan)
    start_prob = (mins / team_max_mins).clip(0.05, 1.0).fillna(0.05)
    df["start_prob"] = start_prob

    # Per-90 stats
    xg       = _safe(df, "xg_p90")
    xa       = _safe(df, "xa_p90")
    cs_prob  = _safe(df, "cs_prob")
    dc_rate  = _safe(df, "dc_hit_rate")
    fix_s    = _safe(df, "fix_score")  # fixture difficulty score

    # ── Per-90 derived stats with small-sample stabilisation ──
    # Raw per-90 rates (bonus, saves, YC, GC) are wildly inflated
    # for low-minute players. E.g. 8 bonus from 120 mins = 6.0/90
    # vs a starter's 20 bonus from 2500 mins = 0.72/90.
    #
    # Fix: use a stabilised denominator — max(actual_mins, 450) —
    # so players with <450 mins (~5 full matches) don't get
    # extreme per-90 rates. This acts like a Bayesian prior,
    # regressing low-sample players toward lower rates.
    stabilised_mins = mins.clip(lower=450)

    saves     = _safe(df, "saves") / stabilised_mins * 90
    saves     = saves.fillna(0)
    bonus_p90 = _safe(df, "bonus") / stabilised_mins * 90
    bonus_p90 = bonus_p90.fillna(0).clip(upper=1.5)  # hard cap: no one averages >1.5 bonus/match
    yc_rate   = _safe(df, "yellow_cards") / stabilised_mins * 90
    yc_rate   = yc_rate.fillna(0)
    ep_next   = _safe(df, "ep_next")  # FPL's own short-term estimate
    xgc_rate  = _safe(df, "goals_conceded") / stabilised_mins * 90
    xgc_rate  = xgc_rate.fillna(1.0)

    # Goal and CS points by position
    goal_pts = df["position"].map({"GKP": 6, "DEF": 6, "MID": 5, "FWD": 4}).fillna(4)
    cs_pts   = df["position"].map({"GKP": 4, "DEF": 4, "MID": 1, "FWD": 0}).fillna(0)
    gc_pen   = df["position"].map({"GKP": -0.5, "DEF": -0.5, "MID": 0, "FWD": 0}).fillna(0)

    # ── Expected FPL points per match (when playing) ──
    xPts_if_playing = (
        2.0                         # appearance points (assume >60 mins)
        + goal_pts * xg             # expected goal points
        + 3.0 * xa                  # expected assist points (3pts all positions)
        + cs_pts * cs_prob          # expected clean sheet points
        + 2.0 * dc_rate             # expected DC points
        + saves / 3.0               # save points (GKP: 1pt per 3 saves)
        + bonus_p90                 # expected bonus points
        - 1.0 * yc_rate             # yellow card deduction
        + gc_pen * xgc_rate         # goals conceded penalty (GKP/DEF)
    )

    # ── Discount by start probability ─────────────────────────
    # xPts represents the EXPECTED points per GW, accounting for
    # the chance the player doesn't play at all.
    xPts = xPts_if_playing * start_prob

    # Fixture difficulty adjustment: scale by fixture_score
    # fix_score is higher for easier fixtures; normalise around 1.0
    fix_mean = fix_s.mean()
    if fix_mean > 0:
        fix_mult = 0.7 + 0.3 * (fix_s / fix_mean)  # range ~0.7 to ~1.3
    else:
        fix_mult = 1.0

    xPts_adjusted = xPts * fix_mult

    # Blend with FPL's own ep_next (short-term prediction) — gives
    # weight to injury/rotation info that our model can't see.
    # v5.4: reduced from 30% → 15% so the richer model dominates;
    # ep_next is a conservative single-GW figure that was pulling
    # regular starters down to 3-4 pts.
    ep_has_data = ep_next > 0
    blended = xPts_adjusted.copy()
    blended[ep_has_data] = (
        0.85 * xPts_adjusted[ep_has_data] +
        0.15 * ep_next[ep_has_data]
    )

    df["xPts"]           = xPts.round(2)
    df["xPts_adjusted"]  = blended.round(2)
    df["fpl_value_score"] = blended  # this is what the transfer engine ranks by

    return df


def apply_position_scores(df):
    """Apply the expected points model to all players."""
    return compute_expected_pts(df)

def apply_risk_appetite(df, appetite):
    own = pd.to_numeric(df["selected_by_percent"], errors="coerce").fillna(0)
    df["ownership"] = own
    # Normalise ownership to 0-1 range for adjustment
    own_max = own.max()
    own_norm = (own / own_max) if own_max > 0 else own
    if   appetite == "safe":         df["fpl_value_score"] += 0.5 * own_norm
    elif appetite == "differential": df["fpl_value_score"] -= 0.5 * own_norm
    return df


# ══════════════════════════════════════════════════════════════════════
#  9.  SINGLE-GW PLAYER RESCORING
# ══════════════════════════════════════════════════════════════════════

def rescore_for_gw(players_base, boot, fixtures, target_gw, cs_map, risk_appetite,
                   team_def_df=None):
    df = players_base.copy()

    fix_df  = compute_fixture_difficulty(boot, fixtures, target_gw - 1, horizon=3)
    fix_map = fix_df.set_index("team_id")[
        ["avg_difficulty","num_fixtures","has_dgw"]
    ].to_dict("index")

    df["avg_fix_diff"] = df["team_id"].map(
        lambda t: fix_map.get(t, {}).get("avg_difficulty", 3.0)
    )
    df["num_fixtures"] = df["team_id"].map(
        lambda t: fix_map.get(t, {}).get("num_fixtures", 3)
    )
    df["has_dgw"]   = df["team_id"].map(
        lambda t: fix_map.get(t, {}).get("has_dgw", 0)
    )
    df["fix_score"] = (6 - df["avg_fix_diff"]) * (df["num_fixtures"] / 3)

    # ── Fixture-specific CS probabilities ──
    if team_def_df is not None:
        # Build opponent map for this GW's fixtures
        gw_fixtures = [
            f for f in fixtures
            if f["event"] and target_gw - 1 < f["event"] <= target_gw + 2
            and not f["finished_provisional"]
        ]
        # Map: team_id -> list of opponent team_ids in upcoming fixtures
        team_opponents = {}
        for f in gw_fixtures:
            h, a = f["team_h"], f["team_a"]
            team_opponents.setdefault(h, []).append(a)
            team_opponents.setdefault(a, []).append(h)

        # For each team, compute avg fixture-specific CS prob across opponents
        fixture_cs = {}
        for tid in df["team_id"].unique():
            opps = team_opponents.get(int(tid), [])
            if opps:
                probs = [compute_fixture_cs_prob(int(tid), opp, team_def_df) for opp in opps]
                fixture_cs[int(tid)] = float(np.mean(probs))
            else:
                fixture_cs[int(tid)] = cs_map.get(int(tid), 0.25)

        df["cs_prob"] = df["team_id"].map(fixture_cs).fillna(0.25)
    else:
        df["cs_prob"] = df["team_id"].map(cs_map).fillna(0.25)

    df = apply_position_scores(df)
    df = apply_risk_appetite(df, risk_appetite)
    return df


# ══════════════════════════════════════════════════════════════════════
#  10.  GW-SPECIFIC CAPTAIN PROJECTION (v5.2 — CS×4 corrected)
# ══════════════════════════════════════════════════════════════════════

def compute_gw_projected_pts(squad_df, target_gw, fixtures):
    """
    Project points per player for a specific GW.

    v5.2 corrected scoring:
      GKP: cs_prob × 4 + saves_p90/3 + 2   (CS=4pts, NOT 6)
      DEF: cs_prob × 4 + xg×6 + xa×3 + dc×2 + 2   (CS=4pts, NOT 6)
      MID: cs_prob × 1 + xg×5 + xa×3 + dc×2 + 2
      FWD: xg×4 + xa×3 + 2

    Fixture difficulty multiplier: diff 3 = 1.0×, ±0.10 per step.
    """
    df = squad_df.copy().reset_index(drop=True)

    gw_fixtures = [
        f for f in fixtures
        if f["event"] == target_gw and not f["finished_provisional"]
    ]
    team_gw_diff  = {}
    teams_playing = set()
    for f in gw_fixtures:
        team_gw_diff[f["team_h"]] = f["team_h_difficulty"]
        team_gw_diff[f["team_a"]] = f["team_a_difficulty"]
        teams_playing.add(f["team_h"])
        teams_playing.add(f["team_a"])

    df["gw_diff"]        = df["team_id"].map(team_gw_diff).fillna(3.0)
    df["fix_multiplier"] = 1.0 + (3.0 - df["gw_diff"]) * 0.10
    df["has_fixture"]    = df["team_id"].isin(teams_playing)

    # ── Start probability: use team-relative minutes share ──
    # v5.4: replaced the old clip(0.5,1.0) approach which was too
    # generous to non-starters — a player with 200 mins was getting
    # at least 50% credit. Now uses the same team-relative method
    # as the main xPts model for consistency.
    team_max_mins = df.groupby("team_id")["minutes"].transform("max")
    team_max_mins = team_max_mins.replace(0, np.nan)
    df["start_prob"] = (df["minutes"] / team_max_mins).clip(0.05, 1.0).fillna(0.05)

    proj_pts = []
    for _, p in df.iterrows():
        if not p["has_fixture"]:
            proj_pts.append(0.0)
            continue

        pos = p.get("position", "")
        fm  = float(p["fix_multiplier"])
        sp  = float(p["start_prob"])
        cs  = float(p.get("cs_prob",     0.25))
        xg  = float(p.get("xg_p90",     0.0))
        xa  = float(p.get("xa_p90",     0.0))
        dc  = float(p.get("dc_hit_rate", 0.0))
        sv  = float(p.get("saves",       0.0))
        sm  = max(float(p.get("minutes", 90.0)), 450.0)  # stabilise like main model

        saves_p90 = sv / sm * 90.0

        # ──────────────────────────────────────────────────────
        # v5.2 CORRECTED: CS = 4pts for GKP/DEF (was 6 in v5.1)
        # ──────────────────────────────────────────────────────
        if pos == "GKP":
            base = (cs * 4.0) + (saves_p90 / 3.0) + 2.0
        elif pos == "DEF":
            base = (cs * 4.0) + (xg * 6.0) + (xa * 3.0) + (dc * 2.0) + 2.0
        elif pos == "MID":
            base = (cs * 1.0) + (xg * 5.0) + (xa * 3.0) + (dc * 2.0) + 2.0
        else:   # FWD
            base = (xg * 4.0) + (xa * 3.0) + 2.0

        proj_pts.append(base * fm * sp)

    df["gw_proj_pts"] = proj_pts
    return df


# ══════════════════════════════════════════════════════════════════════
#  11.  TRANSFER ENGINE
# ══════════════════════════════════════════════════════════════════════

def suggest_transfers_for_gw(squad_ids, scored_players, my_team_df, itb,
                              free_transfers=1, budget_padding=0.0,
                              hit_threshold=0.5, max_transfers=5,
                              horizon_remaining=1):
    """
    Suggest optimal transfers for a GW, including paid transfers (hits).

    A -4 hit is taken for each transfer beyond free_transfers.
    A transfer is only suggested if its projected gain over the remaining
    horizon exceeds the hit cost. For free transfers, any positive gain
    is accepted.

    Args:
        hit_threshold: minimum net gain (after hit cost) per extra transfer
                       over the remaining horizon to justify the hit
        max_transfers: absolute maximum transfers to consider per GW
        horizon_remaining: GWs left in planning window (used to amortise gain)
    """
    available = scored_players[
        (~scored_players["fpl_id"].isin(squad_ids)) &
        (scored_players["status"].isin(["a","d"]))
    ].copy()

    team_counts = my_team_df["team_id"].value_counts().to_dict()

    results = []
    for _, out_p in my_team_df.iterrows():
        sell     = float(out_p.get("sell_price", out_p["price"]))
        buy_price = float(out_p["price"])
        bdgt     = sell + itb + budget_padding
        pos      = out_p["position"]
        s_out    = float(out_p.get("fpl_value_score", 0))
        out_team = int(out_p["team_id"])

        counts_after_out = dict(team_counts)
        counts_after_out[out_team] = counts_after_out.get(out_team, 0) - 1

        def within_team_limit(cand_team_id, _counts=counts_after_out):
            return _counts.get(int(cand_team_id), 0) < 3

        cands = available[
            (available["position"] == pos) &
            (available["price"] <= bdgt) &
            (available["team_id"].apply(within_team_limit))
        ].sort_values("fpl_value_score", ascending=False).head(5)

        for _, cand in cands.iterrows():
            gain = float(cand["fpl_value_score"]) - s_out
            if gain > 0.01:  # lower threshold — hit logic handles the rest
                results.append({
                    "OUT":             out_p["web_name"],
                    "OUT_pos":         pos,
                    "OUT_sell_price":  sell,
                    "OUT_buy_price":   buy_price,
                    "OUT_score":       round(s_out, 3),
                    "OUT_ep_next":     float(out_p.get("ep_next", 0)),
                    "OUT_dc_hit_rate": round(float(out_p.get("dc_hit_rate", 0)), 2),
                    "OUT_fix_diff":    round(float(out_p.get("avg_fix_diff", 0)), 2),
                    "OUT_risk":        out_p.get("rotation_risk", "?"),
                    "IN":              cand["web_name"],
                    "IN_team":         cand["team_short"],
                    "IN_team_id":      int(cand["team_id"]),
                    "IN_pos":          pos,
                    "IN_price":        float(cand["price"]),
                    "IN_score":        round(float(cand["fpl_value_score"]), 3),
                    "IN_ep_next":      float(cand.get("ep_next", 0)),
                    "IN_xPts":         round(float(cand.get("xPts", 0)), 2),
                    "IN_xg_p90":       round(float(cand.get("xg_p90",      0)), 3),
                    "IN_xa_p90":       round(float(cand.get("xa_p90",      0)), 3),
                    "IN_dc_hit_rate":  round(float(cand.get("dc_hit_rate", 0)), 2),
                    "IN_dc_pts_p90":   round(float(cand.get("dc_pts_p90",  0)), 2),
                    "IN_cbit_p90":     round(float(cand.get("cbit_p90",    0)), 2),
                    "IN_fix_diff":     round(float(cand.get("avg_fix_diff",0)), 2),
                    "IN_cs_prob":      round(float(cand.get("cs_prob",     0)), 2),
                    "IN_own_pct":      float(cand.get("ownership",         0)),
                    "IN_risk":         cand.get("rotation_risk", "?"),
                    "score_gain":      round(gain, 3),
                    "price_diff":      round(float(cand["price"]) - sell, 1),
                    "_out_fpl_id":     int(out_p["fpl_id"]),
                    "_out_team_id":    out_team,
                })

    if not results:
        return [], my_team_df.copy(), squad_ids.copy(), 0

    results_df = pd.DataFrame(results).sort_values(
        "score_gain", ascending=False
    ).reset_index(drop=True)

    selected_transfers  = []
    updated_squad       = my_team_df.copy()
    updated_squad_ids   = set(squad_ids)
    current_team_counts = dict(team_counts)
    total_hit_cost      = 0

    for transfer_num in range(1, max_transfers + 1):
        # Determine if this transfer is free or costs a hit
        is_free = transfer_num <= free_transfers
        hit_cost = 0 if is_free else 4

        best = None
        for _, row in results_df.iterrows():
            if row["OUT"] not in updated_squad["web_name"].values:
                continue
            if any(row["IN"] == prev["IN"] for prev in selected_transfers):
                continue

            # Check 3-per-team constraint with all previous transfers
            sim_counts = dict(current_team_counts)
            for prev in selected_transfers:
                sim_counts[prev["_out_team_id"]] = (
                    sim_counts.get(prev["_out_team_id"], 0) - 1
                )
                sim_counts[prev["IN_team_id"]] = (
                    sim_counts.get(prev["IN_team_id"], 0) + 1
                )
            sim_counts[int(row["_out_team_id"])] = (
                sim_counts.get(int(row["_out_team_id"]), 0) - 1
            )
            if sim_counts.get(int(row["IN_team_id"]), 0) + 1 > 3:
                continue

            # Check budget feasibility
            cumulative_itb = itb
            for prev in selected_transfers:
                cumulative_itb += prev["OUT_sell_price"] - prev["IN_price"]
            remaining_budget = cumulative_itb + row["OUT_sell_price"] + budget_padding
            if row["IN_price"] > remaining_budget:
                continue

            # For hits: gain must exceed hit cost amortised over horizon
            # A hit costs 4 pts this GW, but the better player earns points
            # across all remaining GWs. So the gain is score_gain × horizon_remaining
            projected_total_gain = row["score_gain"] * max(horizon_remaining, 1)
            net_gain = projected_total_gain - hit_cost

            if is_free and row["score_gain"] > 0.01:
                best = row
                break
            elif not is_free and net_gain > hit_threshold:
                best = row
                break

        if best is None:
            break

        if not is_free:
            total_hit_cost += 4

        selected_transfers.append(best.to_dict())

        out_fpl_id = int(
            updated_squad.loc[
                updated_squad["web_name"] == best["OUT"], "fpl_id"
            ].values[0]
        )
        in_row = scored_players[scored_players["web_name"] == best["IN"]]
        if not in_row.empty:
            in_player = in_row.iloc[[0]].copy()
            in_player["sell_price"] = in_player["price"]
            updated_squad = updated_squad[updated_squad["fpl_id"] != out_fpl_id]
            updated_squad = pd.concat(
                [updated_squad, in_player], ignore_index=True
            )
            updated_squad_ids.discard(out_fpl_id)
            updated_squad_ids.add(int(in_player.iloc[0]["fpl_id"]))

        results_df = results_df[
            results_df["OUT"] != best["OUT"]
        ].reset_index(drop=True)

    return selected_transfers, updated_squad, updated_squad_ids, total_hit_cost


# ══════════════════════════════════════════════════════════════════════
#  12.  ROLLING 6-GW PLANNER
# ══════════════════════════════════════════════════════════════════════

def run_rolling_plan(players_base, boot, fixtures, cs_map, my_team_df,
                     squad_ids, itb, free_transfers, next_gw,
                     horizon_gws, risk_appetite, budget_padding,
                     team_def_df=None):
    current_squad    = my_team_df.copy()
    current_ids      = set(squad_ids)
    current_itb      = itb
    banked_transfers = free_transfers
    plan_summary     = []

    for gw_offset in range(horizon_gws):
        target_gw     = next_gw + gw_offset
        num_ft        = min(banked_transfers, MAX_TRANSFER_BANK)
        horizon_remaining = horizon_gws - gw_offset

        scored = rescore_for_gw(players_base, boot, fixtures, target_gw,
                                cs_map, risk_appetite, team_def_df=team_def_df)
        scored_squad = scored[scored["fpl_id"].isin(current_ids)].copy()

        # Carry sell_price from current_squad into the rescored squad.
        if "sell_price" in current_squad.columns:
            sp_map = current_squad.set_index("fpl_id")["sell_price"].to_dict()
            scored_squad["sell_price"] = scored_squad["fpl_id"].map(sp_map)
            scored_squad["sell_price"] = scored_squad["sell_price"].fillna(scored_squad["price"])

        transfers, updated_squad, updated_ids, hit_cost = suggest_transfers_for_gw(
            current_ids, scored, scored_squad, current_itb,
            free_transfers=num_ft,
            budget_padding=budget_padding,
            horizon_remaining=horizon_remaining,
        )

        transfers_made = len(transfers)
        free_used = min(transfers_made, num_ft)
        paid_transfers = max(0, transfers_made - num_ft)

        # Captain projection
        cap_options = []
        cap_pool = scored[scored["fpl_id"].isin(updated_ids)].copy()
        if not cap_pool.empty:
            cap_pool = compute_gw_projected_pts(cap_pool, target_gw, fixtures)
            cap_pool = cap_pool[cap_pool["has_fixture"]]
            if not cap_pool.empty:
                top3 = cap_pool.sort_values("gw_proj_pts", ascending=False).head(3)
                for _, p in top3.iterrows():
                    cap_options.append({
                        "name":      p["web_name"],
                        "team":      p.get("team_short", "?"),
                        "proj_pts":  round(float(p["gw_proj_pts"]), 2),
                        "fix_diff":  round(float(p.get("gw_diff", 3.0)), 1),
                        "xg_p90":    round(float(p.get("xg_p90", 0)), 3),
                        "xa_p90":    round(float(p.get("xa_p90", 0)), 3),
                        "cs_prob":   round(float(p.get("cs_prob", 0)), 2),
                        "risk":      p.get("rotation_risk", "?"),
                    })

        plan_summary.append({
            "gw":              target_gw,
            "free_transfers":  num_ft,
            "itb":             round(current_itb, 1),
            "transfers":       transfers,
            "transfers_made":  transfers_made,
            "hit_cost":        hit_cost,
            "paid_transfers":  paid_transfers,
            "captain_options": cap_options,
        })

        current_squad = updated_squad
        current_ids   = updated_ids
        banked_transfers = min(
            banked_transfers - free_used + 1,
            MAX_TRANSFER_BANK
        )
        banked_transfers = max(banked_transfers, 0)

        for t in transfers:
            current_itb = current_itb + t["OUT_sell_price"] - t["IN_price"]

    return plan_summary


# ══════════════════════════════════════════════════════════════════════
#  13.  STREAMLIT APP
# ══════════════════════════════════════════════════════════════════════


def main():
    st.set_page_config(
        page_title="Datumly",
        page_icon="⚽",
        layout="wide",
        initial_sidebar_state="collapsed",
    )

    # ── Embedded logo (base64) ─────────────────────────────
    LOGO_B64 = "iVBORw0KGgoAAAANSUhEUgAAAkYAAADxCAYAAAAujXiIAAAw30lEQVR4nO3dd5wsRbn/8U+RJB8UgQFUsqiAYGBR5BrBSBIkqtAoCqgr/UPa6zUrKmpfpa8jgofU5KAkDxe5igIKAgMigpJBUIQliHAOUYH6/VG9nD7Dzu7sTNV0z+73/Xrt6+yZ7Xn66Z70TFV1FYiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiMgMMzYyaqvOYTZbrOoEREREROpChZGIiEhNjLcWqdWoOiqMRERERAoqjERERGqgvZVIrUbVUGEkIiIiUlBhJCIiUrFOrUNqNRo8FUYiIiIiBRVGIiIiFZqqVUitRoOlwkhERESkoMJIRESkIt22BqnVaHBUGImIiIgUVBiJiIhUYLqtQGo1GgwVRiIiIiIFFUYiIiID1mvrj1qNwlNhJCIiIlJQYSQiIiJSUGEkIiIyQP12h6k7LSwVRiIiIiIFFUYiIiID4qu1R61G4agwEhERESmoMBIRERkA3608ajUKQ4WRiIiISEGFkYiISGChWnfUauSfCiMRERGRggojERGRgEK36qjVyC8VRiIiIiIFFUYiIiKBDKo1R61G/qgwEhERESmoMBIREQlg0K04ajXyQ4WRiIiISMFUnYCIiIhM3OLTaDX1OT1gajESERERKagwEhERESmoMBIREREpqDASERERKagwEhERESmoMBIREREpqDASERERKagwEhERESmoMBIREREpqDASERERKSxRdQJVstb6XnDvLmPM2p5jiohIIcBCqXc1Ws21PceUIaYWIxEREZGCCiMRERGRggojERERkYIKIxEREZHCrB58LTOftfarnkM+bIzJPMcUEZGaUGEkM91XPMe7C8g8xxQRkZpQV5qIiIhIQYWRiIiISEFdaSIiIjXQaDVN1TmIWoxEREREnqPCSERERKSgwkhERESkoMJIREREpKDCSERERKSgwkhERESkoMJIREREpKDCSERERKSgwkhERESkoMJIREREpKDCSERERKRQm7XSrLUG2Bh4E7AJ8ApgTWBVYFlcro8B84GHgVuBG4A/ARcZY+4bfNb1Ms1zuAD4C/Dn4ud3xphbK0hbZEYqvR43wr0W1wFWBFYAXgA8instzgf+BlwHXA/cYYyxVeQMYK1d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"

    # ── Embedded CSS ───────────────────────────────────────
    CSS = """@import url('https://fonts.googleapis.com/css2?family=DM+Sans:ital,opsz,wght@0,9..40,300;0,9..40,400;0,9..40,500;0,9..40,600;0,9..40,700&display=swap');

:root {
    --pink: #e8366d;
    --pink-lt: #f2648a;
    --bg: #111318;
    --card: #1a1d25;
    --card2: #1e222c;
    --border: #2a2e3a;
    --text: #e8eaf0;
    --dim: #8b90a0;
    --green: #3ddc84;
    --red: #ff5c5c;
    --gold: #fbbf24;
}

/* ── Force dark background ── */
.stApp,
.stApp > header,
.main,
.main .block-container,
[data-testid="stAppViewContainer"],
[data-testid="stAppViewBlockContainer"] {
    background-color: var(--bg) !important;
    color: var(--text) !important;
    font-family: 'DM Sans', sans-serif !important;
}

/* Hide default header chrome */
header[data-testid="stHeader"] {
    background: transparent !important;
}
#MainMenu, footer, [data-testid="stToolbar"] {
    visibility: hidden !important;
}

/* ── All text ── */
.stMarkdown, .stMarkdown p, .stMarkdown li,
.stMarkdown h1, .stMarkdown h2, .stMarkdown h3,
.stMarkdown h4, .stMarkdown h5, .stMarkdown h6,
.stText, [data-testid="stText"],
.stCaption, [data-testid="stCaptionContainer"] {
    color: var(--text) !important;
    font-family: 'DM Sans', sans-serif !important;
}
.stCaption, [data-testid="stCaptionContainer"] {
    color: var(--dim) !important;
}

/* ── Branded header ── */
.datumly-header {
    text-align: center;
    padding: 2.5rem 1rem 1.5rem;
    border-bottom: 1px solid var(--border);
    margin-bottom: 1.5rem;
}
.datumly-header img {
    height: 120px;
    margin-bottom: 0.5rem;
}
.datumly-subtitle {
    color: var(--dim);
    font-size: 0.85rem;
    letter-spacing: 0.5px;
    margin-top: 0.25rem;
}
.datumly-metrics-row {
    font-size: 0.75rem;
    color: var(--dim);
    letter-spacing: 2px;
    margin-top: 0.4rem;
}

/* ── App title ── */
.app-title {
    text-align: center;
    margin: 0.5rem 0 1.5rem;
}
.app-title h1 {
    font-family: 'DM Sans', sans-serif;
    font-size: 2rem;
    font-weight: 700;
    color: var(--text);
    margin: 0;
}
.app-title h1 span {
    color: var(--pink);
}
.app-title p {
    color: var(--dim);
    font-size: 0.9rem;
    margin: 0.25rem 0 0;
}

/* ── Stat cards ── */
.stat-cards {
    display: flex;
    gap: 1rem;
    margin: 1.5rem 0;
}
.stat-card {
    flex: 1;
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 1.2rem 1.4rem;
    text-align: left;
}
.stat-card .stat-icon {
    font-size: 1.1rem;
    margin-right: 0.4rem;
    opacity: 0.7;
}
.stat-card .stat-value {
    font-size: 1.8rem;
    font-weight: 700;
    color: var(--text);
    line-height: 1.1;
    margin: 0.3rem 0 0.15rem;
}
.stat-card .stat-value.pink {
    color: var(--pink);
}
.stat-card .stat-label {
    font-size: 0.78rem;
    color: var(--dim);
    font-weight: 500;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}
.stat-card .stat-sub {
    font-size: 0.72rem;
    color: var(--dim);
    margin-top: 0.15rem;
}

/* ── Section headers ── */
.section-hdr {
    font-family: 'DM Sans', sans-serif;
    font-size: 1.15rem;
    font-weight: 600;
    color: var(--text);
    margin: 2rem 0 0.75rem;
    padding-bottom: 0.5rem;
    border-bottom: 1px solid var(--border);
}
.section-hdr .ico {
    margin-right: 0.5rem;
}

/* ── Transfer colours ── */
.transfer-out { color: var(--red); font-weight: 600; }
.transfer-in  { color: var(--green); font-weight: 600; }
.captain-pick { color: var(--gold); font-weight: 700; }

/* ── Instructions panel ── */
.instructions {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 1.2rem 1.5rem;
    margin: 1rem 0;
}
.instructions h4 {
    margin: 0 0 0.75rem;
    color: var(--text);
    font-size: 1rem;
}
.instructions ol {
    color: var(--dim);
    font-size: 0.85rem;
    padding-left: 1.2rem;
    margin: 0;
}
.instructions li { margin-bottom: 0.3rem; }
.instructions .warn {
    color: var(--gold);
    font-weight: 600;
    font-size: 0.82rem;
    margin-top: 0.75rem;
}

/* ── Footer ── */
.datumly-footer {
    text-align: center;
    padding: 2rem 0 1.5rem;
    color: var(--dim);
    font-size: 0.8rem;
    border-top: 1px solid var(--border);
    margin-top: 3rem;
}

/* ══════════════════════════════════════════════════════════
   STREAMLIT COMPONENT OVERRIDES
   ══════════════════════════════════════════════════════════ */

/* ── Metric cards ── */
div[data-testid="stMetric"],
[data-testid="stMetricValue"] {
    background-color: var(--card) !important;
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
    padding: 14px 18px !important;
}
[data-testid="stMetricLabel"] {
    color: var(--dim) !important;
    font-size: 0.78rem !important;
    text-transform: uppercase !important;
    letter-spacing: 0.5px !important;
}
[data-testid="stMetricValue"] {
    color: var(--text) !important;
    font-weight: 700 !important;
    border: none !important;
    padding: 0 !important;
}

/* ── Tabs ── */
.stTabs [data-baseweb="tab-highlight"] {
    background-color: transparent !important;
}
.stTabs [data-baseweb="tab-border"] {
    display: none !important;
}
.stTabs [data-baseweb="tab-list"] {
    gap: 0px !important;
    background: var(--card) !important;
    border-radius: 8px !important;
    padding: 4px !important;
    border: 1px solid var(--border) !important;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 6px !important;
    color: var(--dim) !important;
    font-weight: 500 !important;
    padding: 8px 20px !important;
    background: transparent !important;
}
.stTabs [aria-selected="true"] {
    background: var(--pink) !important;
    color: white !important;
    border-radius: 6px !important;
}

/* ── Buttons ── */
.stButton > button,
button[kind="primary"],
[data-testid="stBaseButton-primary"] {
    background-color: var(--pink) !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    font-size: 0.95rem !important;
    padding: 0.6rem 1.5rem !important;
}
.stButton > button:hover,
button[kind="primary"]:hover,
[data-testid="stBaseButton-primary"]:hover {
    background-color: var(--pink-lt) !important;
    border: none !important;
}

/* ── Number input ── */
.stNumberInput input,
[data-testid="stNumberInput"] input {
    background-color: var(--card2) !important;
    color: var(--text) !important;
    border: 1px solid var(--border) !important;
    border-radius: 6px !important;
}
.stNumberInput label,
[data-testid="stNumberInput"] label {
    color: var(--dim) !important;
}

/* ── Radio buttons ── */
.stRadio > div[role="radiogroup"] {
    background: var(--card) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    padding: 4px 8px !important;
}
.stRadio label,
div[data-baseweb="radio"] label {
    color: var(--dim) !important;
}
div[data-baseweb="radio"] > div:first-child > div {
    border-color: var(--pink) !important;
}
div[data-baseweb="radio"] > div:first-child > div > div {
    background-color: var(--pink) !important;
}

/* ── Selectbox ── */
.stSelectbox label { color: var(--dim) !important; }
.stSelectbox [data-baseweb="select"],
.stSelectbox [data-baseweb="select"] > div {
    background-color: var(--card2) !important;
    border-color: var(--border) !important;
    color: var(--text) !important;
}

/* ── Slider ── */
.stSlider label { color: var(--dim) !important; }

/* ── Expanders ── */
details[data-testid="stExpander"] {
    background: var(--card) !important;
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
    margin-bottom: 0.75rem !important;
}
details[data-testid="stExpander"] summary {
    color: var(--text) !important;
    font-weight: 600 !important;
}
details[data-testid="stExpander"] summary:hover {
    color: var(--pink) !important;
}
details[data-testid="stExpander"] > div[data-testid="stExpanderDetails"] {
    border-top: 1px solid var(--border) !important;
}

/* ── Progress bar ── */
.stProgress > div > div > div {
    background-color: var(--pink) !important;
}
.stProgress > div > div {
    background-color: var(--card) !important;
}

/* ── Dataframe ── */
[data-testid="stDataFrame"] {
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    overflow: hidden !important;
}

/* ── Alerts ── */
[data-testid="stAlert"] {
    background: var(--card) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
}

/* ── Divider ── */
hr {
    border-color: var(--border) !important;
}

/* ── Sidebar ── */
section[data-testid="stSidebar"],
[data-testid="stSidebarContent"] {
    background-color: var(--card) !important;
    border-right: 1px solid var(--border) !important;
}
section[data-testid="stSidebar"] * {
    font-family: 'DM Sans', sans-serif !important;
}
section[data-testid="stSidebar"] .stMarkdown,
section[data-testid="stSidebar"] .stMarkdown p {
    color: var(--text) !important;
}
section[data-testid="stSidebar"] hr {
    border-color: var(--border) !important;
}
"""

    st.markdown(f"<style>{CSS}</style>", unsafe_allow_html=True)

    # ── Branded header ──────────────────────────────────────
    st.markdown(f"""
    <div class="datumly-header">
        <img src="data:image/png;base64,{LOGO_B64}" alt="Datumly" />
        <div class="datumly-subtitle">Data-driven FPL intelligence</div>
        <div class="datumly-metrics-row">xG &nbsp;·&nbsp; xA &nbsp;·&nbsp; CS% &nbsp;·&nbsp; xPts &nbsp;·&nbsp; FDR</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="app-title">
        <h1>⚽ FPL Transfer <span>Optimizer</span> 📊</h1>
        <p>Rolling 6-GW Strategy Planner for Fantasy Premier League</p>
    </div>
    """, unsafe_allow_html=True)

    # ── Input row: Manager ID + Run button ──────────────────
    input_col, spacer, btn_col = st.columns([2, 5, 2])
    with input_col:
        fpl_id = st.number_input(
            "FPL Manager ID:",
            min_value=1, value=210697, step=1,
            help="Find your ID at fantasy.premierleague.com/entry/YOUR_ID/history",
        )
    with btn_col:
        st.markdown("<br>", unsafe_allow_html=True)
        run_btn = st.button("🔍  Run Analysis", type="primary", use_container_width=True)

    # ── Sidebar (advanced settings) ─────────────────────────
    with st.sidebar:
        st.markdown("""
        <div style="text-align:center;padding:1rem 0 0.5rem;">
            <span style="font-size:1.3rem;font-weight:700;color:#e8eaf0;">⚙️ Settings</span>
        </div>
        """, unsafe_allow_html=True)
        horizon_gws = st.slider("Gameweeks to plan ahead", 1, 10, 6)
        risk_appetite = st.selectbox(
            "Risk appetite",
            ["safe", "balanced", "differential"],
            index=1,
        )
        dc_lookback = st.slider("DC stats lookback (GWs)", 3, 15, 8)
        budget_padding = st.number_input(
            "Budget padding (£m)", min_value=0.0, max_value=5.0,
            value=0.0, step=0.1,
        )

    # ── Main area: waiting state ────────────────────────────
    if not run_btn:
        st.markdown("""
        <div class="stat-cards">
            <div class="stat-card">
                <span class="stat-icon" style="color:#e8366d;">↑</span>
                <div class="stat-value pink">—</div>
                <div class="stat-label">Free Transfers</div>
            </div>
            <div class="stat-card">
                <span class="stat-icon">🛡️</span>
                <div class="stat-value">—</div>
                <div class="stat-label">Chips Available</div>
                <div class="stat-sub">Run analysis to see</div>
            </div>
            <div class="stat-card">
                <span class="stat-icon">💰</span>
                <div class="stat-value">£ —</div>
                <div class="stat-label">ITB (£m)</div>
            </div>
            <div class="stat-card">
                <span class="stat-icon">📅</span>
                <div class="stat-value">GW —</div>
                <div class="stat-label">Next GW</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="instructions">
            <h4>👥 Instructions</h4>
            <ol>
                <li>Enter your FPL Manager ID above.</li>
                <li>Adjust settings in the sidebar and click <strong>"Run Analysis"</strong>.</li>
            </ol>
            <div class="warn">⚠️ Note: <strong>Ensure your ID is correct!</strong></div>
        </div>
        """, unsafe_allow_html=True)
        return

    # ── Data loading ────────────────────────────────────────
    progress = st.progress(0, text="Loading FPL data …")

    try:
        boot    = get_bootstrap()
        players = build_fpl_table(boot)
    except Exception as e:
        st.error(f"Failed to load FPL data: {e}")
        return
    progress.progress(0.10, text=f"{len(players)} players loaded.")

    try:
        picks_data, active_gw, next_gw = get_my_team(fpl_id, boot)
    except Exception as e:
        st.error(f"Could not fetch team for manager {fpl_id}: {e}")
        return
    picks = pd.DataFrame(picks_data["picks"])
    itb   = picks_data["entry_history"]["bank"]  / 10
    tv    = picks_data["entry_history"]["value"] / 10
    progress.progress(0.15, text=f"Team loaded. GW{active_gw} → planning from GW{next_gw}")

    free_transfers, chips_available = get_free_transfers(fpl_id, active_gw)
    progress.progress(0.20, text=f"{free_transfers} free transfers banked.")

    fixtures = get_fixtures()
    fix_df   = compute_fixture_difficulty(boot, fixtures, active_gw, horizon_gws)
    progress.progress(0.25, text="Fixtures analysed.")

    team_def = get_team_defensive_stats(boot)
    cs_map   = team_def.set_index("team_id")["cs_prob"].to_dict()
    progress.progress(0.30, text="Clean sheet probabilities computed.")

    progress.progress(0.32, text="Fetching DC stats (this takes ~30s) …")
    dc_bar = st.progress(0, text="Fetching DC stats …")
    dc_df  = fetch_dc_stats(boot, active_gw, lookback=dc_lookback, progress_bar=dc_bar)
    dc_bar.empty()
    progress.progress(0.70, text="DC stats complete.")

    progress.progress(0.72, text="Fetching Understat xG/xA data …")
    us_df = get_understat_stats()
    progress.progress(0.76, text="Understat loaded.")

    progress.progress(0.77, text="Fetching FBref xG/xA data (StatsBomb model) …")
    fb_df = get_fbref_stats()
    progress.progress(0.80, text=f"FBref: {len(fb_df)} players loaded." if not fb_df.empty else "FBref: unavailable.")

    progress.progress(0.82, text="Building player pool …")
    fix_map = fix_df.set_index("team_id")[["avg_difficulty","num_fixtures","has_dgw"]].to_dict("index")
    players["avg_fix_diff"] = players["team_id"].map(lambda t: fix_map.get(t, {}).get("avg_difficulty", 3.0))
    players["num_fixtures"] = players["team_id"].map(lambda t: fix_map.get(t, {}).get("num_fixtures", horizon_gws))
    players["has_dgw"]   = players["team_id"].map(lambda t: fix_map.get(t, {}).get("has_dgw", 0))
    players["fix_score"] = (6 - players["avg_fix_diff"]) * (players["num_fixtures"] / horizon_gws)
    players["cs_prob"]   = players["team_id"].map(cs_map).fillna(0.25)

    players = players.merge(dc_df, on="fpl_id", how="left")
    for col in ["dc_pts_p90", "dc_hit_rate", "cbit_p90"]:
        players[col] = players[col].fillna(0) if col in players.columns else 0.0

    players = match_understat(players, us_df)
    players = match_fbref(players, fb_df)
    players = merge_dual_source_xg(players)
    players = add_risk_flags(players)
    players = apply_position_scores(players)
    players = apply_risk_appetite(players, risk_appetite)

    my_ids     = set(int(x) for x in picks["element"].tolist())
    my_team_df = players[players["fpl_id"].isin(my_ids)].copy().reset_index(drop=True)

    sell_price_map = compute_sell_prices(my_ids, players, fpl_id)
    if sell_price_map:
        my_team_df["sell_price"] = my_team_df["fpl_id"].map(sell_price_map)
        my_team_df["sell_price"] = my_team_df["sell_price"].fillna(my_team_df["price"])
    else:
        my_team_df["sell_price"] = my_team_df["price"]

    progress.progress(0.90, text="Running rolling plan …")

    plan = run_rolling_plan(
        players_base=players, boot=boot, fixtures=fixtures, cs_map=cs_map,
        my_team_df=my_team_df, squad_ids=my_ids, itb=itb,
        free_transfers=free_transfers, next_gw=next_gw,
        horizon_gws=horizon_gws, risk_appetite=risk_appetite,
        budget_padding=budget_padding,
        team_def_df=team_def,
    )

    progress.progress(1.0, text="Analysis complete!")
    progress.empty()

    # ════════════════════════════════════════════════════════
    #  DISPLAY RESULTS
    # ════════════════════════════════════════════════════════

    chips_str = ", ".join(chips_available) if chips_available else "None"
    st.markdown(f"""
    <div class="stat-cards">
        <div class="stat-card">
            <span class="stat-icon" style="color:#e8366d;">↑</span>
            <div class="stat-value pink">{free_transfers}</div>
            <div class="stat-label">Free Transfers</div>
        </div>
        <div class="stat-card">
            <span class="stat-icon">🛡️</span>
            <div class="stat-value" style="font-size:1.4rem;">{len(chips_available)}</div>
            <div class="stat-label">Chips Available</div>
            <div class="stat-sub">{chips_str}</div>
        </div>
        <div class="stat-card">
            <span class="stat-icon">💰</span>
            <div class="stat-value">£{itb:.1f}</div>
            <div class="stat-label">ITB (£m)</div>
            <div class="stat-sub">+£{tv:.1f}m TV</div>
        </div>
        <div class="stat-card">
            <span class="stat-icon">📅</span>
            <div class="stat-value">GW{next_gw}</div>
            <div class="stat-label">Next GW</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Tabbed content ──────────────────────────────────────
    tab_squad, tab_fixtures, tab_insights = st.tabs(["👥 Your Squad ⚽", "Fixture Difficulty 📅", "Insights 📊"])

    with tab_squad:
        show_cols = ["web_name","team_short","price","sell_price","form","xPts","cs_prob",
                     "xg_p90","xa_p90","dc_hit_rate","ownership","fpl_value_score","rotation_risk"]
        show_cols = [c for c in show_cols if c in my_team_df.columns]

        pos_filter = st.radio("Position filter", ["ALL","GKP","DEF","MID","FWD"],
                              horizontal=True, label_visibility="collapsed")

        filtered = my_team_df if pos_filter == "ALL" else my_team_df[my_team_df["position"] == pos_filter]

        if not filtered.empty:
            display = filtered[show_cols].copy()
            nice = {"web_name":"Player","team_short":"Team","price":"Price","sell_price":"Sell",
                    "form":"Form","xPts":"xPts","cs_prob":"CS%","xg_p90":"xG/90","xa_p90":"xA/90",
                    "dc_hit_rate":"DC%","ownership":"Own%","fpl_value_score":"Score","rotation_risk":"Risk"}
            display = display.rename(columns={c: nice.get(c,c) for c in display.columns})
            st.dataframe(display, use_container_width=True, hide_index=True)

    with tab_fixtures:
        st.markdown(f"#### 📅 Fixture Difficulty — GW{active_gw+1} to GW{active_gw+horizon_gws}")
        fs = fix_df.sort_values("avg_difficulty")
        col_easy, col_hard = st.columns(2)
        with col_easy:
            st.markdown("**🟢 Easiest Runs**")
            easy = fs[["name","avg_difficulty","num_fixtures","has_dgw"]].head(8).copy()
            easy.columns = ["Team","Avg Diff","Fixtures","DGW"]
            st.dataframe(easy, use_container_width=True, hide_index=True)
        with col_hard:
            st.markdown("**🔴 Hardest Runs**")
            hard = fs[["name","avg_difficulty","num_fixtures","has_dgw"]].tail(8).copy()
            hard.columns = ["Team","Avg Diff","Fixtures","DGW"]
            st.dataframe(hard, use_container_width=True, hide_index=True)

    with tab_insights:
        st.markdown(f"#### 🛡️ Top DC Earners (last {dc_lookback} GWs)")
        dc_col1, dc_col2 = st.columns(2)
        for col_target, (pos, label) in zip([dc_col1, dc_col2], [("DEF","10 CBIT"), ("MID","12 CBIRT")]):
            sub = players[(players["position"]==pos) & (players["status"].isin(["a","d","n"])) & (players["minutes"]>=180)].sort_values("dc_hit_rate", ascending=False).head(8)
            if sub.empty: continue
            dc_show = [c for c in ["web_name","team_short","price","dc_hit_rate","dc_pts_p90","cbit_p90","cs_prob","rotation_risk"] if c in sub.columns]
            with col_target:
                st.markdown(f"**{pos}** — threshold: {label}/match")
                st.dataframe(sub[dc_show].copy().rename(columns={c: c.replace("_"," ").title() for c in dc_show}), use_container_width=True, hide_index=True)

    # ── Rolling transfer plan ───────────────────────────────
    st.markdown(f"#### 📋 Rolling {horizon_gws}-GW Transfer Plan (GW{next_gw} → GW{next_gw + horizon_gws - 1})")

    for gw_plan in plan:
        gw = gw_plan["gw"]
        hit_cost = gw_plan.get("hit_cost", 0)
        hit_str = f" | ⚠️ -{hit_cost}pt hit" if hit_cost > 0 else ""
        with st.expander(
            f"**GW{gw}** — {gw_plan['transfers_made']} transfer{'s' if gw_plan['transfers_made'] != 1 else ''} | FT: {gw_plan['free_transfers']}{hit_str} | ITB: £{gw_plan['itb']:.1f}m",
            expanded=(gw_plan["transfers_made"] > 0),
        ):
            if not gw_plan["transfers"]:
                st.success(f"No beneficial transfers — hold for GW{gw}.")
            else:
                if hit_cost > 0:
                    st.warning(f"💰 Taking a **-{hit_cost}pt hit** ({gw_plan.get('paid_transfers', 0)} paid transfer{'s' if gw_plan.get('paid_transfers', 0) != 1 else ''}) — projected net gain exceeds cost over the planning horizon.")
                for i, t in enumerate(gw_plan["transfers"], 1):
                    c1, c2, c3 = st.columns([5, 1, 5])
                    with c1:
                        st.markdown(f'''<span class="transfer-out">⬇ OUT: {t["OUT"]}</span> [{t["OUT_pos"]}] sell £{t["OUT_sell_price"]:.1f}m (mkt £{t["OUT_buy_price"]:.1f}m)<br>ep: {t["OUT_ep_next"]:.2f} · DC: {t["OUT_dc_hit_rate"]:.0%} · fix: {t["OUT_fix_diff"]:.2f} · {t["OUT_risk"]}''', unsafe_allow_html=True)
                    with c2:
                        st.markdown("### →")
                    with c3:
                        st.markdown(f'''<span class="transfer-in">⬆ IN: {t["IN"]}</span> [{t["IN_pos"]}] ({t["IN_team"]}) £{t["IN_price"]:.1f}m<br>ep: {t["IN_ep_next"]:.2f} · **xPts: {t["IN_xPts"]:.2f}** · xG: {t["IN_xg_p90"]:.3f} · xA: {t["IN_xa_p90"]:.3f} · DC: {t["IN_dc_hit_rate"]:.0%} · CS: {t["IN_cs_prob"]:.0%}<br>**Score gain: {t["score_gain"]:+.3f}** · Price diff: {t["price_diff"]:+.1f}m''', unsafe_allow_html=True)
                    if i < len(gw_plan["transfers"]):
                        st.divider()

            if gw_plan["captain_options"]:
                st.markdown(f"**🎖️ Captain options for GW{gw}:**")
                cap_df = pd.DataFrame(gw_plan["captain_options"])
                cap_df.columns = ["Player","Team","Proj Pts","Fix Diff","xG/90","xA/90","CS Prob","Risk"]
                st.dataframe(cap_df, use_container_width=True)

    st.markdown("""
    <div class="datumly-footer">
        Built with ⚽ and ❤️ by <strong>Datumly</strong>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
