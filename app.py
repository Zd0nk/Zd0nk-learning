# -*- coding: utf-8 -*-
"""
FPL Transfer Optimizer v5.2 — Rolling 6-GW Transfer Planner (Streamlit)
========================================================================
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
    "Leicester":               "Leicester",
    "Ipswich":                 "Ipswich",
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
    players = pd.DataFrame(boot["elements"]).copy()
    teams   = pd.DataFrame(boot["teams"])[["id","name"]].copy()

    for c in ["expected_goals_conceded","minutes","clean_sheets"]:
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

    gk_agg = gk_agg.merge(teams, left_on="team_id", right_on="id", how="left")
    return gk_agg[["team_id","name","xgc_p90","cs_rate","cs_prob"]].reset_index(drop=True)


# ══════════════════════════════════════════════════════════════════════
#  5.  UNDERSTAT
# ══════════════════════════════════════════════════════════════════════

def get_understat_stats():
    url = "https://understat.com/league/EPL/2024"
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
    Returns the dataframe with 'xPts' and 'fpl_value_score' columns.
    """
    df = df.copy().reset_index(drop=True)

    mins     = _safe(df, "minutes").replace(0, np.nan)
    total_gw = mins / 90  # approximate games played

    # Per-90 stats
    xg       = _safe(df, "xg_p90")
    xa       = _safe(df, "xa_p90")
    cs_prob  = _safe(df, "cs_prob")
    dc_rate  = _safe(df, "dc_hit_rate")
    fix_s    = _safe(df, "fix_score")  # fixture difficulty score
    saves    = _safe(df, "saves") / mins * 90  # saves per 90
    saves    = saves.fillna(0)
    bonus_p90= _safe(df, "bonus") / mins * 90
    bonus_p90= bonus_p90.fillna(0)
    yc_rate  = _safe(df, "yellow_cards") / mins * 90
    yc_rate  = yc_rate.fillna(0)
    ep_next  = _safe(df, "ep_next")  # FPL's own short-term estimate
    xgc_rate = _safe(df, "goals_conceded") / mins * 90
    xgc_rate = xgc_rate.fillna(1.0)

    # Goal and CS points by position
    goal_pts = df["position"].map({"GKP": 6, "DEF": 6, "MID": 5, "FWD": 4}).fillna(4)
    cs_pts   = df["position"].map({"GKP": 4, "DEF": 4, "MID": 1, "FWD": 0}).fillna(0)
    gc_pen   = df["position"].map({"GKP": -0.5, "DEF": -0.5, "MID": 0, "FWD": 0}).fillna(0)

    # ── Expected FPL points per match ──
    xPts = (
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

    # Fixture difficulty adjustment: scale by fixture_score
    # fix_score is higher for easier fixtures; normalise around 1.0
    fix_mean = fix_s.mean()
    if fix_mean > 0:
        fix_mult = 0.7 + 0.3 * (fix_s / fix_mean)  # range ~0.7 to ~1.3
    else:
        fix_mult = 1.0

    xPts_adjusted = xPts * fix_mult

    # Blend with FPL's own ep_next (short-term prediction) — gives
    # weight to injury/rotation info that our model can't see
    # 70% our model, 30% FPL ep_next
    ep_has_data = ep_next > 0
    blended = xPts_adjusted.copy()
    blended[ep_has_data] = (
        0.70 * xPts_adjusted[ep_has_data] +
        0.30 * ep_next[ep_has_data]
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

def rescore_for_gw(players_base, boot, fixtures, target_gw, cs_map, risk_appetite):
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
    df["cs_prob"]   = df["team_id"].map(cs_map).fillna(0.25)

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

    max_mins = df["minutes"].max()
    if max_mins > 0:
        df["mins_factor"] = (df["minutes"] / max_mins).clip(0.5, 1.0)
    else:
        df["mins_factor"] = 1.0

    proj_pts = []
    for _, p in df.iterrows():
        if not p["has_fixture"]:
            proj_pts.append(0.0)
            continue

        pos = p.get("position", "")
        fm  = float(p["fix_multiplier"])
        mf  = float(p["mins_factor"])
        cs  = float(p.get("cs_prob",     0.25))
        xg  = float(p.get("xg_p90",     0.0))
        xa  = float(p.get("xa_p90",     0.0))
        dc  = float(p.get("dc_hit_rate", 0.0))
        sv  = float(p.get("saves",       0.0))
        sm  = max(float(p.get("minutes", 90.0)), 1.0)

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

        proj_pts.append(base * fm * mf)

    df["gw_proj_pts"] = proj_pts
    return df


# ══════════════════════════════════════════════════════════════════════
#  11.  TRANSFER ENGINE
# ══════════════════════════════════════════════════════════════════════

def suggest_transfers_for_gw(squad_ids, scored_players, my_team_df, itb,
                              num_transfers=2, budget_padding=0.0):
    available = scored_players[
        (~scored_players["fpl_id"].isin(squad_ids)) &
        (scored_players["status"].isin(["a","d"]))
    ].copy()

    team_counts = my_team_df["team_id"].value_counts().to_dict()

    results = []
    for _, out_p in my_team_df.iterrows():
        # Use actual sell price (from picks endpoint), not buy price (now_cost).
        # FPL sell price = purchase_price + floor((now_cost - purchase_price) / 2)
        # The picks endpoint provides this as 'selling_price'.
        sell     = float(out_p.get("sell_price", out_p["price"]))
        buy_price = float(out_p["price"])  # current market price (for display)
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
            if gain > 0.05:
                results.append({
                    "OUT":             out_p["web_name"],
                    "OUT_pos":         pos,
                    "OUT_sell_price":  sell,       # what you actually get back
                    "OUT_buy_price":   buy_price,  # current market price
                    "OUT_score":       round(s_out, 3),
                    "OUT_ep_next":     float(out_p.get("ep_next", 0)),
                    "OUT_dc_hit_rate": round(float(out_p.get("dc_hit_rate", 0)), 2),
                    "OUT_fix_diff":    round(float(out_p.get("avg_fix_diff", 0)), 2),
                    "OUT_risk":        out_p.get("rotation_risk", "?"),
                    "IN":              cand["web_name"],
                    "IN_team":         cand["team_short"],
                    "IN_team_id":      int(cand["team_id"]),
                    "IN_pos":          pos,
                    "IN_price":        float(cand["price"]),  # cost to buy
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
        return [], my_team_df.copy(), squad_ids.copy()

    results_df = pd.DataFrame(results).sort_values(
        "score_gain", ascending=False
    ).reset_index(drop=True)

    selected_transfers  = []
    updated_squad       = my_team_df.copy()
    updated_squad_ids   = set(squad_ids)
    current_team_counts = dict(team_counts)

    for _ in range(num_transfers):
        best = None
        for _, row in results_df.iterrows():
            if row["OUT"] not in updated_squad["web_name"].values:
                continue
            if any(row["IN"] == prev["IN"] for prev in selected_transfers):
                continue

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

            best = row
            break

        if best is None:
            break

        selected_transfers.append(best.to_dict())

        out_fpl_id = int(
            updated_squad.loc[
                updated_squad["web_name"] == best["OUT"], "fpl_id"
            ].values[0]
        )
        in_row = scored_players[scored_players["web_name"] == best["IN"]]
        if not in_row.empty:
            in_player = in_row.iloc[[0]].copy()
            # A newly bought player's sell price = their buy price (no profit yet)
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

    return selected_transfers, updated_squad, updated_squad_ids


# ══════════════════════════════════════════════════════════════════════
#  12.  ROLLING 6-GW PLANNER
# ══════════════════════════════════════════════════════════════════════

def run_rolling_plan(players_base, boot, fixtures, cs_map, my_team_df,
                     squad_ids, itb, free_transfers, next_gw,
                     horizon_gws, risk_appetite, budget_padding):
    current_squad    = my_team_df.copy()
    current_ids      = set(squad_ids)
    current_itb      = itb
    banked_transfers = free_transfers
    plan_summary     = []

    for gw_offset in range(horizon_gws):
        target_gw     = next_gw + gw_offset
        num_transfers = min(banked_transfers, MAX_TRANSFER_BANK)

        scored = rescore_for_gw(players_base, boot, fixtures, target_gw, cs_map, risk_appetite)
        scored_squad = scored[scored["fpl_id"].isin(current_ids)].copy()

        # Carry sell_price from current_squad into the rescored squad.
        # rescore_for_gw rebuilds from players_base which has no sell_price,
        # so we need to map it back from current_squad.
        if "sell_price" in current_squad.columns:
            sp_map = current_squad.set_index("fpl_id")["sell_price"].to_dict()
            scored_squad["sell_price"] = scored_squad["fpl_id"].map(sp_map)
            scored_squad["sell_price"] = scored_squad["sell_price"].fillna(scored_squad["price"])

        transfers, updated_squad, updated_ids = suggest_transfers_for_gw(
            current_ids, scored, scored_squad, current_itb,
            num_transfers=min(num_transfers, 2),
            budget_padding=budget_padding,
        )

        transfers_made = len(transfers)

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
            "free_transfers":  num_transfers,
            "itb":             round(current_itb, 1),
            "transfers":       transfers,
            "transfers_made":  transfers_made,
            "captain_options": cap_options,
        })

        current_squad = updated_squad
        current_ids   = updated_ids
        banked_transfers = min(
            banked_transfers - transfers_made + 1,
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
        page_title="FPL Transfer Optimizer v5.2",
        page_icon="⚽",
        layout="wide",
    )

    st.markdown("""
    <style>
    .stApp { background-color: #0e1117; }
    .transfer-out { color: #ff6b6b; font-weight: 600; }
    .transfer-in  { color: #51cf66; font-weight: 600; }
    .captain-pick { color: #ffd43b; font-weight: 700; }
    div[data-testid="stMetric"] {
        background-color: #1a1f2e;
        border: 1px solid #2d3348;
        border-radius: 8px;
        padding: 12px 16px;
    }
    </style>
    """, unsafe_allow_html=True)

    st.title("⚽ FPL Transfer Optimizer v5.2")
    st.caption(
        "Rolling 6-GW transfer planner — corrected scoring per official 2025/26 FPL rules  ·  "
        "CS=4pts DEF/GKP  ·  CS=1pt MID  ·  DC=2pts  ·  Captain projection uses cs×4 (not cs×6)"
    )

    # ── Sidebar ─────────────────────────────────────────────
    with st.sidebar:
        st.header("Settings")
        fpl_id = st.number_input(
            "FPL Manager ID",
            min_value=1, value=210697, step=1,
            help="Find your ID in the URL: fantasy.premierleague.com/entry/YOUR_ID/history"
        )
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

        st.divider()
        st.subheader("v5.3 xPts Model")
        st.markdown("""
        **Expected points per match** based on official scoring:

        | Action | GKP | DEF | MID | FWD |
        |--------|-----|-----|-----|-----|
        | **Appearance** | 2pts | 2pts | 2pts | 2pts |
        | **CS** | 4pts | 4pts | 1pt | — |
        | **Goal** | 6pts | 6pts | 5pts | 4pts |
        | **Assist** | 3pts | 3pts | 3pts | 3pts |
        | **DC** | — | 2pts | 2pts | 2pts |
        | **3 Saves** | 1pt | — | — | — |
        | **Yellow** | -1pt | -1pt | -1pt | -1pt |

        xPts = appearance + (goal_pts × xG/90) + (3 × xA/90) +
        (cs_pts × cs_prob) + (2 × dc_rate) + bonus − yellows

        Then adjusted for fixture difficulty and blended 70/30
        with FPL's own ep_next.
        """)

        run_btn = st.button("▶  RUN ANALYSIS", type="primary", use_container_width=True)

    # ── Main area ───────────────────────────────────────────
    if not run_btn:
        st.info("Enter your FPL Manager ID in the sidebar and click **RUN ANALYSIS** to start.")
        return

    progress = st.progress(0, text="Loading FPL data …")

    # [1] Bootstrap
    try:
        boot    = get_bootstrap()
        players = build_fpl_table(boot)
    except Exception as e:
        st.error(f"Failed to load FPL data: {e}")
        return
    progress.progress(0.10, text=f"{len(players)} players loaded.")

    # [2] Team
    try:
        picks_data, active_gw, next_gw = get_my_team(fpl_id, boot)
    except Exception as e:
        st.error(f"Could not fetch team for manager {fpl_id}: {e}")
        return
    picks = pd.DataFrame(picks_data["picks"])
    itb   = picks_data["entry_history"]["bank"]  / 10
    tv    = picks_data["entry_history"]["value"] / 10
    progress.progress(0.15, text=f"Team loaded. GW{active_gw} → planning from GW{next_gw}")

    # [2b] Free transfers
    free_transfers, chips_available = get_free_transfers(fpl_id, active_gw)
    progress.progress(0.20, text=f"{free_transfers} free transfers banked.")

    # [3] Fixtures
    fixtures = get_fixtures()
    fix_df   = compute_fixture_difficulty(boot, fixtures, active_gw, horizon_gws)
    progress.progress(0.25, text="Fixtures analysed.")

    # [4] CS probability
    team_def = get_team_defensive_stats(boot)
    cs_map   = team_def.set_index("team_id")["cs_prob"].to_dict()
    progress.progress(0.30, text="Clean sheet probabilities computed.")

    # [5] DC stats
    progress.progress(0.32, text="Fetching DC stats (this takes ~30s) …")
    dc_bar = st.progress(0, text="Fetching DC stats …")
    dc_df  = fetch_dc_stats(boot, active_gw, lookback=dc_lookback, progress_bar=dc_bar)
    dc_bar.empty()
    progress.progress(0.70, text="DC stats complete.")

    # [6] Understat
    progress.progress(0.72, text="Fetching Understat xG/xA data …")
    us_df = get_understat_stats()
    progress.progress(0.76, text="Understat loaded.")

    # [6b] FBref (StatsBomb/Opta — independent second source)
    progress.progress(0.77, text="Fetching FBref xG/xA data (StatsBomb model) …")
    fb_df = get_fbref_stats()
    progress.progress(0.80, text=f"FBref: {len(fb_df)} players loaded." if not fb_df.empty else "FBref: unavailable (continuing with Understat only).")

    # [7] Merge & score
    progress.progress(0.82, text="Building player pool …")
    fix_map = fix_df.set_index("team_id")[
        ["avg_difficulty","num_fixtures","has_dgw"]
    ].to_dict("index")
    players["avg_fix_diff"] = players["team_id"].map(
        lambda t: fix_map.get(t, {}).get("avg_difficulty", 3.0)
    )
    players["num_fixtures"] = players["team_id"].map(
        lambda t: fix_map.get(t, {}).get("num_fixtures", horizon_gws)
    )
    players["has_dgw"]   = players["team_id"].map(
        lambda t: fix_map.get(t, {}).get("has_dgw", 0)
    )
    players["fix_score"] = (6 - players["avg_fix_diff"]) * (
        players["num_fixtures"] / horizon_gws
    )
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

    # ── Compute actual sell prices from transfer history ───
    # FPL sell price ≠ buy price. If you bought at £7.0m and the player
    # rose to £7.6m, you only get £7.3m back (half profit, rounded down).
    # We calculate this from the public transfers endpoint + cost_change_start.
    sell_price_map = compute_sell_prices(my_ids, players, fpl_id)
    if sell_price_map:
        my_team_df["sell_price"] = my_team_df["fpl_id"].map(sell_price_map)
        my_team_df["sell_price"] = my_team_df["sell_price"].fillna(my_team_df["price"])
    else:
        my_team_df["sell_price"] = my_team_df["price"]

    progress.progress(0.90, text="Running rolling plan …")

    # ── Rolling plan ────────────────────────────────────────
    plan = run_rolling_plan(
        players_base   = players,
        boot           = boot,
        fixtures       = fixtures,
        cs_map         = cs_map,
        my_team_df     = my_team_df,
        squad_ids      = my_ids,
        itb            = itb,
        free_transfers = free_transfers,
        next_gw        = next_gw,
        horizon_gws    = horizon_gws,
        risk_appetite  = risk_appetite,
        budget_padding = budget_padding,
    )

    progress.progress(1.0, text="Analysis complete!")
    progress.empty()

    # ════════════════════════════════════════════════════════
    #  DISPLAY RESULTS
    # ════════════════════════════════════════════════════════

    # ── Summary metrics ─────────────────────────────────────
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Active GW", f"GW{active_gw}")
    col2.metric("Planning from", f"GW{next_gw}")
    col3.metric("Team Value", f"£{tv:.1f}m")
    col4.metric("In the Bank", f"£{itb:.1f}m")
    col5.metric("Free Transfers", free_transfers)

    if chips_available:
        st.info(f"**Chips available:** {', '.join(chips_available)}")

    # ── Your squad ──────────────────────────────────────────
    st.subheader("👥 Your Current Squad")

    show_cols = ["web_name","team_short","position","price","sell_price","form","ep_next",
                 "xPts","cs_prob","xg_p90","xa_p90","dc_hit_rate","avg_fix_diff",
                 "ownership","fpl_value_score","rotation_risk"]
    show_cols = [c for c in show_cols if c in my_team_df.columns]

    for pos in ["GKP","DEF","MID","FWD"]:
        sub = my_team_df[my_team_df["position"] == pos]
        if sub.empty:
            continue
        st.markdown(f"**{pos}**")
        display = sub[show_cols].copy()
        display.columns = [c.replace("_"," ").title() for c in display.columns]
        st.dataframe(display, use_container_width=True, hide_index=True)

    # ── Fixture overview ────────────────────────────────────
    st.subheader(f"📅 Fixture Difficulty (GW{active_gw+1}–GW{active_gw+horizon_gws})")

    fs = fix_df.sort_values("avg_difficulty")
    col_easy, col_hard = st.columns(2)
    with col_easy:
        st.markdown("**Easiest runs**")
        easy = fs[["name","avg_difficulty","num_fixtures","has_dgw"]].head(8).copy()
        easy.columns = ["Team","Avg Difficulty","Fixtures","DGW"]
        st.dataframe(easy, use_container_width=True, hide_index=True)
    with col_hard:
        st.markdown("**Hardest runs**")
        hard = fs[["name","avg_difficulty","num_fixtures","has_dgw"]].tail(8).copy()
        hard.columns = ["Team","Avg Difficulty","Fixtures","DGW"]
        st.dataframe(hard, use_container_width=True, hide_index=True)

    # ── DC leaders ──────────────────────────────────────────
    st.subheader(f"🛡️ Top DC Earners (last {dc_lookback} GWs)")
    st.caption("DEF threshold: 10 CBIT/match  |  MID threshold: 12 CBIRT/match  |  2pts per match (capped)")

    dc_col1, dc_col2 = st.columns(2)
    for col_target, (pos, label) in zip([dc_col1, dc_col2],
                                         [("DEF","10 CBIT"), ("MID","12 CBIRT")]):
        sub = players[
            (players["position"] == pos) &
            (players["status"].isin(["a","d","n"])) &
            (players["minutes"] >= 180)
        ].sort_values("dc_hit_rate", ascending=False).head(8)
        if sub.empty:
            continue
        dc_show = ["web_name","team_short","price","dc_hit_rate","dc_pts_p90",
                    "cbit_p90","cs_prob","rotation_risk"]
        dc_show = [c for c in dc_show if c in sub.columns]
        with col_target:
            st.markdown(f"**{pos}** — threshold: {label}/match")
            display = sub[dc_show].copy()
            display.columns = [c.replace("_"," ").title() for c in display.columns]
            st.dataframe(display, use_container_width=True, hide_index=True)

    # ── xG Source Comparison (Understat vs FBref) ───────────
    st.subheader("📊 xG Source Comparison — Understat vs FBref (StatsBomb)")
    st.caption(
        "Two independent xG models compared side-by-side. "
        "Large discrepancies may indicate model disagreement — treat projections with more caution. "
        "The optimizer uses a blended average of both sources."
    )

    # Show comparison for players with data from both sources
    compare_cols = ["web_name", "team_short", "position", "price",
                    "us_xg_p90", "fb_xg_p90", "xg_discrepancy",
                    "us_xa_p90", "fb_xa_p90", "xa_discrepancy",
                    "xg_p90", "xa_p90"]
    compare_cols = [c for c in compare_cols if c in players.columns]

    has_both = players[
        (players.get("us_xg_p90", 0) > 0) &
        (players.get("fb_xg_p90", 0) > 0) &
        (players["status"].isin(["a", "d"])) &
        (players["minutes"] >= 450)
    ].copy()

    if not has_both.empty and len(compare_cols) > 6:
        # Tab 1: Biggest discrepancies | Tab 2: Your squad comparison
        tab_disc, tab_squad = st.tabs(["⚠️ Biggest Discrepancies", "👥 Your Squad Comparison"])

        with tab_disc:
            top_disc = has_both.sort_values("xg_discrepancy", ascending=False).head(15)
            disc_display = top_disc[compare_cols].copy()
            disc_display.columns = [
                "Player", "Team", "Pos", "Price",
                "Understat xG/90", "FBref xG/90", "xG Gap",
                "Understat xA/90", "FBref xA/90", "xA Gap",
                "Blended xG/90", "Blended xA/90"
            ][:len(compare_cols)]
            st.dataframe(disc_display, use_container_width=True, hide_index=True)

        with tab_squad:
            squad_compare = my_team_df.copy()
            squad_compare_cols = [c for c in compare_cols if c in squad_compare.columns]
            if squad_compare_cols:
                squad_display = squad_compare[squad_compare_cols].copy()
                squad_display.columns = [
                    "Player", "Team", "Pos", "Price",
                    "Understat xG/90", "FBref xG/90", "xG Gap",
                    "Understat xA/90", "FBref xA/90", "xA Gap",
                    "Blended xG/90", "Blended xA/90"
                ][:len(squad_compare_cols)]
                st.dataframe(squad_display, use_container_width=True, hide_index=True)
            else:
                st.info("Squad comparison data not available.")

        # Source coverage stats
        us_count = int((players.get("us_xg_p90", 0) > 0).sum())
        fb_count = int((players.get("fb_xg_p90", 0) > 0).sum())
        both_count = int(((players.get("us_xg_p90", 0) > 0) & (players.get("fb_xg_p90", 0) > 0)).sum())
        sc1, sc2, sc3 = st.columns(3)
        sc1.metric("Understat coverage", f"{us_count} players")
        sc2.metric("FBref coverage", f"{fb_count} players")
        sc3.metric("Both sources", f"{both_count} players")
    else:
        st.info("FBref data unavailable for this session — using Understat only.")

    # ── Rolling transfer plan ───────────────────────────────
    st.subheader(f"📋 Rolling {horizon_gws}-GW Transfer Plan "
                 f"(GW{next_gw} → GW{next_gw + horizon_gws - 1})")

    for gw_plan in plan:
        gw = gw_plan["gw"]
        with st.expander(
            f"**GW{gw}**  —  "
            f"{gw_plan['transfers_made']} transfer{'s' if gw_plan['transfers_made'] != 1 else ''}  |  "
            f"FT: {gw_plan['free_transfers']}  |  "
            f"ITB: £{gw_plan['itb']:.1f}m",
            expanded=(gw_plan["transfers_made"] > 0),
        ):
            if not gw_plan["transfers"]:
                st.success(f"No beneficial transfers — hold for GW{gw}.")
            else:
                for i, t in enumerate(gw_plan["transfers"], 1):
                    c1, c2, c3 = st.columns([5, 1, 5])
                    with c1:
                        st.markdown(
                            f'<span class="transfer-out">OUT: {t["OUT"]}</span> '
                            f'[{t["OUT_pos"]}] sell £{t["OUT_sell_price"]:.1f}m '
                            f'(mkt £{t["OUT_buy_price"]:.1f}m)  \n'
                            f'ep: {t["OUT_ep_next"]:.2f} · '
                            f'DC: {t["OUT_dc_hit_rate"]:.0%} · '
                            f'fix: {t["OUT_fix_diff"]:.2f} · '
                            f'{t["OUT_risk"]}',
                            unsafe_allow_html=True,
                        )
                    with c2:
                        st.markdown("### →")
                    with c3:
                        st.markdown(
                            f'<span class="transfer-in">IN: {t["IN"]}</span> '
                            f'[{t["IN_pos"]}] ({t["IN_team"]}) £{t["IN_price"]:.1f}m  \n'
                            f'ep: {t["IN_ep_next"]:.2f} · '
                            f'**xPts: {t["IN_xPts"]:.2f}** · '
                            f'xG: {t["IN_xg_p90"]:.3f} · xA: {t["IN_xa_p90"]:.3f} · '
                            f'DC: {t["IN_dc_hit_rate"]:.0%} · '
                            f'CS: {t["IN_cs_prob"]:.0%} · '
                            f'own: {t["IN_own_pct"]:.1f}%  \n'
                            f'**Score gain: {t["score_gain"]:+.3f}** · '
                            f'Price diff: {t["price_diff"]:+.1f}m',
                            unsafe_allow_html=True,
                        )
                    if i < len(gw_plan["transfers"]):
                        st.divider()

            # Captain options
            if gw_plan["captain_options"]:
                st.markdown(f"**🎖️ Captain options for GW{gw}:**")
                cap_df = pd.DataFrame(gw_plan["captain_options"])
                cap_df.index = [f"{'★' if i == 0 else ' '} {i+1}" for i in range(len(cap_df))]
                cap_df.columns = ["Player","Team","Proj Pts","Fix Diff",
                                  "xG/90","xA/90","CS Prob","Risk"]
                st.dataframe(cap_df, use_container_width=True)

    st.success("Analysis complete!")


if __name__ == "__main__":
    main()
