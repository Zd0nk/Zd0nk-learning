# -*- coding: utf-8 -*-
"""
Datumly — Rolling 6-GW Transfer Planner (Streamlit)
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
        page_title="Datumly",
        page_icon="⚽",
        layout="wide",
        initial_sidebar_state="collapsed",
    )

    # ── Embedded logo (base64) ─────────────────────────────
    LOGO_B64 = "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DshFey655BL+9mMfQynFPffcw6ZNm2gECz0y4ygmyzOSJOHOO+80g8GAwERcdNFFYnpymrzM2bZtG1dffbV56KGHaLfbHLdpE+eeey6nnHKK0FrzwP33m+985zv8+Mc/5vbbb+dTn/qUeeMb3yh8qdDSGjE7gXpRLKSyh8MhH/zgB81dd97JmWeeyXlnn8PRRx8thBDceeed5rP/8s+sXr2a//2//zebNm1i/cYNNA+ik4YjY24iEELY1J1ndWSi0lcppVBCIWFRt4KDhUuNutSw02oF7YA7776TrVu3AnDOOefYZzVQdbT1UGIwGNBsWP1RlmX0+32mpqZqXanv+0xNTDEzM8P09DRaaz74wQ+aXdt3sH79el7xilcQBIFodtps3rzZfO7znyfPc2644QZOO/kUzjz9DIQxXHzxxeKMM86gqK7D+z/wASOl5NRTT+Xyyy8XYRXF9aVienq61lomecaf/umfmq1bt6KUYt26dTzlKU/hpJNOEnme0+12zRe+8AW01mzevJk/+7M/M+9617tEN+0SVouF5UApVS0ebJq62WwyHA7rhUIQBMzOz/K7v/u7ptlsMjU1xUknncQFF1xAs9kU1btkbrvtNn784x/zyU9+0hhj2LFjB81ms06NRkFU68je//73m/vuu4/BYMAZZ5zBueeey6mnniq01tw/8v7ccccd/MEf/IF5+9vfLiY7kxgMSZbQbrbrorA0tV0s/umf/snceeednHPOOVxwwQUcVY0zd955p/nqV7/qIuY8+clPZuPGjUxNTNFoNOj3+6Qy5QMf+ICZmZkhDEPWrl3Lc5/7XNrttkiSxOzZs4cvfP7z9Pt9/uqv/sr87u/+rpienKLb7xE3mwzTIbOzs/zlX/6l2bp1K8cff3xtPL1q1SohpeTee+81t9xyC1u2bOGhhx7i937v98wf/MEfCG0WinUcRt/VMcYYYwUJW5qmNKIG/aTP8ccfbzUtrRYPP/wwQgiSLGFmZoabbroJpRQnnHAC73znO4UVNdsJd1ik4oLznsI//dM/meuvu45du3Zx//33s2nTJpqNJmpa0SzaVWPsqE6zrl+/nmOPPRbf98myjEbUYJAMSLKEPXv28P3vf59ut8tFF13E6173OuGIVeAF7J7Zzetf/3rRmZgw1914Aw888AA333qzufD8p4r53jwiFgRRWCvB9yV+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"

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
        with st.expander(
            f"**GW{gw}** — {gw_plan['transfers_made']} transfer{'s' if gw_plan['transfers_made'] != 1 else ''} | FT: {gw_plan['free_transfers']} | ITB: £{gw_plan['itb']:.1f}m",
            expanded=(gw_plan["transfers_made"] > 0),
        ):
            if not gw_plan["transfers"]:
                st.success(f"No beneficial transfers — hold for GW{gw}.")
            else:
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
