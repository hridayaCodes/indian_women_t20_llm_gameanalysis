"""
Script: validate_claims.py
Goal: Approximate validation of LLM responses against ground-truth statistics to estimate fabrication rate.
"""

from pathlib import Path
import json
import re
import pandas as pd
import difflib
from typing import List, Dict, Any


# Data paths
ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = "/Users/hridayamurudkar/Downloads/cwc-odi-2025-india/all_stats_combined.xlsx"
RESPONSES_FILE = ROOT / "Phase2_DataCollection" / "results" / "raw_responses.jsonl"
ANALYSIS_DIR = Path(__file__).resolve().parent / "analysis"
OUT_CSV = ANALYSIS_DIR / "fabrication_flags.csv"


# Require at least two capitalized words (e.g., "Virat Kohli") to reduce false positives
NAME_REGEX = re.compile(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b")
INT_OR_FLOAT_REGEX = re.compile(r"\b\d+(?:\.\d+)?\b")


def find_numeric_claims(text: str):
    """Return list of tuples (matched_string, start, end, float_value)"""
    matches = []
    for m in INT_OR_FLOAT_REGEX.finditer(text or ""):
        s = m.group(0)
        try:
            if '.' in s:
                val = float(s)
            else:
                val = int(s)
        except Exception:
            continue
        matches.append((s, m.start(), m.end(), val))
    return matches


def load_dataframe(data_path: str) -> pd.DataFrame:
    try:
        df = pd.read_excel(data_path)
        return df
    except FileNotFoundError:
        raise
    except Exception as e:
        raise


def extract_player_set(df: pd.DataFrame) -> set:
    # Try common columns
    for col in ["Player", "player_name", "player"]:
        if col in df.columns:
            names = df[col].dropna().astype(str).str.strip().unique().tolist()
            return set(names)
    # fallback: any column containing 'player'
    for c in df.columns:
        if 'player' in c.lower():
            names = df[c].dropna().astype(str).str.strip().unique().tolist()
            return set(names)
    return set()


def get_max_values(df: pd.DataFrame) -> Dict[str, Any]:
    max_runs = None
    max_wkts = None
    # check multiple possible run/wicket column names
    for col in ["Runs", "runs_scored", "RunsScored", "runs"]:
        if col in df.columns:
            try:
                max_runs = int(pd.to_numeric(df[col], errors='coerce').max(skipna=True))
                break
            except Exception:
                pass
    for col in ["Wkts", "wickets", "Wickets", "wkts"]:
        if col in df.columns:
            try:
                max_wkts = int(pd.to_numeric(df[col], errors='coerce').max(skipna=True))
                break
            except Exception:
                pass
    # also attempt to get max economy if present
    max_economy = None
    for col in ["Economy", "economy", "eco", "economy_rate", "econ"]:
        if col in df.columns:
            try:
                max_economy = float(pd.to_numeric(df[col], errors='coerce').max(skipna=True))
                break
            except Exception:
                pass
    return {"max_runs": max_runs, "max_wkts": max_wkts, "max_economy": max_economy}


def load_responses(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Responses file not found: {path}")
    res = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                res.append(json.loads(line))
            except json.JSONDecodeError:
                print("Warning: skipping invalid JSON line in responses")
    return res


def find_candidate_names(text: str) -> List[str]:
    return NAME_REGEX.findall(text or "")


def find_integers(text: str) -> List[int]:
    # return integers parsed from any integer/float-like tokens
    return [int(float(x)) for x in INT_OR_FLOAT_REGEX.findall(text or "")]


def analyze():
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)

    df = load_dataframe(DATA_PATH)
    players_set = extract_player_set(df)
    # Prepare lowercase list for fuzzy matching
    players_list = sorted(list(players_set))
    players_lower = [p.lower() for p in players_list]
    max_vals = get_max_values(df)
    max_runs = max_vals.get("max_runs")
    max_wkts = max_vals.get("max_wkts")
    max_economy = max_vals.get("max_economy")

    responses = load_responses(RESPONSES_FILE)

    records = []

    total_unknown = 0
    total_extreme = 0

    for rec in responses:
        response_id = rec.get("response_id")
        model_name = rec.get("model_name")
        hypothesis_id = rec.get("hypothesis_id")
        condition = rec.get("condition")
        text = rec.get("response_text") or ""

        # A) Unknown players with fuzzy matching
        candidates = find_candidate_names(text)
        # Filter out common words that are not player-like (optional small blacklist)
        blacklist = set(['India', 'England', 'Australia', 'India.', 'India,'])
        candidates = [c for c in candidates if c not in blacklist]

        unknown_players = []
        for c in candidates:
            c_clean = c.strip().strip('.,')
            c_lower = c_clean.lower()
            # exact case-insensitive match
            if c_lower in players_lower:
                continue
            # fuzzy close match (use difflib on lowercase names)
            matches = difflib.get_close_matches(c_lower, players_lower, n=1, cutoff=0.75)
            if matches:
                # matched_name = players_list[players_lower.index(matches[0])]
                # treat as known if a close match exists
                continue
            # otherwise treat as unknown
            unknown_players.append(c_clean)

        has_unknown = len(unknown_players) > 0
        if has_unknown:
            total_unknown += 1

        # B) Numeric claims with context attribution
        numeric_matches = find_numeric_claims(text)
        extreme_count = 0
        extreme_details = []

        # keywords to attribute stat types (lowercase)
        runs_kw = ["run", "runs", "score", "scored", "total", "highest"]
        wkts_kw = ["wicket", "wickets", "wkts", "dismissals"]
        econ_kw = ["economy", "eco", "economy_rate", "economy rate"]

        for s, start, end, val in numeric_matches:
            # grab context window (40 chars each side)
            left = max(0, start - 40)
            right = min(len(text), end + 40)
            window = text[left:right].lower()

            # decide stat type by presence of keywords in the window
            stat_type = None
            if any(k in window for k in runs_kw):
                stat_type = "runs"
            elif any(k in window for k in wkts_kw):
                stat_type = "wkts"
            elif any(k in window for k in econ_kw):
                stat_type = "economy"
            else:
                # fallback heuristics: if the number is small (<=10) prefer wickets
                if isinstance(val, (int,)) and val <= 10:
                    stat_type = "wkts"
                else:
                    stat_type = "runs"

            is_extreme = False
            if stat_type == "runs" and (max_runs is not None):
                if val > (max_runs + 50):
                    is_extreme = True
            if stat_type == "wkts" and (max_wkts is not None):
                if val > (max_wkts + 5):
                    is_extreme = True
            if stat_type == "economy" and (max_economy is not None):
                # economy is usually small; flag if it's unreasonably high (e.g., > max + 20)
                try:
                    if float(val) > (max_economy + 20):
                        is_extreme = True
                except Exception:
                    pass

            if is_extreme:
                extreme_count += 1
                extreme_details.append(f"{s}({stat_type})")

        has_extreme = extreme_count > 0
        if has_extreme:
            total_extreme += 1

        records.append({
            "response_id": response_id,
            "model_name": model_name,
            "hypothesis_id": hypothesis_id,
            "condition": condition,
            "has_unknown_player": has_unknown,
            "unknown_players": ";".join(unknown_players),
            "has_extreme_number": has_extreme,
            "num_extreme_numbers": extreme_count,
            "extreme_details": ";".join(extreme_details),
        })

    out_df = pd.DataFrame(records)
    out_df.to_csv(OUT_CSV, index=False)

    print(f"Total responses checked: {len(responses)}")
    print(f"Total with unknown players: {total_unknown}")
    print(f"Total with extreme numbers: {total_extreme}")


def main():
    analyze()


if __name__ == "__main__":
    main()
