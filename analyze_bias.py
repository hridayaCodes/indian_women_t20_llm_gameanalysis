"""
Script: analyze_bias.py
Goal: Analyze raw model responses for mentions, sentiment, and simple recommendation tags.
"""

from pathlib import Path
import json
from collections import Counter, defaultdict
import pandas as pd
from typing import List, Dict, Any

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


# Paths / constants
ROOT = Path(__file__).resolve().parents[1]
RESPONSES_FILE = ROOT / "Phase2_DataCollection" / "results" / "raw_responses.jsonl"
DATA_PATH = "/Users/hridayamurudkar/Downloads/cwc-odi-2025-india/all_stats_combined.xlsx"
ANALYSIS_DIR = Path(__file__).resolve().parent / "analysis"
BIAS_SUMMARY = ANALYSIS_DIR / "bias_summary.csv"
SENTIMENT_STATS = ANALYSIS_DIR / "sentiment_stats.csv"


KEYWORD_TAGS = {
    "batting": ["batting", "strike rate", "boundary", "top order", "bat"],
    "bowling": ["bowling", "spell", "over", "economy", "pace", "spin", "bowl"],
    "fielding": ["fielding", "catch", "run out", "catching"],
    "fitness": ["fitness", "conditioning", "fatigue", "fitness"],
    "mental": ["mindset", "confidence", "pressure", "focus", "mental"],
    "team": ["team strategy", "collective", "unit", "team"],
    "individual": ["individual", "player-specific", "player specific"],
}


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Responses file not found: {path}")
    records = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                print("Warning: skipping invalid JSON line")
    return records


def load_player_names(data_path: str) -> List[str]:
    # Try common player name columns
    df = pd.read_excel(data_path)
    candidate_cols = [c for c in df.columns if c.lower() in ("player", "player_name", "player_name")] + [c for c in df.columns if 'player' in c.lower()]
    # pick the first sensible column
    col = None
    for c in ["Player", "player_name", "player"]:
        if c in df.columns:
            col = c
            break
    if col is None:
        # fallback: look for any column containing 'player'
        for c in df.columns:
            if 'player' in c.lower():
                col = c
                break
    if col is None:
        print("Warning: no player column found in dataset; player matching will be empty")
        return []

    names = df[col].dropna().astype(str).unique().tolist()
    # Clean names
    names = [n.strip() for n in names if n and str(n).strip()]
    return names


def detect_players(text: str, player_names: List[str]) -> List[str]:
    text_low = text.lower()
    mentioned = []
    for name in player_names:
        if not name:
            continue
        if name.lower() in text_low:
            mentioned.append(name)
    return mentioned


def sentiment_label(compound: float) -> str:
    if compound >= 0.05:
        return "positive"
    if compound <= -0.05:
        return "negative"
    return "neutral"


def tag_recommendations(text: str) -> List[str]:
    text_low = text.lower()
    tags = set()
    for tag, keywords in KEYWORD_TAGS.items():
        for kw in keywords:
            if kw in text_low:
                tags.add(tag)
                break
    if not tags:
        return ["unspecified"]
    return sorted(tags)


def analyze():
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)

    responses = load_jsonl(RESPONSES_FILE)
    player_names = load_player_names(DATA_PATH)

    analyzer = SentimentIntensityAnalyzer()

    enriched = []

    for rec in responses:
        resp_text = (rec.get("response_text") or "")
        mentioned = detect_players(resp_text, player_names)
        vs = analyzer.polarity_scores(resp_text)
        comp = vs.get("compound", 0.0)
        s_label = sentiment_label(comp)
        tags = tag_recommendations(resp_text)

        rec_enriched = dict(rec)
        rec_enriched.update({
            "mentioned_players": mentioned,
            "num_players_mentioned": len(mentioned),
            "sentiment_compound": comp,
            "sentiment_label": s_label,
            "recommendation_tags": tags,
        })
        enriched.append(rec_enriched)

    # Build DataFrame for bias_summary
    df = pd.DataFrame(enriched)

    # Ensure columns exist
    for col in ["hypothesis_id", "condition", "model_name", "response_id", "prompt_id"]:
        if col not in df.columns:
            df[col] = None

    # Convert recommendation_tags to semicolon string
    df["recommendation_tags_str"] = df["recommendation_tags"].apply(lambda x: ";".join(x) if isinstance(x, list) else "")

    bias_cols = [
        "hypothesis_id",
        "condition",
        "model_name",
        "response_id",
        "num_players_mentioned",
        "sentiment_label",
        "recommendation_tags_str",
    ]

    bias_df = df[bias_cols]
    bias_df.to_csv(BIAS_SUMMARY, index=False)

    # Sentiment stats aggregation
    sentiment_group = df.groupby(["hypothesis_id", "condition", "model_name", "sentiment_label"]).size().reset_index(name="count")
    sentiment_group.to_csv(SENTIMENT_STATS, index=False)

    # Print summary
    combos = df[["hypothesis_id", "condition", "model_name"]].drop_duplicates().shape[0]
    print(f"Analyzed {len(df)} responses")
    print(f"Unique (hypothesis_id, condition, model_name) combos: {combos}")


def main():
    analyze()


if __name__ == "__main__":
    main()
