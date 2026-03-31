# engine/matchup_engine.py
# H2H batter-vs-bowler lookup with Bayesian fallback.
# Used in pre-match to generate 3-4 key matchup notes for the brief.
#
# Public API:
#   get_key_matchups_for_brief(our_bowlers, predicted_batters) -> list[MatchupNote]
#   get_matchup(batter, bowler)                                -> MatchupDetail
#   load_matrix()                                              -> cached DataFrame

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Optional

import pandas as pd

# ---------------------------------------------------------------------------
# PATHS
# ---------------------------------------------------------------------------

PROJ_ROOT        = Path(__file__).resolve().parent.parent
MATRIX_PATH      = PROJ_ROOT / "data" / "processed" / "matchup_matrix.parquet"
BALL_BY_BALL_PATH= PROJ_ROOT / "data" / "processed" / "ball_by_ball.parquet"
GPT2_MODEL_PATH  = PROJ_ROOT / "models" / "saved" / "matchup_gpt2"
MAX_NOTE_TOKENS  = 60

# ---------------------------------------------------------------------------
# CONSTANTS
# ---------------------------------------------------------------------------

# Minimum balls before we trust H2H data for a note.
# 8 was too low — 1 dismissal in 8 balls = 12.5% dismissal rate, pure noise.
# At 12 balls a batter has had ~2 meaningful encounters; at 20 we're approaching reliable.
MIN_BALLS_FOR_NOTE       = 12
MIN_BALLS_PHASE_NOTE     = 8     # phase buckets are smaller — still need 8 to avoid single-over noise

# Minimum dismissals to generate a "bowler dominates" note.
# 1 wicket in 12 balls looks dominant numerically but is a single event — require 2+
# unless the ball count is high enough (>= MIN_BALLS_SINGLE_DISMISSAL_OK).
MIN_DISMISSALS_FOR_DOMINANCE  = 2
MIN_BALLS_SINGLE_DISMISSAL_OK = 24   # 1 wicket OK if we have 24+ balls (4 solid encounters)

# Recency weights for season-level ball-by-ball aggregation
SEASON_RECENCY_WEIGHTS = {2025: 2.0, 2024: 1.5, 2023: 1.0}   # anything older → 0.5

# Bayesian prior (global T20 average)
PRIOR_BALLS        = 18      # equivalent prior sample size
PRIOR_SR           = 130.0   # league average SR
PRIOR_DISMISSAL_PCT= 7.5     # 1 wicket per ~13 balls

# "Clear advantage" thresholds for note generation
# bowler_adv = (dismissal_pct/100) - (sr/150), range approx -1.7 to +0.1
BOWLER_ADV_THRESHOLD   = -0.55  # bowler_adv >= -0.55 -> bowler has clear edge
BATTER_ADV_THRESHOLD   = -1.05  # bowler_adv <= -1.05 -> batter has clear edge


# ---------------------------------------------------------------------------
# DATA CLASSES
# ---------------------------------------------------------------------------

@dataclass
class MatchupDetail:
    batter:             str
    bowler:             str
    balls:              int
    runs:               int
    dismissals:         int
    sr:                 float       # raw H2H SR
    dismissal_pct:      float       # raw H2H dismissal %
    dot_pct:            float
    boundary_pct:       float
    bayes_sr:           float       # Bayesian-adjusted SR
    bayes_dismissal_pct:float       # Bayesian-adjusted dismissal %
    bowler_adv:         float       # raw advantage score
    confidence:         str         # "High" | "Medium" | "Low" | "Insufficient"
    summary:            str         # plain-English one-liner


@dataclass
class MatchupNote:
    batter:             str
    bowler:             str
    advantage:          str         # "bowler" | "batter" | "neutral"
    note:               str         # plain-English note for the brief
    confidence:         str         # "High" | "Medium" | "Low"
    balls:              int
    dismissals:         int
    batter_sr:          float


# ---------------------------------------------------------------------------
# UPGRADE 3 — GPT-2 SCOUTING NOTE GENERATOR (lazy-loaded)
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def _load_gpt2():
    """
    Load fine-tuned GPT-2 pipeline (once). Returns None if model missing or
    transformers unavailable.
    """
    if not GPT2_MODEL_PATH.exists():
        return None
    try:
        from transformers import pipeline as hf_pipeline
        return hf_pipeline(
            "text-generation",
            model=str(GPT2_MODEL_PATH),
            tokenizer=str(GPT2_MODEL_PATH),
            device=-1,     # CPU
        )
    except Exception:
        return None


def _generate_note_gpt2(
    batter: str,
    bowler: str,
    balls:  int,
    sr:     float,
    dismissals: int,
    economy:    float,
) -> Optional[str]:
    """
    Generate a scouting note via the fine-tuned GPT-2 model.
    Returns None if model not available (caller falls back to template).
    Uses greedy decoding (no sampling), max MAX_NOTE_TOKENS new tokens.
    """
    pipe = _load_gpt2()
    if pipe is None:
        return None
    prompt = (
        f"Bowler: {bowler}. "
        f"Batter: {batter}. "
        f"Balls: {balls}. "
        f"SR: {sr:.0f}. "
        f"Dismissals: {dismissals}. "
        f"Economy: {economy:.1f}. "
        f"Scouting note: "
    )
    try:
        out = pipe(
            prompt,
            max_new_tokens  = MAX_NOTE_TOKENS,
            do_sample       = False,
            num_return_sequences = 1,
            pad_token_id    = pipe.tokenizer.eos_token_id,
        )
        generated = out[0]["generated_text"][len(prompt):].strip()
        # Truncate at first sentence end
        for sep in (".", "!", "?"):
            idx = generated.find(sep)
            if idx != -1:
                generated = generated[: idx + 1].strip()
                break
        # Reject garbled output: replacement chars, leading non-ASCII, or too short
        if not generated:
            return None
        if "\ufffd" in generated:
            return None
        if len(generated) < 20:
            return None
        # Ensure the note starts with a printable ASCII character
        if not generated[0].isascii() or not generated[0].isprintable():
            return None
        return generated
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Fix 4.3 — DETERMINISTIC TEMPLATE NOTE GENERATOR
# ---------------------------------------------------------------------------

def _generate_note_template(
    batter:     str,
    bowler:     str,
    balls:      int,
    sr:         float,
    bayes_sr:   float,
    dismissals: int,
    dot_pct:    float,
    boundary_pct: float,
    adv_side:   str,        # "bowler" | "batter" | "neutral"
    confidence: str,        # "High" | "Medium" | "Low" | "Insufficient"
    phase_ctx:  str = "",   # optional phase-specific append
) -> str:
    """
    Generate a statistically grounded, deterministic scouting note.
    No ML required — always reproducible and auditable.
    """
    batter_last = batter.split()[-1]
    bowler_last  = bowler.split()[-1]
    conf_flag    = f" [{confidence} confidence — {balls} PSL balls]" if confidence != "High" else ""

    if adv_side == "bowler":
        dismiss_rate = round(dismissals / balls * 100, 1) if balls > 0 else 0.0
        dot_note     = f", {dot_pct:.0f}% dot balls" if dot_pct >= 45 else ""
        note = (
            f"{bowler_last} dominates {batter_last}: "
            f"{dismissals} wicket(s) in {balls} PSL balls (SR {sr:.0f}, {dismiss_rate}% dismissal rate{dot_note}). "
            f"Bowl {bowler_last} early to {batter_last} — this is a key wicket opportunity.{conf_flag}"
        )
    elif adv_side == "batter":
        boundary_note = f", {boundary_pct:.0f}% boundaries" if boundary_pct >= 40 else ""
        note = (
            f"{batter_last} has the edge vs {bowler_last}: "
            f"SR {sr:.0f} in {balls} PSL balls (Bayes-adj {bayes_sr:.0f}{boundary_note}). "
            f"Protect {bowler_last} — avoid this matchup or bowl {bowler_last} at non-striker end.{conf_flag}"
        )
    else:
        note = (
            f"{bowler_last} vs {batter_last}: evenly matched in {balls} PSL balls "
            f"(SR {sr:.0f}, Bayes-adj {bayes_sr:.0f}). "
            f"Match state should drive the decision over head-to-head preference.{conf_flag}"
        )

    if phase_ctx:
        note = note.rstrip(".") + f" {phase_ctx}"
    return note


# ---------------------------------------------------------------------------
# MATRIX LOADER  (cached after first load)
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def load_matrix(matrix_path: Optional[str] = None) -> pd.DataFrame:
    path = Path(matrix_path) if matrix_path else MATRIX_PATH
    return pd.read_parquet(path)


def _reload_matrix() -> None:
    """Clear cache and reload — call after pipeline re-run."""
    load_matrix.cache_clear()
    _load_phase_matchups.cache_clear()


# ---------------------------------------------------------------------------
# PHASE-SPECIFIC + RECENCY-WEIGHTED MATCHUPS (from ball_by_ball)
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def _load_phase_matchups(
    bbb_path: Optional[str] = None,
) -> tuple[dict, dict]:
    """
    Compute phase-specific H2H and recency-weighted career matchups from
    ball_by_ball.parquet. Cached after first load.

    Returns:
        phase_lookup   : {(batter, bowler, phase): {balls, runs, wickets, sr}}
        recency_lookup : {(batter, bowler): recency_bowler_adv}

    Recency weights: 2025 = 2.0×, 2024 = 1.5×, 2023 = 1.0×, older = 0.5×
    This prevents a 2016 dismissal from carrying the same weight as a 2024 one.
    """
    path = Path(bbb_path) if bbb_path else BALL_BY_BALL_PATH
    phase_lookup:   dict = {}
    recency_lookup: dict = {}

    try:
        df = pd.read_parquet(path)
    except Exception:
        return phase_lookup, recency_lookup

    # Work only on legal balls (exclude wides/no-balls from SR calculations)
    legal = df[df["legal_ball"] == True].copy()
    if legal.empty:
        return phase_lookup, recency_lookup

    # --- Phase-specific aggregation ---
    try:
        grp = (
            legal.groupby(["batter", "bowler", "phase"], observed=True)
            .agg(
                balls      = ("legal_ball",  "sum"),
                runs       = ("runs_batter", "sum"),
                wickets    = ("is_wicket",   "sum"),
            )
            .reset_index()
        )
        grp["sr"]            = (grp["runs"] / grp["balls"] * 100).round(1)
        grp["dismissal_pct"] = (grp["wickets"] / grp["balls"] * 100).round(1)

        for _, row in grp.iterrows():
            phase = str(row["phase"]).lower()
            if phase in ("powerplay", "middle", "death"):
                key = (str(row["batter"]), str(row["bowler"]), phase)
                phase_lookup[key] = {
                    "balls":         int(row["balls"]),
                    "runs":          int(row["runs"]),
                    "wickets":       int(row["wickets"]),
                    "sr":            float(row["sr"]),
                    "dismissal_pct": float(row["dismissal_pct"]),
                }
    except Exception:
        pass

    # --- Recency-weighted career totals ---
    try:
        legal["season_weight"] = legal["season"].apply(
            lambda s: SEASON_RECENCY_WEIGHTS.get(int(s), 0.5)
        )
        legal["w_runs"]    = legal["runs_batter"] * legal["season_weight"]
        legal["w_wickets"] = legal["is_wicket"]   * legal["season_weight"]

        rgrp = (
            legal.groupby(["batter", "bowler"], observed=True)
            .agg(
                w_balls   = ("season_weight", "sum"),
                w_runs    = ("w_runs",        "sum"),
                w_wickets = ("w_wickets",     "sum"),
            )
            .reset_index()
        )
        for _, row in rgrp.iterrows():
            wb = float(row["w_balls"])
            if wb <= 0:
                continue
            w_sr  = float(row["w_runs"]) / wb * 100
            w_dis = float(row["w_wickets"]) / wb * 100
            # Positive = bowler advantage, negative = batter advantage
            adv = round((PRIOR_SR - w_sr) * 0.6 + (w_dis - PRIOR_DISMISSAL_PCT) * 3.0, 2)
            recency_lookup[(str(row["batter"]), str(row["bowler"]))] = adv
    except Exception:
        pass

    return phase_lookup, recency_lookup


def _get_phase_context(
    batter: str,
    bowler: str,
    phase_lookup: dict,
) -> str:
    """
    Return the most tactically relevant phase-specific insight for a pair.
    Checks all three phases and surfaces the one with the strongest advantage.
    Returns empty string if no phase has enough balls.

    Examples:
        "PP specialist: 3 wickets in 18 PP balls (SR 72)."
        "Watch out at death: Batter hits SR 195 in 12 death balls."
    """
    PHASE_LABELS = {
        "powerplay": "PP",
        "middle":    "Mid",
        "death":     "Death",
    }
    best_note = ""
    best_abs_adv = 0.0

    for phase, short in PHASE_LABELS.items():
        data = phase_lookup.get((batter, bowler, phase))
        if data is None or data["balls"] < MIN_BALLS_PHASE_NOTE:
            continue

        sr      = data["sr"]
        wickets = data["wickets"]
        balls   = data["balls"]

        # Bowler domination: SR well below average + wickets
        if sr < 90 and wickets >= 2:
            adv = PRIOR_SR - sr + wickets * 10
            if adv > best_abs_adv:
                best_abs_adv = adv
                best_note = (
                    f"{short} weapon: {bowler.split()[-1]} takes {wickets} wickets "
                    f"in {balls} {short} balls (SR {sr:.0f})."
                )

        # Batter domination in this phase
        elif sr > 160 and balls >= MIN_BALLS_PHASE_NOTE:
            adv = sr - PRIOR_SR
            if adv > best_abs_adv:
                best_abs_adv = adv
                best_note = (
                    f"Danger {short}: {batter.split()[-1]} hits SR {sr:.0f} "
                    f"in {balls} {short} balls vs {bowler.split()[-1]}."
                )

    return best_note


# ---------------------------------------------------------------------------
# BAYESIAN SMOOTHING
# ---------------------------------------------------------------------------

def _bayes_sr(balls: int, runs: int) -> float:
    """Bayesian-smoothed SR: shrinks toward league average for small samples."""
    total_balls = balls + PRIOR_BALLS
    total_runs  = runs  + (PRIOR_SR * PRIOR_BALLS / 100)
    return round(total_runs / total_balls * 100, 1)


def _bayes_dismissal_pct(balls: int, dismissals: int) -> float:
    """Bayesian-smoothed dismissal %."""
    total_balls      = balls + PRIOR_BALLS
    total_dismissals = dismissals + (PRIOR_DISMISSAL_PCT * PRIOR_BALLS / 100)
    return round(total_dismissals / total_balls * 100, 1)


def _confidence(balls: int) -> str:
    if balls >= 36:      # 6+ meaningful encounters — genuinely reliable
        return "High"
    elif balls >= 20:    # 3-5 encounters — usable with context
        return "Medium"
    elif balls >= MIN_BALLS_FOR_NOTE:   # 12-19 balls — show with explicit caveat
        return "Low"
    return "Insufficient"


# ---------------------------------------------------------------------------
# SINGLE MATCHUP LOOKUP
# ---------------------------------------------------------------------------

def get_matchup(
    batter: str,
    bowler: str,
    matrix_path: Optional[str] = None,
) -> MatchupDetail:
    """
    Look up H2H stats for one batter-bowler pair.
    Returns a MatchupDetail with Bayesian-adjusted metrics.
    Always returns a result — uses all-prior values if pair not in data.
    """
    matrix = load_matrix(matrix_path)
    row    = matrix[(matrix["batter"] == batter) & (matrix["bowler"] == bowler)]

    if row.empty:
        # No H2H data — return pure Bayesian prior
        return MatchupDetail(
            batter             = batter,
            bowler             = bowler,
            balls              = 0,
            runs               = 0,
            dismissals         = 0,
            sr                 = PRIOR_SR,
            dismissal_pct      = PRIOR_DISMISSAL_PCT,
            dot_pct            = 35.0,
            boundary_pct       = 40.0,
            bayes_sr           = PRIOR_SR,
            bayes_dismissal_pct= PRIOR_DISMISSAL_PCT,
            bowler_adv         = 0.0,
            confidence         = "Insufficient",
            summary            = f"No H2H data ({batter} vs {bowler}) — league average assumed.",
        )

    r = row.iloc[0]
    balls       = int(r["balls"])
    runs        = int(r["runs"])
    dismissals  = int(r["dismissals"])
    sr          = float(r["sr"])
    disp_pct    = float(r["dismissal_pct"])
    dot_pct     = float(r["dot_pct"])
    bnd_pct     = float(r["boundary_pct"])
    adv         = float(r["bowler_adv"])

    b_sr   = _bayes_sr(balls, runs)
    b_disp = _bayes_dismissal_pct(balls, dismissals)
    conf   = _confidence(balls)

    # Plain-English summary
    if conf == "Insufficient":
        summary = (
            f"Only {balls} ball(s) — insufficient data. "
            f"No reliable H2H insight for {batter} vs {bowler}."
        )
    elif adv >= BOWLER_ADV_THRESHOLD:
        summary = (
            f"{bowler} dominates: {dismissals} dismissal(s) in {balls} balls "
            f"(SR {sr:.0f}). Deploy when {batter} arrives."
        )
    elif adv <= BATTER_ADV_THRESHOLD:
        summary = (
            f"{batter} has the edge: SR {sr:.0f} in {balls} balls vs {bowler}. "
            f"Consider switching bowler when {batter} faces {bowler}."
        )
    else:
        summary = (
            f"Competitive matchup: {batter} SR {sr:.0f} in {balls} balls, "
            f"{dismissals} dismissal(s). No clear advantage."
        )

    return MatchupDetail(
        batter             = batter,
        bowler             = bowler,
        balls              = balls,
        runs               = runs,
        dismissals         = dismissals,
        sr                 = sr,
        dismissal_pct      = disp_pct,
        dot_pct            = dot_pct,
        boundary_pct       = bnd_pct,
        bayes_sr           = b_sr,
        bayes_dismissal_pct= b_disp,
        bowler_adv         = adv,
        confidence         = conf,
        summary            = summary,
    )


# ---------------------------------------------------------------------------
# BULK MATCHUP TABLE  (for prep room display)
# ---------------------------------------------------------------------------

def get_matchup_table(
    batters: list[str],
    bowlers: list[str],
    min_balls: int = MIN_BALLS_FOR_NOTE,   # was 4 — raised to match brief threshold
    matrix_path: Optional[str] = None,
) -> pd.DataFrame:
    """
    Returns a DataFrame of all batter×bowler pairs with >= min_balls.
    Useful for rendering the full matchup grid in the prep room.
    """
    matrix = load_matrix(matrix_path)
    df = matrix[
        matrix["batter"].isin(batters)
        & matrix["bowler"].isin(bowlers)
        & (matrix["balls"] >= min_balls)
    ].copy()

    df["bayes_sr"]            = df.apply(lambda r: _bayes_sr(r["balls"], r["runs"]), axis=1)
    df["bayes_dismissal_pct"] = df.apply(lambda r: _bayes_dismissal_pct(r["balls"], r["dismissals"]), axis=1)
    df["confidence"]          = df["balls"].apply(_confidence)

    return df.sort_values("bowler_adv", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------------------
# KEY MATCHUPS FOR BRIEF
# ---------------------------------------------------------------------------

def get_key_matchups_for_brief(
    our_bowlers:         list[str],
    opposition_batters:  list[str],
    max_notes:           int = 4,
    matrix_path:         Optional[str] = None,
) -> list[MatchupNote]:
    """
    Identify the 3-4 most strategically important H2H matchups for the brief.

    Rules (from spec):
    - H2H sample >= 8 balls  (below this = misleading)
    - Clear advantage exists (bowler_adv >= 15 OR <= -15)
    - Prefer higher ball-count pairs (more reliable)
    - Return at most max_notes notes, balanced between bowler and batter advantages

    Returns a list of MatchupNote — plain-English, ready for the brief.
    """
    matrix = load_matrix(matrix_path)

    # Load phase-specific and recency-weighted data
    phase_lookup, recency_lookup = _load_phase_matchups(
        str(BALL_BY_BALL_PATH) if not matrix_path else None
    )

    # Filter to relevant pairs with sufficient data
    df = matrix[
        matrix["bowler"].isin(our_bowlers)
        & matrix["batter"].isin(opposition_batters)
        & (matrix["balls"] >= MIN_BALLS_FOR_NOTE)
    ].copy()

    if df.empty:
        # No pair clears the threshold — return a single informative note so the
        # brief section is never blank.  Coaches need to know WHY there are no notes.
        return [MatchupNote(
            batter     = "—",
            bowler     = "—",
            advantage  = "neutral",
            note       = (
                f"No H2H matchups with sufficient PSL data (≥{MIN_BALLS_FOR_NOTE} balls) "
                "between these squads. Use overall form and venue stats to guide decisions — "
                "do not rely on head-to-head history for this fixture."
            ),
            confidence = "Insufficient",
            balls      = 0,
            dismissals = 0,
            batter_sr  = PRIOR_SR,
        )]

    df["confidence"] = df["balls"].apply(_confidence)
    df["bayes_sr"]   = df.apply(lambda r: _bayes_sr(r["balls"], r["runs"]), axis=1)

    # Override bowler_adv with recency-weighted version when available
    # (career average from 2016 skews older seasons equally; recency version is more predictive)
    def _recency_adv(row: pd.Series) -> float:
        r_adv = recency_lookup.get((row["batter"], row["bowler"]))
        return r_adv if r_adv is not None else float(row["bowler_adv"])

    df["bowler_adv_recency"] = df.apply(_recency_adv, axis=1)

    # Dismissal guard: remove bowler-dominance rows driven by a single wicket
    # on insufficient balls. A 1-wicket note on 12 balls is one lucky delivery —
    # require 2+ dismissals OR enough balls to make 1 dismissal meaningful.
    def _passes_dismissal_guard(row: pd.Series) -> bool:
        d = int(row.get("dismissals", 0))
        b = int(row["balls"])
        if d == 0:
            return True   # batter-advantage rows always pass (0 dismissals is fine)
        if d >= MIN_DISMISSALS_FOR_DOMINANCE:
            return True   # 2+ dismissals — credible regardless of ball count
        # Exactly 1 dismissal: only pass if balls are high enough
        return b >= MIN_BALLS_SINGLE_DISMISSAL_OK

    df = df[df.apply(_passes_dismissal_guard, axis=1)].copy()

    # Split into bowler-advantage and batter-advantage matchups
    bowler_wins = df[df["bowler_adv_recency"] >= BOWLER_ADV_THRESHOLD].sort_values(
        ["bowler_adv_recency", "balls"], ascending=[False, False]
    )
    batter_wins = df[df["bowler_adv_recency"] <= BATTER_ADV_THRESHOLD].sort_values(
        ["bowler_adv_recency", "balls"], ascending=[True, False]
    )

    notes: list[MatchupNote] = []

    def _make_note(row: pd.Series, adv_side: str) -> MatchupNote:
        batter     = row["batter"]
        bowler     = row["bowler"]
        balls      = int(row["balls"])
        dismissals = int(row["dismissals"])
        sr         = float(row["sr"])
        conf       = str(row["confidence"])
        economy    = float(row["runs"]) / balls * 6 if balls > 0 else 8.0

        # Fix 4.3: Deterministic template is the primary note generator.
        # GPT-2 is used only as an optional enhancement when the model exists;
        # template notes are always auditable, reproducible, and stat-grounded.
        bayes_sr     = float(row.get("bayes_sr", sr))
        dot_pct      = float(row.get("dot_pct", 0.0))
        boundary_pct = float(row.get("boundary_pct", 0.0))
        phase_ctx    = _get_phase_context(batter, bowler, phase_lookup)

        note = _generate_note_template(
            batter=batter, bowler=bowler, balls=balls, sr=sr,
            bayes_sr=bayes_sr, dismissals=dismissals,
            dot_pct=dot_pct, boundary_pct=boundary_pct,
            adv_side=adv_side, confidence=conf, phase_ctx=phase_ctx,
        )

        # Optional GPT-2 enhancement — only replaces the template when the
        # model produces a non-trivial output (long enough to be coherent).
        gpt2_note = _generate_note_gpt2(batter, bowler, balls, sr, dismissals, economy)
        if gpt2_note and len(gpt2_note) > len(note) * 0.8:
            note = gpt2_note

        return MatchupNote(
            batter     = batter,
            bowler     = bowler,
            advantage  = adv_side,
            note       = note,
            confidence = conf,
            balls      = balls,
            dismissals = dismissals,
            batter_sr  = sr,
        )

    # Take up to ceil(max_notes * 0.6) bowler wins, rest batter wins
    n_bowler = min(len(bowler_wins), round(max_notes * 0.6))
    n_batter = min(len(batter_wins), max_notes - n_bowler)

    for _, row in bowler_wins.head(n_bowler).iterrows():
        notes.append(_make_note(row, "bowler"))
    for _, row in batter_wins.head(n_batter).iterrows():
        notes.append(_make_note(row, "batter"))

    # Fill remaining slots if under max_notes
    if len(notes) < max_notes:
        all_clear = df[
            df["bowler_adv_recency"].abs() >= BOWLER_ADV_THRESHOLD
        ].sort_values("balls", ascending=False)
        seen = {(n.batter, n.bowler) for n in notes}
        for _, row in all_clear.iterrows():
            key = (row["batter"], row["bowler"])
            if key not in seen and len(notes) < max_notes:
                side = "bowler" if row["bowler_adv_recency"] >= 0 else "batter"
                notes.append(_make_note(row, side))
                seen.add(key)

    return notes[:max_notes]


# ---------------------------------------------------------------------------
# QUICK SELF-TEST
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    matrix = load_matrix()
    print(f"Matchup matrix: {len(matrix):,} pairs loaded")
    print(f"  Pairs with >= {MIN_BALLS_FOR_NOTE} balls: {len(matrix[matrix['balls'] >= MIN_BALLS_FOR_NOTE]):,}")

    # UPGRADE 3: Report GPT-2 status
    gpt2_ready = GPT2_MODEL_PATH.exists()
    print(f"\n  GPT-2 model ({GPT2_MODEL_PATH.name}): {'LOADED' if gpt2_ready else 'NOT FOUND — using template fallback'}")
    if not gpt2_ready:
        print(f"  Run: python models/train_matchup_gpt2.py  to fine-tune")

    # Spot-check a known high-value matchup
    detail = get_matchup("Babar Azam", "Shaheen Shah Afridi")
    print(f"\nMatchup: {detail.batter} vs {detail.bowler}")
    print(f"  Balls: {detail.balls}  Runs: {detail.runs}  Dismissals: {detail.dismissals}")
    print(f"  Raw SR: {detail.sr}  |  Bayes SR: {detail.bayes_sr}")
    print(f"  Confidence: {detail.confidence}")
    print(f"  Summary: {detail.summary}")

    lahore_bowlers  = ["Shaheen Shah Afridi", "Haris Rauf", "Usama Mir", "Mustafizur Rahman"]
    karachi_batters = ["David Warner", "Moeen Ali", "Azam Khan", "Khushdil Shah", "Muhammad Waseem"]

    print(f"\nKey matchup notes (Lahore vs Karachi) [{'GPT-2' if gpt2_ready else 'template'}]:")
    notes = get_key_matchups_for_brief(lahore_bowlers, karachi_batters)
    if notes:
        for n in notes:
            print(f"  [{n.confidence}] {n.note}")
    else:
        print("  No notes with sufficient H2H data found for this combination.")

    tbl = get_matchup_table(karachi_batters, lahore_bowlers, min_balls=4)
    print(f"\nMatchup table ({len(tbl)} pairs with >= 4 balls):")
    if not tbl.empty:
        print(tbl[["batter", "bowler", "balls", "sr", "dismissals", "confidence", "bowler_adv"]].to_string(index=False))
