# utils/claude_advisor.py
# AI T20 Specialist Coach layer — multi-provider, graceful fallback.
#
# Provider priority (first key found in .env wins):
#   1. GROQ_API_KEY       → Groq cloud (Llama 3.3 70B) — FREE tier
#   2. ANTHROPIC_API_KEY  → Claude Sonnet 4.6 — paid
#   3. GEMINI_API_KEY     → Google Gemini 1.5 Flash — FREE tier
#   → No key found: all functions return None (rule-based fallback in app.py)
#
# Provides tactical recommendations for:
#   - Live dugout situation reads (every 4-5 balls)
#   - Pre-match coaching brief (synthesises XI, bowling plan, matchups, weather)
#   - On-demand analysis (opposition, batting tactics, any page)

from __future__ import annotations

import os
import re
from functools import lru_cache
from pathlib import Path as _Path

# Load .env before any provider detection so API keys are available
try:
    from dotenv import load_dotenv as _load_dotenv
    _load_dotenv(_Path(__file__).resolve().parent.parent / ".env", override=False)
except ImportError:
    pass

# ---------------------------------------------------------------------------
# SYSTEM PROMPT — T20 Specialist Head Coach persona
# ---------------------------------------------------------------------------

_T20_SYSTEM = """\
You are an elite T20 cricket specialist and head coach with 15+ years of experience \
at the highest level — PSL, IPL, BBL, and international T20 cricket. \
You have coached championship-winning sides and have deep expertise in:
- Modern power-hitting counters: switch hits, ramps, scoops, 360-degree batting
- Matchup science: exploiting bowler-vs-batter statistical edges
- Phase management: powerplay field rings, middle-over dot ball accumulation, \
death-over yorker execution
- PSL-specific conditions: Lahore's short boundaries, Karachi's large outfield, \
Rawalpindi's high-scoring pace deck, Multan's slow surface
- Dew management: spinner rotation, wet-ball grip, second-innings field settings
- Pressure moments: super overs, knockout chase psychology

Your advice style:
- DIRECT — issue instructions, not suggestions. No "you could consider…"
- SPECIFIC — name the bowler, the over, the field position, the delivery type
- CONTEXTUAL — every instruction must reference the actual numbers in front of you
- CONCISE — 2-4 sentences maximum per tactical call
- MODERN — you understand T20 as played today, not 10 years ago

Talk like a coach to a captain mid-match. No preamble, no sign-off, \
no "As a T20 specialist…" — just the call.\
"""

# ---------------------------------------------------------------------------
# PROVIDER DETECTION & CALL (lazy-loaded, one instance per process)
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def _detect_provider() -> str:
    """Return which provider to use based on available keys. Cached once."""
    if os.environ.get("GROQ_API_KEY", "").strip():
        return "groq"
    if os.environ.get("ANTHROPIC_API_KEY", "").strip():
        return "anthropic"
    if os.environ.get("GEMINI_API_KEY", "").strip():
        return "gemini"
    return "none"


@lru_cache(maxsize=1)
def _groq_client():
    try:
        from groq import Groq
        return Groq(api_key=os.environ.get("GROQ_API_KEY", "").strip())
    except Exception:
        return None


@lru_cache(maxsize=1)
def _anthropic_client():
    try:
        import anthropic
        return anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY", "").strip())
    except Exception:
        return None


def _call(prompt: str, max_tokens: int = 300) -> str | None:
    """Call whichever AI provider is configured. Returns text or None."""
    provider = _detect_provider()

    if provider == "groq":
        client = _groq_client()
        if client is None:
            return None
        try:
            resp = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                max_tokens=max_tokens,
                messages=[
                    {"role": "system", "content": _T20_SYSTEM},
                    {"role": "user",   "content": prompt},
                ],
            )
            return resp.choices[0].message.content.strip()
        except Exception:
            return None

    if provider == "anthropic":
        client = _anthropic_client()
        if client is None:
            return None
        try:
            msg = client.messages.create(
                model="claude-sonnet-4-6",
                max_tokens=max_tokens,
                system=_T20_SYSTEM,
                messages=[{"role": "user", "content": prompt}],
            )
            return msg.content[0].text.strip()
        except Exception:
            return None

    if provider == "gemini":
        try:
            import google.generativeai as genai
            genai.configure(api_key=os.environ.get("GEMINI_API_KEY", "").strip())
            model = genai.GenerativeModel(
                "gemini-1.5-flash",
                system_instruction=_T20_SYSTEM,
            )
            resp = model.generate_content(prompt)
            return resp.text.strip()
        except Exception:
            return None

    return None


# ---------------------------------------------------------------------------
# DUGOUT — live situation read (called every 4-5 balls)
# ---------------------------------------------------------------------------

_PRIORITY_RE = re.compile(r"PRIORITY\s*:\s*(CRITICAL|WARNING|INFO)", re.IGNORECASE)
_HEADLINE_RE = re.compile(r"HEADLINE\s*:\s*(.+?)(?:\n|$)", re.IGNORECASE)
_BODY_RE     = re.compile(r"BODY\s*:\s*(.+?)(?=\nALERT|\Z)", re.IGNORECASE | re.DOTALL)
_ALERT_RE    = re.compile(r"ALERT\s*:\s*(.+?)(?:\n|$)", re.IGNORECASE)


def _parse_dugout_response(text: str) -> dict[str, str]:
    """Parse structured response into badge/headline/body/alert dict."""
    priority = (_PRIORITY_RE.search(text) or ["", "INFO"])[1].upper()
    badge = "crit" if priority == "CRITICAL" else ("warn" if priority == "WARNING" else "info")

    headline_m = _HEADLINE_RE.search(text)
    body_m     = _BODY_RE.search(text)
    alert_m    = _ALERT_RE.search(text)

    headline = headline_m.group(1).strip() if headline_m else text.split("\n")[0][:120]
    body     = body_m.group(1).strip().replace("\n", " ") if body_m else ""
    alert    = alert_m.group(1).strip() if alert_m else ""

    return {"badge": badge, "headline": headline, "body": body, "alert": alert}


def get_dugout_advice(
    over: int,
    score: int,
    wickets: int,
    part_runs: int,
    part_balls: int,
    bowler: str,
    innings: int,
    target: int,
    balls_in_over: int,
    phase: str,            # "pp" | "mid" | "death"
    crr: float,
    sr: int,
    batter1: str = "",
    batter2: str = "",
    weather_note: str = "",
    matchup_note: str = "",
) -> dict[str, str] | None:
    """
    Ask Claude for a live tactical read.
    Returns {badge, headline, body, alert} or None if Claude unavailable.
    """
    phase_label = {"pp": "Powerplay (overs 1-6)", "mid": "Middle overs (7-16)",
                   "death": "Death overs (17-20)"}.get(phase, phase)

    overs_done = (over - 1) + balls_in_over / 6.0
    balls_remaining = max(0, round((20 - overs_done) * 6))

    chase_block = ""
    if innings == 2 and target > 0 and overs_done > 0:
        rem_overs = max(0.0, 20.0 - overs_done)
        need      = target - score
        rrr       = round(need / rem_overs, 2) if rem_overs > 0 else 99.0
        chase_block = (
            f"- Chasing {target}: need {need} runs, {balls_remaining} balls left "
            f"(RRR {rrr})"
        )

    batters_block = ""
    if batter1 or batter2:
        batters_block = f"- Batters at crease: {batter1}{', ' + batter2 if batter2 else ''}"

    weather_block = f"- Weather/dew: {weather_note}" if weather_note else ""
    matchup_block = f"- Key matchup note: {matchup_note}" if matchup_note else ""

    prompt = f"""\
Live T20 match — fielding team situation at over {over}.{balls_in_over}:

- Phase: {phase_label}
- Score: {score}/{wickets} (CRR {crr:.2f})
- Partnership: {part_runs} runs off {part_balls} balls (SR {sr})
- Current bowler: {bowler if bowler else 'unknown'}
{chase_block}
{batters_block}
{weather_block}
{matchup_block}

Respond in this exact format (nothing else):
PRIORITY: CRITICAL | WARNING | INFO
HEADLINE: [one tactical sentence for the captain]
BODY: [2-3 sentences of specific detailed advice]
ALERT: [short CAPS alert text, or leave blank if INFO]\
"""
    raw = _call(prompt, max_tokens=250)
    if raw is None:
        return None
    return _parse_dugout_response(raw)


# ---------------------------------------------------------------------------
# PRE-MATCH COACHING READ
# ---------------------------------------------------------------------------

def get_prematch_coaching_read(
    our_team: str,
    opposition: str,
    venue: str,
    toss_recommendation: str,
    toss_reasoning: str,
    xi_summary: str,           # e.g. "Primary XI: Fakhar, Rizwan, …"
    bowling_plan_summary: str, # e.g. "Shaheen opens; Sikandar mid; Haris death"
    key_matchups: list[str],
    weather_summary: str,
    opposition_top_threats: list[str],
    batting_scenarios_summary: str = "",
) -> str | None:
    """
    Generate a 3-4 sentence pre-match coaching brief synthesising all engine outputs.
    Returns plain text or None if Claude unavailable.
    """
    matchups_block = "\n".join(f"  • {m}" for m in key_matchups[:4]) if key_matchups else "  None identified."
    threats_block  = "\n".join(f"  • {t}" for t in opposition_top_threats[:4]) if opposition_top_threats else "  None identified."

    prompt = f"""\
Pre-match briefing synthesis for a T20 match:

Match: {our_team} vs {opposition} at {venue}
Toss: {toss_recommendation} — {toss_reasoning}
{our_team} XI: {xi_summary}
Bowling plan: {bowling_plan_summary}
Weather/conditions: {weather_summary}

Key matchups to exploit:
{matchups_block}

Opposition top threats:
{threats_block}

{('Batting scenario note: ' + batting_scenarios_summary) if batting_scenarios_summary else ''}

Write a tactical pre-match coaching brief in 4-6 sentences. Cover: (1) what our biggest \
tactical advantage is today, (2) the ONE opposition batter/bowler we must plan specifically \
for, (3) the critical phase (PP/middle/death) where the match will be won or lost today, \
and (4) one specific contingency the captain must remember. Be authoritative, direct, \
match-specific. Speak to your captain, not to a journalist.\
"""
    return _call(prompt, max_tokens=350)


# ---------------------------------------------------------------------------
# FULL COACHING ANALYSIS — AI voice woven into all 4 prep room tabs
# ---------------------------------------------------------------------------

def get_full_coaching_analysis(
    our_team:     str,
    opposition:   str,
    venue:        str,
    xi_options:   list[dict],   # [{label, players:[{name,role,position}]}]
    bowling_plan: list[dict],   # [{over,bowler,phase,note}]
    opp_order:    list[dict],   # [{name,danger,arrival,note}]
    weather_note: str = "",
    toss_rec:     str = "",
    weather:      dict | None = None,  # full weather object from API
) -> dict | None:
    """
    Generate a full coaching analysis to power all 4 prep room tabs.

    Returns:
    {
      "xi": {
        "0": {"reason": str, "players": {name: note}},
        "1": {"reason": str, "players": {name: note}},
        "2": {"reason": str, "players": {name: note}}
      },
      "bowling": {
        "pp":   str,   # powerplay phase reasoning
        "mid":  str,   # middle overs reasoning
        "death": str,  # death overs reasoning
        "contingencies": [str, str, str]
      },
      "opposition": {name: str},   # per-batter coaching note
      "weather_verdict": {
        "xi_impact":      str,  # how weather drives XI choice (spin/pace balance)
        "bowling_impact":  str,  # how weather shapes bowling plan
        "blueprint_note":  str,  # wind/dew/swing effect on field placement
        "alert":           str   # one-line weather warning for captain
      }
    }
    or None if AI unavailable.
    """
    # --- compact XI block ---
    xi_block = ""
    for opt in xi_options[:3]:
        label = opt.get("label", "Option")
        players_str = ", ".join(
            f"{p.get('name','?')} ({p.get('role','?')})"
            for p in opt.get("players", [])[:11]
        )
        xi_block += f"  [{label}] {players_str}\n"

    # --- compact bowling block (include full over-by-over for AI to critique) ---
    pp_bowlers  = [o["bowler"] for o in bowling_plan if o.get("phase") in ("pp",)]
    mid_bowlers = [o["bowler"] for o in bowling_plan if o.get("phase") in ("mid","predeath")]
    dth_bowlers = [o["bowler"] for o in bowling_plan if o.get("phase") == "death"]
    over_lines  = "\n".join(
        f"  Over {o.get('over','?'):>2}: {o.get('bowler','?')} [{o.get('phase','?').upper()}] {o.get('note','')[:50]}"
        for o in bowling_plan[:20]
    )
    bowl_block  = (
        f"  PP (overs 1-6):    {', '.join(pp_bowlers)}\n"
        f"  Middle (7-16):     {', '.join(mid_bowlers)}\n"
        f"  Death (17-20):     {', '.join(dth_bowlers)}\n"
        f"  Full over plan:\n{over_lines}\n"
    )

    # --- compact opposition block ---
    opp_block = "\n".join(
        f"  {o.get('name','?')} [{o.get('danger','?').upper()} danger, arrives {o.get('arrival','?')}]"
        for o in opp_order[:9]
    )

    xi_names_0 = [p.get("name","") for p in (xi_options[0].get("players",[]) if xi_options else [])[:11]]
    xi_names_1 = [p.get("name","") for p in (xi_options[1].get("players",[]) if len(xi_options)>1 else [])[:11]]
    xi_names_2 = [p.get("name","") for p in (xi_options[2].get("players",[]) if len(xi_options)>2 else [])[:11]]

    # --- weather block ---
    w = weather or {}
    humidity       = w.get("humidity", 0)
    temp           = w.get("temp", 0)
    wind_kph       = w.get("wind_kph", 0)
    wind_dir       = w.get("wind_dir", "")
    dew_onset      = w.get("dew_onset_over", 0)
    spinner_pen    = w.get("spinner_penalty", 1.0)
    swing_bon      = w.get("swing_bonus", 1.0)
    dl_needed      = w.get("dl_needed", False)
    wx_warnings    = ", ".join(w.get("warnings", [])) if w.get("warnings") else "none"

    dew_str  = f"DEW expected from over {dew_onset} (spinner penalty {round((1-spinner_pen)*100):.0f}%)" if dew_onset > 0 else "No dew expected"
    swing_str = f"swing bonus {swing_bon:.2f}x" if swing_bon > 1.05 else "minimal swing"

    weather_block = (
        f"  Humidity: {humidity:.0f}%  |  Temp: {temp:.0f}°C  |  Wind: {wind_kph:.0f} km/h {wind_dir}\n"
        f"  {dew_str}  |  {swing_str}\n"
        f"  DL planning needed: {dl_needed}  |  Alerts: {wx_warnings}"
    )

    prompt = f"""\
Pre-match intelligence brief for {our_team} vs {opposition} at {venue}.
Toss: {toss_rec}.

LIVE WEATHER CONDITIONS (from weather API):
{weather_block}

THREE XI OPTIONS:
{xi_block}
BOWLING ALLOCATION:
{bowl_block}
OPPOSITION BATTING ORDER:
{opp_block}

As head coach with 15+ years in T20 franchises, provide coaching analysis for ALL modules.
Weather must directly drive your XI choice (spin vs pace balance) and bowling strategy.
Be specific to THESE players by name. Direct language — no preamble.

Respond ONLY in this exact JSON (no markdown fences, no extra text):
{{
  "xi": {{
    "0": {{
      "reason": "<1-2 sentences: WHY choose this formation today given these weather conditions vs {opposition} at {venue}>",
      "players": {{
        {', '.join(f'"{n}": "<15-word role note>"' for n in xi_names_0[:8])}
      }}
    }},
    "1": {{
      "reason": "<1-2 sentences: WHY this spin-heavy option in these conditions and when to pick it>",
      "players": {{
        {', '.join(f'"{n}": "<15-word role note>"' for n in xi_names_1[:8])}
      }}
    }},
    "2": {{
      "reason": "<1-2 sentences: WHY this pace-heavy option in these conditions and when to pick it>",
      "players": {{
        {', '.join(f'"{n}": "<15-word role note>"' for n in xi_names_2[:8])}
      }}
    }}
  }},
  "bowling": {{
    "pp":   "<2 sentences: PP plan considering humidity {humidity:.0f}% and swing {swing_bon:.2f}x — name bowlers>",
    "mid":  "<2 sentences: middle overs plan with dew considerations — name bowlers>",
    "death": "<2 sentences: death overs plan — dew at over {dew_onset if dew_onset else 'N/A'}, yorker reliability — name bowlers>",
    "contingencies": [
      "<weather-driven trigger: if dew/conditions change, do Y — name the bowler>",
      "<specific trigger: if X happens in match, do Y — name the bowler>",
      "<specific trigger: if X happens in match, do Y — name the bowler>"
    ],
    "over_suggestions": [
      {{"over": <N>, "suggested": "<replacement bowler name>", "replacing": "<current bowler name>", "reason": "<max 12 words>"}},
      {{"over": <N>, "suggested": "<replacement bowler name>", "replacing": "<current bowler name>", "reason": "<max 12 words>"}}
    ]
  }},
  "opposition": {{
    {', '.join(f'"{o.get("name","?")}": "<20-word coaching instruction — consider conditions>"' for o in opp_order[:9])}
  }},
  "weather_verdict": {{
    "xi_impact": "<2 sentences: exactly how today's humidity/dew/wind is pushing you toward one XI over another>",
    "bowling_impact": "<2 sentences: how swing bonus and dew onset change your bowling sequencing today>",
    "blueprint_note": "<1-2 sentences: wind direction {wind_dir} at {wind_kph:.0f} km/h — which end for swing, where to post fielders>",
    "alert": "<one CAPS line: the single most important weather factor the captain must not forget today>"
  }}
}}\
"""
    raw = _call(prompt, max_tokens=1600)
    if raw is None:
        return None

    import json as _json
    import re as _re
    raw = _re.sub(r"^```[a-z]*\s*", "", raw.strip(), flags=_re.MULTILINE)
    raw = _re.sub(r"\s*```$", "", raw.strip(), flags=_re.MULTILINE)
    try:
        result = _json.loads(raw)
        # Ensure over_suggestions is always a list (graceful if AI omits it)
        if "bowling" in result and "over_suggestions" not in result["bowling"]:
            result["bowling"]["over_suggestions"] = []
        return result
    except (_json.JSONDecodeError, ValueError):
        return None


# ---------------------------------------------------------------------------
# HEAD COACH EVALUATION — deep structured brief critique
# ---------------------------------------------------------------------------

def get_coach_evaluation(
    our_team:        str,
    opposition:      str,
    venue:           str,
    xi_players:      list[dict],      # [{name, role, position}]
    bowling_plan:    list[dict],      # [{over, bowler, phase, note}]
    bowler_alloc:    list[dict],      # [{bowler, overs_list, total}]
    opp_order:       list[dict],      # [{name, position, danger, note}]
    scenarios:       list[dict],      # [{id, title, trigger, players:[{name,role}]}]
    toss_rec:        str,
    weather_note:    str,
    contingencies:   list[str],
) -> dict | None:
    """
    Generate a structured head-coach evaluation of all four brief modules.

    Returns a dict:
    {
      "xi":         {"score": int, "justified": str, "concerns": str},
      "bowling":    {"score": int, "justified": str, "concerns": str},
      "opposition": {"score": int, "justified": str, "concerns": str},
      "scenarios":  {"score": int, "content": str},
      "verdict":    str
    }
    or None if AI unavailable.
    """
    # Build compact text blocks for each module
    xi_block = "\n".join(
        f"  {p.get('position','?')}. {p.get('name','?')} — {p.get('role','?')}"
        for p in xi_players[:11]
    )

    bowling_block = "\n".join(
        f"  Over {o.get('over','?')}: {o.get('bowler','?')} [{o.get('phase','?')}] — {o.get('note','')[:60]}"
        for o in bowling_plan[:20]
    )

    alloc_block = "\n".join(
        f"  {a.get('bowler','?')}: overs {a.get('overs_list',[])} ({a.get('total',0)} overs)"
        for a in bowler_alloc[:8]
    )

    opp_block = "\n".join(
        f"  {o.get('position','?')}. {o.get('name','?')} [{o.get('danger','?')} danger] — {o.get('note','')[:80]}"
        for o in opp_order[:9]
    )

    scenario_block = "\n".join(
        f"  [{s.get('id','?')}] {s.get('title','?')}: trigger={s.get('trigger','?')[:70]}"
        for s in scenarios[:4]
    )

    contingency_block = "\n".join(f"  • {c[:100]}" for c in contingencies[:3])

    prompt = f"""\
You are the head coach evaluating the intelligence system's pre-match brief for {our_team} vs {opposition} at {venue}.
Toss recommendation: {toss_rec}. Weather: {weather_note}.

FINAL XI (Option A):
{xi_block}

OVER-BY-OVER BOWLING PLAN:
{bowling_block}

BOWLER ALLOCATION:
{alloc_block}

OPPOSITION BATTING ORDER:
{opp_block}

BATTING SCENARIOS:
{scenario_block}

CONTINGENCY NOTES:
{contingency_block}

Evaluate every module critically as head coach. Be direct — name specific players in your critique.
Identify what the system got RIGHT and what it got WRONG or needs overriding in real match conditions.
Score each module out of 10.

Respond ONLY in this JSON format — no markdown fences, no extra text:
{{
  "xi": {{
    "score": <int 1-10>,
    "justified": "<2-3 sentences on what selections are correct and why>",
    "concerns": "<2-3 sentences on selections you would override and why>"
  }},
  "bowling": {{
    "score": <int 1-10>,
    "justified": "<2-3 sentences on what the plan gets right>",
    "concerns": "<2-3 sentences on bowling decisions you would change>"
  }},
  "opposition": {{
    "score": <int 1-10>,
    "justified": "<2 sentences on correct threat assessments>",
    "concerns": "<2 sentences on under/over-assessed threats>"
  }},
  "scenarios": {{
    "score": <int 1-10>,
    "content": "<3-4 sentences evaluating all four scenario cards — triggers, player roles, and what the dressing room needs to know>"
  }},
  "verdict": "<2-3 sentences — overall assessment, the one change that matters most, and what gives this team the best chance of winning today>"
}}\
"""
    raw = _call(prompt, max_tokens=900)
    if raw is None:
        return None

    # Parse JSON — strip any accidental markdown fences
    import json as _json
    import re as _re
    raw = _re.sub(r"^```[a-z]*\s*", "", raw.strip(), flags=_re.MULTILINE)
    raw = _re.sub(r"\s*```$", "", raw.strip(), flags=_re.MULTILINE)
    try:
        return _json.loads(raw)
    except (_json.JSONDecodeError, ValueError):
        # Fallback: wrap raw text as verdict only
        return {"verdict": raw, "xi": None, "bowling": None, "opposition": None, "scenarios": None}


# ---------------------------------------------------------------------------
# ON-DEMAND ANALYSIS (any page, any context)
# ---------------------------------------------------------------------------

def get_on_demand_advice(
    context_type: str,       # "opposition" | "batting" | "bowling" | "general"
    match_context: str,      # free-form context string
    question: str = "",      # optional specific question
) -> str | None:
    """
    Open-ended tactical advice. Used by /api/claude-advice endpoint.
    Returns plain text advice or None if unavailable.
    """
    type_label = {
        "opposition": "opposition team analysis",
        "batting":    "batting tactics and scenario planning",
        "bowling":    "bowling strategy",
        "general":    "T20 match tactics",
    }.get(context_type, "T20 match tactics")

    prompt = f"""\
You are being asked for {type_label}.

Context:
{match_context}

{('Specific question: ' + question) if question else ''}

Give a direct, actionable tactical recommendation in 3-5 sentences. \
Reference the specific context details provided.\
"""
    return _call(prompt, max_tokens=400)
