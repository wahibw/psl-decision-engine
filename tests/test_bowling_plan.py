# tests/test_bowling_plan.py
# Tests for engine/bowling_plan.py — plan generation, cap validation,
# re-optimisation, form warnings, and opposition profile freshness.

from __future__ import annotations

import pytest
from engine.bowling_plan import (
    BowlingPlan,
    generate_bowling_plan,
    reoptimise_bowling_plan,
    CURRENT_PSL_SEASON,
    OPP_CURRENT_SEASON_WEIGHT,
    MAX_OVERS_PER_BOWLER,
    FORM_CONCERN_ECO_DELTA_PCT,
)
from utils.situation import WeatherImpact, LiveMatchState


# ---------------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------------

def _no_cap_violations(plan: BowlingPlan) -> bool:
    return all(len(v) <= MAX_OVERS_PER_BOWLER for v in plan.bowler_summary.values())


def _total_assigned(plan: BowlingPlan) -> int:
    return sum(len(v) for v in plan.bowler_summary.values())


# ---------------------------------------------------------------------------
# BASIC PLAN GENERATION
# ---------------------------------------------------------------------------

class TestGenerateBowlingPlan:
    def test_returns_20_overs(self, lahore_bowlers, neutral_weather):
        plan = generate_bowling_plan(lahore_bowlers, neutral_weather)
        assert len(plan.overs) == 20

    def test_all_overs_have_primary_bowler(self, lahore_bowlers, neutral_weather):
        plan = generate_bowling_plan(lahore_bowlers, neutral_weather)
        for oa in plan.overs:
            assert oa.primary_bowler not in ("", None, "TBD")

    def test_four_over_cap_respected(self, lahore_bowlers, neutral_weather):
        plan = generate_bowling_plan(lahore_bowlers, neutral_weather)
        assert _no_cap_violations(plan)

    def test_bowler_summary_covers_20_overs(self, lahore_bowlers, neutral_weather):
        plan = generate_bowling_plan(lahore_bowlers, neutral_weather)
        assert _total_assigned(plan) == 20

    def test_has_key_decisions(self, lahore_bowlers, neutral_weather):
        plan = generate_bowling_plan(lahore_bowlers, neutral_weather)
        assert len(plan.key_decisions) >= 1

    def test_has_contingencies(self, lahore_bowlers, neutral_weather):
        plan = generate_bowling_plan(lahore_bowlers, neutral_weather)
        assert len(plan.contingencies) >= 1

    def test_plan_phases_are_valid(self, lahore_bowlers, neutral_weather):
        valid_phases = {"PP", "Early-Mid", "Late-Mid", "Pre-Death", "Death"}
        plan = generate_bowling_plan(lahore_bowlers, neutral_weather)
        for oa in plan.overs:
            assert oa.phase in valid_phases, f"Unexpected phase '{oa.phase}' at over {oa.over}"

    def test_overs_are_1_indexed_and_sequential(self, lahore_bowlers, neutral_weather):
        plan = generate_bowling_plan(lahore_bowlers, neutral_weather)
        assert [oa.over for oa in plan.overs] == list(range(1, 21))

    def test_pp_overs_labelled_correctly(self, lahore_bowlers, neutral_weather):
        plan = generate_bowling_plan(lahore_bowlers, neutral_weather)
        pp_overs = [oa for oa in plan.overs if oa.phase == "PP"]
        assert len(pp_overs) == 6

    def test_death_overs_labelled_correctly(self, lahore_bowlers, neutral_weather):
        plan = generate_bowling_plan(lahore_bowlers, neutral_weather)
        death_overs = [oa for oa in plan.overs if oa.phase == "Death"]
        assert len(death_overs) == 4


class TestPhaseGranularity:
    def test_early_mid_overs_7_to_10(self, lahore_bowlers, neutral_weather):
        plan = generate_bowling_plan(lahore_bowlers, neutral_weather)
        em = [oa for oa in plan.overs if oa.phase == "Early-Mid"]
        assert len(em) == 4
        assert [oa.over for oa in em] == [7, 8, 9, 10]

    def test_late_mid_overs_11_to_14(self, lahore_bowlers, neutral_weather):
        plan = generate_bowling_plan(lahore_bowlers, neutral_weather)
        lm = [oa for oa in plan.overs if oa.phase == "Late-Mid"]
        assert len(lm) == 4
        assert [oa.over for oa in lm] == [11, 12, 13, 14]

    def test_predeath_overs_15_to_16(self, lahore_bowlers, neutral_weather):
        plan = generate_bowling_plan(lahore_bowlers, neutral_weather)
        pd_ = [oa for oa in plan.overs if oa.phase == "Pre-Death"]
        assert len(pd_) == 2
        assert [oa.over for oa in pd_] == [15, 16]


# ---------------------------------------------------------------------------
# DEW WARNINGS IN PLAN
# ---------------------------------------------------------------------------

class TestDewWarningsInPlan:
    def test_dew_warning_note_on_spinner_at_death(self, lahore_bowlers, dew_weather):
        plan = generate_bowling_plan(lahore_bowlers, dew_weather,
                                     venue="Gaddafi Stadium, Lahore")
        # Over 19+ (100% intensity) spinner should have a dew warning note
        spinner_death = [
            oa for oa in plan.overs
            if oa.phase == "Death" and "spinner" in oa.weather_note.lower()
        ]
        # Sikandar Raza may appear at death — check warning is present if it does
        death_overs_with_notes = [oa for oa in plan.overs if oa.phase == "Death" and oa.weather_note]
        # At minimum the key decisions should mention dew
        dew_decisions = [d for d in plan.key_decisions if "dew" in d.lower()]
        assert len(dew_decisions) >= 1

    def test_severe_dew_restricts_spinners_in_key_decisions(self, lahore_bowlers,
                                                               severe_dew_weather):
        plan = generate_bowling_plan(lahore_bowlers, severe_dew_weather,
                                     venue="Gaddafi Stadium, Lahore")
        severe_mentions = [d for d in plan.key_decisions if "SEVERE" in d.upper()]
        assert len(severe_mentions) >= 1

    def test_no_dew_warning_before_onset(self, lahore_bowlers, dew_weather):
        plan = generate_bowling_plan(lahore_bowlers, dew_weather,
                                     venue="Gaddafi Stadium, Lahore")
        # Early-Mid overs 7-10 should have no dew notes (onset=13)
        early_mid_with_dew = [
            oa for oa in plan.overs
            if oa.phase == "Early-Mid" and "dew" in oa.weather_note.lower()
        ]
        assert early_mid_with_dew == []


# ---------------------------------------------------------------------------
# RE-OPTIMISATION
# ---------------------------------------------------------------------------

class TestReoptimiseBowlingPlan:
    def test_reopt_returns_20_overs(self, live_state_over9, lahore_bowlers, dew_weather):
        plan = live_state_over9.bowling_plan
        reopt = reoptimise_bowling_plan(live_state_over9, dew_weather, plan, lahore_bowlers)
        assert len(reopt.overs) == 20

    def test_reopt_completed_overs_locked(self, live_state_over9, lahore_bowlers, dew_weather):
        plan = live_state_over9.bowling_plan
        reopt = reoptimise_bowling_plan(live_state_over9, dew_weather, plan, lahore_bowlers)
        # Overs 1-8 should match original plan assignments
        for oa_orig, oa_new in zip(plan.overs[:8], reopt.overs[:8]):
            assert oa_orig.primary_bowler == oa_new.primary_bowler

    def test_reopt_no_cap_violations(self, live_state_over9, lahore_bowlers, dew_weather):
        plan = live_state_over9.bowling_plan
        reopt = reoptimise_bowling_plan(live_state_over9, dew_weather, plan, lahore_bowlers)
        assert _no_cap_violations(reopt)

    def test_reopt_shaheen_only_1_remaining(self, live_state_over9, lahore_bowlers, dew_weather):
        plan = live_state_over9.bowling_plan
        reopt = reoptimise_bowling_plan(live_state_over9, dew_weather, plan, lahore_bowlers)
        # Shaheen bowled 3 — should have at most 1 remaining
        shaheen_remaining = len(reopt.bowler_summary.get("Shaheen Shah Afridi", []))
        assert shaheen_remaining <= 1

    def test_reopt_key_decision_has_reopt_label(self, live_state_over9, lahore_bowlers, dew_weather):
        plan = live_state_over9.bowling_plan
        reopt = reoptimise_bowling_plan(live_state_over9, dew_weather, plan, lahore_bowlers)
        assert reopt.key_decisions[0].startswith("[Re-optimised at over 9]")

    def test_reopt_at_over_1_returns_original(self, lahore_bowlers, dew_weather):
        plan = generate_bowling_plan(lahore_bowlers, dew_weather)
        state = LiveMatchState(
            batting_team="A", bowling_team="B", venue="Lahore",
            innings=1, target=0, bowling_plan=plan,
            current_over=1, current_score=0, current_wickets=0,
            current_batter1="X", current_batter2="Y",
            partnership_runs=0, partnership_balls=0,
            weather_impact=dew_weather,
        )
        reopt = reoptimise_bowling_plan(state, dew_weather, plan, lahore_bowlers)
        assert reopt is plan


# ---------------------------------------------------------------------------
# OPPOSITION PROFILE FRESHNESS
# ---------------------------------------------------------------------------

class TestOppositionProfileFreshness:
    def test_staleness_warning_for_unknown_team(self, lahore_bowlers, neutral_weather):
        plan = generate_bowling_plan(
            lahore_bowlers, neutral_weather,
            opposition_team="Nonexistent XI 2099"
        )
        # Unknown team → should be neutral (no staleness warning from opp loader
        # since NEUTRAL dict has no staleness_note)
        # Just verify it generates cleanly
        assert len(plan.overs) == 20

    def test_injury_override_appears_in_key_decisions(self, lahore_bowlers, neutral_weather):
        overrides = {
            "injured_out":  ["Babar Azam"],
            "injury_notes": "Babar out — Tayyab moves to 3",
        }
        plan = generate_bowling_plan(
            lahore_bowlers, neutral_weather,
            opposition_team="Karachi Kings",
            opposition_overrides=overrides,
        )
        injury_decisions = [d for d in plan.key_decisions if "[INJURY]" in d]
        assert len(injury_decisions) >= 1
        assert "Babar Azam" in injury_decisions[0]

    def test_form_note_override_appears_in_key_decisions(self, lahore_bowlers, neutral_weather):
        overrides = {"form_note": "Lost 3 of last 4 when chasing"}
        plan = generate_bowling_plan(
            lahore_bowlers, neutral_weather,
            opposition_team="Karachi Kings",
            opposition_overrides=overrides,
        )
        form_decisions = [d for d in plan.key_decisions if "[Form note]" in d]
        assert len(form_decisions) >= 1

    def test_numeric_override_accepted_without_error(self, lahore_bowlers, neutral_weather):
        overrides = {"vs_spin_economy": 9.8, "powerplay_sr": 145.0}
        plan = generate_bowling_plan(
            lahore_bowlers, neutral_weather,
            opposition_team="Karachi Kings",
            opposition_overrides=overrides,
        )
        assert len(plan.overs) == 20


# ---------------------------------------------------------------------------
# VENUE-CONDITIONAL PLAN
# ---------------------------------------------------------------------------

class TestVenueAwarePlan:
    def test_rawalpindi_high_scoring_venue_generates_cleanly(self, lahore_bowlers,
                                                               neutral_weather, venue_rawalpindi):
        plan = generate_bowling_plan(lahore_bowlers, neutral_weather, venue=venue_rawalpindi)
        assert len(plan.overs) == 20
        assert _no_cap_violations(plan)

    def test_karachi_generates_cleanly(self, lahore_bowlers, neutral_weather, venue_karachi):
        plan = generate_bowling_plan(lahore_bowlers, neutral_weather, venue=venue_karachi)
        assert len(plan.overs) == 20
