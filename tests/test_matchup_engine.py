# tests/test_matchup_engine.py
# Tests for engine/matchup_engine.py — ball threshold, dismissal guard,
# confidence tiers, and placeholder note.

from __future__ import annotations

import pytest
from engine.matchup_engine import (
    get_matchup_table,
    get_key_matchups_for_brief,
    MIN_BALLS_FOR_NOTE,
    MIN_BALLS_PHASE_NOTE,
    MIN_DISMISSALS_FOR_DOMINANCE,
    MIN_BALLS_SINGLE_DISMISSAL_OK,
)


class TestMatchupThresholds:
    def test_min_balls_for_note_is_12(self):
        assert MIN_BALLS_FOR_NOTE == 12

    def test_min_balls_phase_note_is_8(self):
        assert MIN_BALLS_PHASE_NOTE == 8

    def test_min_dismissals_for_dominance(self):
        assert MIN_DISMISSALS_FOR_DOMINANCE == 2

    def test_min_balls_single_dismissal_ok(self):
        assert MIN_BALLS_SINGLE_DISMISSAL_OK == 24


BOWLERS  = ["Shaheen Shah Afridi", "Haris Rauf", "Usama Mir", "Sikandar Raza"]
BATTERS  = ["Babar Azam", "Mohammad Rizwan", "Khushdil Shah", "Saim Ayub"]


class TestGetMatchupTable:
    def test_returns_dataframe(self):
        import pandas as pd
        result = get_matchup_table(BATTERS, BOWLERS, min_balls=MIN_BALLS_FOR_NOTE)
        assert isinstance(result, pd.DataFrame)

    def test_all_results_meet_min_balls(self):
        result = get_matchup_table(BATTERS, BOWLERS, min_balls=MIN_BALLS_FOR_NOTE)
        if not result.empty and "balls" in result.columns:
            assert (result["balls"] >= MIN_BALLS_FOR_NOTE).all()

    def test_default_min_balls_matches_constant(self):
        default_result  = get_matchup_table(BATTERS, BOWLERS)
        explicit_result = get_matchup_table(BATTERS, BOWLERS, min_balls=MIN_BALLS_FOR_NOTE)
        assert len(default_result) == len(explicit_result)


class TestGetKeyMatchupsForBrief:
    def test_returns_list(self):
        notes = get_key_matchups_for_brief(
            our_bowlers        = BOWLERS,
            opposition_batters = BATTERS,
        )
        assert isinstance(notes, list)

    def test_returns_matchup_note_objects(self):
        from engine.matchup_engine import MatchupNote
        notes = get_key_matchups_for_brief(
            our_bowlers        = BOWLERS[:2],
            opposition_batters = BATTERS[:2],
        )
        for note in notes:
            assert isinstance(note, MatchupNote)

    def test_no_sub_threshold_notes(self):
        """All returned notes must have balls >= MIN_BALLS_FOR_NOTE or be placeholder."""
        notes = get_key_matchups_for_brief(
            our_bowlers        = BOWLERS,
            opposition_batters = BATTERS,
        )
        for note in notes:
            if note.balls > 0:   # 0 = placeholder note
                assert note.balls >= MIN_BALLS_FOR_NOTE, (
                    f"Note for {note.bowler} vs {note.batter} has {note.balls} balls "
                    f"(threshold {MIN_BALLS_FOR_NOTE})"
                )

    def test_empty_inputs_returns_list(self):
        notes = get_key_matchups_for_brief(
            our_bowlers=[], opposition_batters=[]
        )
        assert isinstance(notes, list)
