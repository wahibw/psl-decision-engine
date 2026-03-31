# tests/test_live_state.py
# Tests for LiveMatchState derived field calculation and advance_over().

from __future__ import annotations

import pytest
from utils.situation import LiveMatchState, WeatherImpact


def _make_state(over: int, score: int, wickets: int,
                innings: int = 1, target: int = 0,
                weather: WeatherImpact | None = None) -> LiveMatchState:
    return LiveMatchState(
        batting_team      = "Team A",
        bowling_team      = "Team B",
        venue             = "Gaddafi Stadium, Lahore",
        innings           = innings,
        target            = target,
        bowling_plan      = None,
        current_over      = over,
        current_score     = score,
        current_wickets   = wickets,
        current_batter1   = "Batter 1",
        current_batter2   = "Batter 2",
        partnership_runs  = 10,
        partnership_balls = 6,
        weather_impact    = weather,
    )


class TestDerivedFields:
    def test_phase_powerplay(self):
        state = _make_state(4, 30, 0)
        assert state.phase == "powerplay"

    def test_phase_middle(self):
        state = _make_state(10, 80, 1)
        assert state.phase == "middle"

    def test_phase_death(self):
        state = _make_state(17, 140, 3)
        assert state.phase == "death"

    def test_balls_remaining_over1(self):
        state = _make_state(1, 0, 0)
        assert state.balls_remaining == 120   # 20 overs, 0 completed

    def test_balls_remaining_over11(self):
        state = _make_state(11, 80, 2)
        # 10 overs completed = 60 balls bowled → 60 remaining
        assert state.balls_remaining == 60

    def test_crr_zero_at_over1(self):
        state = _make_state(1, 0, 0)
        assert state.crr == pytest.approx(0.0)

    def test_crr_calculated_correctly(self):
        # over 11 → 10 overs completed, score 80
        state = _make_state(11, 80, 2)
        assert state.crr == pytest.approx(8.0)

    def test_rrr_innings2(self):
        state = _make_state(16, 120, 3, innings=2, target=175)
        # 55 needed, 30 balls remaining → RRR = 55 / 5 = 11.0
        assert state.rrr == pytest.approx(11.0, abs=0.1)

    def test_rrr_zero_in_innings1(self):
        state = _make_state(10, 80, 2, innings=1)
        assert state.rrr == pytest.approx(0.0)

    def test_dew_active_false_before_onset(self, dew_weather):
        state = _make_state(12, 80, 1, weather=dew_weather)
        assert state.dew_active is False

    def test_dew_active_true_at_onset(self, dew_weather):
        state = _make_state(13, 100, 2, weather=dew_weather)
        assert state.dew_active is True


class TestOversRemaining:
    def test_fresh_bowler_has_4_left(self, neutral_weather):
        state = _make_state(7, 50, 1, weather=neutral_weather)
        state.overs_bowled_by = {"Shaheen Shah Afridi": 0}
        assert state.overs_remaining_for("Shaheen Shah Afridi") == 4

    def test_bowler_with_3_overs_has_1_left(self, neutral_weather):
        state = _make_state(15, 110, 3, weather=neutral_weather)
        state.overs_bowled_by = {"Haris Rauf": 3}
        assert state.overs_remaining_for("Haris Rauf") == 1

    def test_capped_bowler_has_0_left(self, neutral_weather):
        state = _make_state(18, 145, 4, weather=neutral_weather)
        state.overs_bowled_by = {"Usama Mir": 4}
        assert state.overs_remaining_for("Usama Mir") == 0


class TestAdvanceOver:
    def test_advance_increments_current_over(self, neutral_weather):
        state = _make_state(8, 60, 1, weather=neutral_weather)
        state.advance_over(
            score=68, wickets=1, batter1="X", batter2="Y",
            partnership_runs=8, partnership_balls=6,
            bowler_this_over="Haris Rauf",
        )
        assert state.current_over == 9

    def test_advance_records_bowler_overs(self, neutral_weather):
        state = _make_state(8, 60, 1, weather=neutral_weather)
        state.advance_over(
            score=68, wickets=1, batter1="X", batter2="Y",
            partnership_runs=8, partnership_balls=6,
            bowler_this_over="Haris Rauf",
        )
        assert state.overs_bowled_by.get("Haris Rauf", 0) == 1

    def test_advance_computes_wickets_this_over(self, neutral_weather):
        state = _make_state(8, 60, 2, weather=neutral_weather)
        state.advance_over(
            score=65, wickets=4, batter1="New1", batter2="New2",
            partnership_runs=0, partnership_balls=0,
        )
        assert state.wickets_this_over == 2


class TestDisplayProperties:
    def test_display_score(self):
        state = _make_state(10, 75, 3)
        assert state.display_score == "75/3"

    def test_display_over(self):
        state = _make_state(14, 100, 2)
        assert state.display_over == "Over 14"

    def test_partnership_sr(self):
        state = _make_state(10, 80, 2)
        state.partnership_runs  = 30
        state.partnership_balls = 20
        assert state.partnership_sr == pytest.approx(150.0)
