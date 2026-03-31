# tests/test_weather.py
# Tests for WeatherImpact dew gradient methods and LiveMatchState.dew_intensity.

import pytest
from utils.situation import WeatherImpact, LiveMatchState


class TestDewGradient:
    def test_no_dew_returns_zero_intensity(self, neutral_weather):
        assert neutral_weather.dew_probability_at(15) == 0.0
        assert neutral_weather.dew_probability_at(20) == 0.0

    def test_onset_over_itself_is_zero_intensity(self, dew_weather):
        # At onset over (13), intensity is still 0 — dew only builds AFTER onset
        assert dew_weather.dew_probability_at(13) == 0.0

    def test_pre_onset_over_is_zero(self, dew_weather):
        assert dew_weather.dew_probability_at(12) == 0.0
        assert dew_weather.dew_probability_at(1) == 0.0

    def test_gradient_builds_linearly(self, dew_weather):
        # onset=13, gradient=4 → 25% at 14, 50% at 15, 75% at 16, 100% at 17
        assert dew_weather.dew_probability_at(14) == pytest.approx(0.25)
        assert dew_weather.dew_probability_at(15) == pytest.approx(0.50)
        assert dew_weather.dew_probability_at(16) == pytest.approx(0.75)
        assert dew_weather.dew_probability_at(17) == pytest.approx(1.00)

    def test_intensity_caps_at_1(self, dew_weather):
        assert dew_weather.dew_probability_at(18) == pytest.approx(1.0)
        assert dew_weather.dew_probability_at(20) == pytest.approx(1.0)

    def test_spinner_penalty_at_is_1_before_onset(self, dew_weather):
        assert dew_weather.spinner_penalty_at(12) == pytest.approx(1.0)
        assert dew_weather.spinner_penalty_at(13) == pytest.approx(1.0)

    def test_spinner_penalty_blends_to_full_at_gradient_end(self, dew_weather):
        # Full penalty is 0.60 — should reach it at onset+4=17
        assert dew_weather.spinner_penalty_at(17) == pytest.approx(0.60, abs=0.001)
        assert dew_weather.spinner_penalty_at(19) == pytest.approx(0.60, abs=0.001)

    def test_spinner_penalty_partial_at_midgradient(self, dew_weather):
        # At over 15 (50% intensity): 1.0 - 0.5*(1.0-0.6) = 0.80
        assert dew_weather.spinner_penalty_at(15) == pytest.approx(0.80, abs=0.001)

    def test_severe_dew_full_penalty_earlier(self, severe_dew_weather):
        # onset=10, gradient=4 → full at over 14
        assert severe_dew_weather.dew_probability_at(14) == pytest.approx(1.0)
        assert severe_dew_weather.spinner_penalty_at(14) == pytest.approx(0.40, abs=0.001)


class TestLiveMatchStateDewIntensity:
    def _make_state(self, over: int, weather: WeatherImpact) -> LiveMatchState:
        return LiveMatchState(
            batting_team="A", bowling_team="B", venue="Lahore",
            innings=1, target=0, bowling_plan=None,
            current_over=over, current_score=50, current_wickets=1,
            current_batter1="X", current_batter2="Y",
            partnership_runs=10, partnership_balls=8,
            weather_impact=weather,
        )

    def test_dew_intensity_zero_before_onset(self, dew_weather):
        state = self._make_state(12, dew_weather)
        assert state.dew_intensity == pytest.approx(0.0)
        assert state.dew_active is False

    def test_dew_active_true_at_onset(self, dew_weather):
        state = self._make_state(13, dew_weather)
        assert state.dew_active is True
        assert state.dew_intensity == pytest.approx(0.0)

    def test_dew_intensity_builds_after_onset(self, dew_weather):
        state = self._make_state(15, dew_weather)
        assert state.dew_intensity == pytest.approx(0.50)

    def test_dew_intensity_full_at_plus4(self, dew_weather):
        state = self._make_state(17, dew_weather)
        assert state.dew_intensity == pytest.approx(1.0)

    def test_no_dew_intensity_without_weather(self):
        state = LiveMatchState(
            batting_team="A", bowling_team="B", venue="Lahore",
            innings=1, target=0, bowling_plan=None,
            current_over=18, current_score=150, current_wickets=4,
            current_batter1="X", current_batter2="Y",
            partnership_runs=20, partnership_balls=12,
        )
        assert state.dew_intensity == pytest.approx(0.0)
