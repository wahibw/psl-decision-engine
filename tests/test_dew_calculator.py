# tests/test_dew_calculator.py
# Tests for weather/dew_calculator.py — risk classification, onset over,
# venue adjustments, and public assess_dew() API.

from __future__ import annotations

import pytest
from weather.dew_calculator import (
    assess_dew,
    DewAssessment,
    _base_risk,
    _onset_over,
    _spinner_warning,
)
from utils.situation import WeatherReading
from datetime import datetime


def _reading(temp_c: float, dewpoint_c: float, humidity_pct: float) -> WeatherReading:
    return WeatherReading(
        temp_c           = temp_c,
        dewpoint_c       = dewpoint_c,
        humidity_pct     = humidity_pct,
        wind_kph         = 10.0,
        wind_dir         = "N",
        condition        = "Clear",
        fetched_at       = datetime(2025, 3, 15, 19, 0),
        rain_probability = 0,
        precip_mm        = 0.0,
    )


class TestBaseRisk:
    def test_severe_risk_tight_spread_high_humidity(self):
        risk = _base_risk(dew_spread=1.0, humidity_pct=82)
        assert risk == "Severe"

    def test_high_risk_moderate_spread(self):
        risk = _base_risk(dew_spread=2.5, humidity_pct=72)
        assert risk == "High"

    def test_medium_risk(self):
        risk = _base_risk(dew_spread=3.5, humidity_pct=66)
        assert risk == "Medium"

    def test_none_risk_dry(self):
        risk = _base_risk(dew_spread=8.0, humidity_pct=30)
        assert risk == "None"


class TestOnsetOver:
    def test_severe_base_onset_is_10(self):
        onset = _onset_over(dew_spread=1.0, risk="Severe", venue="", temp_c=22)
        assert onset == 10

    def test_high_base_onset_is_12(self):
        onset = _onset_over(dew_spread=2.5, risk="High", venue="", temp_c=22)
        assert onset == 12

    def test_lahore_shifts_onset_earlier(self):
        # Lahore offset is -1
        onset_generic = _onset_over(2.5, "High", venue="",                    temp_c=22)
        onset_lahore  = _onset_over(2.5, "High", venue="Gaddafi Stadium, Lahore", temp_c=22)
        assert onset_lahore == onset_generic - 1

    def test_dubai_shifts_onset_much_earlier(self):
        # Dubai offset is -2
        onset_generic = _onset_over(2.5, "High", venue="",                                  temp_c=22)
        onset_dubai   = _onset_over(2.5, "High", venue="Dubai International Cricket Stadium", temp_c=22)
        assert onset_dubai == onset_generic - 2

    def test_cold_night_accelerates_onset(self):
        warm = _onset_over(2.0, "High", venue="", temp_c=25)
        cold = _onset_over(2.0, "High", venue="", temp_c=12)
        assert cold < warm

    def test_onset_never_below_6(self):
        onset = _onset_over(0.5, "Severe", venue="Dubai International Cricket Stadium", temp_c=10)
        assert onset >= 6

    def test_onset_never_above_19(self):
        onset = _onset_over(4.5, "Low", venue="National Stadium, Karachi", temp_c=30)
        assert onset <= 19


class TestSpinnerWarning:
    def test_warning_if_onset_before_16(self):
        assert _spinner_warning(onset_over=13, risk="High") is True

    def test_no_warning_if_onset_17_or_later(self):
        assert _spinner_warning(onset_over=17, risk="Low") is False

    def test_no_warning_if_no_dew(self):
        assert _spinner_warning(onset_over=0, risk="None") is False


class TestAssessDew:
    def test_day_match_returns_no_dew(self):
        r = _reading(30, 20, 55)
        result = assess_dew(r, is_night_match=False)
        assert result.dew_risk == "None"
        assert result.onset_over == 0

    def test_dry_conditions_return_no_dew(self):
        r = _reading(32, 15, 35)
        result = assess_dew(r, is_night_match=True)
        assert result.dew_risk == "None"

    def test_humid_night_returns_dew(self):
        r = _reading(22, 20, 80)
        result = assess_dew(r, venue="Gaddafi Stadium, Lahore", is_night_match=True)
        assert result.has_dew
        assert result.onset_over > 0

    def test_critical_over_is_onset_plus_3(self):
        r = _reading(22, 20, 80)
        result = assess_dew(r, is_night_match=True)
        if result.has_dew:
            assert result.critical_over == min(result.onset_over + 3, 19)

    def test_toss_advice_present(self):
        r = _reading(22, 20, 80)
        result = assess_dew(r, is_night_match=True)
        assert result.toss_advice.strip() != ""

    def test_details_non_empty_when_dew(self):
        r = _reading(22, 20, 80)
        result = assess_dew(r, is_night_match=True)
        if result.has_dew:
            assert len(result.details) >= 1

    def test_no_dew_assessment_factory(self):
        nd = DewAssessment.no_dew()
        assert nd.dew_risk == "None"
        assert nd.onset_over == 0
        assert nd.has_dew is False
