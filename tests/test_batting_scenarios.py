# tests/test_batting_scenarios.py
# Tests for engine/batting_scenarios.py — venue-aware scenario cards.

from __future__ import annotations

import pytest
from engine.batting_scenarios import (
    generate_batting_scenarios,
    BattingScenario,
    _load_venue_stats,
    _venue_scenario_targets,
)


LAHORE_BATTERS = [
    "Fakhar Zaman", "Abdullah Shafique", "Sikandar Raza",
    "Shaheen Shah Afridi", "Hussain Talat", "Tayyab Tahir",
    "Usama Mir", "Haris Rauf",
]

OPP_BOWLING = {
    "vs_spin_economy": 8.8,
    "vs_pace_economy": 7.2,
    "pace_overs_pct":  50.0,
    "spin_overs_pct":  20.0,
}


class TestVenueStats:
    def test_lahore_returns_non_default(self):
        vstats = _load_venue_stats("Gaddafi Stadium, Lahore")
        assert vstats["avg_first_score"] > 150

    def test_rawalpindi_is_high_scoring(self):
        vstats = _load_venue_stats("Rawalpindi Cricket Stadium")
        assert vstats["avg_first_score"] >= 190

    def test_unknown_venue_returns_defaults(self):
        vstats = _load_venue_stats("Nonexistent Ground")
        assert vstats["avg_first_score"] == pytest.approx(168.0)

    def test_empty_venue_returns_defaults(self):
        vstats = _load_venue_stats("")
        assert vstats["avg_first_score"] == pytest.approx(168.0)


class TestVenueScenarioTargets:
    def test_ideal_pp_target_above_par_pp(self):
        vstats = _load_venue_stats("Gaddafi Stadium, Lahore")
        vt = _venue_scenario_targets(vstats)
        # Ideal PP target should be ~10% above par PP
        assert vt["ideal_pp_target"] > vstats["avg_pp_score"]

    def test_collapse_pp_max_below_par_pp(self):
        vstats = _load_venue_stats("Gaddafi Stadium, Lahore")
        vt = _venue_scenario_targets(vstats)
        assert vt["collapse_pp_max"] < vstats["avg_pp_score"]

    def test_low_score_total_below_par(self):
        vstats = _load_venue_stats("Gaddafi Stadium, Lahore")
        vt = _venue_scenario_targets(vstats)
        assert vt["low_score_total"] < vt["par_total"]

    def test_rawalpindi_venue_tag_says_high_scoring(self):
        vstats = _load_venue_stats("Rawalpindi Cricket Stadium")
        vt = _venue_scenario_targets(vstats)
        assert "high-scoring" in vt["venue_tag"]


class TestGenerateBattingScenarios:
    @pytest.fixture
    def cards(self, neutral_weather):
        return generate_batting_scenarios(
            LAHORE_BATTERS, OPP_BOWLING,
            "Gaddafi Stadium, Lahore", neutral_weather
        )

    def test_returns_4_cards(self, cards):
        assert len(cards) == 4

    def test_card_ids_are_a_b_c_d(self, cards):
        assert [c.scenario_id for c in cards] == ["A", "B", "C", "D"]

    def test_all_cards_have_key_message(self, cards):
        for c in cards:
            assert c.key_message.strip() != ""

    def test_all_cards_have_trigger(self, cards):
        for c in cards:
            assert c.trigger.strip() != ""

    def test_batting_order_populated(self, cards):
        for c in cards:
            assert len(c.batting_order) >= 1

    def test_venue_par_in_card_a_message(self, cards):
        card_a = next(c for c in cards if c.scenario_id == "A")
        # Key message or trigger should reference the venue par (~175 for Lahore)
        assert "174" in card_a.key_message or "175" in card_a.key_message \
               or "par" in card_a.key_message.lower() or "175" in card_a.trigger

    def test_chase_pct_in_card_c_lahore(self, neutral_weather):
        """Lahore has ~42% chase win rate — card C should mention it."""
        cards = generate_batting_scenarios(
            LAHORE_BATTERS, OPP_BOWLING, "Gaddafi Stadium, Lahore", neutral_weather
        )
        card_c = next(c for c in cards if c.scenario_id == "C")
        assert "42" in card_c.key_message or "42" in card_c.trigger

    def test_rawalpindi_cards_have_higher_pp_target(self, neutral_weather):
        cards = generate_batting_scenarios(
            LAHORE_BATTERS, OPP_BOWLING, "Rawalpindi Cricket Stadium", neutral_weather
        )
        card_a = next(c for c in cards if c.scenario_id == "A")
        # Rawalpindi PP par ~58, ideal PP target ~64 — trigger should reference 60+
        # Key message should mention high-scoring ground or par ~194
        assert "high-scoring" in card_a.key_message.lower() or "194" in card_a.key_message \
               or "64" in card_a.trigger

    def test_unknown_venue_falls_back_gracefully(self, neutral_weather):
        cards = generate_batting_scenarios(
            LAHORE_BATTERS, OPP_BOWLING, "Random Stadium", neutral_weather
        )
        assert len(cards) == 4

    def test_weather_note_swing_on_cards_a_d(self, neutral_weather):
        """Swing weather should add a note to cards A and D."""
        from utils.situation import WeatherImpact
        swing_weather = WeatherImpact(
            spinner_penalty=1.0, swing_bonus=1.25, pace_bounce_bonus=1.0,
            yorker_reliability=1.0, dl_planning_needed=False,
            dew_onset_over=0, warnings=[],
        )
        cards = generate_batting_scenarios(
            LAHORE_BATTERS, OPP_BOWLING, "Gaddafi Stadium, Lahore", swing_weather
        )
        card_a = next(c for c in cards if c.scenario_id == "A")
        card_d = next(c for c in cards if c.scenario_id == "D")
        assert "swing" in (card_a.weather_note + card_d.weather_note).lower()
