from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import tools
from agent import Agent
from models.energy import DatabaseManager, EnergyUsage, SolarGeneration, populate_berlin_sample_data


class FakeResponse:
    def __init__(self, payload):
        self.payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self.payload


def test_database_setup_and_exclusive_date_range(tmp_path):
    manager = DatabaseManager(tmp_path / "energy.db")
    counts = populate_berlin_sample_data(manager, days=2)
    session = manager.get_session()
    try:
        assert session.query(EnergyUsage).count() == counts["usage_records"]
        assert session.query(SolarGeneration).count() == counts["solar_records"]
    finally:
        session.close()


def test_electricity_prices_are_berlin_euro_time_of_use():
    result = tools.get_electricity_prices.invoke({"date": "2026-08-12"})
    assert result["currency"] == "EUR"
    assert result["timezone"] == "Europe/Berlin"
    assert len(result["hourly_rates"]) == 24
    assert result["hourly_rates"][0]["period"] == "off_peak"
    assert result["hourly_rates"][18]["period"] == "peak"


def test_live_weather_response_is_normalized(monkeypatch):
    def fake_get(url, params, timeout):
        if "geocoding" in url:
            return FakeResponse({"results": [{"name": "Berlin", "country": "Germany", "latitude": 52.52, "longitude": 13.41}]})
        return FakeResponse(
            {
                "timezone": "Europe/Berlin",
                "current": {"time": "2026-08-12T12:00", "temperature_2m": 22.0, "relative_humidity_2m": 45, "wind_speed_10m": 12, "weather_code": 2, "cloud_cover": 35, "shortwave_radiation": 650},
                "hourly": {
                    "time": ["2026-08-12T12:00"],
                    "temperature_2m": [22.0],
                    "relative_humidity_2m": [45],
                    "wind_speed_10m": [12],
                    "weather_code": [2],
                    "cloud_cover": [35],
                    "shortwave_radiation": [650],
                },
            }
        )

    tools.clear_weather_cache()
    monkeypatch.setattr(tools.requests, "get", fake_get)
    result = tools.get_weather_forecast.invoke({"location": "Berlin, Germany", "days": 1})
    assert result["data_source"] == "Open Meteo"
    assert result["fallback"] is False
    assert result["hourly"][0]["solar_irradiance_w_m2"] == 650


def test_weather_fallback_is_clear(monkeypatch):
    def broken_get(*args, **kwargs):
        raise tools.requests.RequestException("network unavailable")

    tools.clear_weather_cache()
    monkeypatch.setattr(tools.requests, "get", broken_get)
    result = tools.get_weather_forecast.invoke({"location": "Berlin, Germany", "days": 1})
    assert result["fallback"] is True
    assert result["data_source"] == "deterministic_fallback"
    assert len(result["hourly"]) == 24


def test_savings_supports_price_shifting():
    result = tools.calculate_energy_savings.invoke(
        {"device_type": "dishwasher", "current_usage_kwh": 1.4, "optimized_usage_kwh": 1.4, "price_per_kwh": 0.46, "optimized_price_per_kwh": 0.24}
    )
    assert result["energy_savings_kwh"] == 0
    assert result["savings_eur"] == 0.31
    assert result["annual_savings_eur"] > 100


def test_all_seven_energy_documents_are_available():
    documents = tools.load_energy_tip_documents()
    assert len(documents) == 7
    assert {document.metadata["source"] for document in documents} >= {
        "tip_device_best_practices.txt",
        "tip_energy_savings.txt",
        "tip_hvac_optimization.txt",
        "tip_smart_home_automation.txt",
        "tip_renewable_energy_integration.txt",
        "tip_seasonal_energy_management.txt",
        "tip_energy_storage_optimization.txt",
    }


def test_agent_contract_and_tool_list():
    agent = Agent("You are a clear energy advisor for Berlin homes.")
    assert agent.graph is not None
    assert agent.model_name == "gpt-5.6-luna"
    assert {"get_weather_forecast", "get_electricity_prices", "search_energy_tips"}.issubset(agent.get_agent_tools())
