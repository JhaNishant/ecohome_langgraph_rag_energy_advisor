"""Simple, editable household preferences for personalized energy advice."""

from __future__ import annotations

import json
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PREFERENCES_PATH = PROJECT_ROOT / "data" / "user_preferences.json"

DEFAULT_PREFERENCES: dict[str, Any] = {
    "location": "Berlin, Germany",
    "timezone": "Europe/Berlin",
    "currency": "EUR",
    "ev": {"departure_time": "07:30", "target_charge_kwh": 16.0, "allow_midday_charging": True},
    "comfort": {"minimum_temperature_c": 20.0, "maximum_temperature_c": 23.0, "quiet_hours": ["22:00", "07:00"]},
    "battery": {"reserve_percent": 25, "use_during_peak_hours": True},
    "priorities": {"maximize_solar": True, "avoid_peak_prices": True, "carbon_aware": True},
}


def load_user_preferences(path: str | Path = PREFERENCES_PATH) -> dict[str, Any]:
    """Load the sample profile while keeping sensible defaults for missing values."""
    preference_path = Path(path)
    if not preference_path.exists():
        return deepcopy(DEFAULT_PREFERENCES)
    with preference_path.open(encoding="utf-8") as handle:
        saved = json.load(handle)

    merged = deepcopy(DEFAULT_PREFERENCES)
    for section, value in saved.items():
        if isinstance(value, dict) and isinstance(merged.get(section), dict):
            merged[section] = {**merged[section], **value}
        else:
            merged[section] = value
    try:
        datetime.strptime(str(merged["ev"]["departure_time"]), "%H:%M")
        if float(merged["ev"]["target_charge_kwh"]) < 0:
            raise ValueError("EV target charge must be zero or greater")
        if float(merged["comfort"]["minimum_temperature_c"]) > float(merged["comfort"]["maximum_temperature_c"]):
            raise ValueError("minimum comfort temperature cannot exceed the maximum")
        if not 0 <= int(merged["battery"]["reserve_percent"]) <= 100:
            raise ValueError("battery reserve must be between 0 and 100")
        quiet_hours = merged["comfort"]["quiet_hours"]
        if not isinstance(quiet_hours, list) or len(quiet_hours) != 2:
            raise ValueError("quiet_hours must contain a start and end time")
        for value in quiet_hours:
            datetime.strptime(str(value), "%H:%M")
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"Invalid user preferences: {exc}") from exc
    return merged
