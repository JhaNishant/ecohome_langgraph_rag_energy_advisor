"""Tools used by the EcoHome Berlin energy advisor."""

from __future__ import annotations

import math
import os
import re
import shutil
from datetime import datetime, timedelta
from functools import lru_cache
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import requests
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.tools import tool
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from models.energy import DEFAULT_DB_PATH, DatabaseManager, berlin_price_for_hour
from models.preferences import load_user_preferences


PROJECT_ROOT = Path(__file__).resolve().parent
load_dotenv(PROJECT_ROOT / ".env")
DATA_DIR = PROJECT_ROOT / "data"
DOCUMENTS_DIR = DATA_DIR / "documents"
VECTORSTORE_DIR = DATA_DIR / "vectorstore"
BERLIN_TIMEZONE = ZoneInfo("Europe/Berlin")
DEFAULT_LOCATION = "Berlin, Germany"
WEATHER_CODES = {
    0: "clear_sky",
    1: "mainly_clear",
    2: "partly_cloudy",
    3: "overcast",
    45: "foggy",
    48: "foggy",
    51: "light_drizzle",
    53: "drizzle",
    55: "heavy_drizzle",
    61: "light_rain",
    63: "rain",
    65: "heavy_rain",
    71: "light_snow",
    73: "snow",
    75: "heavy_snow",
    80: "rain_showers",
    81: "rain_showers",
    82: "heavy_rain_showers",
    95: "thunderstorm",
}

db_manager = DatabaseManager(DEFAULT_DB_PATH)


def _error(message: str) -> dict[str, Any]:
    return {"error": message}


def _parse_date(value: str) -> datetime:
    return datetime.strptime(value, "%Y-%m-%d")


def _weather_condition(code: int | None) -> str:
    return WEATHER_CODES.get(code or 0, "unknown")


@lru_cache(maxsize=32)
def _geocode_location(location: str) -> dict[str, Any]:
    response = requests.get(
        "https://geocoding-api.open-meteo.com/v1/search",
        params={"name": location, "count": 1, "language": "en", "format": "json"},
        timeout=12,
    )
    response.raise_for_status()
    results = response.json().get("results", [])
    if not results:
        raise ValueError(f"No location found for '{location}'")
    return results[0]


@lru_cache(maxsize=64)
def _fetch_forecast(latitude: float, longitude: float, days: int) -> dict[str, Any]:
    response = requests.get(
        "https://api.open-meteo.com/v1/forecast",
        params={
            "latitude": latitude,
            "longitude": longitude,
            "forecast_days": days,
            "timezone": "auto",
            "current": "temperature_2m,relative_humidity_2m,wind_speed_10m,weather_code,cloud_cover,shortwave_radiation",
            "hourly": "temperature_2m,relative_humidity_2m,wind_speed_10m,weather_code,cloud_cover,shortwave_radiation",
        },
        timeout=12,
    )
    response.raise_for_status()
    return response.json()


def clear_weather_cache() -> None:
    """Clear cached Open Meteo responses. Useful for focused tests."""
    _geocode_location.cache_clear()
    _fetch_forecast.cache_clear()


def _fallback_weather(location: str, days: int, reason: str) -> dict[str, Any]:
    now = datetime.now(BERLIN_TIMEZONE).replace(minute=0, second=0, microsecond=0)
    hourly: list[dict[str, Any]] = []
    for offset in range(days * 24):
        timestamp = now + timedelta(hours=offset)
        daylight = max(0.0, math.sin(math.pi * (timestamp.hour - 6) / 12))
        cloud_cover = 35 if timestamp.day % 2 else 62
        irradiance = round(760 * daylight * (1 - cloud_cover / 150), 1)
        condition = "partly_cloudy" if cloud_cover < 50 else "cloudy"
        hourly.append(
            {
                "time": timestamp.isoformat(),
                "hour": timestamp.hour,
                "temperature_c": round(16 + 7 * daylight, 1),
                "condition": condition,
                "humidity_percent": 62,
                "wind_speed_kmh": 14,
                "cloud_cover_percent": cloud_cover,
                "solar_irradiance_w_m2": irradiance,
            }
        )
    return {
        "location": location,
        "forecast_days": days,
        "timezone": "Europe/Berlin",
        "data_source": "deterministic_fallback",
        "fallback": True,
        "warning": f"Open Meteo was unavailable: {reason}",
        "current": hourly[0],
        "hourly": hourly,
    }


@tool
def get_weather_forecast(location: str = DEFAULT_LOCATION, days: int = 3) -> dict[str, Any]:
    """Get live hourly weather and solar radiation data, defaulting to Berlin, Germany."""
    if not location or not location.strip():
        return _error("location must be a nonempty city or place name")
    if not 1 <= days <= 7:
        return _error("days must be between 1 and 7")
    try:
        place = _geocode_location(location.strip())
        forecast = _fetch_forecast(float(place["latitude"]), float(place["longitude"]), days)
        hourly_data = forecast.get("hourly", {})
        values = zip(
            hourly_data.get("time", []),
            hourly_data.get("temperature_2m", []),
            hourly_data.get("relative_humidity_2m", []),
            hourly_data.get("wind_speed_10m", []),
            hourly_data.get("weather_code", []),
            hourly_data.get("cloud_cover", []),
            hourly_data.get("shortwave_radiation", []),
        )
        hourly = [
            {
                "time": timestamp,
                "hour": int(timestamp[11:13]),
                "temperature_c": temperature,
                "condition": _weather_condition(code),
                "humidity_percent": humidity,
                "wind_speed_kmh": wind_speed,
                "cloud_cover_percent": cloud_cover,
                "solar_irradiance_w_m2": solar_radiation or 0.0,
            }
            for timestamp, temperature, humidity, wind_speed, code, cloud_cover, solar_radiation in values
        ]
        if not hourly:
            return _error("Open Meteo returned no hourly forecast rows")
        current = forecast.get("current", {})
        current_result = {
            "time": current.get("time", hourly[0]["time"]),
            "temperature_c": current.get("temperature_2m", hourly[0]["temperature_c"]),
            "condition": _weather_condition(current.get("weather_code")),
            "humidity_percent": current.get("relative_humidity_2m", hourly[0]["humidity_percent"]),
            "wind_speed_kmh": current.get("wind_speed_10m", hourly[0]["wind_speed_kmh"]),
            "cloud_cover_percent": current.get("cloud_cover", hourly[0]["cloud_cover_percent"]),
            "solar_irradiance_w_m2": current.get("shortwave_radiation", hourly[0]["solar_irradiance_w_m2"]),
        }
        return {
            "location": f"{place.get('name', location)}, {place.get('country', '')}".strip(", "),
            "coordinates": {"latitude": place["latitude"], "longitude": place["longitude"]},
            "forecast_days": days,
            "timezone": forecast.get("timezone", "Europe/Berlin"),
            "data_source": "Open Meteo",
            "fallback": False,
            "current": current_result,
            "hourly": hourly,
        }
    except Exception as exc:
        return _fallback_weather(location, days, str(exc))


@tool
def get_electricity_prices(date: str | None = None) -> dict[str, Any]:
    """Return reproducible Berlin time of use electricity prices in EUR per kWh."""
    try:
        selected_date = _parse_date(date).date() if date else datetime.now(BERLIN_TIMEZONE).date()
    except ValueError:
        return _error("date must use YYYY-MM-DD format")

    def period(hour: int) -> str:
        if hour < 6:
            return "off_peak"
        if hour < 8 or hour >= 21:
            return "shoulder"
        if hour < 16:
            return "standard"
        return "peak"

    rates = [
        {
            "hour": hour,
            "rate_eur_per_kwh": berlin_price_for_hour(hour),
            "period": period(hour),
            "demand_charge_eur": 0.0,
        }
        for hour in range(24)
    ]
    return {
        "date": selected_date.isoformat(),
        "location": "Berlin, Germany",
        "timezone": "Europe/Berlin",
        "pricing_type": "deterministic_time_of_use",
        "currency": "EUR",
        "unit": "EUR_per_kWh",
        "hourly_rates": rates,
    }


@tool
def query_energy_usage(start_date: str, end_date: str, device_type: str | None = None) -> dict[str, Any]:
    """Query historical household electricity consumption for an inclusive date range."""
    try:
        start_dt = _parse_date(start_date)
        end_dt = _parse_date(end_date) + timedelta(days=1)
        if end_dt <= start_dt:
            return _error("end_date must be on or after start_date")
        records = db_manager.get_usage_by_date_range(start_dt, end_dt)
        if device_type:
            records = [record for record in records if record.device_type.lower() == device_type.lower()]
        return {
            "start_date": start_date,
            "end_date": end_date,
            "device_type": device_type,
            "total_records": len(records),
            "total_consumption_kwh": round(sum(record.consumption_kwh for record in records), 2),
            "total_cost_eur": round(sum(record.cost_eur for record in records), 2),
            "records": [
                {
                    "timestamp": record.timestamp.isoformat(),
                    "consumption_kwh": record.consumption_kwh,
                    "device_type": record.device_type,
                    "device_name": record.device_name,
                    "cost_eur": record.cost_eur,
                }
                for record in records
            ],
        }
    except Exception as exc:
        return _error(f"Failed to query energy usage: {exc}")


@tool
def query_solar_generation(start_date: str, end_date: str) -> dict[str, Any]:
    """Query historical rooftop solar generation for an inclusive date range."""
    try:
        start_dt = _parse_date(start_date)
        end_dt = _parse_date(end_date) + timedelta(days=1)
        if end_dt <= start_dt:
            return _error("end_date must be on or after start_date")
        records = db_manager.get_generation_by_date_range(start_dt, end_dt)
        return {
            "start_date": start_date,
            "end_date": end_date,
            "total_records": len(records),
            "total_generation_kwh": round(sum(record.generation_kwh for record in records), 2),
            "average_daily_generation_kwh": round(
                sum(record.generation_kwh for record in records) / max(1, (end_dt - start_dt).days), 2
            ),
            "records": [
                {
                    "timestamp": record.timestamp.isoformat(),
                    "generation_kwh": record.generation_kwh,
                    "weather_condition": record.weather_condition,
                    "temperature_c": record.temperature_c,
                    "solar_irradiance_w_m2": record.solar_irradiance,
                }
                for record in records
            ],
        }
    except Exception as exc:
        return _error(f"Failed to query solar generation: {exc}")


@tool
def get_recent_energy_summary(hours: int = 24) -> dict[str, Any]:
    """Summarize the most recent energy use and solar production."""
    if not 1 <= hours <= 24 * 90:
        return _error("hours must be between 1 and 2160")
    try:
        usage_records = db_manager.get_recent_usage(hours)
        generation_records = db_manager.get_recent_generation(hours)
        device_breakdown: dict[str, dict[str, Any]] = {}
        for record in usage_records:
            item = device_breakdown.setdefault(record.device_type, {"consumption_kwh": 0.0, "cost_eur": 0.0, "records": 0})
            item["consumption_kwh"] += record.consumption_kwh
            item["cost_eur"] += record.cost_eur
            item["records"] += 1
        for item in device_breakdown.values():
            item["consumption_kwh"] = round(item["consumption_kwh"], 2)
            item["cost_eur"] = round(item["cost_eur"], 2)
        weather = generation_records[-1].weather_condition if generation_records else "unknown"
        return {
            "time_period_hours": hours,
            "usage": {
                "total_consumption_kwh": round(sum(record.consumption_kwh for record in usage_records), 2),
                "total_cost_eur": round(sum(record.cost_eur for record in usage_records), 2),
                "device_breakdown": device_breakdown,
            },
            "generation": {
                "total_generation_kwh": round(sum(record.generation_kwh for record in generation_records), 2),
                "latest_weather_condition": weather,
            },
        }
    except Exception as exc:
        return _error(f"Failed to get recent energy summary: {exc}")


@tool
def get_user_preferences() -> dict[str, Any]:
    """Return the editable household preferences used for personalized recommendations."""
    try:
        return load_user_preferences()
    except Exception as exc:
        return _error(f"Failed to load user preferences: {exc}")


def _hours_to_window(hours: list[int]) -> str:
    if not hours:
        return "No suitable window"
    return f"{min(hours):02d}:00–{max(hours) + 1:02d}:00"


def build_tomorrow_plan(
    weather: dict[str, Any], prices: dict[str, Any], preferences: dict[str, Any]
) -> dict[str, Any]:
    """Create a transparent, deterministic day plan from weather, pricing, and preferences."""
    tomorrow = (datetime.now(BERLIN_TIMEZONE).date() + timedelta(days=1)).isoformat()
    hourly_weather = [row for row in weather.get("hourly", []) if str(row.get("time", "")).startswith(tomorrow)]
    if not hourly_weather:
        hourly_weather = weather.get("hourly", [])[:24]

    solar_hours = [
        int(row["hour"])
        for row in hourly_weather
        if float(row.get("solar_irradiance_w_m2") or 0) >= 250
    ]
    price_rows = prices.get("hourly_rates", [])
    off_peak_hours = [int(row["hour"]) for row in price_rows if row.get("period") == "off_peak"]
    peak_hours = [int(row["hour"]) for row in price_rows if row.get("period") == "peak"]
    solar_window = _hours_to_window(solar_hours)
    off_peak_window = _hours_to_window(off_peak_hours)
    peak_window = _hours_to_window(peak_hours)

    ev = preferences["ev"]
    comfort = preferences["comfort"]
    battery = preferences["battery"]
    priorities = preferences["priorities"]
    solar_first = bool(priorities.get("maximize_solar")) and bool(ev.get("allow_midday_charging")) and bool(solar_hours)
    ev_window = solar_window if solar_first else off_peak_window
    ev_reason = "uses forecast rooftop solar" if solar_first else "uses the cheapest grid period before departure"
    estimated_solar_kwh = round(min(float(ev["target_charge_kwh"]), max(0, len(solar_hours) * 1.5)), 1)
    shifted_kwh = round(min(float(ev["target_charge_kwh"]), 6.0), 1)
    carbon = calculate_carbon_impact.invoke(
        {"shifted_energy_kwh": shifted_kwh, "solar_energy_kwh": estimated_solar_kwh}
    )
    source = weather.get("data_source", "unknown")
    confidence = "high" if source == "Open Meteo" and len(solar_hours) >= 3 else "medium"
    if weather.get("fallback"):
        confidence = "limited"

    return {
        "date": tomorrow,
        "location": preferences["location"],
        "confidence": confidence,
        "data_inputs": {
            "weather_source": source,
            "solar_window": solar_window,
            "off_peak_window": off_peak_window,
            "peak_window": peak_window,
            "quiet_hours": comfort["quiet_hours"],
        },
        "plan": [
            {
                "device": "EV charger",
                "window": ev_window,
                "action": f"Charge up to {ev['target_charge_kwh']} kWh before {ev['departure_time']}",
                "why": ev_reason,
            },
            {
                "device": "Flexible appliances",
                "window": solar_window,
                "action": "Run dishwasher, washing, or water heating during this window",
                "why": "matches the strongest forecast solar period while staying outside quiet hours",
            },
            {
                "device": "HVAC",
                "window": solar_window,
                "action": f"Preheat or precool within {comfort['minimum_temperature_c']}–{comfort['maximum_temperature_c']}°C",
                "why": "uses lower cost daytime energy without giving up comfort",
            },
            {
                "device": "Home battery",
                "window": peak_window,
                "action": f"Keep {battery['reserve_percent']}% in reserve and use stored energy during peak hours",
                "why": "reduces exposure to the highest grid price period",
            },
        ],
        "estimated_impact": {
            "shifted_energy_kwh": shifted_kwh,
            "estimated_solar_energy_kwh": estimated_solar_kwh,
            "carbon": carbon,
        },
        "assumptions": [
            "Solar energy is estimated from forecast radiation, not a site specific panel model.",
            "Carbon impact uses a transparent 350 g CO2e per kWh grid estimate.",
            "The plan recommends schedules only and does not operate devices.",
        ],
    }


@tool
def get_personalized_tomorrow_plan(location: str = DEFAULT_LOCATION) -> dict[str, Any]:
    """Create an explainable Berlin energy plan using weather, prices, and saved preferences."""
    try:
        preferences = load_user_preferences()
        if location and location != DEFAULT_LOCATION:
            preferences["location"] = location
        tomorrow = (datetime.now(BERLIN_TIMEZONE).date() + timedelta(days=1)).isoformat()
        weather = get_weather_forecast.invoke({"location": preferences["location"], "days": 3})
        prices = get_electricity_prices.invoke({"date": tomorrow})
        if weather.get("error"):
            return weather
        if prices.get("error"):
            return prices
        return build_tomorrow_plan(weather, prices, preferences)
    except Exception as exc:
        return _error(f"Failed to create a personalized tomorrow plan: {exc}")


def _embedding_client() -> OpenAIEmbeddings:
    api_key = os.getenv("VOCAREUM_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("Set VOCAREUM_API_KEY before building or searching the energy tips index")
    return OpenAIEmbeddings(
        model=os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-large"),
        api_key=api_key,
        base_url=os.getenv("OPENAI_BASE_URL", "https://openai.vocareum.com/v1"),
    )


def load_energy_tip_documents() -> list[Document]:
    documents = []
    for path in sorted(DOCUMENTS_DIR.glob("*.txt")):
        documents.append(Document(page_content=path.read_text(encoding="utf-8"), metadata={"source": path.name}))
    if not documents:
        raise RuntimeError(f"No energy tip documents found in {DOCUMENTS_DIR}")
    return documents


def build_energy_tip_vectorstore(rebuild: bool = False) -> Chroma:
    """Create or load the persistent Chroma knowledge base."""
    if rebuild and VECTORSTORE_DIR.exists():
        shutil.rmtree(VECTORSTORE_DIR)
    embeddings = _embedding_client()
    database_file = VECTORSTORE_DIR / "chroma.sqlite3"
    if database_file.exists():
        return Chroma(
            collection_name="ecohome_energy_tips",
            persist_directory=str(VECTORSTORE_DIR),
            embedding_function=embeddings,
        )
    documents = load_energy_tip_documents()
    splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=120)
    chunks = splitter.split_documents(documents)
    VECTORSTORE_DIR.mkdir(parents=True, exist_ok=True)
    return Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name="ecohome_energy_tips",
        persist_directory=str(VECTORSTORE_DIR),
    )


def _keyword_overlap(query: str, content: str) -> float:
    query_terms = {term for term in re.findall(r"[a-zA-Z]{3,}", query.lower())}
    content_terms = set(re.findall(r"[a-zA-Z]{3,}", content.lower()))
    if not query_terms:
        return 0.0
    return round(len(query_terms & content_terms) / len(query_terms), 3)


@tool
def search_energy_tips(query: str, max_results: int = 5) -> dict[str, Any]:
    """Search EcoHome's energy saving knowledge base and return sourced tips."""
    if not query or not query.strip():
        return _error("query must be nonempty")
    if not 1 <= max_results <= 10:
        return _error("max_results must be between 1 and 10")
    try:
        vectorstore = build_energy_tip_vectorstore()
        candidates = vectorstore.similarity_search_with_relevance_scores(query, k=max_results * 2)
        ranked = []
        for document, semantic_score in candidates:
            keyword_score = _keyword_overlap(query, document.page_content)
            hybrid_score = round((0.75 * max(0.0, semantic_score)) + (0.25 * keyword_score), 3)
            ranked.append((document, max(0.0, semantic_score), keyword_score, hybrid_score))
        ranked.sort(key=lambda item: item[3], reverse=True)
        documents = ranked[:max_results]
        return {
            "query": query,
            "total_results": len(documents),
            "retrieval_method": "hybrid semantic search with keyword re ranking",
            "tips": [
                {
                    "rank": index,
                    "content": document.page_content,
                    "source": document.metadata.get("source", "unknown"),
                    "citation": f"[{document.metadata.get('source', 'unknown')}]",
                    "semantic_score": round(semantic_score, 3),
                    "keyword_score": keyword_score,
                    "hybrid_score": hybrid_score,
                }
                for index, (document, semantic_score, keyword_score, hybrid_score) in enumerate(documents, start=1)
            ],
        }
    except Exception as exc:
        return _error(f"Failed to search energy tips: {exc}")


@tool
def calculate_energy_savings(
    device_type: str,
    current_usage_kwh: float,
    optimized_usage_kwh: float | None = None,
    price_per_kwh: float = 0.35,
    optimized_price_per_kwh: float | None = None,
) -> dict[str, Any]:
    """Calculate euro savings from using less energy or moving use to a cheaper period."""
    optimized_usage = current_usage_kwh if optimized_usage_kwh is None else optimized_usage_kwh
    optimized_price = price_per_kwh if optimized_price_per_kwh is None else optimized_price_per_kwh
    if not device_type.strip():
        return _error("device_type must be nonempty")
    if min(current_usage_kwh, optimized_usage, price_per_kwh, optimized_price) < 0:
        return _error("usage and price values must be zero or greater")
    current_cost = current_usage_kwh * price_per_kwh
    optimized_cost = optimized_usage * optimized_price
    savings_eur = current_cost - optimized_cost
    return {
        "device_type": device_type,
        "current_usage_kwh": round(current_usage_kwh, 2),
        "optimized_usage_kwh": round(optimized_usage, 2),
        "energy_savings_kwh": round(current_usage_kwh - optimized_usage, 2),
        "current_price_eur_per_kwh": round(price_per_kwh, 3),
        "optimized_price_eur_per_kwh": round(optimized_price, 3),
        "current_cost_eur": round(current_cost, 2),
        "optimized_cost_eur": round(optimized_cost, 2),
        "savings_eur": round(savings_eur, 2),
        "savings_percentage": round((savings_eur / current_cost) * 100, 1) if current_cost else 0.0,
        "annual_savings_eur": round(savings_eur * 365, 2),
    }


@tool
def calculate_carbon_impact(
    shifted_energy_kwh: float,
    solar_energy_kwh: float,
    grid_intensity_g_per_kwh: float = 350.0,
) -> dict[str, Any]:
    """Estimate avoided grid emissions from solar use or moving flexible energy."""
    if min(shifted_energy_kwh, solar_energy_kwh, grid_intensity_g_per_kwh) < 0:
        return _error("energy and grid intensity values must be zero or greater")
    avoided_grid_kwh = min(shifted_energy_kwh, solar_energy_kwh)
    avoided_kg = avoided_grid_kwh * grid_intensity_g_per_kwh / 1000
    return {
        "shifted_energy_kwh": round(shifted_energy_kwh, 2),
        "solar_energy_kwh": round(solar_energy_kwh, 2),
        "avoided_grid_energy_kwh": round(avoided_grid_kwh, 2),
        "grid_intensity_g_co2e_per_kwh": round(grid_intensity_g_per_kwh, 1),
        "estimated_avoided_kg_co2e": round(avoided_kg, 2),
        "method": "transparent demo estimate using a configurable grid intensity",
    }


TOOL_KIT = [
    get_weather_forecast,
    get_electricity_prices,
    query_energy_usage,
    query_solar_generation,
    get_recent_energy_summary,
    get_user_preferences,
    get_personalized_tomorrow_plan,
    search_energy_tips,
    calculate_energy_savings,
    calculate_carbon_impact,
]
