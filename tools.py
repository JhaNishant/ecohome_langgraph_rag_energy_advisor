"""Tools used by the EcoHome Berlin energy advisor."""

from __future__ import annotations

import math
import os
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


@tool
def search_energy_tips(query: str, max_results: int = 5) -> dict[str, Any]:
    """Search EcoHome's energy saving knowledge base and return sourced tips."""
    if not query or not query.strip():
        return _error("query must be nonempty")
    if not 1 <= max_results <= 10:
        return _error("max_results must be between 1 and 10")
    try:
        vectorstore = build_energy_tip_vectorstore()
        documents = vectorstore.similarity_search(query, k=max_results)
        return {
            "query": query,
            "total_results": len(documents),
            "tips": [
                {
                    "rank": index,
                    "content": document.page_content,
                    "source": document.metadata.get("source", "unknown"),
                    "citation": f"[{document.metadata.get('source', 'unknown')}]",
                }
                for index, document in enumerate(documents, start=1)
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


TOOL_KIT = [
    get_weather_forecast,
    get_electricity_prices,
    query_energy_usage,
    query_solar_generation,
    get_recent_energy_summary,
    search_energy_tips,
    calculate_energy_savings,
]
