"""SQLite models and repeatable Berlin sample data for EcoHome."""

from __future__ import annotations

import random
from datetime import date as Date, datetime, timedelta
from pathlib import Path
from sqlalchemy import Column, DateTime, Float, Integer, String, create_engine
from sqlalchemy.orm import declarative_base, sessionmaker


Base = declarative_base()
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DB_PATH = PROJECT_ROOT / "data" / "energy_data.db"


class EnergyUsage(Base):
    """Hourly household electricity consumption."""

    __tablename__ = "energy_usage"

    id = Column(Integer, primary_key=True)
    timestamp = Column(DateTime, nullable=False, index=True)
    consumption_kwh = Column(Float, nullable=False)
    device_type = Column(String(50), nullable=False)
    device_name = Column(String(100), nullable=False)
    cost_eur = Column(Float, nullable=False)


class SolarGeneration(Base):
    """Hourly rooftop solar generation and the conditions behind it."""

    __tablename__ = "solar_generation"

    id = Column(Integer, primary_key=True)
    timestamp = Column(DateTime, nullable=False, index=True)
    generation_kwh = Column(Float, nullable=False)
    weather_condition = Column(String(50), nullable=False)
    temperature_c = Column(Float, nullable=False)
    solar_irradiance = Column(Float, nullable=False)


class DatabaseManager:
    """Small database gateway used by the notebooks and agent tools."""

    def __init__(self, db_path: str | Path = DEFAULT_DB_PATH):
        self.db_path = Path(db_path).expanduser().resolve()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.engine = create_engine(f"sqlite:///{self.db_path}")
        self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)

    def create_tables(self) -> None:
        Base.metadata.create_all(bind=self.engine)

    def reset_database(self) -> None:
        Base.metadata.drop_all(bind=self.engine)
        Base.metadata.create_all(bind=self.engine)

    def get_session(self):
        return self.SessionLocal()

    def add_usage_record(
        self,
        timestamp: datetime,
        consumption_kwh: float,
        device_type: str,
        device_name: str,
        cost_eur: float,
    ) -> EnergyUsage:
        session = self.get_session()
        try:
            record = EnergyUsage(
                timestamp=timestamp,
                consumption_kwh=consumption_kwh,
                device_type=device_type,
                device_name=device_name,
                cost_eur=cost_eur,
            )
            session.add(record)
            session.commit()
            session.refresh(record)
            return record
        finally:
            session.close()

    def add_generation_record(
        self,
        timestamp: datetime,
        generation_kwh: float,
        weather_condition: str,
        temperature_c: float,
        solar_irradiance: float,
    ) -> SolarGeneration:
        session = self.get_session()
        try:
            record = SolarGeneration(
                timestamp=timestamp,
                generation_kwh=generation_kwh,
                weather_condition=weather_condition,
                temperature_c=temperature_c,
                solar_irradiance=solar_irradiance,
            )
            session.add(record)
            session.commit()
            session.refresh(record)
            return record
        finally:
            session.close()

    def get_usage_by_date_range(self, start_date: datetime, end_date: datetime):
        session = self.get_session()
        try:
            return (
                session.query(EnergyUsage)
                .filter(EnergyUsage.timestamp >= start_date, EnergyUsage.timestamp < end_date)
                .order_by(EnergyUsage.timestamp)
                .all()
            )
        finally:
            session.close()

    def get_generation_by_date_range(self, start_date: datetime, end_date: datetime):
        session = self.get_session()
        try:
            return (
                session.query(SolarGeneration)
                .filter(SolarGeneration.timestamp >= start_date, SolarGeneration.timestamp < end_date)
                .order_by(SolarGeneration.timestamp)
                .all()
            )
        finally:
            session.close()

    def get_recent_usage(self, hours: int = 24):
        end_time = datetime.now().replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)
        return self.get_usage_by_date_range(end_time - timedelta(hours=hours), end_time)

    def get_recent_generation(self, hours: int = 24):
        end_time = datetime.now().replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)
        return self.get_generation_by_date_range(end_time - timedelta(hours=hours), end_time)


def berlin_price_for_hour(hour: int, pricing_date: Date | datetime | None = None) -> float:
    """Return a repeatable Berlin tariff that varies by hour and calendar date.

    The model combines time of use windows with a seasonal, weekday, and small
    day specific market factor. It is deliberately deterministic: asking for
    the same date always returns the same tariff, while different dates can
    produce different prices without requiring a paid pricing API.
    """
    if not 0 <= hour <= 23:
        raise ValueError("hour must be between 0 and 23")
    if pricing_date is None:
        selected_date = datetime.now().date()
    elif isinstance(pricing_date, datetime):
        selected_date = pricing_date.date()
    else:
        selected_date = pricing_date

    if 0 <= hour < 6:
        base_rate = 0.24
    elif 6 <= hour < 8:
        base_rate = 0.32
    elif 8 <= hour < 16:
        base_rate = 0.35
    elif 16 <= hour < 21:
        base_rate = 0.46
    else:
        base_rate = 0.30

    seasonal_factor = {
        12: 1.14, 1: 1.14, 2: 1.14,
        3: 0.98, 4: 0.98, 5: 0.98,
        6: 0.88, 7: 0.88, 8: 0.88,
        9: 1.04, 10: 1.04, 11: 1.04,
    }[selected_date.month]
    weekday_factor = 0.86 if selected_date.weekday() >= 5 else 1.0
    daily_market_factor = 0.94 + ((selected_date.toordinal() * 17) % 13) / 100
    return round(base_rate * seasonal_factor * weekday_factor * daily_market_factor, 3)


def populate_berlin_sample_data(manager: DatabaseManager, days: int = 30, seed: int = 42) -> dict:
    """Reset and fill the database with repeatable, realistic hourly demo data."""
    if days < 1:
        raise ValueError("days must be at least 1")

    manager.reset_database()
    randomizer = random.Random(seed)
    start = (datetime.now().replace(minute=0, second=0, microsecond=0) - timedelta(days=days - 1)).replace(hour=0)
    weather_cycle = [("sunny", 1.0), ("partly_cloudy", 0.62), ("cloudy", 0.28), ("rainy", 0.10)]
    device_profiles = {
        "EV": ("Home EV Charger", 1.15),
        "HVAC": ("Heat Pump", 0.62),
        "appliance": ("Flexible Appliances", 0.38),
    }
    usage_count = 0
    solar_count = 0

    for day_offset in range(days):
        day = start + timedelta(days=day_offset)
        condition, weather_factor = weather_cycle[day_offset % len(weather_cycle)]
        temperature_base = 8 + ((day_offset * 3) % 14)
        for hour in range(24):
            timestamp = day.replace(hour=hour)
            price = berlin_price_for_hour(hour, timestamp.date())
            ev_load = 6.8 if hour in {0, 1, 2, 3, 4, 5} else 1.1 if hour in {11, 12, 13, 14} else 0.18
            hvac_load = 1.25 if hour in {6, 7, 17, 18, 19, 20} else 0.62
            appliance_load = 1.45 if hour in {19, 20, 21} else 0.58
            base_loads = {"EV": ev_load, "HVAC": hvac_load, "appliance": appliance_load}
            for device_type, base_load in base_loads.items():
                device_name, variation = device_profiles[device_type]
                consumption = round(max(0.08, base_load * randomizer.uniform(1 - variation * 0.15, 1 + variation * 0.15)), 3)
                manager.add_usage_record(
                    timestamp,
                    consumption,
                    device_type,
                    device_name,
                    round(consumption * price, 3),
                )
                usage_count += 1

            if 6 <= hour <= 18:
                solar_curve = max(0, 1 - abs(hour - 12) / 6)
                irradiance = round(820 * solar_curve * weather_factor, 1)
                generation = round(max(0, 4.8 * solar_curve * weather_factor * randomizer.uniform(0.92, 1.08)), 3)
                manager.add_generation_record(
                    timestamp,
                    generation,
                    condition,
                    round(temperature_base + randomizer.uniform(-2.0, 2.0), 1),
                    irradiance,
                )
                solar_count += 1

    return {"days": days, "usage_records": usage_count, "solar_records": solar_count}
