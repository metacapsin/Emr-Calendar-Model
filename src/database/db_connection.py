from pathlib import Path
from typing import Optional

import os
import yaml
from dotenv import load_dotenv
from pymongo import MongoClient
from pymongo.database import Database

# Load .env — try project root first, then cwd
_env_path = Path(__file__).resolve().parents[3] / ".env"
if not _env_path.exists():
    _env_path = Path(".") / ".env"
load_dotenv(dotenv_path=_env_path, override=False)

_client: Optional[MongoClient] = None
_db: Optional[Database] = None


def _load_config() -> dict:
    cfg_path = Path("configs/config.yaml")
    if cfg_path.exists():
        with cfg_path.open("r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    return {}


def get_mongo_uri() -> str:
    """Read URI from .env — already percent-encoded by user."""
    uri = os.getenv("MONGO_URI")
    if not uri:
        raise RuntimeError("MONGO_URI not set in .env")
    return uri


def get_db_name() -> str:
    return os.getenv("MONGO_DB_NAME") or _load_config().get("database", {}).get("db_name", "admin")


def get_client() -> MongoClient:
    global _client
    if _client is None:
        uri = get_mongo_uri()
        _client = MongoClient(uri, serverSelectionTimeoutMS=10000)
    return _client


def get_database() -> Database:
    global _db
    if _db is None:
        _db = get_client()[get_db_name()]
    return _db


def get_collection_names() -> dict:
    cfg = _load_config()
    cols = cfg.get("database", {}).get("collections", {})
    return {
        "patients":           cols.get("patients",           "patients"),
        "providers":          cols.get("providers",          "providers"),
        "appointments":       cols.get("appointments",       "appointments"),
        "provider_schedules": cols.get("provider_schedules", "provider_schedules"),
        "slot_statistics":    cols.get("slot_statistics",    "slot_statistics"),
    }


def get_db():
    """FastAPI dependency — yields MongoDB database instance."""
    db = get_database()
    try:
        yield db
    finally:
        pass  # MongoClient is a singleton — do not close per-request


def init_db() -> None:
    """Verify connection and ensure indexes exist."""
    db = get_database()
    cols = get_collection_names()

    # Patients — unique index on patient_encoded
    db[cols["patients"]].create_index("patient_encoded", unique=True, background=True)

    # Providers — unique index on provider_encoded
    db[cols["providers"]].create_index("provider_encoded", unique=True, background=True)

    # Appointments — compound index for fast provider+date lookups
    db[cols["appointments"]].create_index(
        [("provider_encoded", 1), ("appt_date", 1), ("appt_hour", 1)],
        background=True,
    )
    db[cols["appointments"]].create_index("patient_encoded", background=True)

    # Slot statistics — unique compound index
    db[cols["slot_statistics"]].create_index(
        [("provider_encoded", 1), ("weekday", 1), ("hour", 1)],
        unique=True,
        background=True,
    )

    # Provider schedules — compound index
    db[cols["provider_schedules"]].create_index(
        [("provider_encoded", 1), ("blocked_date", 1)],
        unique=True,
        background=True,
    )

    from src.utils.logger import get_logger
    get_logger(__name__).info(
        "MongoDB connected: db=%s", get_db_name()
    )
