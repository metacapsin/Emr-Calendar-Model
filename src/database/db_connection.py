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
    uri = os.getenv("MONGO_URI")
    if not uri:
        raise RuntimeError("MONGO_URI not set in .env")
    return uri


def get_db_name() -> str:
    return os.getenv("MONGO_DB_NAME") or _load_config().get("database", {}).get("db_name", "dev")


def get_client() -> MongoClient:
    global _client
    if _client is None:
        _client = MongoClient(get_mongo_uri(), serverSelectionTimeoutMS=10000)
    return _client


def get_database() -> Database:
    global _db
    if _db is None:
        _db = get_client()[get_db_name()]
    return _db


def get_collection_names() -> dict:
    cols = _load_config().get("database", {}).get("collections", {})
    return {
        "patients":           cols.get("patients",           "patient-details"),
        "providers":          cols.get("providers",          "users"),
        "appointments":       cols.get("appointments",       "appointments"),
        "provider_schedules": cols.get("provider_schedules", "provider_schedules"),
        "slot_statistics":    cols.get("slot_statistics",    "slot_statistics"),
    }


def get_db():
    """FastAPI dependency — yields MongoDB database instance. Read-only."""
    db = get_database()
    try:
        yield db
    finally:
        pass


def init_db() -> None:
    """Verify MongoDB connection only — NO writes, NO index creation."""
    get_database().command("ping")
    from src.utils.logger import get_logger
    get_logger(__name__).info("MongoDB connected: db=%s", get_db_name())
