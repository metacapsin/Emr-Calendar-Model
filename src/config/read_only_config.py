"""
Read-Only Configuration & Write Control Layer
==============================================
Centralized configuration manager for DB write operations.
When DB_WRITE_ENABLED is false, all MongoDB write operations are blocked.

Usage:
    from src.config.read_only_config import is_write_enabled, log_write_blocked
    
    if not is_write_enabled():
        log_write_blocked("function_name", "operation_type")
        return safe_default
"""
import yaml
from pathlib import Path
from contextlib import contextmanager
from typing import Any

from src.utils.logger import get_logger

logger = get_logger(__name__)
_write_enabled = None


def load_write_config() -> bool:
    """
    Load DB_WRITE_ENABLED from config.yaml.
    Caches result for performance.
    Logs warning if read-only mode is active.
    """
    global _write_enabled
    if _write_enabled is not None:
        return _write_enabled
    
    # Try multiple config paths
    cfg_path = Path("configs/config.yaml")
    if not cfg_path.exists():
        # Fallback to parent directory
        cfg_path = Path(__file__).resolve().parents[2] / "configs" / "config.yaml"
    
    if cfg_path.exists():
        with cfg_path.open("r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        _write_enabled = cfg.get("DB_WRITE_ENABLED", False)
    else:
        logger.warning("Config file not found, defaulting to READ-ONLY mode")
        _write_enabled = False
    
    if not _write_enabled:
        logger.warning("=" * 70)
        logger.warning("READ-ONLY MODE ENABLED: All DB write operations will be skipped")
        logger.warning("=" * 70)
    
    return _write_enabled


def is_write_enabled() -> bool:
    """Check if database write operations are permitted."""
    return load_write_config()


def log_write_blocked(func_name: str, operation: str) -> None:
    """
    Log a blocked write operation with structured information.
    
    Args:
        func_name: Name of the function attempting the write
        operation: Type of operation (e.g., 'insert_one', 'update_one')
    """
    logger.warning(
        "DB WRITE BLOCKED: %s attempted %s (read-only mode)",
        func_name,
        operation
    )


@contextmanager
def write_guard(operation_name: str):
    """
    Context manager that blocks write operations in read-only mode.
    Provides an extra safety layer for critical sections.
    
    Usage:
        with write_guard("critical_operation"):
            db.collection.insert_one(doc)
    
    Raises:
        PermissionError: If write operation is attempted in read-only mode
    """
    if not is_write_enabled():
        log_write_blocked(operation_name, "write operation")
        raise PermissionError(
            f"Write operation blocked: {operation_name} (read-only mode)"
        )
    yield


def get_write_mode_status() -> dict:
    """
    Return comprehensive write mode status for monitoring.
    
    Returns:
        dict with write_enabled status and mode description
    """
    enabled = is_write_enabled()
    return {
        "write_enabled": enabled,
        "mode": "READ_WRITE" if enabled else "READ_ONLY",
        "description": (
            "Full database access enabled" if enabled 
            else "All write operations blocked - production safe mode"
        )
    }
