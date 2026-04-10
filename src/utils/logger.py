import json
import logging
import sys
from datetime import datetime, timezone
from typing import Any, Dict, Optional


class JSONFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        log_obj: Dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info:
            log_obj["exception"] = self.formatException(record.exc_info)
        if hasattr(record, "extra"):
            log_obj.update(record.extra)
        return json.dumps(log_obj)


def get_logger(name: str, level: str = "INFO") -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JSONFormatter())
    logger.addHandler(handler)
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    logger.propagate = False
    return logger


def log_request(logger: logging.Logger, endpoint: str, payload: Dict[str, Any]) -> None:
    logger.info("API request", extra={"extra": {"endpoint": endpoint, "payload_keys": list(payload.keys())}})


def log_prediction(logger: logging.Logger, patient_id: Optional[int], provider_id: Optional[int], slot_count: int) -> None:
    logger.info(
        "Prediction completed",
        extra={"extra": {"patient_id": patient_id, "provider_id": provider_id, "slots_returned": slot_count}},
    )
