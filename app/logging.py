import json
import logging
from typing import Any, Dict

from app.utils.phi import redact_phi


class PHIRedactionFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        if isinstance(record.args, dict):
            record.args = redact_phi(record.args)
        elif isinstance(record.args, tuple):
            record.args = tuple(
                redact_phi(arg) if isinstance(arg, dict) else arg for arg in record.args
            )
        for attr in ["message", "msg"]:
            value = getattr(record, attr, None)
            if isinstance(value, dict):
                redacted = redact_phi(value)
                setattr(record, attr, json.dumps(redacted))
        return True


def configure_logging(level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger("appeals")
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.addFilter(PHIRedactionFilter())
        formatter = logging.Formatter("%(asctime)s %(levelname)s %(name)s - %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    logger.setLevel(level)
    return logger
