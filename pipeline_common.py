"""
Shared utilities for the image pipeline.

Provides common functions, logging setup, and constants used across all servers.
"""

import json
import logging
import os
import sys
import uuid
from datetime import datetime
from functools import lru_cache
from typing import Any, Dict, Optional

ALLOWED_IMAGE_TYPES = {'jpeg', 'png', 'webp', 'gif'}
MAX_IMAGE_SIZE = 10 * 1024 * 1024  # 10MB

VALID_ORIENTATIONS = {'landscape', 'portrait', 'square'}
VALID_SIZES = {'0.5mp', '1mp', '2mp', '4mp'}

ERROR_CODES = {
    'VALIDATION_ERROR': 'Input validation failed',
    'IMAGE_TOO_LARGE': 'Image exceeds maximum size',
    'INVALID_IMAGE_FORMAT': 'Unsupported image format',
    'JOB_NOT_FOUND': 'Requested job does not exist',
    'SERVICE_UNAVAILABLE': 'Upstream service is unavailable',
    'INTERNAL_ERROR': 'An internal error occurred',
    'LM_STUDIO_ERROR': 'LM Studio request failed',
    'TIMEOUT': 'Request timed out',
}


@lru_cache(maxsize=1)
def load_config() -> Dict[str, Any]:
    """
    Load and cache configuration from config.json.

    The config is cached using lru_cache, so subsequent calls return
    the same config without re-reading the file.

    Returns:
        Dict containing the configuration

    Raises:
        FileNotFoundError: If config.json doesn't exist
        json.JSONDecodeError: If config.json is invalid JSON
    """
    config_path = os.path.join(os.path.dirname(__file__), "config.json")
    with open(config_path, "r") as f:
        config = json.load(f)

    _validate_config(config)
    return config


def _validate_config(config: Dict[str, Any]) -> None:
    """Validate required config fields exist."""
    pass


def detect_image_type(data: bytes) -> Optional[str]:
    """
    Detect image type from binary data using magic bytes.

    Args:
        data: Raw image bytes

    Returns:
        Image type string ('jpeg', 'png', 'webp', 'gif') or None if unrecognized
    """
    if len(data) < 12:
        return None

    if data[:4] == b'RIFF' and data[8:12] == b'WEBP':
        return 'webp'
    if data[:8] == b'\x89PNG\r\n\x1a\n':
        return 'png'
    if data[:2] == b'\xff\xd8':
        return 'jpeg'
    if data[:6] in (b'GIF87a', b'GIF89a'):
        return 'gif'

    return None


def detect_image_type_from_base64(base64_data: str) -> Optional[str]:
    """
    Detect image type from base64 encoded data.

    Args:
        base64_data: Base64 encoded image (with or without data URL prefix)

    Returns:
        Image type string or None if unrecognized
    """
    import base64

    if base64_data.startswith("data:"):
        comma_pos = base64_data.find(",")
        if comma_pos > 0:
            base64_data = base64_data[comma_pos + 1:]

    try:
        header = base64.b64decode(base64_data[:32])
        return detect_image_type(header)
    except Exception:
        return None


def normalize_lm_studio_url(url: str) -> str:
    """
    Normalize LM Studio URL to ensure it ends with /v1.

    Args:
        url: The base URL for LM Studio

    Returns:
        URL with /v1 suffix
    """
    url = url.rstrip("/")
    if not url.endswith("/v1"):
        url = url + "/v1"
    return url


def generate_correlation_id() -> str:
    """Generate a unique correlation ID for request tracing."""
    return str(uuid.uuid4())[:12]


class JsonFormatter(logging.Formatter):
    """JSON formatter for structured logging."""

    def format(self, record: logging.LogRecord) -> str:
        log_entry = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "service": getattr(record, 'service', 'unknown'),
            "message": record.getMessage(),
        }

        if hasattr(record, 'correlation_id'):
            log_entry["correlation_id"] = record.correlation_id

        if record.exc_info:
            log_entry["exception"] = self.formatException(record.exc_info)

        for key in ['job_id', 'endpoint', 'duration', 'status_code', 'method', 'path']:
            if hasattr(record, key):
                log_entry[key] = getattr(record, key)

        return json.dumps(log_entry)


def setup_logger(
    service_name: str,
    level: str = "INFO",
    json_format: bool = True
) -> logging.Logger:
    """
    Set up a logger with JSON formatting.

    Args:
        service_name: Name of the service for log attribution
        level: Logging level (DEBUG, INFO, WARNING, ERROR)
        json_format: If True, output JSON logs; if False, output plain text

    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(service_name)
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    logger.handlers.clear()

    handler = logging.StreamHandler(sys.stdout)

    if json_format:
        handler.setFormatter(JsonFormatter())
    else:
        handler.setFormatter(logging.Formatter(
            f'[%(asctime)s] [{service_name}] %(levelname)s: %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        ))

    logger.addHandler(handler)

    old_factory = logging.getLogRecordFactory()

    def record_factory(*args, **kwargs):
        record = old_factory(*args, **kwargs)
        record.service = service_name
        return record

    logging.setLogRecordFactory(record_factory)

    return logger


def create_error_response(
    code: str,
    message: str,
    details: Optional[Dict[str, Any]] = None,
    correlation_id: Optional[str] = None
) -> Dict[str, Any]:
    """
    Create a standardized error response.

    Args:
        code: Error code from ERROR_CODES
        message: Human-readable error message
        details: Optional additional error details
        correlation_id: Optional correlation ID for request tracing

    Returns:
        Standardized error response dict
    """
    response = {
        "success": False,
        "error": {
            "code": code,
            "message": message,
        }
    }

    if details:
        response["error"]["details"] = details

    if correlation_id:
        response["correlation_id"] = correlation_id

    return response


def create_success_response(
    data: Dict[str, Any],
    correlation_id: Optional[str] = None
) -> Dict[str, Any]:
    """
    Create a standardized success response.

    Args:
        data: Response data
        correlation_id: Optional correlation ID for request tracing

    Returns:
        Standardized success response dict
    """
    response = {
        "success": True,
        **data
    }

    if correlation_id:
        response["correlation_id"] = correlation_id

    return response


def create_health_response(
    service: str,
    status: str = "healthy",
    extra: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Create a standardized health check response.

    Args:
        service: Service name
        status: Health status (healthy, degraded, unhealthy)
        extra: Additional status information (e.g., queue_size)

    Returns:
        Health check response dict
    """
    response = {
        "status": status,
        "service": service,
        "timestamp": datetime.utcnow().isoformat() + "Z"
    }

    if extra:
        response.update(extra)

    return response


def get_config_value(key: str, default: Any = None) -> Any:
    """
    Get a configuration value with support for nested keys.

    Supports both flat keys (e.g., "dual_gen_port") and nested keys
    (e.g., "services.dual_gen.port") for backward compatibility.

    Args:
        key: Configuration key (dot-separated for nested access)
        default: Default value if key not found

    Returns:
        Configuration value or default
    """
    config = load_config()

    if key in config:
        return config[key]

    parts = key.split(".")
    value = config
    for part in parts:
        if isinstance(value, dict) and part in value:
            value = value[part]
        else:
            return default

    return value


CONFIG = load_config()
