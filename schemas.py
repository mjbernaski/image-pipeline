"""
Input validation schemas for the image pipeline.

Provides validation functions and structured error responses.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

VALID_ORIENTATIONS = {'landscape', 'portrait', 'square'}
VALID_SIZES = {'0.5mp', '1mp', '2mp', '4mp'}
VALID_PROMPT_MODES = {'same', 'different'}

LIMITS = {
    'count': {'min': 1, 'max': 100},
    'steps': {'min': 1, 'max': 150},
    'strength': {'min': 0.0, 'max': 1.0},
    'guidance_scale': {'min': 0.0, 'max': 30.0},
    'temperature': {'min': 0.0, 'max': 2.0},
}


@dataclass
class ValidationError:
    """Represents a single validation error."""
    field: str
    message: str
    code: str = "VALIDATION_ERROR"


@dataclass
class ValidationResult:
    """Result of validation containing errors or validated data."""
    is_valid: bool
    errors: List[ValidationError] = field(default_factory=list)
    data: Dict[str, Any] = field(default_factory=dict)

    def add_error(self, field: str, message: str, code: str = "VALIDATION_ERROR") -> None:
        self.errors.append(ValidationError(field=field, message=message, code=code))
        self.is_valid = False

    def to_error_response(self, correlation_id: Optional[str] = None) -> Dict[str, Any]:
        """Convert validation errors to API error response format."""
        response = {
            "success": False,
            "error": {
                "code": "VALIDATION_ERROR",
                "message": "Input validation failed",
                "details": {
                    "fields": [
                        {"field": e.field, "message": e.message, "code": e.code}
                        for e in self.errors
                    ]
                }
            }
        }
        if correlation_id:
            response["correlation_id"] = correlation_id
        return response


def validate_count(value: Any) -> Tuple[bool, Optional[int], Optional[str]]:
    """
    Validate the count parameter.

    Args:
        value: Value to validate

    Returns:
        Tuple of (is_valid, parsed_value, error_message)
    """
    try:
        count = int(value)
        if count < LIMITS['count']['min']:
            return False, None, f"count must be at least {LIMITS['count']['min']}"
        if count > LIMITS['count']['max']:
            return False, None, f"count must be at most {LIMITS['count']['max']}"
        return True, count, None
    except (TypeError, ValueError):
        return False, None, "count must be an integer"


def validate_steps(value: Any) -> Tuple[bool, Optional[int], Optional[str]]:
    """
    Validate the steps parameter.

    Args:
        value: Value to validate

    Returns:
        Tuple of (is_valid, parsed_value, error_message)
    """
    try:
        steps = int(value)
        if steps < LIMITS['steps']['min']:
            return False, None, f"steps must be at least {LIMITS['steps']['min']}"
        if steps > LIMITS['steps']['max']:
            return False, None, f"steps must be at most {LIMITS['steps']['max']}"
        return True, steps, None
    except (TypeError, ValueError):
        return False, None, "steps must be an integer"


def validate_strength(value: Any) -> Tuple[bool, Optional[float], Optional[str]]:
    """
    Validate the strength parameter for img2img.

    Args:
        value: Value to validate

    Returns:
        Tuple of (is_valid, parsed_value, error_message)
    """
    try:
        strength = float(value)
        if strength < LIMITS['strength']['min']:
            return False, None, f"strength must be at least {LIMITS['strength']['min']}"
        if strength > LIMITS['strength']['max']:
            return False, None, f"strength must be at most {LIMITS['strength']['max']}"
        return True, strength, None
    except (TypeError, ValueError):
        return False, None, "strength must be a number"


def validate_guidance_scale(value: Any) -> Tuple[bool, Optional[Union[float, str]], Optional[str]]:
    """
    Validate the guidance_scale parameter.

    Args:
        value: Value to validate (can be number or "random")

    Returns:
        Tuple of (is_valid, parsed_value, error_message)
    """
    if value == "random":
        return True, "random", None
    if value is None:
        return True, None, None
    try:
        gs = float(value)
        if gs < LIMITS['guidance_scale']['min']:
            return False, None, f"guidance_scale must be at least {LIMITS['guidance_scale']['min']}"
        if gs > LIMITS['guidance_scale']['max']:
            return False, None, f"guidance_scale must be at most {LIMITS['guidance_scale']['max']}"
        return True, gs, None
    except (TypeError, ValueError):
        return False, None, "guidance_scale must be a number or 'random'"


def validate_orientation(value: Any) -> Tuple[bool, Optional[str], Optional[str]]:
    """
    Validate the orientation parameter.

    Args:
        value: Value to validate

    Returns:
        Tuple of (is_valid, parsed_value, error_message)
    """
    if value is None:
        return True, "landscape", None
    if value in VALID_ORIENTATIONS:
        return True, value, None
    return False, None, f"orientation must be one of: {', '.join(sorted(VALID_ORIENTATIONS))}"


def validate_size(value: Any) -> Tuple[bool, Optional[str], Optional[str]]:
    """
    Validate the size parameter.

    Args:
        value: Value to validate

    Returns:
        Tuple of (is_valid, parsed_value, error_message)
    """
    if value is None:
        return True, "1mp", None
    if value in VALID_SIZES:
        return True, value, None
    return False, None, f"size must be one of: {', '.join(sorted(VALID_SIZES))}"


def validate_prompt_mode(value: Any) -> Tuple[bool, Optional[str], Optional[str]]:
    """
    Validate the prompt_mode parameter.

    Args:
        value: Value to validate

    Returns:
        Tuple of (is_valid, parsed_value, error_message)
    """
    if value is None:
        return True, "same", None
    if value in VALID_PROMPT_MODES:
        return True, value, None
    return False, None, f"prompt_mode must be one of: {', '.join(sorted(VALID_PROMPT_MODES))}"


def validate_seed(value: Any) -> Tuple[bool, Optional[int], Optional[str]]:
    """
    Validate the seed parameter.

    Args:
        value: Value to validate (can be None, empty string, or integer)

    Returns:
        Tuple of (is_valid, parsed_value, error_message)
    """
    if value is None or value == "":
        return True, None, None
    try:
        seed = int(value)
        if seed < 0:
            return False, None, "seed must be a non-negative integer"
        return True, seed, None
    except (TypeError, ValueError):
        return False, None, "seed must be an integer"


def validate_temperature(value: Any) -> Tuple[bool, Optional[float], Optional[str]]:
    if value is None or value == "":
        return True, None, None
    try:
        temp = float(value)
        if temp < LIMITS['temperature']['min']:
            return False, None, f"temperature must be at least {LIMITS['temperature']['min']}"
        if temp > LIMITS['temperature']['max']:
            return False, None, f"temperature must be at most {LIMITS['temperature']['max']}"
        return True, temp, None
    except (TypeError, ValueError):
        return False, None, "temperature must be a number"


def validate_generation_request(data: Dict[str, Any]) -> ValidationResult:
    """
    Validate a complete generation request.

    Args:
        data: Request data dictionary

    Returns:
        ValidationResult with validated data or errors
    """
    result = ValidationResult(is_valid=True, data={})

    valid, count, error = validate_count(data.get("count", 1))
    if valid:
        result.data["count"] = count
    else:
        result.add_error("count", error)

    valid, steps, error = validate_steps(data.get("steps", 25))
    if valid:
        result.data["steps"] = steps
    else:
        result.add_error("steps", error)

    valid, strength, error = validate_strength(data.get("strength", 0.75))
    if valid:
        result.data["strength"] = strength
    else:
        result.add_error("strength", error)

    valid, guidance_scale, error = validate_guidance_scale(data.get("guidance_scale"))
    if valid:
        result.data["guidance_scale"] = guidance_scale
    else:
        result.add_error("guidance_scale", error)

    valid, orientation, error = validate_orientation(data.get("orientation"))
    if valid:
        result.data["orientation"] = orientation
    else:
        result.add_error("orientation", error)

    valid, size, error = validate_size(data.get("size"))
    if valid:
        result.data["size"] = size
    else:
        result.add_error("size", error)

    valid, prompt_mode, error = validate_prompt_mode(data.get("prompt_mode"))
    if valid:
        result.data["prompt_mode"] = prompt_mode
    else:
        result.add_error("prompt_mode", error)

    valid, seed, error = validate_seed(data.get("seed"))
    if valid:
        result.data["seed"] = seed
    else:
        result.add_error("seed", error)

    valid, temperature, error = validate_temperature(data.get("temperature"))
    if valid:
        result.data["temperature"] = temperature
    else:
        result.add_error("temperature", error)

    result.data["prompt"] = data.get("prompt", "").strip() if isinstance(data.get("prompt"), str) else ""
    result.data["prompt2"] = data.get("prompt2", "").strip() if isinstance(data.get("prompt2"), str) else ""

    use_random = data.get("random", False)
    if isinstance(use_random, str):
        use_random = use_random.lower() in ("true", "1", "yes")
    result.data["use_random"] = bool(use_random)

    if not result.data["use_random"] and not result.data["prompt"]:
        result.add_error("prompt", "Prompt is required when not using random mode")

    result.data["image_base64"] = data.get("image") or data.get("image_base64")

    model = data.get("model", "")
    if isinstance(model, str):
        model = model.strip()
    result.data["model"] = model or None

    return result
