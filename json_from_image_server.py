"""
JSON from Image Server

Vision analysis service that converts images to structured JSON scene descriptions.
"""

import json
import base64
import re
import io

from flask import Flask, request, jsonify, render_template, g
import requests
from PIL import Image

from pipeline_common import (
    load_config, normalize_llm_url, setup_logger,
    create_error_response, create_health_response, generate_correlation_id
)

CONFIG = load_config()

LLM_URL = normalize_llm_url(CONFIG.get("llm_url", "http://localhost:11434"))
DEFAULT_MODEL = CONFIG.get("vision_model", "")
GENERATOR_ENDPOINT = f"http://127.0.0.1:{CONFIG.get('dual_gen_port', 5050)}"
GENERATOR_PATH = "/api/generate"

logger = setup_logger("json_from_image", level=CONFIG.get("logging_level", "INFO"))

app = Flask(__name__, template_folder="templates")
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024

SYSTEM_PROMPT = """You are analyzing a photograph or image. Output ONLY valid JSON (no markdown, no code fences) matching this exact schema:
{
  "scene": "string - overall scene/environment description",
  "props": "string - foreground elements and props visible",
  "subjects": [
    {
      "description": "string - detailed subject description",
      "position": "string - where in the frame (e.g. center foreground, left third)",
      "color_palette": ["string - colors associated with this subject"]
    }
  ],
  "style": "string - photographic style (e.g. cinematic, editorial, product photography)",
  "color_palette": ["string - overall color palette, 3-5 colors"],
  "lighting": "string - lighting setup description",
  "mood": "string - emotional tone/atmosphere",
  "film_stock": "string - film stock or post-processing look (e.g. Kodak Portra 400, Fuji Velvia, matte desaturated)",
  "era": "string - time period or aesthetic era (e.g. 1970s, modern, retro-futuristic, Victorian)",
  "background": "string - background description",
  "composition": "string - compositional technique",
  "time_weather": "string - time of day and weather conditions",
  "camera": {
    "angle": "string - camera angle (e.g. low, eye-level, high, dutch)",
    "distance": "string - shot type (e.g. close-up, medium shot, wide shot)",
    "lens": "string - estimated focal length (e.g. 35mm, 50mm, 85mm, 200mm)",
    "f_number": "string - estimated aperture (e.g. f/1.4, f/2.8, f/8)",
    "iso": "number - estimated ISO value",
    "shutter_speed": "string - estimated shutter speed",
    "aspect_ratio": "string - frame aspect ratio",
    "focus": "string - focus description"
  }
}
Analyze what you see in the image and fill every field. Be specific and technically accurate with camera setting estimates. Output only the JSON."""


@app.before_request
def before_request():
    """Add correlation ID to all requests."""
    g.correlation_id = request.headers.get('X-Correlation-ID') or generate_correlation_id()


@app.after_request
def after_request(response):
    """Add correlation ID header to responses."""
    if hasattr(g, 'correlation_id'):
        response.headers['X-Correlation-ID'] = g.correlation_id
    return response


@app.route("/")
def index():
    return render_template("json_from_image.html")


@app.route("/health")
def health():
    """Health check endpoint."""
    try:
        resp = requests.get(f"{LLM_URL}/models", timeout=3)
        lm_status = "connected" if resp.status_code == 200 else "error"
    except:
        lm_status = "disconnected"

    return jsonify(create_health_response(
        service="json_from_image",
        status="healthy" if lm_status == "connected" else "degraded",
        extra={"llm": lm_status, "model": DEFAULT_MODEL}
    ))


@app.route("/api/models")
def list_models():
    try:
        resp = requests.get(f"{LLM_URL}/models", timeout=5)
        data = resp.json()
        data["default_model"] = DEFAULT_MODEL
        return jsonify(data)
    except Exception as e:
        logger.error(f"Failed to list models", extra={
            'correlation_id': g.correlation_id,
            'error': str(e),
        })
        return jsonify(create_error_response(
            "LLM_ERROR",
            str(e),
            details={"default_model": DEFAULT_MODEL},
            correlation_id=g.correlation_id
        )), 502


def convert_to_jpeg(image_bytes):
    """Convert image to JPEG format."""
    img = Image.open(io.BytesIO(image_bytes))
    if img.mode in ('RGBA', 'LA', 'P'):
        img = img.convert('RGB')
    output = io.BytesIO()
    img.save(output, format='JPEG', quality=90)
    return output.getvalue()


@app.route("/api/analyze", methods=["POST"])
def analyze():
    """Analyze an image and return structured JSON description."""
    correlation_id = g.correlation_id

    logger.debug(f"Analyze request received", extra={
        'correlation_id': correlation_id,
        'content_type': request.content_type,
        'has_files': bool(request.files),
    })

    if "image" in request.files:
        file = request.files["image"]
        raw_bytes = file.read()
        mime = file.content_type or "image/jpeg"
        if mime == "image/webp":
            raw_bytes = convert_to_jpeg(raw_bytes)
            mime = "image/jpeg"
        image_data = base64.b64encode(raw_bytes).decode("utf-8")
    elif request.is_json and request.json.get("image_base64"):
        image_data = request.json["image_base64"]
        mime = request.json.get("mime", "image/jpeg")
        if mime == "image/webp":
            raw_bytes = base64.b64decode(image_data)
            raw_bytes = convert_to_jpeg(raw_bytes)
            image_data = base64.b64encode(raw_bytes).decode("utf-8")
            mime = "image/jpeg"
    else:
        return jsonify(create_error_response(
            "VALIDATION_ERROR",
            "No image provided",
            correlation_id=correlation_id
        )), 400

    model = request.form.get("model") or (request.json or {}).get("model", "") or DEFAULT_MODEL
    steering = request.form.get("steering", "").strip() or (request.json or {}).get("steering", "").strip()

    data_url = f"data:{mime};base64,{image_data}"

    system = SYSTEM_PROMPT
    if steering:
        system += f"\n\nIMPORTANT: The following creative direction MUST heavily influence your descriptions across all fields: \"{steering}\""

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Analyze this image and describe it as a detailed scene JSON."},
                    {"type": "image_url", "image_url": {"url": data_url}},
                ],
            },
        ],
        "temperature": 0.3,
        "max_tokens": 4096,
    }

    logger.info(f"Sending to LLM", extra={
        'correlation_id': correlation_id,
        'model': model,
        'has_steering': bool(steering),
    })

    try:
        resp = requests.post(
            f"{LLM_URL}/chat/completions",
            json=payload,
            timeout=120,
        )
        if resp.status_code != 200:
            logger.error(f"LLM error", extra={
                'correlation_id': correlation_id,
                'status_code': resp.status_code,
                'mime': mime,
            })
            return jsonify(create_error_response(
                "LLM_ERROR",
                f"LLM returned {resp.status_code}",
                details={"response": resp.text[:500]},
                correlation_id=correlation_id
            )), resp.status_code

        resp.raise_for_status()
        result = resp.json()
        msg = result["choices"][0]["message"]
        content = (msg.get("content") or "").strip()
        if not content:
            content = (msg.get("reasoning_content") or msg.get("reasoning") or "").strip()
        content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL)
        content = re.sub(r'<reasoning>.*?</reasoning>', '', content, flags=re.DOTALL)
        content = content.strip()
        content = re.sub(r"^```(?:json)?\s*", "", content)
        content = re.sub(r"\s*```\s*$", "", content)

        try:
            scene_json = json.loads(content)
        except json.JSONDecodeError:
            match = re.search(r"\{[\s\S]*\}", content)
            if match:
                scene_json = json.loads(match.group(0))
            else:
                logger.error(f"Failed to parse response", extra={
                    'correlation_id': correlation_id,
                    'raw_content': content[:200],
                })
                return jsonify(create_error_response(
                    "LLM_ERROR",
                    "Failed to parse model response",
                    details={"raw": content[:500]},
                    correlation_id=correlation_id
                )), 422

        logger.info(f"Analysis complete", extra={
            'correlation_id': correlation_id,
            'scene': scene_json.get('scene', '')[:50],
        })
        return jsonify(scene_json)

    except requests.exceptions.Timeout:
        logger.error(f"LLM timeout", extra={'correlation_id': correlation_id})
        return jsonify(create_error_response(
            "TIMEOUT",
            "LLM request timed out",
            correlation_id=correlation_id
        )), 504

    except requests.exceptions.ConnectionError:
        logger.error(f"Cannot connect to LLM", extra={
            'correlation_id': correlation_id,
            'url': LLM_URL,
        })
        return jsonify(create_error_response(
            "SERVICE_UNAVAILABLE",
            f"Cannot connect to LLM at {LLM_URL}",
            correlation_id=correlation_id
        )), 502

    except Exception as e:
        logger.error(f"Unexpected error", extra={
            'correlation_id': correlation_id,
            'error': str(e),
        })
        return jsonify(create_error_response(
            "INTERNAL_ERROR",
            str(e),
            correlation_id=correlation_id
        )), 500


@app.route("/api/send", methods=["POST"])
def send_to_generator():
    """Send analyzed scene to the generator."""
    correlation_id = g.correlation_id
    scene_data = request.json
    payload = {"prompt": json.dumps(scene_data, indent=2), "random": False, "count": 1}

    logger.info(f"Sending to generator", extra={
        'correlation_id': correlation_id,
        'endpoint': GENERATOR_ENDPOINT,
    })

    try:
        resp = requests.post(
            f"{GENERATOR_ENDPOINT}{GENERATOR_PATH}",
            json=payload,
            timeout=30,
        )
        return jsonify(resp.json())
    except Exception as e:
        logger.error(f"Generator send failed", extra={
            'correlation_id': correlation_id,
            'error': str(e),
        })
        return jsonify(create_error_response(
            "SERVICE_UNAVAILABLE",
            str(e),
            correlation_id=correlation_id
        )), 502


def main():
    host = CONFIG.get("json_from_image_host", "0.0.0.0")
    port = CONFIG.get("json_from_image_port", 4000)

    logger.info(f"Starting JSON from Image server", extra={
        'host': host,
        'port': port,
        'llm_url': LLM_URL,
        'model': DEFAULT_MODEL,
    })

    app.run(host=host, port=port, debug=False, threaded=True)


if __name__ == "__main__":
    main()
