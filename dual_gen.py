"""
Image generation and download utilities.

Handles communication with generation endpoints and result logging.
"""

import csv
import os
import time
import json
import requests
from datetime import datetime

from pipeline_common import load_config, detect_image_type_from_base64, setup_logger

CONFIG = load_config()
ENDPOINTS = CONFIG.get("endpoints", [])

logger = setup_logger("dual_gen_worker", level=CONFIG.get("logging_level", "INFO"))

DEFAULT_CONFIG = {
    "orientation": "landscape",
    "size": "1mp",
    "steps": 25,
    "batch": 1
}

LOG_FILE = os.path.join(os.path.dirname(__file__), "generation_log.csv")


def log_result(result, prompt):
    """Log generation result to CSV file."""
    file_exists = os.path.isfile(LOG_FILE)

    with open(LOG_FILE, "a", newline="", encoding="utf-8") as csvfile:
        fieldnames = ["timestamp", "endpoint", "prompt", "seed", "filename", "duration", "status", "error"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        if not file_exists:
            writer.writeheader()

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        if result["success"]:
            writer.writerow({
                "timestamp": timestamp,
                "endpoint": result["endpoint"]["name"],
                "prompt": prompt,
                "seed": result["stats"].get("seed"),
                "filename": result["local_path"],
                "duration": f"{result['duration']:.2f}",
                "status": "Success",
                "error": ""
            })
        else:
            writer.writerow({
                "timestamp": timestamp,
                "endpoint": result["endpoint"]["name"],
                "prompt": prompt,
                "seed": "",
                "filename": "",
                "duration": "",
                "status": "Failed",
                "error": result.get("error", "Unknown error")
            })


def generate_and_download(endpoint, prompt, image_base64=None, orientation=None, size=None, steps=None, seed=None, strength=0.75, guidance_scale=None):
    """
    Send generation request to endpoint and download result.

    Args:
        endpoint: Endpoint config dict with ip, port, name
        prompt: Generation prompt
        image_base64: Optional base64 input image for img2img
        orientation: Image orientation (landscape, portrait, square)
        size: Image size (0.5mp, 1mp, 2mp, 4mp)
        steps: Number of generation steps
        seed: Optional seed for reproducibility
        strength: img2img strength (0-1)
        guidance_scale: Classifier-free guidance scale

    Returns:
        Dict with success status, local_path, endpoint, stats, duration or error
    """
    base_url = f"http://{endpoint['ip']}:{endpoint['port']}"
    api_url = f"{base_url}/generate"

    payload = DEFAULT_CONFIG.copy()
    payload["prompt"] = prompt
    if orientation:
        payload["orientation"] = orientation
    if size:
        payload["size"] = size
    if steps:
        payload["steps"] = steps
    if seed is not None:
        payload["seed"] = seed
    if guidance_scale is not None:
        payload["guidance_scale"] = guidance_scale

    if image_base64:
        if not image_base64.startswith("data:"):
            image_type = detect_image_type_from_base64(image_base64)
            mime_map = {
                'jpeg': 'image/jpeg',
                'png': 'image/png',
                'webp': 'image/webp',
                'gif': 'image/gif',
            }
            mime = mime_map.get(image_type, 'image/png')
            image_base64 = f"data:{mime};base64,{image_base64}"

        payload["input_image"] = image_base64
        payload["strength"] = strength
        detected_mime = image_base64.split(';')[0].split(':')[1] if image_base64.startswith('data:') else 'unknown'
        logger.debug(f"Including input_image", extra={
            'endpoint': endpoint['name'],
            'mime': detected_mime,
            'size': len(image_base64),
            'strength': strength,
        })

    logger.debug(f"Sending request", extra={
        'endpoint': endpoint['name'],
        'payload_keys': [k for k in payload.keys() if k != 'input_image'],
    })

    try:
        start_time = time.time()
        response = requests.post(api_url, json=payload, timeout=720)

        if response.status_code != 200:
            try:
                error_data = response.json()
                error_msg = error_data.get('error', response.text[:500])
            except:
                error_msg = response.text[:500]
            logger.error(f"API Error", extra={
                'endpoint': endpoint['name'],
                'status_code': response.status_code,
                'error': error_msg,
            })
            return {"success": False, "error": f"{response.status_code}: {error_msg}", "endpoint": endpoint}

        data = response.json()

        if not data.get("success"):
            return {"success": False, "error": f"API Error: {data.get('error')}", "endpoint": endpoint}

        image_info = data["images"][0]
        image_filename = image_info["filename"]
        download_url = f"{base_url}/images/{image_filename}"

        logger.debug(f"Downloading image", extra={'endpoint': endpoint['name'], 'image_file': image_filename})
        img_response = requests.get(download_url)
        img_response.raise_for_status()

        output_dir = CONFIG.get("output_directory", ".")
        os.makedirs(output_dir, exist_ok=True)
        local_filename = f"gen_{endpoint['ip'].replace('.', '_')}_{image_filename}"
        local_path = os.path.join(output_dir, local_filename)
        with open(local_path, "wb") as f:
            f.write(img_response.content)

        duration = time.time() - start_time
        logger.info(f"Generation complete", extra={
            'endpoint': endpoint['name'],
            'duration': round(duration, 1),
            'output_file': local_filename,
        })

        return {
            "success": True,
            "local_path": local_path,
            "endpoint": endpoint,
            "stats": image_info,
            "duration": duration
        }

    except Exception as e:
        logger.error(f"Generation failed", extra={'endpoint': endpoint['name'], 'error': str(e)})
        return {"success": False, "error": str(e), "endpoint": endpoint}
