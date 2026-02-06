"""
Dual Image Generation Server

Main web UI and job queue orchestration for the image pipeline.
"""

import os
import signal
import sys
import base64
import threading
import time
import uuid
from datetime import datetime
from functools import wraps

import requests
from flask import Flask, render_template, request, jsonify, send_from_directory, Blueprint, g

from pipeline_common import (
    load_config, detect_image_type, setup_logger, create_error_response,
    create_health_response, generate_correlation_id,
    ALLOWED_IMAGE_TYPES, MAX_IMAGE_SIZE
)
from job_queue import PersistentJobQueue, JobStatus, JobState
from schemas import validate_generation_request

import prompt_gen
from dual_gen import generate_and_download, log_result

CONFIG = load_config()
ENDPOINTS = CONFIG.get("endpoints", [])

logger = setup_logger("dual_gen", level=CONFIG.get("logging_level", "INFO"))

app = Flask(__name__, template_folder="templates", static_folder="static")

api_v1 = Blueprint('api_v1', __name__, url_prefix='/api/v1')

db_path = CONFIG.get("database_path", os.path.join(os.path.dirname(__file__), "jobs.db"))
job_queue = PersistentJobQueue(db_path)
job_state = JobState()

worker_thread = None
shutdown_event = threading.Event()


@app.before_request
def before_request():
    """Add correlation ID to all requests."""
    g.correlation_id = request.headers.get('X-Correlation-ID') or generate_correlation_id()


@app.after_request
def after_request(response):
    """Add correlation ID and deprecation headers to responses."""
    if hasattr(g, 'correlation_id'):
        response.headers['X-Correlation-ID'] = g.correlation_id

    if hasattr(g, 'deprecated') and g.deprecated:
        response.headers['X-API-Deprecated'] = 'true'
        logger.warning(
            "Deprecated API endpoint accessed",
            extra={'correlation_id': g.correlation_id, 'path': request.path, 'method': request.method}
        )

    return response


def deprecated_route(f):
    """Decorator to mark routes as deprecated."""
    @wraps(f)
    def decorated(*args, **kwargs):
        g.deprecated = True
        return f(*args, **kwargs)
    return decorated


def run_generation(job):
    """Execute a generation job."""
    job_id = job.id
    payload = job.payload

    prompt = payload.get("prompt", "")
    use_random = payload.get("use_random", False)
    steering_concept = payload.get("steering_concept")
    count = payload.get("count", 1)
    image_base64 = payload.get("image_base64")
    prompt_mode = payload.get("prompt_mode", "same")
    prompt2 = payload.get("prompt2", "")
    orientation = payload.get("orientation", "landscape")
    size = payload.get("size", "1mp")
    steps = payload.get("steps", 25)
    seed = payload.get("seed")
    strength = payload.get("strength", 0.75)
    guidance_scale = payload.get("guidance_scale")

    job_state.current_job_id = job_id
    results = []
    started_at = time.time()

    endpoint_status = {ep["name"]: {"state": "pending", "start_time": None, "elapsed": None} for ep in ENDPOINTS}
    llm_status = {"state": "idle", "start_time": None, "elapsed": None, "model": None, "source": None}

    job_queue.update(
        job_id,
        endpoint_status=endpoint_status,
        llm_status=llm_status,
    )

    logger.info(f"Starting generation job", extra={
        'job_id': job_id,
        'count': count,
        'use_random': use_random,
    })

    for i in range(count):
        if shutdown_event.is_set():
            logger.info(f"Job interrupted by shutdown", extra={'job_id': job_id})
            break

        endpoint_prompts = {}

        if prompt_mode == "different":
            if use_random:
                llm_status = {"state": "generating", "start_time": time.time(), "elapsed": None, "model": None, "source": None}
                job_queue.update(job_id, llm_status=llm_status)

                for ep in ENDPOINTS:
                    llm_result = prompt_gen.generate_prompt(steering_concept=steering_concept, image_base64=image_base64, return_details=True)
                    endpoint_prompts[ep["name"]] = llm_result["prompt"]

                llm_status = {
                    "state": "done" if llm_result["source"] == "llm" else "fallback",
                    "start_time": llm_status["start_time"],
                    "elapsed": llm_result["elapsed"],
                    "model": llm_result["model"],
                    "source": llm_result["source"],
                    "mode": llm_result["mode"],
                    "error": llm_result.get("error")
                }
                job_queue.update(job_id, llm_status=llm_status)
            else:
                endpoint_prompts[ENDPOINTS[0]["name"]] = prompt
                endpoint_prompts[ENDPOINTS[1]["name"]] = prompt2 or prompt
        else:
            if use_random:
                llm_status = {"state": "generating", "start_time": time.time(), "elapsed": None, "model": None, "source": None}
                job_queue.update(job_id, llm_status=llm_status)

                llm_result = prompt_gen.generate_prompt(steering_concept=steering_concept, image_base64=image_base64, return_details=True)
                current_prompt = llm_result["prompt"]

                llm_status = {
                    "state": "done" if llm_result["source"] == "llm" else "fallback",
                    "start_time": llm_status["start_time"],
                    "elapsed": llm_result["elapsed"],
                    "model": llm_result["model"],
                    "source": llm_result["source"],
                    "mode": llm_result["mode"],
                    "error": llm_result.get("error")
                }
                job_queue.update(job_id, llm_status=llm_status)
            else:
                current_prompt = prompt
            for ep in ENDPOINTS:
                endpoint_prompts[ep["name"]] = current_prompt

        job_queue.update(
            job_id,
            current_run=i + 1,
            current_prompt=endpoint_prompts.get(ENDPOINTS[0]["name"], ""),
            endpoint_prompts=endpoint_prompts,
        )

        start_time = time.time()
        endpoint_status = {ep["name"]: {"state": "generating", "start_time": start_time, "elapsed": None} for ep in ENDPOINTS}
        job_queue.update(job_id, endpoint_status=endpoint_status)

        import concurrent.futures
        import random as rng
        gs = rng.choice([1, 2, 3.5, 5, 7, 10]) if guidance_scale == "random" else guidance_scale

        with concurrent.futures.ThreadPoolExecutor() as executor:
            future_to_endpoint = {
                executor.submit(generate_and_download, ep, endpoint_prompts[ep["name"]], image_base64, orientation, size, steps, seed, strength, gs): ep
                for ep in ENDPOINTS
            }

            run_results = []
            for future in concurrent.futures.as_completed(future_to_endpoint):
                ep = future_to_endpoint[future]
                res = future.result()
                res["prompt_used"] = endpoint_prompts[ep["name"]]
                run_results.append(res)
                log_result(res, endpoint_prompts[ep["name"]])
                elapsed = time.time() - start_time

                endpoint_status[ep["name"]] = {
                    "state": "done" if res.get("success") else "error",
                    "start_time": start_time,
                    "elapsed": round(elapsed, 1)
                }
                job_queue.update(job_id, endpoint_status=endpoint_status)

        results.append({
            "prompt": endpoint_prompts.get(ENDPOINTS[0]["name"], ""),
            "endpoint_prompts": endpoint_prompts,
            "guidance_scale": gs,
            "images": run_results
        })
        job_queue.update(job_id, results=results)

    total_elapsed = round(time.time() - started_at, 1)
    job_queue.complete(job_id, results, total_elapsed)
    job_state.current_job_id = None

    logger.info(f"Generation job completed", extra={
        'job_id': job_id,
        'duration': total_elapsed,
        'count': count,
    })


def queue_worker():
    """Worker thread that processes jobs from the queue."""
    logger.info("Queue worker started")

    while not shutdown_event.is_set():
        job = job_queue.dequeue(timeout=1.0)
        if job is None:
            continue

        if job.status == JobStatus.CANCELLED:
            continue

        try:
            run_generation(job)
        except Exception as e:
            logger.error(f"Generation failed", extra={'job_id': job.id, 'error': str(e)})
            job_queue.fail(job.id, str(e))
            job_state.current_job_id = None

    logger.info("Queue worker stopped")


def graceful_shutdown(signum, frame):
    """Handle shutdown signals gracefully."""
    logger.info(f"Received signal {signum}, initiating graceful shutdown")
    shutdown_event.set()
    job_queue.shutdown()

    if worker_thread and worker_thread.is_alive():
        logger.info("Waiting for worker thread to finish...")
        worker_thread.join(timeout=30)

    logger.info("Shutdown complete")
    sys.exit(0)


@app.route("/")
def index():
    return render_template("dual_gen.html")


@app.route("/gallery")
def gallery():
    return render_template("gallery.html")


@app.route("/health")
def health():
    """Health check endpoint."""
    queue_status = job_queue.get_queue_status()
    return jsonify(create_health_response(
        service="dual_gen",
        status="healthy",
        extra={"queue_size": queue_status["pending"], "current_job": queue_status["current_job_id"]}
    ))


@api_v1.route("/generate", methods=["POST"])
def api_v1_generate():
    """API v1 generate endpoint with validation."""
    return _handle_generate()


@app.route("/api/generate", methods=["POST"])
@deprecated_route
def api_generate():
    """Legacy generate endpoint (deprecated)."""
    return _handle_generate()


def _handle_generate():
    """Common handler for generate requests."""
    correlation_id = g.correlation_id
    image_base64 = None

    logger.debug(f"Generate request received", extra={
        'correlation_id': correlation_id,
        'content_type': request.content_type,
    })

    if request.content_type and request.content_type.startswith("multipart/form-data"):
        data = {
            "prompt": request.form.get("prompt", "").strip(),
            "prompt2": request.form.get("prompt2", "").strip(),
            "random": request.form.get("random", "").lower() in ("true", "1", "yes"),
            "count": request.form.get("count", 1),
            "prompt_mode": request.form.get("prompt_mode", "same"),
            "orientation": request.form.get("orientation", "landscape"),
            "size": request.form.get("size", "1mp"),
            "steps": request.form.get("steps", 25),
            "seed": request.form.get("seed", ""),
            "strength": request.form.get("strength", 0.75),
            "guidance_scale": request.form.get("guidance_scale", ""),
        }

        if data["guidance_scale"] == "random":
            pass
        elif data["guidance_scale"]:
            data["guidance_scale"] = float(data["guidance_scale"])
        else:
            data["guidance_scale"] = None

        if "image" in request.files:
            image_file = request.files["image"]
            if image_file.filename:
                image_data = image_file.read()

                if len(image_data) > MAX_IMAGE_SIZE:
                    return jsonify(create_error_response(
                        "IMAGE_TOO_LARGE",
                        f"Image too large. Maximum size is {MAX_IMAGE_SIZE // (1024*1024)}MB",
                        correlation_id=correlation_id
                    )), 400

                image_type = detect_image_type(image_data)
                if image_type not in ALLOWED_IMAGE_TYPES:
                    return jsonify(create_error_response(
                        "INVALID_IMAGE_FORMAT",
                        f"Invalid image format: {image_type or 'unknown'}. Supported: JPEG, PNG, WebP, GIF",
                        correlation_id=correlation_id
                    )), 400

                image_base64 = base64.b64encode(image_data).decode("utf-8")
                logger.debug(f"Received image", extra={
                    'correlation_id': correlation_id,
                    'image_name': image_file.filename,
                    'size': len(image_data),
                })
    else:
        data = request.json or {}
        image_base64 = data.get("image")

    validation = validate_generation_request(data)
    if not validation.is_valid:
        return jsonify(validation.to_error_response(correlation_id)), 400

    validated = validation.data
    if image_base64:
        validated["image_base64"] = image_base64

    if validated["prompt_mode"] == "different" and not validated["use_random"] and not validated.get("prompt2"):
        validated["prompt2"] = validated["prompt"]

    job_id = str(uuid.uuid4())[:8]

    payload = {
        "prompt": validated["prompt"],
        "prompt2": validated.get("prompt2", ""),
        "use_random": validated["use_random"],
        "steering_concept": validated["prompt"] if validated["use_random"] else None,
        "count": validated["count"],
        "image_base64": validated.get("image_base64"),
        "prompt_mode": validated["prompt_mode"],
        "orientation": validated["orientation"],
        "size": validated["size"],
        "steps": validated["steps"],
        "seed": validated["seed"],
        "strength": validated["strength"],
        "guidance_scale": validated["guidance_scale"],
    }

    job = job_queue.enqueue(job_id, payload)

    logger.info(f"Job queued", extra={
        'correlation_id': correlation_id,
        'job_id': job_id,
        'count': validated["count"],
    })

    return jsonify({
        "success": True,
        "job_id": job_id,
        "status": "queued",
        "queue_position": job_queue.size(),
        "correlation_id": correlation_id,
    })


@api_v1.route("/status/<job_id>")
def api_v1_status(job_id):
    """API v1 job status endpoint."""
    return _handle_status(job_id)


@app.route("/api/status/<job_id>")
@deprecated_route
def api_status(job_id):
    """Legacy status endpoint (deprecated)."""
    return _handle_status(job_id)


def _handle_status(job_id):
    """Common handler for status requests."""
    job = job_queue.get(job_id)
    if job is None:
        return jsonify(create_error_response(
            "JOB_NOT_FOUND",
            "Job not found",
            correlation_id=g.correlation_id
        )), 404
    return jsonify(job.to_dict())


@api_v1.route("/jobs")
def api_v1_jobs():
    """API v1 list all jobs."""
    return _handle_jobs()


@app.route("/api/jobs")
@deprecated_route
def api_jobs():
    """Legacy jobs endpoint (deprecated)."""
    return _handle_jobs()


def _handle_jobs():
    """Common handler for jobs list."""
    all_jobs = job_queue.get_all_jobs()
    return jsonify([
        job.to_dict()
        for category in ['pending', 'running', 'completed']
        for job in all_jobs[category]
    ])


@api_v1.route("/queue")
def api_v1_queue():
    """API v1 queue status."""
    return _handle_queue()


@app.route("/api/queue")
@deprecated_route
def api_queue():
    """Legacy queue endpoint (deprecated)."""
    return _handle_queue()


def _handle_queue():
    """Common handler for queue status."""
    all_jobs = job_queue.get_all_jobs()
    status = job_queue.get_queue_status()

    return jsonify({
        "pending": [job.to_dict() for job in all_jobs["pending"]],
        "running": [job.to_dict() for job in all_jobs["running"]],
        "completed": [job.to_dict() for job in all_jobs["completed"][:20]],
        "current_job_id": status["current_job_id"],
        "queue_size": status["pending"],
    })


@api_v1.route("/queue/<job_id>", methods=["DELETE"])
def api_v1_cancel_job(job_id):
    """API v1 cancel job."""
    return _handle_cancel_job(job_id)


@app.route("/api/queue/<job_id>", methods=["DELETE"])
@deprecated_route
def api_cancel_job(job_id):
    """Legacy cancel endpoint (deprecated)."""
    return _handle_cancel_job(job_id)


def _handle_cancel_job(job_id):
    """Common handler for job cancellation."""
    job = job_queue.get(job_id)
    if job is None:
        return jsonify(create_error_response(
            "JOB_NOT_FOUND",
            "Job not found",
            correlation_id=g.correlation_id
        )), 404

    if job.status != JobStatus.QUEUED:
        return jsonify(create_error_response(
            "VALIDATION_ERROR",
            "Can only cancel queued jobs",
            correlation_id=g.correlation_id
        )), 400

    job_queue.cancel(job_id)

    logger.info(f"Job cancelled", extra={
        'correlation_id': g.correlation_id,
        'job_id': job_id,
    })

    return jsonify({"success": True, "job_id": job_id})


@api_v1.route("/queue/clear", methods=["POST"])
def api_v1_clear_queue():
    """API v1 clear queue."""
    return _handle_clear_queue()


@app.route("/api/queue/clear", methods=["POST"])
@deprecated_route
def api_clear_queue():
    """Legacy clear queue endpoint (deprecated)."""
    return _handle_clear_queue()


def _handle_clear_queue():
    """Common handler for clearing queue."""
    cleared = job_queue.clear_queue()
    logger.info(f"Queue cleared", extra={
        'correlation_id': g.correlation_id,
        'cleared_count': cleared,
    })
    return jsonify({"success": True, "cleared": cleared})


@app.route("/images/<path:filename>")
def serve_image(filename):
    output_dir = CONFIG.get("output_directory", ".")
    return send_from_directory(output_dir, filename)


@api_v1.route("/endpoints")
def api_v1_endpoints():
    """API v1 endpoints status."""
    return _handle_endpoints()


@app.route("/api/endpoints")
@deprecated_route
def api_endpoints():
    """Legacy endpoints status (deprecated)."""
    return _handle_endpoints()


def _handle_endpoints():
    """Common handler for endpoints status."""
    results = []
    for ep in ENDPOINTS:
        status = {"name": ep["name"], "ip": ep["ip"], "port": ep["port"]}
        try:
            url = f"http://{ep['ip']}:{ep['port']}/"
            resp = requests.get(url, timeout=3)
            resp.raise_for_status()
            status["status"] = "online"
        except requests.exceptions.Timeout:
            status["status"] = "timeout"
        except requests.exceptions.ConnectionError:
            status["status"] = "offline"
        except requests.exceptions.HTTPError:
            status["status"] = "error"
        except Exception as e:
            logger.warning(f"Unexpected error checking endpoint {ep['name']}", extra={'error': str(e)})
            status["status"] = "unknown"
        results.append(status)
    return jsonify(results)


@api_v1.route("/gallery")
def api_v1_gallery():
    """API v1 gallery endpoint."""
    return _handle_gallery()


@app.route("/api/gallery")
@deprecated_route
def api_gallery():
    """Legacy gallery endpoint (deprecated)."""
    return _handle_gallery()


def _handle_gallery():
    """Common handler for gallery."""
    output_dir = CONFIG.get("output_directory", ".")
    if not os.path.exists(output_dir):
        return jsonify([])

    images = []
    for f in os.listdir(output_dir):
        if f.startswith("._"):
            continue
        if f.lower().endswith((".png", ".jpg", ".jpeg", ".webp")):
            filepath = os.path.join(output_dir, f)
            stat = os.stat(filepath)
            images.append({
                "filename": f,
                "url": f"/images/{f}",
                "created": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                "mtime": stat.st_mtime,
                "size": stat.st_size
            })
    images.sort(key=lambda x: x["mtime"], reverse=True)
    limit = min(int(request.args.get("limit", 100)), 10000)
    return jsonify(images[:limit])


app.register_blueprint(api_v1)


def main():
    global worker_thread

    signal.signal(signal.SIGTERM, graceful_shutdown)
    signal.signal(signal.SIGINT, graceful_shutdown)

    recovered = job_queue.recover_interrupted()
    if recovered:
        logger.info(f"Recovered {len(recovered)} interrupted jobs")

    host = CONFIG.get("dual_gen_host", "0.0.0.0")
    port = CONFIG.get("dual_gen_port", 5050)

    worker_thread = threading.Thread(target=queue_worker, daemon=False)
    worker_thread.start()

    logger.info(f"Starting Dual Image Generator Web UI", extra={
        'host': host,
        'port': port,
    })

    app.run(host=host, port=port, debug=False, threaded=True)


if __name__ == "__main__":
    main()
