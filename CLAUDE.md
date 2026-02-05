# Image Pipeline

Multi-service image generation orchestration system with LLM-powered prompt generation.

## Architecture

```
┌─────────────────────┐     ┌─────────────────────┐
│   Create JSON       │     │  JSON from Image    │
│   (Port 3030)       │     │   (Port 4000)       │
│   - Manual editor   │     │   - Vision analysis │
│   - Proxy to dual   │     │   - Scene → JSON    │
└────────┬────────────┘     └────────┬────────────┘
         │                           │
         └───────────┬───────────────┘
                     │
              ┌──────▼──────┐
              │  Dual Gen   │
              │ (Port 5050) │
              │ - Job queue │
              │ - UI        │
              └──────┬──────┘
                     │
        ┌────────────┼────────────┐
        │            │            │
   ┌────▼────┐  ┌────▼────┐  ┌────▼────┐
   │ LM      │  │Endpoint │  │Endpoint │
   │ Studio  │  │   1     │  │   2     │
   └─────────┘  └─────────┘  └─────────┘
```

## Files

| File | Purpose |
|------|---------|
| `pipeline_common.py` | Shared utilities: config, logging, error responses, image detection |
| `job_queue.py` | SQLite-backed persistent job queue |
| `schemas.py` | Input validation |
| `dual_gen_server.py` | Main server: job orchestration, web UI |
| `dual_gen.py` | Generation execution and result logging |
| `prompt_gen.py` | LLM-based prompt generation |
| `json_from_image_server.py` | Vision analysis service |
| `create_json_server.py` | Manual JSON editor/proxy |

## Quick Start

```bash
./start_all.sh
```

Health checks:
```bash
curl http://localhost:5050/health
curl http://localhost:4000/health
curl http://localhost:3030/health
```

## API Endpoints

### Dual Gen Server (Port 5050)

**New API (v1):**
- `POST /api/v1/generate` - Submit generation job
- `GET /api/v1/status/<job_id>` - Get job status
- `GET /api/v1/queue` - Get queue status
- `DELETE /api/v1/queue/<job_id>` - Cancel job
- `GET /api/v1/gallery` - List generated images
- `GET /api/v1/endpoints` - Endpoint status

**Legacy API (deprecated, logs warnings):**
- Same endpoints at `/api/*` without version prefix

**Health:**
- `GET /health` - Service health check

### JSON from Image (Port 4000)

- `POST /api/analyze` - Analyze image → JSON
- `POST /api/send` - Send to generator
- `GET /api/models` - List LM Studio models
- `GET /health` - Service health

### Create JSON (Port 3030)

- `GET /` - Manual JSON editor UI
- `GET /health` - Service health
- Proxies `/api/*` to Dual Gen
- Proxies `/lm/*` to LM Studio

## Configuration

All settings in `config.json`:

```json
{
    "version": "1.0",
    "output_directory": "/path/to/output",
    "logging_level": "INFO",
    "database_path": "jobs.db",

    "lm_studio_url": "http://localhost:11434",
    "lm_studio_model": "model-name",
    "vision_model": "vision-model-name",

    "dual_gen_port": 5050,
    "json_from_image_port": 4000,
    "create_json_port": 3030,

    "endpoints": [
        {"ip": "192.168.5.40", "port": 2222, "name": "Endpoint 1"}
    ]
}
```

## Common Modifications

### Adding a new generation endpoint

Edit `config.json`:
```json
"endpoints": [
    {"ip": "192.168.5.40", "port": 2222, "name": "Endpoint 1"},
    {"ip": "192.168.5.50", "port": 2222, "name": "Endpoint 3"}
]
```

### Changing LM Studio model

Edit `config.json`:
```json
"lm_studio_model": "new-model-name"
```

### Adjusting validation limits

Edit `config.json`:
```json
"validation_limits": {
    "count": {"min": 1, "max": 200},
    "steps": {"min": 1, "max": 200}
}
```

### Enabling debug logging

Edit `config.json`:
```json
"logging_level": "DEBUG"
```

## Error Response Format

All errors follow this structure:

```json
{
    "success": false,
    "error": {
        "code": "VALIDATION_ERROR",
        "message": "Human readable message",
        "details": {}
    },
    "correlation_id": "abc123"
}
```

Error codes:
- `VALIDATION_ERROR` - Input validation failed
- `IMAGE_TOO_LARGE` - Image exceeds 10MB
- `INVALID_IMAGE_FORMAT` - Unsupported format
- `JOB_NOT_FOUND` - Job doesn't exist
- `SERVICE_UNAVAILABLE` - Upstream down
- `LM_STUDIO_ERROR` - LLM request failed
- `TIMEOUT` - Request timed out
- `INTERNAL_ERROR` - Unexpected error

## Correlation IDs

All requests include `X-Correlation-ID` header for tracing. Pass the header to trace requests across services:

```bash
curl -H "X-Correlation-ID: my-trace-123" http://localhost:5050/api/v1/generate
```

## Job Recovery

The SQLite-backed queue persists jobs across restarts. If the server crashes mid-generation:

1. Jobs with status `RUNNING` are recovered to `QUEUED`
2. They will be retried on next startup
3. Completed jobs remain in the database

Check recovered jobs:
```bash
sqlite3 jobs.db "SELECT id, status FROM jobs WHERE status = 'queued'"
```

## Graceful Shutdown

Press Ctrl+C or send SIGTERM:
1. Servers stop accepting new requests
2. Current job finishes (if reasonable time)
3. Worker threads drain
4. Database connections close cleanly
