"""
Create JSON Server

Manual JSON editor and proxy server for the image pipeline.
"""

import http.server
import json
import urllib.request
import socket
import os

from pipeline_common import load_config, setup_logger, create_health_response

CONFIG = load_config()

UPSTREAM = f"http://127.0.0.1:{CONFIG.get('dual_gen_port', 5050)}"
LLM_UPSTREAM = CONFIG.get("llm_url", "http://localhost:11434")
PORT = CONFIG.get("create_json_port", 3030)
BIND = CONFIG.get("create_json_host", "0.0.0.0")

logger = setup_logger("create_json", level=CONFIG.get("logging_level", "INFO"))


class Handler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=os.path.join(os.path.dirname(__file__), "templates"), **kwargs)

    def log_message(self, format, *args):
        logger.debug(f"{args[0]} {args[1]}", extra={'status_code': args[1] if len(args) > 1 else None})

    def do_POST(self):
        if self.path.startswith('/lm/'):
            self._proxy_lm()
        elif self.path.startswith('/api/'):
            self._proxy()
        else:
            self.send_error(404)

    def do_GET(self):
        if self.path == '/' or self.path == '':
            self.path = '/create_json.html'
            super().do_GET()
        elif self.path == '/health':
            self._serve_health()
        elif self.path == '/config.json':
            self._serve_config()
        elif self.path.startswith('/lm/'):
            self._proxy_lm()
        elif self.path.startswith('/api/'):
            self._proxy()
        else:
            super().do_GET()

    def do_DELETE(self):
        if self.path.startswith('/lm/'):
            self._proxy_lm()
        elif self.path.startswith('/api/'):
            self._proxy()
        else:
            self.send_error(404)

    def _serve_health(self):
        """Serve health check response."""
        upstream_status = "unknown"
        try:
            req = urllib.request.Request(f"{UPSTREAM}/health", method='GET')
            with urllib.request.urlopen(req, timeout=3) as resp:
                upstream_status = "connected" if resp.status == 200 else "error"
        except:
            upstream_status = "disconnected"

        health = create_health_response(
            service="create_json",
            status="healthy" if upstream_status == "connected" else "degraded",
            extra={"upstream": upstream_status, "upstream_url": UPSTREAM}
        )
        data = json.dumps(health).encode('utf-8')
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', len(data))
        self.end_headers()
        self.wfile.write(data)

    def _serve_config(self):
        client_config = {
            "endpoint": UPSTREAM,
            "generatePath": "/api/generate",
            "lmModel": CONFIG.get("llm_model", "")
        }
        data = json.dumps(client_config).encode('utf-8')
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', len(data))
        self.end_headers()
        self.wfile.write(data)

    def _proxy_lm(self):
        path = self.path[3:]
        url = LLM_UPSTREAM + path
        length = int(self.headers.get('Content-Length', 0))
        body = self.rfile.read(length) if length else None

        logger.debug(f"Proxying to LLM", extra={'path': path, 'method': self.command})

        req = urllib.request.Request(
            url,
            data=body,
            method=self.command,
            headers={'Content-Type': self.headers.get('Content-Type', 'application/json')}
        )
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                data = resp.read()
                self.send_response(resp.status)
                self.send_header('Content-Type', resp.headers.get('Content-Type', 'application/json'))
                self.end_headers()
                self.wfile.write(data)
        except urllib.error.HTTPError as e:
            logger.error(f"LLM proxy error", extra={'status_code': e.code, 'path': path})
            self.send_response(e.code)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(e.read())
        except Exception as e:
            logger.error(f"LLM proxy failed", extra={'error': str(e), 'path': path})
            self.send_error(502, 'Cannot reach LLM')

    def _proxy(self):
        url = UPSTREAM + self.path
        length = int(self.headers.get('Content-Length', 0))
        body = self.rfile.read(length) if length else None

        logger.debug(f"Proxying to upstream", extra={'path': self.path, 'method': self.command})

        req = urllib.request.Request(
            url,
            data=body,
            method=self.command,
            headers={'Content-Type': self.headers.get('Content-Type', 'application/json')}
        )
        try:
            with urllib.request.urlopen(req) as resp:
                data = resp.read()
                self.send_response(resp.status)
                self.send_header('Content-Type', resp.headers.get('Content-Type', 'application/json'))
                self.end_headers()
                self.wfile.write(data)
        except urllib.error.HTTPError as e:
            logger.error(f"Upstream proxy error", extra={'status_code': e.code, 'path': self.path})
            self.send_response(e.code)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(e.read())
        except Exception as e:
            logger.error(f"Upstream proxy failed", extra={'error': str(e), 'path': self.path})
            self.send_error(502, 'Cannot reach upstream server')


def main():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(('8.8.8.8', 80))
        local_ip = s.getsockname()[0]
        s.close()
    except Exception:
        local_ip = '127.0.0.1'

    logger.info(f"Starting Create JSON server", extra={
        'port': PORT,
        'upstream': UPSTREAM,
        'lm_upstream': LLM_UPSTREAM,
        'local_ip': local_ip,
    })

    http.server.HTTPServer((BIND, PORT), Handler).serve_forever()


if __name__ == "__main__":
    main()
