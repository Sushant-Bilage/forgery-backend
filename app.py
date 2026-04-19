"""
Document Forgery Detection — REST API Backend
"""

import json
import uuid
import time
import traceback
from pathlib import Path
from http.server import HTTPServer, BaseHTTPRequestHandler
import os
import tempfile

from detector import DocumentForgeryDetector, report_to_dict

REPORTS: dict = {}
detector = DocumentForgeryDetector()

CORS_HEADERS = {
    "Access-Control-Allow-Origin": "*",
    "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
    "Access-Control-Allow-Headers": "Content-Type, Authorization",
    "Content-Type": "application/json",
}

SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}


class ForgeryAPIHandler(BaseHTTPRequestHandler):

    def log_message(self, fmt, *args):
        print(f"[{self.address_string()}] {fmt % args}")

    # ───────── helpers ─────────

    def send_json(self, data: dict, status: int = 200):
        body = json.dumps(data).encode("utf-8")
        self.send_response(status)
        for k, v in CORS_HEADERS.items():
            self.send_header(k, v)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def send_error_json(self, message: str, status: int = 400):
        self.send_json({"error": message, "status": status}, status)

    # ───────── CORS ─────────

    def do_OPTIONS(self):
        self.send_response(204)
        for k, v in CORS_HEADERS.items():
            self.send_header(k, v)
        self.end_headers()

    # ───────── GET ─────────

    def do_GET(self):
        path = self.path.split("?")[0]

        if path == "/health":
            self.send_json({
                "status": "ok",
                "service": "Forgery Detection API"
            })

        elif path.startswith("/report/"):
            rid = path.replace("/report/", "")
            if rid in REPORTS:
                self.send_json(REPORTS[rid])
            else:
                self.send_error_json("Report not found", 404)

        else:
            self.send_error_json("Endpoint not found", 404)

    # ───────── POST ─────────

    def do_POST(self):
        if self.path == "/detect":
            self._handle_detect()
        else:
            self.send_error_json("Endpoint not found", 404)

    # ───────── DETECT ─────────

    def _handle_detect(self):
        try:
            # read raw image data
            length = int(self.headers.get("Content-Length", 0))
            file_data = self.rfile.read(length)

            filename = self.headers.get("X-Filename", "document.jpg")

            # save temp file
            with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
                tmp.write(file_data)
                tmp_path = tmp.name

            # run detection
            report = detector.detect(tmp_path)
            result = report_to_dict(report)

            # store report
            report_id = str(uuid.uuid4())
            REPORTS[report_id] = result
            result["report_id"] = report_id

            # send response
            self.send_json(result)

        except Exception as e:
            traceback.print_exc()
            self.send_error_json(f"Detection failed: {e}", 500)


# ───────── RUN SERVER ─────────

def run_server(host="0.0.0.0", port=8000):
    server = HTTPServer((host, port), ForgeryAPIHandler)
    print(f"Server running on http://{host}:{port}")
    server.serve_forever()


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    run_server(port=port)
