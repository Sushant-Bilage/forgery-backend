"""
Document Forgery Detection — REST API Backend
================================================
Run with:  python app.py
Endpoints:
  POST /detect          — upload a document image for analysis
  GET  /health          — health check
  GET  /methods         — list detection methods and weights
  GET  /report/{id}     — retrieve a cached report by ID
"""

import json
import uuid
import time
import traceback
from pathlib import Path
from http.server import HTTPServer, BaseHTTPRequestHandler
import cgi
import io
import os
import tempfile

from detector import DocumentForgeryDetector, report_to_dict

REPORTS:  dict = {}
detector = DocumentForgeryDetector()

CORS_HEADERS = {
    "Access-Control-Allow-Origin":  "*",
    "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
    "Access-Control-Allow-Headers": "Content-Type, Authorization",
    "Content-Type": "application/json",
}

SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}


class ForgeryAPIHandler(BaseHTTPRequestHandler):

    def log_message(self, fmt, *args):
        print(f"[{self.address_string()}] {fmt % args}")

    # ── helpers ───────────────────────────────────────────────────────────────

    def send_json(self, data: dict, status: int = 200):
        body = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        for k, v in CORS_HEADERS.items():
            self.send_header(k, v)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def send_error_json(self, message: str, status: int = 400):
        self.send_json({"error": message, "status": status}, status)

    # ── CORS pre-flight ───────────────────────────────────────────────────────

    def do_OPTIONS(self):
        self.send_response(204)
        for k, v in CORS_HEADERS.items():
            self.send_header(k, v)
        self.end_headers()

    # ── GET ───────────────────────────────────────────────────────────────────

    def do_GET(self):
        path = self.path.split("?")[0]

        if path == "/health":
            self.send_json({
                "status":         "ok",
                "service":        "Document Forgery Detection API",
                "version":        "2.0.0",
                "cached_reports": len(REPORTS)
            })

        elif path.startswith("/report/"):
            rid = path.replace("/report/", "")
            if rid in REPORTS:
                self.send_json(REPORTS[rid])
            else:
                self.send_error_json(f"Report '{rid}' not found", 404)

        elif path == "/methods":
            self.send_json({"methods": [
                {"id": "ela",              "name": "Error Level Analysis",
                 "weight": 0.25, "description": "Detects editing artifacts via re-compression analysis"},
                {"id": "noise_analysis",   "name": "Noise Pattern Analysis",
                 "weight": 0.20, "description": "Detects spliced regions via inconsistent noise floors"},
                {"id": "font_consistency", "name": "Font Consistency Analysis",
                 "weight": 0.15, "description": "Detects text insertion via font metric inconsistencies"},
                {"id": "copy_move",        "name": "Copy-Move Detection",
                 "weight": 0.15, "description": "Detects cloned regions using keypoint self-matching"},
                {"id": "layout_anomaly",   "name": "Layout Anomaly Detection",
                 "weight": 0.10, "description": "Detects irregular margins and spacing from text insertion"},
                {"id": "metadata",         "name": "Metadata Forensics",
                 "weight": 0.08, "description": "Detects file-level tampering signals"},
                {"id": "ocr_confidence",   "name": "OCR Confidence Analysis",
                 "weight": 0.07, "description": "Low OCR confidence indicates degraded or tampered text"},
            ]})

        else:
            self.send_error_json("Endpoint not found", 404)

    # ── POST ──────────────────────────────────────────────────────────────────

    def do_POST(self):
        if self.path.split("?")[0] == "/detect":
            self._handle_detect()
        else:
            self.send_error_json("Endpoint not found", 404)

    def _handle_detect(self):
        try:
            content_type = self.headers.get("Content-Type", "")

            # — multipart/form-data upload ————————————————————————————————————
            if "multipart/form-data" in content_type:
                length = int(self.headers.get("Content-Length", 0))
                body   = self.rfile.read(length)
                fs     = cgi.FieldStorage(
                    fp=io.BytesIO(body),
                    environ={"REQUEST_METHOD": "POST",
                             "CONTENT_TYPE":   content_type,
                             "CONTENT_LENGTH": str(length)},
                    keep_blank_values=True
                )
                if "file" not in fs:
                    self.send_error_json("No 'file' field in request.")
                    return
                item      = fs["file"]
                filename  = item.filename or "document.jpg"
                file_data = item.file.read()

            # — raw binary upload ─────────────────────────────────────────────
            elif "image/" in content_type or "application/octet-stream" in content_type:
                length    = int(self.headers.get("Content-Length", 0))
                file_data = self.rfile.read(length)
                filename  = self.headers.get("X-Filename", "document.jpg")

            else:
                self.send_error_json(
                    "Send multipart/form-data with a 'file' field, "
                    "or raw bytes with an image/* Content-Type."
                )
                return

            ext = Path(filename).suffix.lower()
            if ext not in SUPPORTED_EXTS:
                self.send_error_json(
                    f"Unsupported file type '{ext}'. "
                    f"Supported: {', '.join(sorted(SUPPORTED_EXTS))}"
                )
                return

            with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
                tmp.write(file_data)
                tmp_path = tmp.name

            try:
                t0     = time.time()
                print(f"\nAnalysing: {filename}")
                report = detector.detect(tmp_path)
                elapsed = time.time() - t0

                result = report_to_dict(report)
                result["processing_time_sec"] = round(elapsed, 2)
                result["report_id"]           = str(uuid.uuid4())
                REPORTS[result["report_id"]]  = result

                print(f"  Done in {elapsed:.1f}s — verdict: {report.verdict} "
                      f"(score={report.overall_score:.3f})")
                self.send_json(result)

            finally:
                os.unlink(tmp_path)

        except Exception as e:
            traceback.print_exc()
            self.send_error_json(f"Detection failed: {e}", 500)


def run_server(host: str = "0.0.0.0", port: int = 8000):
    server = HTTPServer((host, port), ForgeryAPIHandler)
    print(f"""
╔══════════════════════════════════════════════════╗
║  Document Forgery Detection API  v2.0            ║
║  Running on http://{host}:{port}              ║
╠══════════════════════════════════════════════════╣
║  POST /detect        analyse a document          ║
║  GET  /health        health check                ║
║  GET  /methods       list detection methods      ║
║  GET  /report/{{id}}  get cached report           ║
╚══════════════════════════════════════════════════╝
    """)
    server.serve_forever()


if __name__ == "__main__":
    run_server()
