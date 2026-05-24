#!/usr/bin/env python3
"""
dashboard.py — DO NOT MODIFY. Minimal stdlib web server that serves the latest
plot.html (written by train.py) at http://127.0.0.1:8050. The measurement
command starts it automatically; if the port is already bound (a server is
already running) this process simply exits. No dependencies.
"""

import http.server
import socketserver
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PORT = 8050


class Handler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        f = HERE / "plot.html"
        if f.exists():
            data = f.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
        else:
            data = b"<h2>No plot yet - run train.py.</h2>"
            self.send_response(503)
            self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *args):
        pass


def main():
    try:
        with socketserver.TCPServer(("127.0.0.1", PORT), Handler) as srv:
            srv.serve_forever()
    except OSError:
        sys.exit(0)  # port already in use → a server is already running


if __name__ == "__main__":
    main()
