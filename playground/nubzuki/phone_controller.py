"""Touch controller served over HTTP, for running the policy without a gamepad.

Exposes the same read/fresh/close surface as `XboxController`, so the
simulation loop cannot tell the two apart. A phone on the same network opens
the page, drags two virtual sticks, and each sample carries a timestamp: unlike
the gamepad path, `fresh()` here reports on the arrival of real input, so a
phone that walks out of Wi-Fi range is detected instead of silently freezing
the last command.
"""

from __future__ import annotations

import json
import socket
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from playground.nubzuki.page import CONTROL_PAGE


AXES = ("left_x", "left_y", "right_x", "right_y")


def local_address() -> str:
    """Best guess at the address a phone on the same network should open."""
    probe = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        probe.connect(("192.0.2.1", 1))  # TEST-NET-1: routed nowhere, sends nothing.
        return probe.getsockname()[0]
    except OSError:
        return "127.0.0.1"
    finally:
        probe.close()


class PhoneController:
    def __init__(self, host: str = "0.0.0.0", port: int = 8765, timeout_s: float = 0.5,
                 target_label: str = "시뮬레이터"):
        self.timeout_s = float(timeout_s)
        self._lock = threading.Lock()
        self._axes = {name: 0.0 for name in AXES}
        self._a_pressed = False
        self._b_pressed = False
        self.control_mode = "walk"
        self._last_input = 0.0
        self._connected = False
        self.target_label = str(target_label)
        controller = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_args):
                pass  # The training console is not a web server log.

            def _send(self, code, body=b"", content_type="text/plain; charset=utf-8"):
                self.send_response(code)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                if body:
                    self.wfile.write(body)

            def do_GET(self):
                if self.path.split("?")[0] in ("/", "/index.html"):
                    page = CONTROL_PAGE.replace("__TARGET_LABEL__", controller.target_label)
                    self._send(200, page.encode("utf-8"), "text/html; charset=utf-8")
                    return
                self._send(404)

            def do_POST(self):
                if self.path.split("?")[0] != "/input":
                    self._send(404)
                    return
                try:
                    length = int(self.headers.get("Content-Length", "0"))
                    payload = json.loads(self.rfile.read(length) or b"{}")
                    controller.submit(payload)
                except (ValueError, TypeError) as error:
                    self._send(400, str(error).encode("utf-8"))
                    return
                self._send(204)

        self.server = ThreadingHTTPServer((host, port), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.url = f"http://{local_address()}:{self.server.server_address[1]}"

    def submit(self, payload: dict) -> None:
        """Accept one sample from the page, clamped to the ranges we promise."""
        axes = {}
        for name in AXES:
            value = float(payload.get(name, 0.0))
            if value != value:  # NaN from a malformed page never reaches the policy.
                value = 0.0
            axes[name] = max(-1.0, min(1.0, value))
        with self._lock:
            self._axes = axes
            mode = str(payload.get("mode", "walk"))
            self.control_mode = mode if mode in ("walk", "head") else "walk"
            self._a_pressed = bool(payload.get("a", False))
            self._b_pressed = bool(payload.get("b", False))
            self._last_input = time.monotonic()
            self._connected = True

    def read(self) -> tuple[dict[str, float], bool, bool]:
        with self._lock:
            if self._connected and time.monotonic() - self._last_input > self.timeout_s:
                # Hold no stale deflection: an unreachable phone recentres the head.
                self._axes = {name: 0.0 for name in AXES}
                self._a_pressed = False
            return dict(self._axes), self._a_pressed, self._b_pressed

    def fresh(self) -> bool:
        """True only while samples are actually arriving."""
        with self._lock:
            if not self._connected:
                return False
            return time.monotonic() - self._last_input <= self.timeout_s

    def waiting(self) -> bool:
        with self._lock:
            return not self._connected

    def mode(self) -> str:
        with self._lock:
            return self.control_mode

    def close(self) -> None:
        self.server.shutdown()
        self.server.server_close()
