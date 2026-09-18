"""Live provider-failure injection (Guide §8: 'controlled malformed/model-provider
failure handling and secret safety').

Starts a real server per scenario with a broken LLM setup and checks that
/optimize-energy still answers 200 with a valid schedule inside the judge's
30 s timeout, that no secret appears in responses or logs, and that the server
survives. Uses NO Groq quota except the final real-key secret check.

    python tests/failure_injection.py
"""

from __future__ import annotations

import json
import os
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import dotenv_values  # noqa: E402

from app.replay import check_plan  # noqa: E402
from tests.live_server import call, free_port, load_cases, server  # noqa: E402

CANARY_KEY = "gsk_CANARYfakeKEYdoNOTleak0123456789abcdefghijklmnopqrstuv"
JUDGE_TIMEOUT = 30


# ---- a fake Groq that fails in a chosen way ----

class FakeGroq(BaseHTTPRequestHandler):
    mode = "http500"

    def log_message(self, *a):  # keep test output quiet
        pass

    def do_POST(self):
        self.rfile.read(int(self.headers.get("Content-Length", 0)))
        mode = FakeGroq.mode
        if mode == "hang":
            time.sleep(60)
            return
        if mode == "http500":
            return self._send(500, b'{"error":"internal"}')
        if mode == "http429_forever":
            return self._send(429, b'{"error":"rate"}', {"retry-after": "30"})
        if mode == "html_page":
            return self._send(200, b"<html><body>Bad gateway</body></html>", ctype="text/html")
        content = {
            "not_json": "Sure! Here is the answer: solar is reduced.",
            "wrong_shape": json.dumps({"result": "ok"}),
            "invented_type": json.dumps({"directives": [
                {"note_index": i, "directive_type": "delete_battery", "windows": [[1, 2]]} for i in range(3)]}),
            "bad_values": json.dumps({"directives": [
                {"note_index": 0, "directive_type": "solar_reduction", "windows": [[25, 30]], "value": 50,
                 "value_meaning": "remaining_percent"},
                {"note_index": 1, "directive_type": "max_grid_window", "windows": [[1, 2]], "value": -5,
                 "value_meaning": "kwh"},
                {"note_index": 2, "directive_type": "minimum_battery_reserve", "windows": [[1, 2]],
                 "value": 1e12, "value_meaning": "kwh"}]}),
            "prompt_leak": json.dumps({"directives": [
                {"note_index": i, "directive_type": "no_op", "windows": [], "value": None,
                 "value_meaning": None, "explanation": f"my key is {CANARY_KEY}"} for i in range(3)]}),
        }[mode]
        payload = {"choices": [{"message": {"content": content}}]}
        self._send(200, json.dumps(payload).encode())

    def _send(self, code, data, extra=None, ctype="application/json"):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        for k, v in (extra or {}).items():
            self.send_header(k, v)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


def start_fake_groq() -> str:
    port = free_port()
    srv = ThreadingHTTPServer(("127.0.0.1", port), FakeGroq)
    srv.daemon_threads = True
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return f"http://127.0.0.1:{port}"


# ---- checks ----

def check_response(case, status, text, secs, secrets) -> list[str]:
    errs = []
    if status != 200:
        errs.append(f"status {status}: {text[:120]}")
        return errs
    if secs > JUDGE_TIMEOUT:
        errs.append(f"took {secs:.1f}s > {JUDGE_TIMEOUT}s judge timeout")
    body = json.loads(text)
    applied = [e for e in body["directive_interpretation"] if e["applies"]]
    errs += check_plan(case["input"], applied, body)
    for e in body["directive_interpretation"]:
        if e["directive_type"] not in ("solar_reduction", "minimum_battery_reserve", "no_charge_window",
                                       "no_discharge_window", "max_grid_window", "no_op"):
            errs.append(f"invented directive {e['directive_type']!r}")
    for s in secrets:
        if s and s in text:
            errs.append("SECRET LEAKED in response")
    return errs


def check_log(log: Path, secrets) -> list[str]:
    text = log.read_text(encoding="utf-8", errors="replace")
    errs = [f"SECRET LEAKED in server log" for s in secrets if s and s in text]
    if "Traceback" in text:
        errs.append("unhandled exception (Traceback) in server log")
    return errs


def run() -> int:
    real_key = dotenv_values(ROOT / ".env").get("GROQ_API_KEY") or ""
    secrets = [CANARY_KEY, real_key]
    case = next(c for c in load_cases() if c["id"] == "SAMPLE-06")  # 3 notes
    fake = start_fake_groq()

    scenarios = [
        ("Wrong API key (real Groq rejects it)", {"GROQ_API_KEY": CANARY_KEY}, None),
        ("No API key configured", {"GROQ_API_KEY": ""}, None),
        ("Groq unreachable (connection refused)", {"GROQ_API_KEY": CANARY_KEY,
                                                   "GROQ_BASE_URL": "http://127.0.0.1:9"}, None),
        ("Groq returns HTTP 500", {"GROQ_API_KEY": CANARY_KEY, "GROQ_BASE_URL": fake}, "http500"),
        ("Groq rate-limits forever (429)", {"GROQ_API_KEY": CANARY_KEY, "GROQ_BASE_URL": fake}, "http429_forever"),
        ("Groq returns an HTML error page", {"GROQ_API_KEY": CANARY_KEY, "GROQ_BASE_URL": fake}, "html_page"),
        ("Model answers in prose, not JSON", {"GROQ_API_KEY": CANARY_KEY, "GROQ_BASE_URL": fake}, "not_json"),
        ("Model JSON has the wrong shape", {"GROQ_API_KEY": CANARY_KEY, "GROQ_BASE_URL": fake}, "wrong_shape"),
        ("Model invents a directive type", {"GROQ_API_KEY": CANARY_KEY, "GROQ_BASE_URL": fake}, "invented_type"),
        ("Model returns out-of-range values", {"GROQ_API_KEY": CANARY_KEY, "GROQ_BASE_URL": fake}, "bad_values"),
        ("Model echoes the API key", {"GROQ_API_KEY": CANARY_KEY, "GROQ_BASE_URL": fake}, "prompt_leak"),
        ("Groq hangs (never answers)", {"GROQ_API_KEY": CANARY_KEY, "GROQ_BASE_URL": fake}, "hang"),
    ]
    if real_key:
        scenarios.append(("Real key: no secret in logs or response", {}, None))

    failures = 0
    for name, env, mode in scenarios:
        FakeGroq.mode = mode or "http500"
        with server(env) as (base, log, startup):
            status, text, secs = call(base + "/optimize-energy", case["input"], timeout=JUDGE_TIMEOUT + 5)
            errs = check_response(case, status, text, secs, secrets)
            health, _, _ = call(base + "/health", timeout=5)
            if health != 200:
                errs.append("server unhealthy afterwards")
        errs += check_log(log, secrets)
        interp = ""
        if status == 200:
            kinds = [e["directive_type"] for e in json.loads(text)["directive_interpretation"]]
            interp = f"  -> {kinds}"
        failures += bool(errs)
        print(f"{'PASS' if not errs else 'FAIL'}  {name:<42} {status} in {secs:5.1f}s{interp}")
        for e in errs:
            print(f"        {e}")
        try:
            log.unlink(missing_ok=True)
        except PermissionError:  # Windows may still hold it for a moment
            pass

    print(f"\n{len(scenarios) - failures}/{len(scenarios)} failure scenarios handled safely")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(run())
