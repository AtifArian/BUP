"""Start a fresh GridWise server in a subprocess for live tests.

Each server gets its own port, environment, empty interpretation cache and log
file, so tests measure cold, realistic behaviour.
"""

from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from contextlib import contextmanager
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def call(url: str, body: dict | bytes | None = None, headers: dict | None = None,
         timeout: float = 35) -> tuple[int | None, str, float]:
    """POST (or GET when body is None). Returns (status, text, seconds)."""
    data = body if isinstance(body, bytes) or body is None else json.dumps(body).encode()
    hdrs = {"Content-Type": "application/json"} if body is not None else {}
    hdrs.update(headers or {})
    req = urllib.request.Request(url, data=data, headers=hdrs)
    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, r.read().decode("utf-8", "replace"), time.perf_counter() - t0
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode("utf-8", "replace"), time.perf_counter() - t0
    except Exception as e:  # timeout, connection refused, ...
        return None, f"{type(e).__name__}: {e}", time.perf_counter() - t0


@contextmanager
def server(env_overrides: dict[str, str] | None = None, workers: int = 1):
    """Yield (base_url, log_path, startup_seconds) for a fresh server."""
    port = free_port()
    env = {**os.environ, **(env_overrides or {})}
    fd, name = tempfile.mkstemp(prefix="gridwise_", suffix=".log")
    os.close(fd)
    log = Path(name)
    cmd = [sys.executable, "-m", "uvicorn", "app.main:app",
           "--host", "127.0.0.1", "--port", str(port), "--workers", str(workers)]
    with open(log, "w", encoding="utf-8") as fh:
        proc = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=fh, stderr=subprocess.STDOUT)
    base = f"http://127.0.0.1:{port}"
    t0 = time.perf_counter()
    try:
        while True:
            status, _, _ = call(base + "/health", timeout=2)
            if status == 200:
                break
            if proc.poll() is not None or time.perf_counter() - t0 > 60:
                raise RuntimeError(f"server did not start; see {log}")
            time.sleep(0.2)
        yield base, log, time.perf_counter() - t0
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()


def load_cases() -> list[dict]:
    """Public samples plus the edge pack (if present) as uniform request bodies."""
    cases = []
    samples = json.loads((ROOT / "BUP_CSE_FEST_2026_Preli_Public_Sample_Cases.json")
                         .read_text(encoding="utf-8"))["cases"]
    for c in samples:
        cases.append({"id": c["id"], "input": c["input"],
                      "truth": c["expected_output"]["directive_interpretation"]})
    pack = ROOT / "GridWise_Edge_Test_Pack" / "gridwise_edge_cases.json"
    if pack.exists():
        for c in json.loads(pack.read_text(encoding="utf-8"))["valid_cases"]:
            cases.append({"id": c["id"], "input": c["input"], "truth": c["expected_interpretation"]})
    return cases
