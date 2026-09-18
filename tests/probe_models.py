"""Probe every configured model with one tiny interpretation request.

Reads .env. Prints, per model, whether it answered valid JSON and how long it
took, so you can confirm keys/quotas before judging:

    python tests/probe_models.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
load_dotenv(ROOT / ".env")

from app import llm  # noqa: E402
from app.interpreter import SYSTEM_PROMPT, _user_prompt, normalise  # noqa: E402

NOTE = "Do not charge the battery between 2 PM and 4 PM."


def main() -> int:
    models = llm.chain()
    if not models:
        print("no provider configured: set GROQ_API_KEY and/or GEMINI_API_KEY in .env")
        return 1
    print(f"{len(models)} model(s) in chain order:\n")
    ok = 0
    for m in models:
        llm._RATE_LIMITED.clear()
        t0 = time.perf_counter()
        try:
            out = llm.chat_json(SYSTEM_PROMPT, _user_prompt([NOTE], 500, 50),
                                deadline=time.monotonic() + 20, models=[m])
            entry = normalise(out["directives"][0], 500, 50)
            good = entry["structured_adjustment"] == {"hours": [14, 15]}
            status = "OK " if good else f"WRONG {entry['structured_adjustment']}"
            ok += good
        except Exception as e:  # noqa: BLE001 - report everything
            status = f"FAIL {type(e).__name__}: {str(e)[:80]}"
        print(f"{status:<40} {time.perf_counter() - t0:6.2f}s  {m.id}")
    print(f"\n{ok}/{len(models)} models answered correctly")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
