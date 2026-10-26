"""Regenerate output_examples/ from input_examples/ in sample mode, in-process.

    python scripts/regenerate_examples.py

Calls the FastAPI app directly with sample-mode settings and a temporary logs directory, so it
never reaches a server already running on port 8000 (which may be in real mode) and makes no
Nightshift, Zetaris or network calls.
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fastapi.testclient import TestClient  # noqa: E402
from litagent import api  # noqa: E402
from litagent.config import Settings  # noqa: E402


def main() -> int:
    with tempfile.TemporaryDirectory() as logs:
        settings = Settings(sample_mode=True, logs_dir=logs)
        api.app.dependency_overrides[api.get_settings] = lambda: settings
        try:
            client = TestClient(api.app)
            for path in sorted((ROOT / "input_examples").glob("*.json")):
                body = client.post("/run", json=json.loads(path.read_text(encoding="utf-8"))).json()
                (ROOT / "output_examples" / path.name).write_text(json.dumps(body, indent=2) + "\n",
                                                                  encoding="utf-8")
                print(f"{path.name}: {body['status']} / {body.get('output', {}).get('status')}")
        finally:
            api.app.dependency_overrides.clear()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
