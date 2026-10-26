"""Entry point: serves POST /run (see litagent/api.py) on 0.0.0.0:$PORT, default 8000.

    SAMPLE_MODE=true python run.py
    curl -X POST localhost:8000/run -H 'Content-Type: application/json' -d @input_examples/01_normal_low_resource_languages.json

The command line version still works: python -m litagent "your question" --sample-mode
"""

from __future__ import annotations

import os

import uvicorn
from dotenv import load_dotenv

load_dotenv()

from litagent.trace import setup_logging  # noqa: E402


def main() -> None:
    setup_logging()
    uvicorn.run("litagent.api:app", host="0.0.0.0", port=int(os.getenv("PORT", "8000")))


if __name__ == "__main__":
    main()
