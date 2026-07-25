"""Load .env from the repo root into os.environ.

Called at orchestrator startup so credentials (ANTHROPIC_API_KEY, E2B_API_KEY, ...) can
live in a gitignored .env. The gateway runs as a subprocess with the orchestrator's env
inherited, so loading here is enough for the whole run. A no-op if python-dotenv isn't
installed or .env is absent.
"""

from __future__ import annotations

from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent


def load_env() -> None:
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    env_path = _REPO / ".env"
    if env_path.exists():
        load_dotenv(env_path)
