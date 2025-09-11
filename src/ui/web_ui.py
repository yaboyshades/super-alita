from __future__ import annotations

import sys
from pathlib import Path

from src.core.proc import run


def launch_ui(port: int = 8081) -> None:
    """Launch the Streamlit web UI.

    Spawns: python -m streamlit run ui_web_interface.py --server.port <port>
    Blocks until the UI process exits.
    """
    repo_root = Path(__file__).resolve().parents[2]
    ui_script = repo_root / "ui_web_interface.py"
    cmd = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(ui_script),
        "--server.port",
        str(port),
        "--server.headless",
        "true",
    ]
    run(cmd)

