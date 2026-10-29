"""Start the Streamlit dashboard from Python or a PyInstaller Windows bundle."""

import sys
from pathlib import Path


if getattr(sys, "frozen", False):
    code_dir = Path(sys._MEIPASS) / "Code"
else:
    code_dir = Path(__file__).resolve().parent

sys.path.insert(0, str(code_dir))

# Import dashboard modules up front so PyInstaller can discover their dependencies.
import app_gui  # noqa: E402,F401
import evaluation  # noqa: E402,F401
import export_results  # noqa: E402,F401
import fan_beam  # noqa: E402,F401
import iterative_reconstruction  # noqa: E402,F401
import parallel_beam  # noqa: E402,F401
import phantom  # noqa: E402,F401
from streamlit.web import cli as stcli  # noqa: E402


if __name__ == "__main__":
    sys.argv = [
        "streamlit",
        "run",
        str(code_dir / "app_gui.py"),
        "--global.developmentMode=false",
        "--server.headless=true",
    ]
    stcli.main()
