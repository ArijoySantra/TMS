"""Double-click launcher for the desktop app (opens without a console window)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from traffic_ai.gui.app import main  # noqa: E402

sys.exit(main())
