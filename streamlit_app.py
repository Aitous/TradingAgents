import sys
from pathlib import Path

# Ensure repo root is on sys.path for imports when running on Streamlit Cloud
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tradingagents.ui.dashboard import main

main()
