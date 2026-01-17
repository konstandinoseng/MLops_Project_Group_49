import sys
from pathlib import Path

# Ensure `src` is on the Python path so `import project` works without installation
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))
