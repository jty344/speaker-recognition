import os
import tempfile
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TEST_TMP = PROJECT_ROOT / ".tmp" / "pytest"
TEST_TMP.mkdir(parents=True, exist_ok=True)
os.environ["TMPDIR"] = str(TEST_TMP)
tempfile.tempdir = str(TEST_TMP)

