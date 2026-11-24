# Make the post's scripts and the shared examplekit importable from the tests.
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parent), str(__import__("pathlib").Path(__file__).resolve().parents[3] / "shared")]
