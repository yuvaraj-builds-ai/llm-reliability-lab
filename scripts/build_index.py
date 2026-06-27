from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.retrieval import build_index
print(f"Built index with {build_index()} chunks.")
