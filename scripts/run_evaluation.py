from pathlib import Path
import json, sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.evaluation import run_evaluation
print(json.dumps(run_evaluation(), indent=2))
