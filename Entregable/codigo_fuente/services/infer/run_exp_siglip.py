import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))

from services.infer.experiment_utils import run_full_experiment

if __name__ == "__main__":
    run_full_experiment("siglip", ROOT)

