import numpy as np

from src.train import run_all, run_fold
from src.utils import prepare_args

if __name__ == "__main__":
    args = prepare_args()
    scores = []
    # run_all(-1, args)
    for fold in range(0, 4):
        # if fold in (0, 1, 2):
        #     continue
        fold_score = run_fold(fold, args)
        scores.append(fold_score)
    print(f"Cross-validation score: {np.mean(scores):.4f}+/-{np.std(scores):.4f}")
