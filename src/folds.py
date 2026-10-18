import numpy as np

def stratified_folds(y, n_folds=5, seed=42):
    rng = np.random.default_rng(seed)
    parts = [[] for _ in range(n_folds)]
    unique_categories = np.unique(y)
    for cat in unique_categories:
        idx = rng.permutation(np.where(y == cat)[0])    
        for i, chunk in enumerate(np.array_split(idx, n_folds)):
            parts[i].append(chunk)

    n = len(y)
    for chunk in parts:
        val_idx = np.sort(np.concatenate(chunk))
        mask = np.ones(n, dtype=bool)
        mask[val_idx] = False
        train_idx = np.where(mask)[0]
        yield train_idx, val_idx
