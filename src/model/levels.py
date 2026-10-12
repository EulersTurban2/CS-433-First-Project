import copy
import numpy as np
import model.model as model


from src.folds import stratified_folds
from src.metrics import auroc

class Level:
    def __init__(self, level_name, models, passthrough = False, seed = 42, folds = 5):
        self.level_name = level_name
        self.models = models
        self.passthrough = passthrough
        self.seed = seed
        self.folds = folds
        self.fitted_models = None
        self.oof_predictions = None
        self.scores = {}

    def fit_transform(self, X_train, y_train, X_orig=None):
        if X_orig is None:
            X_orig = X_train
        names = list(self.models)
        oof = np.full((X_train.shape[0], len(names)), np.nan)

        for train_idx, val_idx in stratified_folds(y_train, n_folds = self.folds, seed = self.seed):
            X_tr, y_tr = X_train[train_idx], y_train[train_idx]
            X_val, y_val = X_train[val_idx], y_train[val_idx]

            for i, name in enumerate(names):
                model = copy.deepcopy(self.models[name])
                model.fit(X_tr, y_tr, X_val, y_val)
                oof[val_idx, i] = model.predict(X_val)

        self.fitted_models = {name: copy.deepcopy(self.models[name]).fit(X_train, y_train) for name in names }
        self.oof_predictions = oof
        self.scores = {name : auroc(y_train, oof[:, i]) for i, name in enumerate(names) }
        return self._assemble(oof, X_orig)

    def transform(self, X, X_orig=None):
        if self.fitted_models is None:
            raise ValueError("Level has not been fitted yet. Call fit_transform first.")
        if X_orig is None:
            X_orig = X
        preds = np.column_stack([model.predict(X) for model in self.fitted_models.values()])
        return self._assemble(preds, X_orig)

    def _assemble(self, preds, X_orig):
        parts = [np.ones((preds.shape[0], 1)), preds]
        if self.passthrough and X_orig is not None:
            parts.append(X_orig[:, 1:])
        return np.concatenate(parts, axis=1)