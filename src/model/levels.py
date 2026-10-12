import numpy as np
import model.model as model

class Level:
    def __init__(self, level_name, models, passthrough = False, seed = 42, folds = 2):
        self.level_name = level_name
        self.models = models
        self.passthrough = passthrough
        self.seed = seed
        self.folds = folds
        self.fitted_models = None
        self.oof_predictions = None
        self.scores = {}

    def fit_transform(self, X_train, y_train, X_orig=None):
        pass

    def transform(self, X, X_orig=None):
        pass

    def _assemble(self, preds, X_orig):
        pass