import copy
import numpy as np

class Stack:
    def __init__(self, levels, final_model):
        self.levels = levels            #A list of Level instances
        self.final_model = final_model  #A Model instance
        self.fitted_final = None

    def fit(self, X_train, y_train, X_val, y_val):
        inputs = X_train
        for level in self.levels:
            inputs = level.fit_transform(inputs, y_train, X_orig=X_train)

        val_inputs = None
        if X_val is not None and y_val is not None:
            val_inputs = self._forward(X_val)

        self.fitted_final = copy.deepcopy(self.final_model)
        self.fitted_final.fit(inputs, y_train, val_inputs, y_val)
        return self

    def predict(self, X):
        if self.fitted_final is None:
            raise ValueError("Final model not fitted yet, call .fit method!")
        return self.fitted_final.predict(self._forward(X))

    def _forward(self, X):
        inputs = X
        for level in self.levels:
            inputs = level.transform(inputs, X_orig=X)
        return inputs

    @property
    def history(self):
        return self.fitted_final.history if self.fitted_final is not None else []

    @property
    def scores(self):
        return {level.level_name: level.scores for level in self.levels}