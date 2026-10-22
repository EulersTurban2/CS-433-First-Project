import numpy as np
import src.implementations as impl


class Model:
    def __init__(self, model_name, fn, loss_fn, chunk=50, **params):
        self.model_name = model_name
        self.fn = fn
        self.loss_fn = loss_fn
        self.chunk = chunk
        self.params = params

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        params = dict(self.params)
        if "max_iters" not in params:
            self.w, loss = self.fn(y_train, X_train, **params)
            self.history = [{"done": 1, "total": 1, "loss": loss}]
            return self
        total = params.pop("max_iters")
        w = np.zeros(X_train.shape[1])
        self.history, done = [], 0
        while done < total:
            chunk_iters = min(self.chunk, total - done)
            w, loss = self.fn(
                y_train, X_train, initial_w=w, max_iters=chunk_iters, **params
            )
            done += chunk_iters
            entry = {"done": done, "total": total, "loss": loss}
            if X_val is not None and y_val is not None:
                val_loss = self.loss_fn(y_val, X_val, w)
                entry["val_loss"] = val_loss
            self.history.append(entry)
        self.w = w
        return self

    def predict(self, X):
        res = X.dot(self.w)
        if self.loss_fn == impl.logistic_loss:
            return impl.sigmoid(res)
        return res
