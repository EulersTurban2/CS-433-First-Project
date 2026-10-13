import json

import src.data_processing.preprocessor as preprocessor

"""
To fasten up experiment execution, we read from a json file, which contains all the parameters for the experiment.
It is possible to change the parameters in the json file, and the experiment will be executed with the new parameters!

Example of a json-config file:

{
  "name": "stack_baseline",
  "seed": 42,

  "preprocess": [
    {"step": "drop_columns", "columns": ["IDATE", "IMONTH", "IDAY", "IYEAR", "FMONTH",
                                         "SEQNO", "_PSU", "DISPCODE", "_STSTR", "_STRWT",
                                         "_RAWRAKE", "_WT2RAKE", "_LLCPWT"]},
    {"step": "fill_missing_values", "numeric": "median", "categorical": "mode"},
    {"step": "one_hot", "max_categories": 10},
    {"step": "standardize"}
  ],

  "models": {
    "logreg_light": {"name": "reg_logistic_regression", "lambda_": 0.001, "gamma": 0.1, "max_iters": 2000},
    "logreg_heavy": {"name": "reg_logistic_regression", "lambda_": 0.1,   "gamma": 0.1, "max_iters": 2000},
    "ridge":        {"name": "ridge_regression",        "lambda_": 0.01},
    "meta":         {"name": "logistic_regression",     "gamma": 0.1,     "max_iters": 500}
  },

  "stack": [
    ["logreg_light", "logreg_heavy", "ridge"],
    ["meta"]
  ],

  "eval": {
    "cv": {"folds": 5, "stratified": true, "shuffle": true},
    "metrics": ["f1", "accuracy", "precision", "recall"],
    "threshold_search": {"optimize": "f1", "min": 0.05, "max": 0.95, "step": 0.01},
    "save": ["oof_predictions", "history", "submission"]
  }
}
"""

class ConfigBuilder:
    def __init__(self, config_file):
        with open(config_file, 'r') as f:
            self.config = json.load(f)

    def get(self, key, default=None):
        return self.config.get(key, default)

    def preprocess_config(self):
        return self.config.get("preprocess", {})

    def return_seed(self):
        return self.config.get("seed", 42)

    def return_experiment_name(self):
        return self.config.get("name", "default_experiment")

    def return_model_config(self):
        return self.config.get("model", {})

    def return_model_stack(self):
        return self.config.get("stack", [])

    def return_eval_config(self):
        return self.config.get("eval", {})


class ConfigRunner:
    def __init__(self, config_builder:ConfigBuilder):
        self.config_builder = config_builder

    def run_data_preprocess(self, X_train, X_test, y_train, train_ids, test_ids, feature_names):
        preprocess_steps = self.config_builder.preprocess_config()
        preprocessor_instance = preprocessor.Preprocessor(X_train, y_train, X_test, train_ids, test_ids, feature_names)

        for step in preprocess_steps:
            fn_name = step.get("step")
            args = {k: v for k, v in step.items() if k != "step"}
            preprocessor_instance.run(fn_name, **args)

        return preprocessor_instance.get_processed_data(), preprocessor_instance



    

