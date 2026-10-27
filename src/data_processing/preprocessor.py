import os
import numpy as np

# Planned preprocessor helpers (not part of the graded functions):
# Helpers:
#   - save_history(path)
#   - plot_missing(), plot_histogram(name), plot_correlation()
#
# More advanced:
#   Column selection:
#     - drop_correlated_cols(threshold)
#   Rows (train only):
#     - drop_nan_rows(threshold)
#     - drop_duplicate_rows()
#     - drop_outlier_rows(z)
#   Encoding:
#     - ordinal_encode(columns)
#     - frequency_encode(columns)
#     - target_encode(columns)       (per fold)
#     - bin_numeric(column, bins)
#   Transforms / scaling:
#     - clip_outliers(lower, upper)
#     - log_transform(columns)
#     - min_max_scale()
#     - robust_scale()


class Preprocessor:

    """
    Preprocessor class for data preprocessing tasks.
    Contains:
        1. drop_features: Drop specified features from the dataset.
        2. standardize: Standardize the dataset to have zero mean and unit variance.
        3. one_hot_encode: One-hot encode categorical features in the dataset.
        4. fill_missing_values: Fill missing values in the dataset with specified strategy.
        5. drop_duplicates: Drop duplicate rows from the dataset.
        6. remove_outliers: Remove outliers from the dataset based on z-score threshold.
        7. drop_nan(threshold): Drop features with a percentage of NaN values above the specified threshold.
        8. drop_nan_rows(threshold): Drop rows with a percentage of NaN values above the specified threshold.
        9. map_dict: Map values in a specified feature to new values based on a provided mapping dictionary.
        10. map_func: Map values in a specified feature to new values based on a provided mapping function.
        11. copy / fold(tr, va): A new Preprocessor on the same data, or on one CV fold (validation rows as test set).
        12. clean(drop_above): Basic cleaning: drop_nan, drop_constant, median fill, standardize.
        13. add_bias: Add a column of ones (the intercept) in front of the features.
        14. undersample(ratio, seed): Keep all positive training rows and ratio times as many negatives.
    """

    def __init__(self, x_train, y_train, x_test, train_ids, test_ids, feature_names):
        self.x_train = x_train.astype(float, copy=True)
        self.y_train = np.asarray(y_train).copy()
        self.x_test = x_test.astype(float, copy=True)
        self.train_ids = np.asarray(train_ids).copy()
        self.test_ids = np.asarray(test_ids).copy()
        self.feature_names = list(feature_names)

    # ids + data are only glued together when someone asks for them,
    # rebuilding them after every step copied the whole matrix each time
    @property
    def x_ids_train(self):
        return np.concatenate((self.train_ids.reshape(-1, 1), self.x_train), axis=1)

    @property
    def x_ids_test(self):
        return np.concatenate((self.test_ids.reshape(-1, 1), self.x_test), axis=1)

    def copy(self):
        return Preprocessor(
            self.x_train,
            self.y_train,
            self.x_test,
            self.train_ids,
            self.test_ids,
            self.feature_names,
        )

    def fold(self, tr, va):
        return Preprocessor(
            self.x_train[tr],
            self.y_train[tr],
            self.x_train[va],
            self.train_ids[tr],
            self.train_ids[va],
            self.feature_names,
        )

    def keep_columns(self, keep):
        dropped = [n for n, k in zip(self.feature_names, keep) if not k]
        self.x_train = self.x_train[:, keep]
        self.x_test = self.x_test[:, keep]
        self.feature_names = [n for n, k in zip(self.feature_names, keep) if k]
        return dropped

    def keep_train_rows(self, keep):
        removed = self.train_ids[~keep].tolist()
        self.x_train = self.x_train[keep]
        self.y_train = self.y_train[keep]
        self.train_ids = self.train_ids[keep]
        return removed

    # TESTED, WORKING FINALLY
    def drop_features(self, features_to_drop):
        missing = [f for f in features_to_drop if f not in self.feature_names]
        if missing:
            print(f"drop_features: not found, skipped {missing}")
        drop = set(features_to_drop)
        keep = np.array([n not in drop for n in self.feature_names], dtype=bool)
        return self.keep_columns(keep)

    # TESTED
    def standardize(self):
        mean = np.nanmean(self.x_train, axis=0)
        std = np.nanstd(self.x_train, axis=0)
        no_values = np.isnan(mean)
        mean[no_values] = 0.0
        std[no_values | (std == 0)] = 1.0
        self.x_train -= mean
        self.x_train /= std
        self.x_test -= mean
        self.x_test /= std

    # TESTED
    def one_hot_encode(self, categorical_features):
        categories = {}
        positions = [self.feature_names.index(f) for f in categorical_features]
        new_train, new_test, new_names = [], [], []
        for feature, position in zip(categorical_features, positions):
            col = self.x_train[:, position]
            cats = np.unique(col[~np.isnan(col)])

            new_train.append((self.x_train[:, [position]] == cats).astype(float))
            new_test.append((self.x_test[:, [position]] == cats).astype(float))
            new_names += [f"{feature}={c:g}" for c in cats]
            categories[feature] = cats.tolist()

        self.x_train = np.hstack([np.delete(self.x_train, positions, axis=1)] + new_train)
        self.x_test = np.hstack([np.delete(self.x_test, positions, axis=1)] + new_test)
        encoded = set(categorical_features)
        self.feature_names = [n for n in self.feature_names if n not in encoded] + new_names
        return categories

    # TESTED
    def fill_missing_values(self, strategy="mean"):
        if strategy == "mean":
            fill = np.nanmean(self.x_train, axis=0)
        elif strategy == "median":
            fill = np.nanmedian(self.x_train, axis=0)
        else:
            raise ValueError("Strategy not recognized. Use 'mean' or 'median'.")

        self.x_train = np.where(np.isnan(self.x_train), fill, self.x_train)
        self.x_test = np.where(np.isnan(self.x_test), fill, self.x_test)

    # TESTED
    def drop_duplicates(self):
        filled = np.where(np.isnan(self.x_train), np.inf, self.x_train)
        _, first = np.unique(filled, axis=0, return_index=True)
        keep = np.zeros(len(self.x_train), dtype=bool)
        keep[first] = True
        return self.keep_train_rows(keep)

    # TESTED
    def remove_outliers(self, z_threshold):
        mean = np.nanmean(self.x_train, axis=0)
        std = np.nanstd(self.x_train, axis=0)

        valid_values = ~np.isnan(std) & (std > 0)

        z_scores = np.abs(
            (self.x_train[:, valid_values] - mean[valid_values]) / std[valid_values]
        )
        outlier = (z_scores >= z_threshold).any(axis=1)
        return self.keep_train_rows(~outlier)

    # TESTED
    def drop_nan(self, threshold):
        nan_percentage = np.isnan(self.x_train).mean(axis=0)
        features_to_drop = nan_percentage <= threshold
        self.keep_columns(features_to_drop)

    def drop_constant(self):
        observed = np.isfinite(self.x_train)
        lowest = np.where(observed, self.x_train, np.inf).min(axis=0)
        highest = np.where(observed, self.x_train, -np.inf).max(axis=0)
        self.keep_columns(lowest < highest)

    # TESTED
    def drop_nan_rows(self, threshold):
        nan_percentage = np.isnan(self.x_train).mean(axis=1)
        return self.keep_train_rows(nan_percentage <= threshold)

    # TESTED (times 2)
    def map_dict(self, feature, mapping_dict):
        mapping, has_nan, nan_value = {}, False, None
        for k, v in mapping_dict.items():
            new = np.nan if v is None else float(v)
            if str(k).lower() == "nan":
                has_nan, nan_value = True, new
            else:
                mapping[float(k)] = new

        # masks come from the old column, so 1 -> 2 and 2 -> 1 doesn't chain
        j = self.feature_names.index(feature)
        for x in (self.x_train, self.x_test):
            old = x[:, j].copy()
            for k, v in mapping.items():
                x[old == k, j] = v
            if has_nan:
                x[np.isnan(old), j] = nan_value
        return mapping

    # TESTED
    def map_func(self, feature, mapping_func):
        j = self.feature_names.index(feature)
        f = np.vectorize(mapping_func, otypes=[float])
        self.x_train[:, j] = f(self.x_train[:, j])
        self.x_test[:, j] = f(self.x_test[:, j])

    def map_special_codes(self, groups):
        for group in groups:
            for f in group["columns"]:
                if f not in self.feature_names:
                    print(f"map_special_codes: '{f}' not found, skipped")
                    continue
                self.map_dict(f, group["map"])

    def clean(self, drop_above=0.90):
        self.drop_nan(drop_above)
        self.drop_constant()
        self.fill_missing_values("median")
        self.standardize()
        return self.x_train, self.x_test, self.feature_names

    def add_bias(self):
        self.x_train = np.column_stack([np.ones(len(self.x_train)), self.x_train])
        self.x_test = np.column_stack([np.ones(len(self.x_test)), self.x_test])
        self.feature_names = ["bias"] + self.feature_names

    def undersample(self, ratio, seed):
        rng = np.random.default_rng(seed)
        pos = np.flatnonzero(self.y_train == 1)
        neg = np.flatnonzero(self.y_train != 1)
        take = min(len(neg), int(round(ratio * len(pos))))
        chosen = np.concatenate([pos, rng.choice(neg, take, replace=False)])
        rng.shuffle(chosen)
        self.x_train = self.x_train[chosen]
        self.y_train = self.y_train[chosen]
        self.train_ids = self.train_ids[chosen]


    def labels_to_01(self):
        self.y_train = np.where(self.y_train > 0, 1, 0)

    """
    Getters for processed data, training data, and test data.

    The following methods return the processed training and test datasets, along with the training labels and

    """

    def get_processed_data(self):
        self.train_ids = self.train_ids.astype(int)
        self.test_ids = self.test_ids.astype(int)
        return self.x_train, self.x_test, self.y_train, self.train_ids, self.test_ids

    def get_train_data(self):
        self.train_ids = self.train_ids.astype(int)
        return self.x_train, self.y_train, self.train_ids

    def get_test_data(self):
        self.test_ids = self.test_ids.astype(int)
        return self.x_test, self.test_ids

    # Runner
    def run(self, name, *args, **kwargs):
        if hasattr(self, name):
            method = getattr(self, name)
            return method(*args, **kwargs)
        else:
            raise ValueError(f"Method {name} not found in Preprocessor class!")

    # Save result
    def export(self, path):
        os.makedirs(path, exist_ok=True)
        header = ",".join(["Id"] + self.feature_names)
        files = {
            "x_train.csv": (
                np.column_stack([self.train_ids, self.x_train]),
                header,
                "%.10g",
            ),
            "x_test.csv": (
                np.column_stack([self.test_ids, self.x_test]),
                header,
                "%.10g",
            ),
            "y_train.csv": (
                np.column_stack([self.train_ids, self.y_train]),
                "Id,_MICHD",
                "%d",
            ),
        }
        paths = []
        for filename, (table, head, f) in files.items():
            new_path = os.path.join(path, filename)
            np.savetxt(new_path, table, delimiter=",", header=head, comments="", fmt=f)
            paths.append(new_path)
        return paths


if __name__ == "__main__":
    pass
