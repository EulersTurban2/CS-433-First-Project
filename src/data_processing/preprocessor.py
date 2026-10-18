import os
import numpy as np

# TODO: Preprocessor (factory) functions
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
    """
    
    def __init__(self, x_train, y_train, x_test, train_ids, test_ids, feature_names):
        self.x_train = x_train.astype(float, copy=True)
        self.y_train = np.asarray(y_train).copy()
        self.x_test = x_test.astype(float, copy=True)
        self.train_ids = np.asarray(train_ids).copy()
        self.test_ids = np.asarray(test_ids).copy()
        self.feature_names = list(feature_names)
        
        self.x_ids_train = np.concatenate((self.train_ids.reshape(-1, 1), self.x_train), axis=1)
        self.x_ids_test = np.concatenate((self.test_ids.reshape(-1, 1), self.x_test), axis=1)


    def keep_columns(self, keep):
        dropped = [n for n, k in zip(self.feature_names, keep) if not k]
        self.x_train = self.x_train[:, keep]
        self.x_test = self.x_test[:, keep]
        self.x_ids_train = np.concatenate((self.train_ids.reshape(-1, 1), self.x_train), axis=1)
        self.x_ids_test = np.concatenate((self.test_ids.reshape(-1, 1), self.x_test), axis=1)
        self.feature_names = [n for n, k in zip(self.feature_names, keep) if k]
        return dropped

    def keep_train_rows(self, keep):
        removed = self.train_ids[~keep].tolist()
        self.x_train = self.x_train[keep]
        self.y_train = self.y_train[keep]
        self.train_ids = self.train_ids[keep]
        self.x_ids_train = np.concatenate((self.train_ids.reshape(-1, 1), self.x_train), axis=1)
        return removed

    #TESTED, WORKING FINALLY
    def drop_features(self, features_to_drop):
        missing = [f for f in features_to_drop if f not in self.feature_names]
        if missing:
            print(f"drop_features: not found, skipped {missing}")
        drop = set(features_to_drop)
        keep = np.array([n not in drop for n in self.feature_names], dtype=bool)
        return self.keep_columns(keep)

    #TESTED
    def standardize(self):
        mean = np.nanmean(self.x_ids_train[:, 1:], axis=0)
        std = np.nanstd(self.x_ids_train[:, 1:], axis=0)
        no_values = np.isnan(mean)
        mean[no_values] = 0.0
        std[no_values | (std == 0)] = 1.0
        self.x_train = (self.x_train - mean) / std 
        self.x_test = (self.x_test - mean) / std
        self.x_ids_train = np.concatenate((self.train_ids.reshape(-1, 1), self.x_train), axis=1)
        self.x_ids_test = np.concatenate((self.test_ids.reshape(-1, 1), self.x_test), axis=1)


    #TESTED
    def one_hot_encode(self, categorical_features):
        categories = {}
        for feature in categorical_features:
            position = self.feature_names.index(feature)
            col = self.x_train[:, position]
            cats = np.unique(col[~np.isnan(col)])

            new_train = (self.x_train[:, [position]] == cats).astype(float)
            new_test = (self.x_test[:, [position]] == cats).astype(float)
            self.x_train = np.hstack([np.delete(self.x_train, position, axis=1), new_train])
            self.x_test = np.hstack([np.delete(self.x_test, position, axis=1), new_test])
            self.feature_names = [n for n in self.feature_names if n != feature] + [f"{feature}={c:g}" for c in cats]
            categories[feature] = cats.tolist()

        self.x_ids_train = np.concatenate((self.train_ids.reshape(-1, 1), self.x_train), axis=1)
        self.x_ids_test = np.concatenate((self.test_ids.reshape(-1, 1), self.x_test), axis=1)
        return categories

    #TESTED
    def fill_missing_values(self, strategy='mean'):
        if strategy == 'mean':
            mean = np.nanmean(self.x_train, axis=0)
            inds_train = np.where(np.isnan(self.x_train))
            self.x_train[inds_train] = np.take(mean, inds_train[1])

            inds_test = np.where(np.isnan(self.x_test))
            self.x_test[inds_test] = np.take(mean, inds_test[1])
        elif strategy == 'median':
            median = np.nanmedian(self.x_train, axis=0)
            inds_train = np.where(np.isnan(self.x_train))
            self.x_train[inds_train] = np.take(median, inds_train[1])

            inds_test = np.where(np.isnan(self.x_test))
            self.x_test[inds_test] = np.take(median, inds_test[1])
        else:
            raise ValueError("Strategy not recognized. Use 'mean' or 'median'.")

        self.x_ids_train = np.concatenate((self.train_ids.reshape(-1, 1), self.x_train), axis=1)
        self.x_ids_test = np.concatenate((self.test_ids.reshape(-1, 1), self.x_test), axis=1)

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
        
        z_scores = np.abs((self.x_train[:, valid_values] - mean[valid_values]) / std[valid_values])
        outlier = (z_scores >= z_threshold).any(axis=1)
        return self.keep_train_rows(~outlier)

    # TESTED
    def drop_nan(self, threshold):
        nan_percentage = np.isnan(self.x_train).mean(axis=0)
        features_to_drop = nan_percentage <= threshold
        self.keep_columns(features_to_drop)

    # TESTED
    def drop_nan_rows(self, threshold):
        nan_percentage = np.isnan(self.x_ids_train[:, 1:]).mean(axis=1)  # Exclude the ID column
        rows_to_keep = np.where(nan_percentage <= threshold)[0]
        self.x_ids_train = self.x_ids_train[rows_to_keep]
        self.x_train = self.x_ids_train[:, 1:]
        self.train_ids = self.train_ids[rows_to_keep]
        self.y_train = self.y_train[rows_to_keep]

    # TESTED
    def map_dict(self, feature, mapping_dict):
        mapping = {float(k): (np.nan if v is None else float(v)) for k, v in mapping_dict.items()}
        self.map_func(feature, lambda v: mapping.get(v, v))

    # TESTED
    def map_func(self, feature, mapping_func):
        j = self.feature_names.index(feature)
        f = np.vectorize(mapping_func, otypes=[float])
        self.x_train[:, j] = f(self.x_train[:, j])
        self.x_test[:, j] = f(self.x_test[:, j])
        self.x_ids_train = np.concatenate((self.train_ids.reshape(-1, 1), self.x_train), axis=1)
        self.x_ids_test = np.concatenate((self.test_ids.reshape(-1, 1), self.x_test), axis=1)
    
    def labels_to_01(self):
        self.y_train = np.where(self.y_train > 0, 1, 0)

    """
    Getters for processed data, training data, and test data.

    The following methods return the processed training and test datasets, along with the training labels and 

    """

    def get_processed_data(self):
        self.train_ids = self.x_ids_train[:, 0].astype(int)
        self.test_ids = self.x_ids_test[:, 0].astype(int)
        return self.x_train, self.x_test, self.y_train, self.train_ids, self.test_ids

    def get_train_data(self):
        self.train_ids = self.x_ids_train[:, 0].astype(int)
        return self.x_train, self.y_train, self.train_ids

    def get_test_data(self):
        self.test_ids = self.x_ids_test[:, 0].astype(int)
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
            "x_train.csv": (np.column_stack([self.train_ids, self.x_train]), header, "%.10g"),
            "x_test.csv": (np.column_stack([self.test_ids, self.x_test]), header, "%.10g"),
            "y_train.csv": (np.column_stack([self.train_ids, self.y_train]), "Id,_MICHD", "%d"),
        }
        paths = []
        for filename, (table, head, f) in files.items():
            new_path = os.path.join(path, filename)
            np.savetxt(new_path, table, delimiter=",", header=head, comments="", fmt=f)
            paths.append(new_path)
        return paths



if __name__ == "__main__":
    pass