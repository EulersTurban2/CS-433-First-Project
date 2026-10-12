import numpy as np


# TODO: Preprocessor (factory) functions
# Helpers:
#   - run(config)                    call steps from config in order, record history
#   - get_data()                     return x_train, x_test, names
#   - summary()                      shape + NaNs left
#   - save_history(path)
#   - describe_column(name)          unique values, NaN rate, min/max
#   - plot_missing(), plot_histogram(name), plot_correlation()
#
# More advanced:
#   Column selection:
#     - keep_columns(columns)
#     - drop_constant_cols(max_frac)
#     - drop_correlated_cols(threshold)
#   Rows (train only):
#     - drop_nan_rows(threshold)
#     - drop_duplicate_rows()
#     - drop_outlier_rows(z)
#   Cleaning:
#     - replace_values(column, mapping)
#     - decode_units(columns)        ALCDAY5, fruit/veg, WEIGHT2, HEIGHT3, exercise
#     - fix_decimals(columns)        e.g. _BMI5 / 100
#   Missing values:
#     - add_missing_indicators(min_rate)
#     - impute_knn(k)                slow in pure numpy
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
#   Feature engineering:
#     - polynomial_features(degree)
#     - interaction_features(pairs)
#     - custom_features()            e.g. BMI from weight and height
#     - pca(n_components)
#     - add_bias()
#   Feature selection (per fold):
#     - select_by_target_corr(k)
#     - select_by_mutual_info(k)
#   Class imbalance (per fold, train only):
#     - undersample(ratio)
#     - oversample(ratio)
#   Labels:
#     - labels_to_01(), labels_to_pm1()

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
    
    def __init__(self, x_train, y_train, x_test, train_ids, test_ids):
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.train_ids = train_ids
        self.test_ids = test_ids

        self.x_ids_train = np.concatenate((self.train_ids.reshape(-1, 1), self.x_train), axis=1)
        self.x_ids_test = np.concatenate((self.test_ids.reshape(-1, 1), self.x_test), axis=1)


    def drop_features(self, features_to_drop):
        self.x_ids_train = np.delete(self.x_ids_train, features_to_drop, axis=1)
        self.x_ids_test = np.delete(self.x_ids_test, features_to_drop, axis=1)

    def standardize(self):
        mean = np.mean(self.x_ids_train[:, 1:], axis=0)  # Exclude the ID column
        std = np.std(self.x_ids_train[:, 1:], axis=0)
        self.x_train = (self.x_train - mean) / std 
        self.x_test = (self.x_test - mean) / std

        self.x_ids_train = np.concatenate((self.train_ids.reshape(-1, 1), self.x_train), axis=1)
        self.x_ids_test = np.concatenate((self.test_ids.reshape(-1, 1), self.x_test), axis=1)

    def one_hot_encode(self, categorical_features):
        for feature in categorical_features:
            unique_values = np.unique(self.x_ids_train[:, feature])
            one_hot_train = np.zeros((self.x_ids_train.shape[0], len(unique_values)))
            one_hot_test = np.zeros((self.x_ids_test.shape[0], len(unique_values)))

            for i, value in enumerate(unique_values):
                one_hot_train[:, i] = (self.x_ids_train[:, feature] == value).astype(int)
                one_hot_test[:, i] = (self.x_ids_test[:, feature] == value).astype(int)

            self.x_ids_train = np.delete(self.x_ids_train, feature, axis=1)
            self.x_ids_test = np.delete(self.x_ids_test, feature, axis=1)

            self.x_ids_train = np.hstack((self.x_ids_train, one_hot_train))
            self.x_ids_test = np.hstack((self.x_ids_test, one_hot_test))

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

    def drop_duplicates(self):
        self.x_train = np.unique(self.x_train, axis=0)
        self.x_test = np.unique(self.x_test, axis=0)

    def remove_outliers(self, z_threshold):
        mean = np.mean(self.x_train, axis=0)
        std = np.std(self.x_train, axis=0)
        z_scores = np.abs((self.x_train - mean) / std)
        self.x_train = self.x_train[(z_scores < z_threshold).all(axis=1)]
        self.y_train = self.y_train[(z_scores < z_threshold).all(axis=1)]

        self.x_ids_test = np.concatenate((self.test_ids.reshape(-1, 1), self.x_test), axis=1)
        self.x_ids_train = np.concatenate((self.train_ids.reshape(-1, 1), self.x_train), axis=1)

    def drop_nan(self, threshold):
        nan_percentage = np.isnan(self.x_train).mean(axis=0)
        features_to_drop = np.where(nan_percentage > threshold)[0]
        self.drop_features(features_to_drop)

    def drop_nan_rows(self, threshold):
        nan_percentage = np.isnan(self.x_ids_train[:, 1:]).mean(axis=1)  # Exclude the ID column
        rows_to_keep = np.where(nan_percentage <= threshold)[0]
        self.x_ids_train = self.x_ids_train[rows_to_keep]
        self.y_train = self.y_train[rows_to_keep]

    def map_dict(self, feature, mapping_dict):
        self.x_ids_train[:, feature] = np.vectorize(mapping_dict.get)(self.x_ids_train[:, feature])
        self.x_ids_test[:, feature] = np.vectorize(mapping_dict.get)(self.x_ids_test[:, feature])

    def map_func(self, feature, mapping_func):
        self.x_ids_train[:, feature] = np.vectorize(mapping_func)(self.x_ids_train[:, feature])
        self.x_ids_test[:, feature] = np.vectorize(mapping_func)(self.x_ids_test[:, feature])

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




if __name__ == "__main__":
    pass