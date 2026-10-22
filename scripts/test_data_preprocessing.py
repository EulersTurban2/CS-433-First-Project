import copy
import unittest
import numpy as np


import src.config as config
import src.data_processing.preprocessor as preprocessor

nan = np.nan
NAMES = ["age", "weight", "mostly_nan", "const", "cat", "yesno", "days", "all_nan"]

def make_data():
    x_train = np.array([
        
        [20,   60,    nan,       5,    1,  1,    88,  nan],   
        [30,   62,    nan,       5,    2,  2,    5,   nan],   
        [40,   65,    nan,       5,    3,  1,    30,  nan],   
        [50,   70,    nan,       5,    1,  2,    88,  nan],   
        [60,   68,    nan,       5,    2,  7,    77,  nan],   
        [70,   72,    nan,       5,    3,  9,    99,  nan],   
        [80,   300,   nan,       5,    1,  1,    10,  nan],   
        [nan,  nan,   nan,       5,    2,  9,    88,  nan],   
        [25,   64,    1,         5,    3,  1,    2,   nan],   
        [25,   64,    1,         5,    3,  1,    2,   nan],  
    ], dtype=float)
    x_test = np.array([
        [33,   70,    nan,       5,    1,  1,    88,  1],     
        [nan,  80,    3,         5,    4,  2,    15,  2],     
        [90,   nan,   nan,       6,    2,  9,    77,  nan],   
        [18,   55,    nan,       5,    3,  1,    3,   4],     
    ], dtype=float)
    y_train = np.array([1, -1, -1, 1, -1, -1, -1, 1, -1, -1])
    train_ids = np.arange(100, 110)
    test_ids = np.arange(200, 204)
    return x_train, y_train, x_test, train_ids, test_ids, list(NAMES)


def make_prep():
    return preprocessor.Preprocessor(*make_data())

def check_rows(test: unittest.TestCase, prep: preprocessor.Preprocessor, data):
    for k, row_id in enumerate(prep.train_ids):
        i = row_id - 100
        np.testing.assert_array_equal(prep.x_train[k], data[0][i])
        test.assertEqual(prep.y_train[k], data[1][i])
    np.testing.assert_array_equal(prep.x_ids_train[:, 0], prep.train_ids)
    np.testing.assert_array_equal(prep.x_ids_train[:, 1:], prep.x_train)
    np.testing.assert_array_equal(prep.x_test, data[2])
    np.testing.assert_array_equal(prep.test_ids, data[4])

def check_drop_nan(test:unittest.TestCase, threshold, names):
    x_train, y_train, x_test, *_ = make_data()
    prep = make_prep()
    prep.drop_nan(threshold)
    test.assertEqual(prep.feature_names, names)
    keep = [NAMES.index(n) for n in names]
    np.testing.assert_array_equal(prep.x_train, x_train[:, keep])
    np.testing.assert_array_equal(prep.x_test, x_test[:, keep])
    np.testing.assert_array_equal(prep.x_ids_train[:, 1:], prep.x_train)
    np.testing.assert_array_equal(prep.x_ids_test[:, 1:], prep.x_test)

def check_only_column_changed(test:unittest.TestCase, prep:preprocessor.Preprocessor, j):
    x_train, _, x_test, *_ = make_data()
    others = [k for k in range(len(NAMES)) if k != j]
    np.testing.assert_array_equal(prep.x_train[:, others], x_train[:, others])
    np.testing.assert_array_equal(prep.x_test[:, others], x_test[:, others])
    np.testing.assert_array_equal(prep.x_ids_train[:, 1:], prep.x_train)
    np.testing.assert_array_equal(prep.x_ids_test[:, 1:], prep.x_test)
    test.assertEqual(prep.feature_names, NAMES)

class TestUnit(unittest.TestCase):
    
    def test_drop_features(self):
        columns = ["mostly_nan"]
        NAMES_copy = copy.deepcopy(NAMES)
        NAMES_copy.remove("mostly_nan")
        prep = make_prep()
        prep.drop_features(columns)
        self.assertEqual(prep.feature_names, NAMES_copy)


    def test_mean_fill_missing_values(self):
        x_train, y_train, x_test, *_ = make_data()
        prep = make_prep()
        prep.fill_missing_values(strategy="mean")
        self.assertAlmostEqual(prep.x_train[7,NAMES.index("age")], 400/9)
        self.assertAlmostEqual(prep.x_test[1,NAMES.index("age")], 400/9)

        # no unchanged values (must be tested out as well)
        observed_train, observed_test = ~np.isnan(x_train), ~np.isnan(x_test)
        np.testing.assert_array_equal(prep.x_train[observed_train], x_train[observed_train])
        np.testing.assert_array_equal(prep.x_test[observed_test], x_test[observed_test])

        #shapes must change the same
        self.assertEqual(prep.x_train.shape, (10,8))
        self.assertEqual(prep.x_test.shape, (4,8))
        self.assertEqual(prep.feature_names, NAMES)

    def test_median_fill_missing_values(self):
        x_train, y_train, x_test, *_ = make_data()
        prep = make_prep()
        prep.fill_missing_values(strategy="median")
        self.assertEqual(prep.x_train[7,NAMES.index("age")], 40.0)
        self.assertEqual(prep.x_test[1,NAMES.index("age")], 40.0)
        self.assertEqual(prep.x_train[7, NAMES.index("weight")], 65.0)

        # no unchanged values (must be tested out as well)
        observed_train, observed_test = ~np.isnan(x_train), ~np.isnan(x_test)
        np.testing.assert_array_equal(prep.x_train[observed_train], x_train[observed_train])
        np.testing.assert_array_equal(prep.x_test[observed_test], x_test[observed_test])

        #shapes must change the same
        self.assertEqual(prep.x_train.shape, (10,8))
        self.assertEqual(prep.x_test.shape, (4,8))
        self.assertEqual(prep.feature_names, NAMES)
        
    def test_standardize(self):
        prep = make_prep()
        prep.standardize()

        # we test whether the mean is 0 and variance is 1
        varying = [j for j in range(len(NAMES)) if j not in (NAMES.index("const"), NAMES.index("all_nan"), NAMES.index("mostly_nan"))]
        np.testing.assert_allclose(np.nanmean(prep.x_train[:, varying], axis=0), 0, atol=1e-12)
        np.testing.assert_allclose(np.nanstd(prep.x_train[:, varying], axis=0), 1, atol=1e-12)

        # shapes stay the same
        self.assertEqual(prep.x_train.shape, (10, 8))
        self.assertEqual(prep.x_test.shape, (4, 8))
        self.assertEqual(prep.feature_names, NAMES)
        np.testing.assert_array_equal(prep.x_ids_train[:, 1:], prep.x_train)
        np.testing.assert_array_equal(prep.x_ids_test[:, 1:], prep.x_test)

    def test_removes_row_107(self):
        data, prep = make_data(), make_prep()
        prep.drop_nan_rows(0.4)
        self.assertEqual(prep.train_ids.tolist(), [100, 101, 102, 103, 104, 105, 106, 108, 109])
        self.assertEqual(int((prep.y_train == 1).sum()), 2)   
        check_rows(self, prep, data)

    def test_threshold_one_keeps_everything(self):
        data, prep = make_data(), make_prep()
        prep.drop_nan_rows(1.0)
        self.assertEqual(len(prep.train_ids), 10)
        check_rows(self, prep, data)

    def test_drop_nan_half(self):
        check_drop_nan(self, 0.5, ["age", "weight", "const", "cat", "yesno", "days"])

    def test_drop_nan_one_keeps_everything(self):
        check_drop_nan(self, 1.0, NAMES)

    def test_drop_zero_keeps_only_complete(self):
        check_drop_nan(self, 0.0,  ["const", "cat", "yesno", "days"])

    def test_remove_outliers_removes_weight_outlier(self):
        data, prep = make_data(), make_prep()
        prep.remove_outliers(2.5)                         
        self.assertEqual(prep.train_ids.tolist(), [100, 101, 102, 103, 104, 105, 107, 108, 109])
        check_rows(self, prep, data)

    def test_remove_outliers_high_threshold_removes_nothing(self):
        data, prep = make_data(), make_prep()
        prep.remove_outliers(3.0)                         
        self.assertEqual(len(prep.train_ids), 10)
        check_rows(self, prep, data)

    def test_map_func_float_results(self):
        days = NAMES.index("days")
        prep = make_prep()
        prep.map_func("days", lambda v: v / 2)
        np.testing.assert_array_equal(prep.x_train[:, days], [44, 2.5, 15, 44, 38.5, 49.5, 5, 44, 1, 1])
        np.testing.assert_array_equal(prep.x_test[:, days], [44, 7.5, 38.5, 1.5])
        check_only_column_changed(self, prep, days)

    def test_map_func_nan_stays_nan(self):
        age = NAMES.index("age")
        prep = make_prep()
        prep.map_func("age", lambda v: v + 1)
        self.assertTrue(np.isnan(prep.x_train[7, age]))           
        self.assertTrue(np.isnan(prep.x_test[1, age]))            
        self.assertEqual(prep.x_train[0, age], 21)
        check_only_column_changed(self, prep, age)

    def test_map_dict_does_not_chain(self):
        yesno = NAMES.index("yesno")
        prep = make_prep()
        prep.map_dict("yesno", {1: 2, 2: 1})               
        np.testing.assert_array_equal(prep.x_train[:, yesno], [2, 1, 2, 1, 7, 9, 2, 9, 2, 2])
        np.testing.assert_array_equal(prep.x_test[:, yesno], [2, 1, 9, 2])

    def test_drop_duplicates_keeps_first_and_order(self):
        data = make_data()
        data[0][6] = data[0][2]                            
        prep = preprocessor.Preprocessor(*data)
        prep.drop_duplicates()
        self.assertEqual(prep.train_ids.tolist(), [100, 101, 102, 103, 104, 105, 107, 108])
        check_rows(self, prep, data)

    def test_drop_duplicates_ids_stay_integers(self):
        prep = make_prep()
        prep.drop_duplicates()
        self.assertTrue(np.issubdtype(prep.train_ids.dtype, np.integer))
        self.assertTrue(np.issubdtype(prep.test_ids.dtype, np.integer))

    def test_one_hot_names_and_order(self):
        prep = make_prep()
        prep.one_hot_encode(["cat"])
        self.assertEqual(prep.feature_names,
                         ["age", "weight", "mostly_nan", "const", "yesno", "days", "all_nan",
                          "cat=1", "cat=2", "cat=3"])

    def test_one_hot_train_values(self):
        prep = make_prep()
        prep.one_hot_encode(["cat"])
        expected = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 0, 0], [0, 1, 0],
                             [0, 0, 1], [1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 0, 1]], dtype=float)
        np.testing.assert_array_equal(prep.x_train[:, -3:], expected)
        np.testing.assert_array_equal(prep.x_train[:, -3:].sum(axis=1), np.ones(10))  

    def test_one_hot_test_values_and_unseen_category(self):
        prep = make_prep()
        prep.one_hot_encode(["cat"])
        expected = np.array([[1, 0, 0], [0, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float) 
        np.testing.assert_array_equal(prep.x_test[:, -3:], expected)

if __name__ == "__main__":
    unittest.main()