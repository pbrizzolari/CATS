import pandas as pd
from pandas.testing import assert_frame_equal
import numpy as np
import unittest
import sklearn.linear_model as linear_model
from typing import Union, Type, Tuple
from repeated_CV_builder import DCV

class testing_DCV(unittest.TestCase):
    def setUp(self) -> None:
        self.data = pd.DataFrame([[0],[1],[2],[3],[0],[1],[2],[3],[0],[1],[2],[3],[0],[1],[2],[3],[0],[1],[2],[3]])
        self.classes = np.array([0,0,1,1,0,0,1,1,0,0,1,1,0,0,1,1,0,0,1,1])
        self.model = linear_model.LogisticRegression()
        self.model.fit(self.data, self.classes)
        self.dcv_model = DCV(self.model)
        self.dcv_model.hyperParams["C"] = [0,1]
        self.dcv_model.num_features = "all"




    def test_class_cleaner(self):
        target = pd.DataFrame([0,1,2,3])
        pd_df = pd.DataFrame([0,1,2,3])
        pd_sr = pd.Series([0,1,2,3])
        pd_id = pd.DataFrame([0,1,2,3]).index
        np_arr = np.array([0,1,2,3], dtype=np.int64)

        assert_frame_equal(DCV.__class_cleaner__(None,pd_df),target)
        assert_frame_equal(DCV.__class_cleaner__(None,pd_sr),target)
        assert_frame_equal(DCV.__class_cleaner__(None,pd_id),target)
        assert_frame_equal(DCV.__class_cleaner__(None,np_arr),target)

    def test_class_cleaner_error(self):
        self.assertRaises(TypeError,DCV.__class_cleaner__, [0,1], (0,1))

    def test_data_cleaner_error(self):
        self.assertRaises(TypeError,DCV.__data_cleaner__, [0,1], (0,1))

    def test_model(self):
        self.assertEqual(self.model.predict([[-1]]),[0])
        self.assertEqual(self.model.predict([[5]]),[1])

    def test_hyperparams(self):
        target = {"C":[0,1]}
        self.assertEqual(self.dcv_model.hyperParams,target)




if __name__ == '__main__':
    unittest.main()