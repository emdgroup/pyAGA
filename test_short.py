import unittest

import pandas as pd

import parameter_study


class MainTest(unittest.TestCase):
    def test_success_T_world(self):
        """
        Test the T-world, meaning the 20x10-world with translational invariance.
        Results from this world have been published in the corresponding submission.
        :return:
        """
        config_name = (
            "parameter_study/test_config_files/parameter_study_success_T_world.ini"
        )
        results_df: pd.DataFrame = parameter_study.main(
            running_as_test=True, config_name=config_name, study="20x10"
        )
        self.assertEqual(1, results_df.shape[0])
        self.assertEqual("exact", results_df["is_group_order_correct"].iloc[0])
        self.assertTrue(results_df["fundamental_generators_contained"].iloc[0])

    def test_failure_T_world(self):
        """
        Test the T-world, meaning the 20x10-world with translational invariance.
        Results from this world have been published in the corresponding submission.
        :return:
        """
        config_name = (
            "parameter_study/test_config_files/parameter_study_failure_T_world.ini"
        )
        results_df: pd.DataFrame = parameter_study.main(
            running_as_test=True, config_name=config_name, study="20x10"
        )
        self.assertEqual(1, results_df.shape[0])
        self.assertEqual("too few", results_df["is_group_order_correct"].iloc[0])
        self.assertFalse(results_df["fundamental_generators_contained"].iloc[0])


if __name__ == '__main__':
    unittest.main()