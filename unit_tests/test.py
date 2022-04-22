import unittest

import pandas as pd

import calculate_automorphisms


class MainTest(unittest.TestCase):
    def test_success_T_world(self):
        """
        Test the T-world, meaning the 20x10-world with translational invariance.
        Results from this world have been published in the corresponding submission.
        :return:
        """
        config_name = "unit_tests/test_config_files/success_T_world.ini"
        results_df: pd.DataFrame = calculate_automorphisms.main(
            running_as_test=True, config_name=config_name, study="20x10"
        )
        self.assertEqual(1, results_df.shape[0])
        self.assertEqual("exact", results_df["is_group_order_correct"].iloc[0])
        self.assertTrue(results_df["fundamental_generators_contained"].iloc[0])

    def test_success_TR1_world(self):
        """
        Test the TR1-world, meaning the 15x15-world with translational and rotational invariance.
        Results from this world have been published in the corresponding submission.
        :return:
        """
        config_name = "unit_tests/test_config_files/success_TR1_world.ini"
        results_df: pd.DataFrame = calculate_automorphisms.main(
            running_as_test=True, config_name=config_name, study="15x15_rotations"
        )
        self.assertEqual(1, results_df.shape[0])
        self.assertEqual("exact", results_df["is_group_order_correct"].iloc[0])
        self.assertTrue(results_df["fundamental_generators_contained"].iloc[0])

    def test_success_TR2_world(self):
        """
        Test the TR2-world, meaning the 15x15-world with translational and rotational invariance.
        The letters which exhibit axial symmetry have been removed.
        Results from this world have been published in the corresponding submission.
        :return:
        """
        config_name = "unit_tests/test_config_files/success_TR2_world.ini"
        results_df: pd.DataFrame = calculate_automorphisms.main(
            running_as_test=True,
            config_name=config_name,
            study="no_axsym_15x15_rotations",
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
        config_name = "unit_tests/test_config_files/failure_T_world.ini"
        results_df: pd.DataFrame = calculate_automorphisms.main(
            running_as_test=True, config_name=config_name, study="20x10"
        )
        self.assertEqual(1, results_df.shape[0])
        self.assertEqual("too few", results_df["is_group_order_correct"].iloc[0])
        self.assertFalse(results_df["fundamental_generators_contained"].iloc[0])

    def test_failure_TR1_world(self):
        """
        Test the TR1-world, meaning the 15x15-world with translational and rotational invariance.
        Results from this world have been published in the corresponding submission.
        :return:
        """
        config_name = "unit_tests/test_config_files/failure_TR1_world.ini"
        results_df: pd.DataFrame = calculate_automorphisms.main(
            running_as_test=True, config_name=config_name, study="15x15_rotations"
        )
        self.assertEqual(1, results_df.shape[0])
        self.assertEqual("too few", results_df["is_group_order_correct"].iloc[0])
        self.assertFalse(results_df["fundamental_generators_contained"].iloc[0])

    def test_failure_TR2_world(self):
        """
        Test the TR2-world, meaning the 15x15-world with translational and rotational invariance.
        The letters which exhibit axial symmetry have been removed.
        Results from this world have been published in the corresponding submission.
        :return:
        """
        config_name = "unit_tests/test_config_files/failure_TR2_world.ini"
        results_df: pd.DataFrame = calculate_automorphisms.main(
            running_as_test=True,
            config_name=config_name,
            study="no_axsym_15x15_rotations",
        )
        self.assertEqual(1, results_df.shape[0])
        self.assertEqual("too few", results_df["is_group_order_correct"].iloc[0])
        self.assertFalse(results_df["fundamental_generators_contained"].iloc[0])


if __name__ == "__main__":
    unittest.main()
