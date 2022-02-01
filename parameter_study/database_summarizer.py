import os
import re

import pandas as pd

testcase = "15x15_rotations"
regex_str = f"{testcase}.*xlsx"
expected_permutation_group_order = 900

dataframes = []
for excel_sheet in os.listdir("results"):
    if re.match(regex_str, excel_sheet) is not None:
        df = pd.read_excel(f"results/{excel_sheet}")

        dataframes.append(df)



big_df = pd.concat(dataframes)
timeout_or_skip = big_df[
    (big_df["num_found_trafos"] == "skipped")
    | (big_df["num_found_trafos"] == "Timeout")
]
big_df.drop(timeout_or_skip.index, inplace=True)

big_df.rename(
    columns=lambda col: "local_index" if col == "Unnamed: 0" else col, inplace=True
)

# Remove duplicate rows
big_df.drop_duplicates(inplace=True)

success_df = big_df[
    (big_df["fundamental_generators_contained"].astype(bool) == True)
    & (
            big_df["permutation_group_order"].astype(int) == expected_permutation_group_order
    )
]

too_many_df = big_df[
    (big_df["fundamental_generators_contained"].astype(bool) == True)
    & (
            big_df["permutation_group_order"].astype(int) > expected_permutation_group_order
    )
]

total_failure_df = pd.concat([big_df, success_df, too_many_df]).drop_duplicates(keep=False)

sorting = ["percentage_observation", "error_value_limit", "kde_bandwidth", "trafo_fault_tolerance_ratio"]

success_df = success_df.sort_values(by=sorting)
too_many_df = too_many_df.sort_values(by=sorting)
total_failure_df = total_failure_df.sort_values(by=sorting)

success_df.to_excel(f"results/{testcase}_summary_success.xlsx")
too_many_df.to_excel(f"results/{testcase}_summary_too_many_transformations.xlsx")
total_failure_df.to_excel(f"results/{testcase}_summary_total_failure.xlsx")

