import os
import re

import pandas as pd

testcase = "no_axsym_15x15_rotations"
# testcase = "20x10"
# testcase = "15x15_rotations"
# testcase = "13x7_letters_indiv_colors"
# testcase = "11x7_letters_indiv_colors"
regex_str = f"{testcase}.*xlsx"
expected_permutation_group_order = 900

sub_directory = "jobarray_12779489"

dataframes = []
for excel_sheet in os.listdir(f"results/{sub_directory}"):
    if re.match(regex_str, excel_sheet) is not None:
        df = pd.read_excel(f"results/{sub_directory}/{excel_sheet}")
        dataframes.append(df)


big_df = pd.concat(dataframes, ignore_index=True)
timeout_or_skip_df = big_df[
    (big_df["permutation_group_order"] == "skipped")
    | (big_df["permutation_group_order"] == "Timeout")
]
big_df.drop(timeout_or_skip_df.index, inplace=True)

big_df.rename(
    columns=lambda col: "local_index" if col == "Unnamed: 0" else col, inplace=True
)

# Remove duplicate rows
big_df.drop_duplicates(inplace=True)

success_df = big_df[
    (big_df["fundamental_generators_contained"].astype(bool) == True)
    & (
        big_df["permutation_group_order"].astype(int)
        == expected_permutation_group_order
    )
]

too_many_df = big_df[
    (big_df["fundamental_generators_contained"].astype(bool) == True)
    & (big_df["permutation_group_order"].astype(int) > expected_permutation_group_order)
]

total_failure_df = pd.concat([big_df, success_df, too_many_df]).drop_duplicates(
    keep=False
)

sorting = [
    "percentage_observation",
    "error_value_limit",
    "kde_bandwidth",
    "trafo_fault_tolerance_ratio",
]

big_df = big_df.sort_values(by=sorting)
success_df = success_df.sort_values(by=sorting)
too_many_df = too_many_df.sort_values(by=sorting)
total_failure_df = total_failure_df.sort_values(by=sorting)
timeout_or_skip_df = timeout_or_skip_df.sort_values(by=sorting)

os.makedirs(f"results/{sub_directory}/summaries", exist_ok=True)
big_df.to_excel(f"results/{sub_directory}/summaries/{testcase}_summary_everything.xlsx")
success_df.to_excel(f"results/{sub_directory}/summaries/{testcase}_summary_success.xlsx")
too_many_df.to_excel(
    f"results/{sub_directory}/summaries/{testcase}_summary_too_many_transformations.xlsx"
)
total_failure_df.to_excel(f"results/{sub_directory}/summaries/{testcase}_summary_total_failure.xlsx")
timeout_or_skip_df.to_excel(
    f"results/{sub_directory}/summaries/{testcase}_summary_timeout_or_skip.xlsx"
)
