import os
import re

import pandas as pd

# testcase = "no_axsym_15x15_rotations"
# testcase = "20x10"
# testcase = "15x15_rotations"
from matplotlib import pyplot as plt

testcase = "13x7_letters_indiv_colors"
# testcase = "11x7_letters_indiv_colors"
regex_str = f"{testcase}.*xlsx"
expected_permutation_group_order = 1092
plot = False
sub_directory = "jobarray_13x7_combined"

dataframes = []
for excel_sheet in os.listdir(f"results/{sub_directory}"):
    if re.match(regex_str, excel_sheet) is not None:
        df = pd.read_excel(f"results/{sub_directory}/{excel_sheet}")
        dataframes.append(df)


big_df = pd.concat(dataframes, ignore_index=True)
big_df.rename(
    columns=lambda col: "local_index" if col == "Unnamed: 0" else col, inplace=True
)

sorting = [
    "percentage_observation",
    "error_value_limit",
    "kde_bandwidth",
    "trafo_fault_tolerance_ratio",
]

big_df = big_df.sort_values(by=sorting)

os.makedirs(f"results/{sub_directory}/summaries", exist_ok=True)
big_df.to_excel(f"results/{sub_directory}/summaries/{testcase}_summary_everything.xlsx")

if plot:
    status_files_started = []
    runtime_labels = {}

    status_files_finished = []
    fig, ax = plt.subplots()
    correctness_colors = {
        "exact": "green",
        "too many": "black",
        "too few": "red",
        "exact (Timeout)": "purple",
        "too few (Timeout)": "pink",
    }
    for correctness, color in correctness_colors.items():
        # plot exact points
        df = big_df[big_df["is_group_order_correct"] == correctness]
        ax.plot(
            df["percentage_observation"],
            df["trafo_fault_tolerance_ratio"],
            color=color,
            linestyle="None",
            marker="o",
        )
    plt.show()
    fig.savefig("13x7_evl=0_005.pdf")
