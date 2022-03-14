import pandas as pd

import matrices_error_values

from mipsym.mip import Norm

if __name__ == "__main__":
    percentages = ["100", "99.9", "99.0", "98.0", "95.0", "90.0", "85.0", "80.0", "75.0", "70.0", "65.0", "60.0", "55.0",
                   "50.0", "40.0", "30.0", "20.0", "10.0", "5.0"]
    testcases = ["two_letter_words_20x10",
                 "two_letter_words_15x15_rotations",
                 "two_letter_words_no_axsym_15x15_rotations",
                 "two_letter_words_no_axsym_13x7_letters_indiv_colors"]
    long_short_name = {
        "20x10": "two_letter_words_20x10",
         "15x15_rotations": "two_letter_words_15x15_rotations",
         "no_axsym_15x15_rotations": "two_letter_words_no_axsym_15x15_rotations",
         "13x7_letters_indiv_colors": "two_letter_words_no_axsym_13x7_letters_indiv_colors",
    }
    df_20x10 = pd.read_excel("data/20x10_summary_everything.xlsx")
    df_15x15 = pd.read_excel("data/15x15_rotations_summary_everything.xlsx")
    df_no_axsym_15x15 = pd.read_excel("data/no_axsym_15x15_rotations_summary_everything.xlsx")
    dataframes = [df_20x10, df_15x15, df_no_axsym_15x15]
    error_values = matrices_error_values.main(testcases, percentages)

    for index, testcase in enumerate(testcases[:3]):
        df = dataframes[index]
        error_val = error_values[testcase]
        for norm in Norm:
            norm_values = dict(error_val[norm])
            df[norm] = df["percentage_observation"].apply(lambda x: norm_values[str(x)] if x != 100.0 else norm_values["100"])
        df.to_excel(f"data/{list(long_short_name.keys())[index]}_summary_everything_with_Norm.xlsx", engine="xlsxwriter")

