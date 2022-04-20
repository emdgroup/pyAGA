import ast
import glob
import os
import sys
import time

import matplotlib.pyplot as plt

try:
    jobarray_id = sys.argv[1]
except IndexError as e:
    print(e)
    print("Please pass a job array as an argument")
    sys.exit(1)

jobarray_dirname = f"parameter_study/results/jobarray_{jobarray_id}"
if not os.path.isdir(jobarray_dirname):
    print("Job array folder does not exist")
    sys.exit(1)

fig, ax = plt.subplots()
ax.set_xlabel("percentages")
ax.set_ylabel("fault tolerances")
status_files_todo = []
for status_file in glob.glob(f"{jobarray_dirname}/status_todo*"):
    with open(status_file, "r") as file:
        lines = file.readlines()
        params = ast.literal_eval(lines[2])
        status_files_todo.append(params)
        ax.plot(params[1], params[3], marker="o", color="grey")

time_now = time.time()
status_files_started = []
runtime_labels = {}
for status_file in glob.glob(f"{jobarray_dirname}/status_started*"):
    with open(status_file, "r") as file:
        lines = file.readlines()
        runtime = time_now - float(lines[1])
        runtime = int(runtime)
        params = ast.literal_eval(lines[2])
        status_files_started.append((runtime, params))
        ax.plot(params[1], params[3], marker="o", color="yellow")
        runtime_labels[params] = plt.text(params[1], params[3], str(runtime))

status_files_finished = []
for status_file in glob.glob(f"{jobarray_dirname}/status_finished*"):
    with open(status_file, "r") as file:
        lines = file.readlines()
        params = ast.literal_eval(lines[2])
        status_files_finished.append((params, lines[4]))
        result = lines[4].split(":")[1].strip()
        if result == "exact":
            color = "green"
        elif result == "too many":
            color = "black"
        elif result == "too few":
            color = "red"
        elif result == "exact (Timeout)":
            color = "purple"
        elif result == "too few (Timeout)":
            color = "pink"
        else:
            assert False
        ax.plot(params[1], params[3], marker="o", color=color)
        runtime_labels[params].set_visible(False)

fig.savefig(f"{jobarray_dirname}/current_state_{jobarray_id}.pdf")




