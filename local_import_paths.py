import sys


def import_paths():
    paths = [
    r"C:\Users\M305822\OneDrive - MerckGroup\PycharmProjects\integer_programming_for_transformations",
    r"C:\Users\M290244@eu.merckgroup.com\OneDrive - MerckGroup\Programming\integer_programming_for_transformations",
    r"../integer_programming_for_transformations",
    ]
    for path in paths:
        sys.path.append(path)