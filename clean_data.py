#!/usr/bin/python3

import sys
import os
osl = os.listdir
ospj = os.path.join


def clean_dir(dir, filetype=""):
    """
    :param dir: directory to clean
    :param filetype: filetype to clean from that directory. if empty, cleans
    all files EXCEPT for .gitignore.
    :returns: None. this is a data/directory cleaning utility function that
    just deletes all filetype-type files from the given dir.
    """
    for file in osl(dir):
        if filetype != "" and file.endswith(filetype) or filetype == "" and file != ".gitignore":
            os.remove(ospj(dir, file))


if __name__ == "__main__":
    # ensure data deletion should happen
    ans = input("are you sure you want to delete all data? ([Y]/n): ")
    if "n" in ans.lower(): sys.exit()

    # delete data
    synth_data_folders = [
        "data/synthetic/test",
        "data/synthetic/train",
        "data/encoded/test",
        "data/encoded/train",
    ]
    for folder in synth_data_folders:
        clean_dir(folder)
