import data.utils as utils
import sys

# ensure data deletion should happen
ans = input("are you sure you want to delete all data? ([Y]/n): ")
if "n" in ans.lower(): sys.exit()

# delete data
kg_path = "./knowledge_graph"
synth_data_folders = [
    "data/synthetic/test",
    "data/synthetic/train",
    "data/encoded/test",
    "data/encoded/train",
    f"{kg_path}/data/input/",
    f"{kg_path}/data/output/kg/",
    f"{kg_path}/data/output/ner",
    f"{kg_path}/data/result/"
]
for folder in synth_data_folders:
    utils.clean_dir(folder)