import os
import shutil

root = "/home/guogy/dataset"
datasets = ["DAVIS","DAVSOD"]

for dataset in datasets:
    for video in os.listdir(os.path.join(root, dataset)):
        del_path = os.path.join(root, dataset, video, "mask")
        if os.path.exists(del_path):
            shutil.rmtree(del_path)
            print(f"Directory {del_path} has been removed.")
        else:
            print(f"Directory {del_path} does not exist.")

print("All done!")