import os
import seg_image as seg

root = "/home/guogy/dataset/"
datasets = ["DAVIS","DAVSOD"]

PL = "PL"
Prob = "Prob"
Img = "Imgs"

# 模型
checkpoint = "checkpoints/sam2.1_hiera_large.pt"
cfg = "sam2.1_hiera_l.yaml"
Seg = seg.Seg(checkpoint, cfg)

for dataset in datasets:
    os.chdir(root + dataset)
    for dir in os.listdir():
        os.chdir(root + dataset + "/" + dir)
        if not os.path.exists(PL):
            os.makedirs(PL)
            print("make dir", PL)
        if not os.path.exists(Prob):
            os.makedirs(Prob)
            print("make dir", Prob)
        for image in os.listdir(Img):
            print("find", image)
            Seg.make_seg(root + dataset +"/"+ dir, image)
            print("seg", root + dataset +"/"+ dir, image)
