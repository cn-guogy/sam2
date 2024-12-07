import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor


class Seg:
    # 初始化模型
    def __init__(self, checkpoint, cfg):
        torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()
        if torch.cuda.get_device_properties(0).major >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        self.set_model(checkpoint, cfg)
        print("init")

    # 加载模型
    def set_model(self, checkpoint, cfg):
        sam2_model = build_sam2(cfg, checkpoint)
        predictor = SAM2ImagePredictor(sam2_model)
        self.predictor = predictor
        print("set model")

    # 生成分割
    def make_seg(self, dir, image):
        image_path, gt_path, mask_path, prob_path = self.set_path(dir, image)
        img = self.set_image(image_path)
        input_point, input_label = self.set_point(gt_path)
        masks, scores, logits = self.seg(input_point, input_label, return_logits=True)
        self.save_mask(masks, mask_path)
        self.save_probabilities(masks, prob_path)
        print("make seg over")

    # 加载图片
    def set_image(self, image):
        image = Image.open(image)
        image = np.array(image.convert("RGB"))
        self.predictor.set_image(image)
        print("set image")
        return image

    # 加载标签
    def set_point(self, gt_path):
        point_coords = []
        point_labels = []
        image = Image.open(gt_path)
        image = np.array(image.convert("L"))
        white_pixels = np.argwhere(image == 255)
        for i in range(0, len(white_pixels), 600):
            y, x = white_pixels[i]
            point_coords.append([x, y])
            point_labels.append(1)
        point_coords = np.array(point_coords)
        point_labels = np.array(point_labels)
        print("set point")
        return point_coords, point_labels

    # 分割
    def seg(self, input_point, input_label, multimask_output=False, return_logits=True):
        masks, score, logits = self.predictor.predict(
            point_coords=input_point,
            point_labels=input_label,
            multimask_output=multimask_output,
            return_logits=return_logits,
        )
        print("seg")
        return masks, score, logits

    # 保存分割
    def save_mask(self, masks, path, threshold=0, if_filter=True):
        masks_tensor = torch.tensor(masks)
        if if_filter:
            masks_tensor[masks_tensor < threshold] = 0
            masks_tensor[masks_tensor > threshold] = 1
        masks_image = masks_tensor.squeeze().cpu().numpy() * 255
        masks_image = masks_image.astype(np.uint8)
        plt.imsave(path, masks_image, cmap="gray")
        print("save mask")

    # 保存概率
    def save_probabilities(self, masks, path, threshold=0.5, if_filter=True):
        masks_tensor = torch.tensor(masks)
        probabilities = torch.sigmoid(masks_tensor)
        if if_filter:
            probabilities[probabilities < threshold] = 0
        probabilities_image = probabilities.squeeze().cpu().numpy() * 255
        probabilities_image = probabilities_image.astype(np.uint8)
        plt.imsave(path, probabilities_image, cmap="gray")
        print("save probabilities")

    # 设置路径
    def set_path(self, dir, image):
        image_path = dir + "/Imgs/" + image
        gt_path = dir + "/gt/" + image
        save_mask_path = dir + "/PL/" + image
        save_probabilities_path = dir + "/Prob/" + image
        print("set path")
        return image_path, gt_path, save_mask_path, save_probabilities_path
