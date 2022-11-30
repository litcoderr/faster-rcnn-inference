import os
import cv2
import torch
import numpy as np
from pathlib import Path
from PIL import Image
from torchvision.transforms import ToTensor
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights


class Label():
    def __init__(self):
        self.fpath = Path("/faster-rcnn-inference/labels.txt")
        self.data = self.get_data()

    def __call__(self, idx):
        return self.data[idx]

    def get_data(self):
        data = {}
        with open(self.fpath, "r") as f:
            for line in f:
                idx, name = tuple(line.split(": "))
                data[int(idx)] = name.split("\n")[0]
        return data


class Extractor():
    def __init__(self, imgs_path, save_path):
        self.score_threshold = 0.3
        self.model = FRCNNModel.get_instance()
        self.label = Label()

        self.save_path = Path(save_path)

        self.img_names = os.listdir(imgs_path)
        self.img_paths = [Path(imgs_path) / name for name in self.img_names]
        self.img_tensors = [ToTensor()(Image.open(img_path)).to("cuda") for img_path in self.img_paths]

    def extract(self):
        preds = self.model(self.img_tensors)

        np_images = self.tensors_to_npys()
        for pred, np_image, img_name in zip(preds, np_images, self.img_names):
            cv_img = np.transpose(np_image, (1, 2, 0)) * 255
            cv_img = cv2.cvtColor(cv_img, cv2.COLOR_RGB2BGR)

            boxes = pred['boxes'].cpu().detach().numpy()  # [n_boxes, 4 (x1, y1, x2, y2)]
            labels = pred['labels'].cpu().detach().numpy()  # [n_boxes]
            scores = pred['scores'].cpu().detach().numpy()  # [n_boxes]

            for box_i in range(boxes.shape[0]):
                x1, y1, x2, y2 = (int(coord) for coord in boxes[box_i])
                label = self.label(labels[box_i])
                score = scores[box_i]

                if score > self.score_threshold:
                    color = (255, 0, 0)
                    cv_img = cv2.rectangle(cv_img, (x1, y1), (x2, y2), color, 3)
                    cv_img = cv2.putText(cv_img, f'{label}', (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

            # write image
            cv2.imwrite(str(self.save_path / img_name), cv_img)

    def tensors_to_npys(self):
        npy_list = []
        for tensor in self.img_tensors:
            npy = tensor.to("cpu").detach().numpy()
            npy_list.append(npy)
        return npy_list
class FRCNNModel:
    model = None

    def __init__(self):
        pass

    @classmethod
    def get_instance(cls):
        if not cls.model:
            model = fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT)
            model = model.to("cuda")
            model.eval()
            cls.model = model
        return cls.model
