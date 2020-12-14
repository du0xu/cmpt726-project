"""
The dataset class.
"""
import os

import numpy as np
import torch
import torch.utils.data as data
from skimage import io, transform

WIDTH = 200
HEIGHT = 400


def resize(img, kps):
    h_old, w_old, _ = img.shape
    img_new = transform.resize(img, (HEIGHT, WIDTH)).astype(np.float32)
    kps_new = (kps * [WIDTH / w_old, HEIGHT / h_old]).astype(np.float32)
    return img_new, kps_new


def numpy_to_tensor(img, label):
    return torch.from_numpy(img.transpose(2, 0, 1)), torch.from_numpy(label)


class ImageAndKeypointDataset(data.Dataset):
    def __init__(self, data_path):
        super().__init__()

        # Read keypoint coordinates
        self.data_path = data_path
        all_files = os.listdir(self.data_path)
        self.dataset = []
        dat_files = filter(lambda x: x.endswith(".dat"), all_files)
        for dat_file in dat_files:
            with open(os.path.join(data_path, dat_file), "r") as dat:
                dat_num = dat_file.split(".")[0]
                # Each line in the DAT file contains 42 floating point numbers,
                # representing (x, y) coordinates of 21 keypoints
                lines = dat.readlines()
                for line_num, line in enumerate(lines):
                    keypoints = np.array(list(map(float, line.split()))).reshape(-1, 2)
                    datapoint = {
                        "img_file": f"{dat_num}_{line_num}.jpg",
                        "keypoints": keypoints
                    }
                    self.dataset.append(datapoint)

    def __getitem__(self, index):
        datapoint = self.dataset[index]
        img = io.imread(os.path.join(self.data_path, datapoint["img_file"]))
        kps = datapoint["keypoints"]
        img_resized, kps_resized = resize(img, kps)
        label = kps_resized.flatten()
        return numpy_to_tensor(img_resized, label)

    def __len__(self):
        return len(self.dataset)
