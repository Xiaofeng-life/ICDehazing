import sys
sys.path.append("..")
from methods.ICDehazing.ICDehazing.DCP import dcp_dehazing
import argparse
import os
import cv2
import numpy as np


class Options:
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.parser.add_argument("--dataset", type=str)
        self.parser.add_argument("--root_4KDehazing_hazy", type=str, default="../../dataset/RESIDE/4KDehazing/hazy/")
        self.parser.add_argument("--root_4KDehazing_DCP", type=str, default="../../dataset/RESIDE/4KDehazing/DCP/")
    def parse(self):
        par = self.parser.parse_args()
        return par


if __name__ == "__main__":
    parser = Options()
    config = parser.parse()

    root_4KDehazing_hazy = config.root_4KDehazing_hazy
    root_4KDehazing_DCP = config.root_4KDehazing_DCP

    for file in os.listdir(root_4KDehazing_hazy):
        hazy_img = cv2.imread(os.path.join(root_4KDehazing_hazy, file)).astype("float32") / 255
        pred = dcp_dehazing(hazy_img) * 255
        pred[pred > 255] = 255
        pred[pred < 0] = 0
        cv2.imwrite(os.path.join(root_4KDehazing_DCP, file), pred.astype(np.uint8))