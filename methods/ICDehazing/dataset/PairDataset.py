from PIL import Image
import os
import numpy as np
import torch
import torch.utils.data as data
import torchvision.transforms as tt


class PairDataset(data.Dataset):
    """
    """

    def __init__(self, data_root, hazy_folder="hazy", gt_folder="GT", trans_hazy=None, trans_gt=None):

        self.data_root = data_root
        self.hazy_folder = hazy_folder
        self.gt_folder = gt_folder
        self.hazy_list = os.listdir(os.path.join(data_root, hazy_folder))

        if trans_hazy:
            self.trans_hazy = trans_hazy
        else:
            self.trans_hazy = tt.Compose([tt.Resize((self.img_size, self.img_size)),
                                          tt.ToTensor()])

        if trans_gt:
            self.trans_gt = trans_gt
        else:
            self.trans_gt = tt.Compose([tt.Resize((self.img_size, self.img_size)),
                                        tt.ToTensor()])

    def __getitem__(self, index):
        data_hazy_path = os.path.join(self.data_root,
                                      self.hazy_folder,
                                      self.hazy_list[index])
        data_gt_path = os.path.join(self.data_root,
                                    self.gt_folder,
                                    self.hazy_list[index].replace("hazy", "GT"))

        data_gt = Image.open(data_gt_path)
        data_hazy = Image.open(data_hazy_path)

        data_gt = data_gt.resize((256, 256), Image.ANTIALIAS)
        data_hazy = data_hazy.resize((256, 256), Image.ANTIALIAS)

        data_gt = (np.asarray(data_gt) / 255.0)
        data_hazy = (np.asarray(data_hazy) / 255.0)

        data_gt = torch.from_numpy(data_gt).float()
        data_hazy = torch.from_numpy(data_hazy).float()

        tar_data = {"hazy": data_hazy.permute(2, 0, 1), "gt": data_gt.permute(2, 0, 1)}
        return tar_data

    def __len__(self):
        return len(self.hazy_list)
