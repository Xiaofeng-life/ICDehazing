import os
import torchvision.transforms as tt
import torch.utils.data as data
import torchvision.transforms as tfs
from torchvision.transforms import functional as FF
from PIL import Image
import random


def pair_augmentation(img, target, new_size):

    if random.random() > 0.5:
        img = FF.hflip(img)
        target = FF.hflip(target)
    if random.random() > 0.5:
        img = FF.vflip(img)
        target = FF.vflip(target)

    if random.random() > 0.5 and img.size[0] > new_size[0] and img.size[1] > new_size[1]:
        i, j, h, w = tfs.RandomCrop.get_params(img, output_size=new_size)
        img = FF.crop(img, i, j, h, w)
        target = FF.crop(target, i, j, h, w)

    return img, target



class DCPDataset(data.Dataset):
    def __init__(self, path, img_size, if_train, trans_hazy=None, trans_gt=None, if_half_crop=False):
        super(DCPDataset, self).__init__()
        self.haze_imgs_dir = os.listdir(path)
        self.haze_imgs = [os.path.join(path, img) for img in self.haze_imgs_dir]

        self.img_size = img_size
        self.crop_size = img_size
        self.if_half_crop = if_half_crop
        self.if_train = if_train
        self.trans_hazy = None
        self.trans_gt = None
        if trans_hazy:
            self.trans_hazy = trans_hazy
        else:
            self.trans_hazy = tt.Compose([tt.Resize(self.img_size),
                                     tt.ToTensor(),
                                          tt.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        if trans_gt:
            self.trans_gt = trans_gt
        else:
            self.trans_gt = tt.Compose([tt.Resize(self.img_size),
                                        tt.ToTensor(),
                                        tt.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    def __getitem__(self, index):
        data_hazy = Image.open(self.haze_imgs[index]).convert('RGB')
        img = self.haze_imgs[index]

        clear_name = img.replace("hazy", "DCP", 1)
        data_gt = Image.open(clear_name).convert('RGB')
        data_gt = tt.CenterCrop(data_hazy.size[::-1])(data_gt)

        if self.if_half_crop:
            self.crop_size = (int(data_hazy.size[0] / 2), int(data_hazy.size[1] / 2))

        if self.if_train:
            data_hazy, data_gt = pair_augmentation(data_hazy, data_gt, self.crop_size)
        data_hazy = self.trans_hazy(data_hazy)
        data_gt = self.trans_gt(data_gt)
        # tar_data = {"hazy": data_hazy, "gt": data_gt,
        #             "name": img.split("/")[-1],
        #             "hazy_path": self.haze_imgs[index]}

        return data_hazy, data_gt

    def __len__(self):
        return len(self.haze_imgs)