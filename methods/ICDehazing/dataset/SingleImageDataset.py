from torch.utils.data import Dataset
import os
from PIL import Image
import torchvision.transforms as T
from torchvision.transforms import functional as FF
import random


def single_augmentation(img, new_size):

    if random.random() > 0.5:
        img = FF.hflip(img)
    if random.random() > 0.5:
        img = FF.vflip(img)

    if random.random() > 0.5 and img.size[0] > new_size[0] and img.size[1] > new_size[1]:
        i, j, h, w = T.RandomCrop.get_params(img, output_size=new_size)
        img = FF.crop(img, i, j, h, w)

    return img

class SingleImageDataset(Dataset):
    """
    --folder
      --1.png
      --2.png
    """

    def __init__(self, dataroot, img_size, if_train, if_half_crop=False):
        super().__init__()

        self.dataroot = dataroot
        self.files = os.listdir(dataroot)
        self.files.sort()
        self.if_train = if_train
        self.img_size = img_size
        self.trans = T.Compose([T.Resize(img_size),
                                T.ToTensor(),
                                T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                ])

        self.crop_size = img_size
        self.if_half_crop = if_half_crop

    def __getitem__(self, index):
        img = Image.open(os.path.join(self.dataroot, self.files[index])).convert("RGB")

        if self.if_half_crop:
            self.crop_size = (int(img.size[0] / 2), int(img.size[1] / 2))

        if self.if_train:
           img = single_augmentation(img, self.crop_size)
        img = self.trans(img)
        return img

    def __len__(self):
        return len(self.files)


if __name__ == "__main__":
    img = Image.open("../demo_img/000007.jpg")
    single_augmentation(img, (20000, 100))