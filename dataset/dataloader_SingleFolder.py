from torch.utils.data import Dataset
import os
from PIL import Image
import torchvision.transforms as T
from torchvision.transforms import functional as FF
import random
from torch.utils.data import DataLoader


def single_augmentation(img, new_size):

    if random.random() > 0.5:
        img = FF.hflip(img)
    if random.random() > 0.5:
        img = FF.vflip(img)

    # if random.random() > 0.5:
    #     i, j, h, w = tfs.RandomCrop.get_params(img, output_size=(new_size, new_size))
    #     img = FF.crop(img, i, j, h, w)

    return img


class SingleImageDataset(Dataset):
    """
    --data_root
      --1.png
      --2.png
    """

    def __init__(self, data_root, img_size, if_aug):
        super().__init__()

        self.dataroot = data_root
        self.files = os.listdir(data_root)
        self.files.sort()
        self.if_aug = if_aug
        self.img_size = img_size
        self.trans = T.Compose([T.Resize(img_size),
                                T.ToTensor(),
                                T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                ])

    def __getitem__(self, index):
        img = Image.open(os.path.join(self.dataroot, self.files[index])).convert("RGB")
        if self.if_aug:
           img = single_augmentation(img, new_size=self.img_size)
        img = self.trans(img)
        return img

    def __len__(self):
        return len(self.files)


def get_single_image_folder(data_root, img_h, img_w, batch_size, num_workers, shuffle, if_aug):
    """
    """
    dataset = SingleImageDataset(data_root=data_root, img_size=[img_h, img_w], if_aug=if_aug)
    dataloader = DataLoader(dataset=dataset, batch_size=batch_size,
                                    pin_memory=True, shuffle=shuffle,
                                    num_workers=num_workers, drop_last=True)
    return dataloader