import os.path
import sys

sys.path.append("..")
from torch.utils.data import DataLoader
from .DCPDataset import DCPDataset
from dataset.dataloader_SingleFolder import get_single_image_folder
from dataset.dataloader_ImageDehazing import get_train_val_loader
from dataset_path_config import get_path_dict_ImageDehazing


def get_train_val_loader_ICDehazing(dataset, img_h, img_w, train_batch_size, val_batch_size, num_workers,
                                    if_half_crop, train_drop_last=True):
    path_dict = get_path_dict_ImageDehazing()

    img_size = [img_h, img_w]
    dataset_A = DCPDataset(os.path.join(path_dict[dataset]["train"], "hazy/"), img_size, True, if_half_crop=if_half_crop)
    dataloader_train_A = DataLoader(dataset=dataset_A, batch_size=train_batch_size,
                                    pin_memory=True, shuffle=True,
                                    num_workers=num_workers, drop_last=train_drop_last)

    dataloader_train_B = get_single_image_folder(data_root=os.path.join(path_dict[dataset]["train"], "clear/"),
                                                 img_h=img_h, img_w=img_w,
                                                 batch_size=train_batch_size, num_workers=num_workers,
                                                 shuffle=True, if_aug=True)

    _, val_loader = get_train_val_loader(dataset=dataset, img_h=img_h, img_w=img_w,
                                         train_batch_size=val_batch_size,
                                         num_workers=num_workers, if_flip=False, if_crop=False, crop_h=0, crop_w=0)

    return dataloader_train_A, dataloader_train_B, val_loader
