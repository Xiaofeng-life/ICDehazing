# -*- coding: utf-8 -*-
import sys
sys.path.append("..")
import torch
import os
import torch.nn as nn
from utils import save
from metric import cal_psnr_ssim
import argparse
from methods.ICDehazing.dataset.get_train_val_loader_ICDehazing import get_train_val_loader_ICDehazing


class InfOptions():
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.parser.add_argument("--dataset", type=str)
        self.parser.add_argument("--img_w", type=int)
        self.parser.add_argument("--img_h", type=int)
        self.parser.add_argument("--val_batch_size", type=int, default=1)
        self.parser.add_argument("--results_dir", type=str)
        self.parser.add_argument("--net", type=str, default="ICDehazing")
        self.parser.add_argument("--pth_path", type=str)
        self.parser.add_argument("--num_workers", type=int, default=0)

        self.parser.add_argument("--if_pair", type=str, default="True")
        self.parser.add_argument("--if_mul_alpha", type=str)

    def parse(self):
        parser = self.parser.parse_args()
        return parser


def inference_one2one(net: nn.Module, dataloader, save_dir,
                      denorm_func_hazy=None, denorm_func_clear=None,
                      denorm_func_pred=None, norm_func_hazy=None,
                      alp=0.5, if_save_all=True):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net.eval()

    num_samples = 0
    total_ssim = 0
    total_psnr = 0
    with torch.no_grad():
        for data in dataloader:
            hazy = data["hazy"].to(device)
            clear = data["gt"].to(device)

            if norm_func_hazy:
                hazy = norm_func_hazy(hazy)

            pred = net(hazy, alp)[0]

            if denorm_func_clear:
                clear = denorm_func_clear(clear)
            if denorm_func_pred:
                pred = denorm_func_pred(pred)
            if denorm_func_hazy:
                hazy = denorm_func_hazy(hazy)

            pred = torch.clamp(pred, min=0, max=1)
            num_samples += 1

            cur_psnr = cal_psnr_ssim.cal_batch_psnr(pred=pred,
                                                       gt=clear)
            cur_ssim = cal_psnr_ssim.cal_batch_ssim(pred=pred,
                                                       gt=clear)
            total_psnr += cur_psnr
            total_ssim += cur_ssim

            print("cur psnr: {}, cur ssim: {}".format(cur_psnr, cur_ssim))

            save.save_image(image_tensor=pred[0],
                            out_name=os.path.join(save_dir, "images", data["name"][0]))
            if if_save_all:
                hazy = torch.clamp(hazy, min=0, max=1)
                clear = torch.clamp(clear, min=0, max=1)
                save.save_image(image_tensor=hazy[0],
                                out_name=os.path.join(save_dir, "hazy", data["name"][0]))
                save.save_image(image_tensor=clear[0],
                                out_name=os.path.join(save_dir, "clear", data["name"][0]))

        psnr = total_psnr / num_samples
        ssim = total_ssim / num_samples
        with open(os.path.join(save_dir, "metrics", "ssim_psnr.txt"), mode='a') as f:
            info = "Final metrics, SSIM: " + str(ssim) + ", PSNR: " + str(psnr) + "\n"
            f.write(info)
            f.close()


if __name__ == "__main__":
    config = InfOptions().parse()

    # alpha = [0.42, 0.44, 0.45, 0.46, 0.47, 0.48, 0.49, 0.50, 0.51, 0.52, 0.53, 0.54, 0.56]
    alpha = [0.56]

    denorm_func_hazy = None
    denorm_func_clear = None
    denorm_func_pred = None
    norm_func_hazy = None

    network = None

    if config.net == "ICDehazing":
        from methods.ICDehazing.ICDehazing.ICDehazing import get_FormerX2Y
        network = get_FormerX2Y().cuda()
        def denorm(x):
            x = (x + 1) / 2
            return x
        def norm(x):
            x = x * 2 - 1
            return x
        denorm_func_pred = denorm
        norm_func_hazy = norm
        denorm_func_hazy = denorm

    else:
        raise ValueError("model {} not supported!".format(config.net))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # define and reload model
    net = network.to(device)
    net.load_state_dict(torch.load(config.pth_path))
    top_results_dir = config.results_dir
    if not os.path.exists(top_results_dir):
        os.mkdir(top_results_dir)
    #
    img_size = [config.img_w, config.img_h]

    _, _, val_loader = get_train_val_loader_ICDehazing(dataset=config.dataset,
                                                       img_h=config.img_h,
                                                       img_w=config.img_w,
                                                       train_batch_size=1,
                                                       val_batch_size=config.val_batch_size,
                                                       num_workers=config.num_workers,
                                                       if_half_crop=False)

    if config.if_mul_alpha == "True":
        for alp in alpha:
            results_dir = os.path.join(top_results_dir, str(alp).split(".")[1])
            if not os.path.exists(results_dir):
                os.mkdir(results_dir)
                os.mkdir(os.path.join(results_dir, "images"))
                os.mkdir(os.path.join(results_dir, "metrics"))

            inference_one2one(net=network,
                              dataloader=val_loader,
                              save_dir=results_dir,
                              denorm_func_hazy=denorm_func_hazy,
                              denorm_func_clear=denorm_func_clear,
                              denorm_func_pred=denorm_func_pred,
                              norm_func_hazy=norm_func_hazy,
                              alp=alp,
                              if_save_all=False)

    else:
        os.mkdir(os.path.join(top_results_dir, "metrics"))
        os.mkdir(os.path.join(top_results_dir, "hazy"))
        os.mkdir(os.path.join(top_results_dir, "clear"))
        os.mkdir(os.path.join(top_results_dir, "images"))
        inference_one2one(net=network,
                          dataloader=val_loader,
                          save_dir=top_results_dir,
                          denorm_func_hazy=denorm_func_hazy,
                          denorm_func_clear=denorm_func_clear,
                          denorm_func_pred=denorm_func_pred,
                          norm_func_hazy=norm_func_hazy,
                          if_save_all=True)
