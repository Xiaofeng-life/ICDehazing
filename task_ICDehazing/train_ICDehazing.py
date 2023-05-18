import sys
sys.path.append("..")

import itertools
import torch.optim as optim
from methods.ICDehazing.ICDehazing.options_ICDehazing import Options
from methods.ICDehazing.ICDehazing.discriminator import set_requires_grad
from methods.ICDehazing.dataset.get_train_val_loader_ICDehazing import get_train_val_loader_ICDehazing
import torch.nn as nn
import torch
from utils.writer import LossWriter, save_config_as_json
from utils.save import save_image
import os
from metric import cal_psnr_ssim
from methods.ICDehazing.loss.PerceptualLoss import PerceptualLoss
from methods.ICDehazing.loss.gp_loss import gradient_penalty
from utils.make_dir import make_train_dir


def train(prior_per_weight, data_iter_A, data_iter_B):
    iteration = 0
    for epo in range(1, config.total_epoches + 1):
        if prior_per_func:
            prior_per_weight = prior_per_weight * prior_decay

        generator_x2y.train()
        generator_y2x.train()
        for _ in range(len(dataloader_train_A)):
            try:
                data_X, data_DCP_X = next(data_iter_A)
            except StopIteration:
                data_iter_A = iter(dataloader_train_A)
                data_X, data_DCP_X = next(data_iter_A)
            try:
                data_Y = next(data_iter_B)
            except StopIteration:
                data_iter_B = iter(dataloader_train_B)
                data_Y = next(data_iter_B)

            # prepare data
            image_x = data_X.to(device)
            image_dcp_x = data_DCP_X.to(device)
            image_y = data_Y.to(device)

            # save loss
            loss_items = {}

            # #################################################
            # train generator
            optimizer_G.zero_grad()

            #
            generated_x2y = generator_x2y(image_x)
            d_out_fake_x2y = discriminator_y(generated_x2y)

            # dcp loss
            if prior_per_func:
                g_loss_dcp_x2y = prior_per_weight * prior_per_func(generated_x2y, image_dcp_x)

            g_loss_x2y = gan_loss_func(d_out_fake_x2y, real_label)

            # deleted if necessary
            identity_x = generator_y2x(image_x)
            identity_x_loss = identity_loss_func(identity_x, image_x) * config.idt_w

            #
            generated_y2x = generator_y2x(image_y)
            d_out_fake_y2x = discriminator_x(generated_y2x)
            g_loss_y2x = gan_loss_func(d_out_fake_y2x, real_label)

            # deleted if necessary
            identity_y = generator_x2y(image_y)
            identity_y_loss = identity_loss_func(identity_y, image_y) * config.idt_w

            # ##########################################################
            # cycle loss
            cycle_x2y2x = generator_y2x(generated_x2y)
            g_loss_cycle_x2y2x = cycle_loss_func(cycle_x2y2x, image_x) * config.cyc_w

            cycle_y2x2y = generator_x2y(generated_y2x)
            g_loss_cycle_y2x2y = cycle_loss_func(cycle_y2x2y, image_y) * config.cyc_w

            # generator total loss
            g_loss = g_loss_y2x + g_loss_x2y + identity_y_loss + identity_x_loss + g_loss_cycle_x2y2x + g_loss_cycle_y2x2y

            if prior_per_func:
                g_loss += g_loss_dcp_x2y
                loss_writer.add("DCPLoss", g_loss_dcp_x2y.item(), iteration)

            loss_items["G/cycle_x2y2x"] = g_loss_cycle_x2y2x.item()
            loss_items["G/cycle_y2x2y"] = g_loss_cycle_y2x2y.item()
            loss_items["G/dis_x2y"] = g_loss_x2y.item()
            loss_items["G/dis_y2x"] = g_loss_y2x.item()
            loss_items["G/idt_x"] = identity_x_loss.item()
            loss_items["G/idt_y"] = identity_y_loss.item()

            if prior_per_func:
                loss_items["G/dcp"] = g_loss_dcp_x2y.item()

            # update the parameter of generator
            set_requires_grad([discriminator_x, discriminator_y], False)
            g_loss.backward()
            optimizer_G.step()

            # #################################################
            set_requires_grad([discriminator_x, discriminator_y], True)
            optimizer_D.zero_grad()

            # part two: train discriminator_x
            d_out_real_x = discriminator_x(image_x)
            d_real_loss_x = gan_loss_func(d_out_real_x, real_label)

            #
            d_out_fake_y2x_ = discriminator_x(generated_y2x.detach())
            d_fake_loss_y2x_ = gan_loss_func(d_out_fake_y2x_, fake_label)

            #
            d_loss_gp_1 = gradient_penalty(discriminator_x,
                                           x_real=image_x, x_fake=generated_y2x)

            # discriminator_x total loss
            d_loss_x = (d_real_loss_x + d_fake_loss_y2x_) * 0.5 + d_loss_gp_1
            d_loss_x.backward()

            loss_items["DX/real_x"] = d_real_loss_x.item()
            loss_items["DX/fake_y2x"] = d_fake_loss_y2x_.item()
            loss_items["DX/gp"] = d_loss_gp_1.item()

            # ##############################################
            # part three: train discriminator _y
            #
            d_out_real_y = discriminator_y(image_y)
            d_real_loss_y = gan_loss_func(d_out_real_y, real_label)

            #
            d_out_fake_x2y_ = discriminator_y(generated_x2y.detach())
            d_fake_loss_x2y_ = gan_loss_func(d_out_fake_x2y_, fake_label)

            #
            d_loss_gp_2 = gradient_penalty(discriminator_y,
                                           x_real=image_y, x_fake=generated_x2y)

            # discriminator_y total loss
            d_loss_y = (d_real_loss_y + d_fake_loss_x2y_) * 0.5 + d_loss_gp_2
            d_loss_y.backward()

            loss_items["DY/real_y"] = d_real_loss_y.item()
            loss_items["DY/fake_x2y"] = d_fake_loss_x2y_.item()

            # update parameters of discriminators
            optimizer_D.step()

            loss_writer.add("G loss", g_loss.item(), iteration)
            loss_writer.add("D loss X", d_loss_x.item(), iteration)
            loss_writer.add("D loss Y", d_loss_y.item(), iteration)

            if iteration % 10 == 0:
                log = "Eepch [{}], Iteration [{}]".format(epo, iteration)
                for tag, value in loss_items.items():
                    log += ", {}: {:.8f}".format(tag, value)
                log += "\n \n"
                print(log)

            # add
            iteration += 1

        # if iteration % len(dataloader_train_B) == 0:
        #     eval(iteration=iteration)


# def eval(iteration):
#         # eval mode, eval_batch_size is set to 1.
#         generator_x2y.eval()
#         generator_y2x.eval()
#         with torch.no_grad():
#             num_samples = 0
#             total_ssim = 0
#             total_psnr = 0
#
#             for data in val_loader:
#                 image_x = (data["hazy"] * 2 - 1).to(device)
#                 image_y = (data["gt"] * 2 - 1).to(device)
#
#                 # first
#                 image_x2y, _, _, _ = generator_x2y(image_x)
#                 image_x2y2x = generator_y2x(image_x2y)
#                 result = torch.cat((image_x, image_x2y, image_y, image_x2y2x), dim=3)
#
#                 # second
#                 image_y2x = generator_y2x(image_y)
#                 image_y2x2y, _, _, _ = generator_x2y(image_y2x)
#                 result_another = torch.cat((image_y, image_y2x, image_x, image_y2x2y), dim=3)
#
#                 save_image(result[0].squeeze(),
#                            os.path.join(res_dir, "cat_images", data["name"][0].split(".")[0] + "_A2B2A.png"),
#                            last_acti="tanh")
#
#                 save_image(result_another[0].squeeze(),
#                            os.path.join(res_dir, "cat_images", data["name"][0].split(".")[0] + "_B2A2B.png"),
#                            last_acti="tanh")
#
#                 num_samples += 1
#
#                 image_x2y = (image_x2y + 1) / 2
#                 image_y = (image_y + 1) / 2
#                 total_psnr += cal_psnr_ssim.cal_batch_psnr(pred=image_x2y, gt=image_y)
#                 total_ssim += cal_psnr_ssim.cal_batch_ssim(pred=image_x2y, gt=image_y)
#
#             psnr = total_psnr / num_samples
#             ssim = total_ssim / num_samples
#             with open(os.path.join(res_dir, "metrics/metric.txt"), mode='a') as f:
#                 info = str(iteration) + " " + str(ssim) + " " + str(psnr) + "\n"
#                 f.write(info)
#                 f.close()
#
#             print("Log: ||iterations: {}||, ||PSNR {:.4}||, ||SSIM {:.4}||".format(iteration, psnr, ssim))
#
#         generator_x2y.train()
#         generator_y2x.train()
#
#         if config.save_w == "True":
#             torch.save(generator_x2y.state_dict(), os.path.join(res_dir, "models", str(iteration) + "_x2y.pth"))
#             torch.save(generator_y2x.state_dict(), os.path.join(res_dir, "models", str(iteration) + "_y2x.pth"))


if __name__ == "__main__":
    config = Options().parse()

    res_dir = config.results_dir
    device = torch.device('cuda')
    make_train_dir(res_dir=res_dir)
    save_config_as_json(save_path=os.path.join(res_dir, "configs", "config.txt"), config=config)
    loss_writer = LossWriter(os.path.join(res_dir, "losses"))

    # half crop for high-resolution images
    if_half_crop = False
    if config.dataset == "4KDehazing":
        if_half_crop = True

    img_size = [config.img_w, config.img_h]
    dataloader_train_A, dataloader_train_B, val_loader = get_train_val_loader_ICDehazing(dataset=config.dataset,
                                                                                         img_h=config.img_h,
                                                                                         img_w=config.img_w,
                                                                                         train_batch_size=config.train_batch_size,
                                                                                         val_batch_size=config.val_batch_size,
                                                                                         num_workers=config.num_workers,
                                                                                         if_half_crop=if_half_crop)

    # which can also be created by label smoothing
    real_label = torch.ones(size=(config.train_batch_size, 1, 16, 16), requires_grad=False).to(device)
    fake_label = torch.zeros(size=(config.train_batch_size, 1, 16, 16), requires_grad=False).to(device)

    # define the networks
    generator_x2y = None
    generator_y2x = None
    discriminator_x = None
    discriminator_y = None

    if config.model == "ICDehazing":
        from methods.ICDehazing.ICDehazing.ICDehazing import get_FormerX2Y, get_FormerY2X
        from methods.ICDehazing.ICDehazing.discriminator import Discriminator
        generator_x2y = get_FormerX2Y().cuda()
        generator_y2x = get_FormerY2X().cuda()
        discriminator_x = Discriminator().cuda()
        discriminator_y = Discriminator().cuda()

    # 4: define the optimizers
    optimizer_G = optim.Adam(itertools.chain(generator_x2y.parameters(), generator_y2x.parameters()),
                             lr=config.g_lr, betas=(config.beta1, config.beta2))
    optimizer_D = optim.Adam(itertools.chain(discriminator_x.parameters(), discriminator_y.parameters()),
                             lr=config.d_lr, betas=(config.beta1, config.beta2))

    # cycle loss config
    cycle_loss_func = None
    if config.rec_loss == "L1":
        cycle_loss_func = nn.L1Loss()
    elif config.rec_loss == "L2":
        cycle_loss_func = nn.MSELoss()
    elif config.rec_loss == "Per":
        cycle_loss_func = PerceptualLoss(pre_train=True, model_path="vgg19-dcbb9e9d.pth").cuda()

    prior_per_func = None
    pre_train = True
    prior_per_weight = 0
    prior_decay = 0
    cwd = os.getcwd()
    f_cwd = os.path.abspath(os.path.dirname(cwd))
    if config.prior_per == "True":
        prior_per_func = PerceptualLoss(pre_train=pre_train, model_path=os.path.join(f_cwd, "pretrained_models/vgg19-dcbb9e9d.pth")).cuda()
        # prior_per_func = nn.MSELoss()
        prior_per_weight = config.prior_per_weight
        prior_decay = config.prior_decay

    identity_loss_func = nn.MSELoss()
    gan_loss_func = nn.MSELoss()
    data_iter_A = iter(dataloader_train_A)
    data_iter_B = iter(dataloader_train_B)

    train(prior_per_weight=prior_per_weight, data_iter_A=data_iter_A, data_iter_B=data_iter_B)