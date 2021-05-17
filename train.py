import torch
from apex import amp
from tqdm import tqdm

import config
import dataloaders.dataloaders as dataloaders
import models.losses as losses
import models.models as models
import utils.utils as utils
from utils.fid_scores import FIDCalculator


def run():
    torch.multiprocessing.freeze_support()
    print('loop')
    torch.multiprocessing.freeze_support()

    # --- read options ---#
    opt = config.read_arguments(train=True)

    # --- create utils ---#
    timer = utils.Timer(opt)
    visualizer_losses = utils.LossesSaver(opt)
    losses_computer = losses.LossesComputer(opt)
    dataloader, dataloader_val = dataloaders.get_dataloaders(opt)
    im_saver = utils.ImageSaver(opt)
    fid_computer = FIDCalculator(opt, dataloader_val)

    # --- create models ---#
    model = models.OASIS(opt)
    if opt.gpu_ids != "-1":
        gpus = list(map(int, opt.gpu_ids.split(",")))
        model.to(gpus[0])

    # --- create optimizers ---#
    G_params = list(model.netG.parameters())
    D_params = list(model.netD.parameters())

    optimizerG = torch.optim.Adam(G_params, lr=opt.lr_g, betas=(opt.beta1, opt.beta2))
    optimizerD = torch.optim.Adam(D_params, lr=opt.lr_d, betas=(opt.beta1, opt.beta2))

    [model], [optimizerD, optimizerG] = amp.initialize(
        [model], [optimizerD, optimizerG], loss_scale=1,
        opt_level=opt.opt_level, num_losses=2)
    optimizerD._lazy_init_maybe_master_weights()
    optimizerG._lazy_init_maybe_master_weights()

    model = models.put_on_multi_gpus(model, opt)

    # --- the training loop ---#
    already_started = False
    start_epoch, start_iter = utils.get_start_iters(opt.loaded_latest_iter, len(dataloader))
    for epoch in range(start_epoch, opt.num_epochs):
        t = tqdm(dataloader, total=len(dataloader))
        for i, data_i in enumerate(t):
            if not already_started and i < start_iter:
                continue
            already_started = True
            cur_iter = epoch * len(dataloader) + i
            data = models.preprocess_input(opt, data_i)

            # --- generator update ---#
            model.zero_grad()
            loss_G, losses_G_list = model(**data, mode="losses_G", losses_computer=losses_computer)
            loss_G, losses_G_list = loss_G.mean(), [loss.mean() if loss is not None else None for loss in losses_G_list]
            with amp.scale_loss(loss_G, optimizerD, loss_id=0, model=model) as loss_G_scaled:
                loss_G_scaled.backward()
            # loss_G.backward()
            optimizerG.step()

            # --- discriminator update ---#
            model.zero_grad()
            loss_D, losses_D_list = model(**data, mode="losses_D", losses_computer=losses_computer)
            loss_D, losses_D_list = loss_D.mean(), [loss.mean() if loss is not None else None for loss in losses_D_list]
            with amp.scale_loss(loss_D, optimizerD, loss_id=1, model=model) as loss_D_scaled:
                loss_D_scaled.backward()
            # loss_D.backward()
            optimizerD.step()
            t.set_description_str(f'G loss: {loss_G:.4f}, D loss: {loss_D:.4f}', refresh=False)

            # --- stats update ---#
            if not opt.no_EMA:
                model.update_ema(cur_iter, dataloader, opt)
            if cur_iter > 0:
                if cur_iter % opt.freq_print == 0:
                    im_saver.visualize_batch(model, data['image'], data['label'], cur_iter)
                    timer(epoch, cur_iter)
                if cur_iter % opt.freq_save_ckpt == 0:
                    model.save_networks(cur_iter)
                if cur_iter % opt.freq_save_latest == 0:
                    model.save_networks(cur_iter, latest=True)
                if cur_iter % opt.freq_fid == 0:
                    is_best = fid_computer.update(model, cur_iter)
                    if is_best:
                        model.save_networks(cur_iter, best=True)
            visualizer_losses(cur_iter, losses_G_list + losses_D_list)

    # --- after training ---#
    model.update_ema(cur_iter, dataloader, opt, force_run_stats=True)
    model.save_networks(cur_iter)
    model.save_networks(cur_iter, latest=True)
    is_best = fid_computer.update(model, cur_iter)
    if is_best:
        model.save_networks(cur_iter, best=True)

    print("The training has successfully finished")


if __name__ == '__main__':
    run()
