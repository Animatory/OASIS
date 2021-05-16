import torch

import config
import dataloaders.dataloaders as dataloaders
import models.losses as losses
import models.models as models
import utils.utils as utils
from tqdm import tqdm
from utils.fid_scores import FIDCalculator

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
model = models.put_on_multi_gpus(model, opt)

# --- create optimizers ---#
G_params = list(model.module.netG.parameters())
D_params = list(model.module.netD.parameters())
optimizerG = torch.optim.Adam(G_params, lr=opt.lr_g, betas=(opt.beta1, opt.beta2))
optimizerD = torch.optim.Adam(D_params, lr=opt.lr_d, betas=(opt.beta1, opt.beta2))

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

        #     --- generator update ---#
        model.zero_grad()
        loss_G, losses_G_list = model(**data, mode="losses_G", losses_computer=losses_computer)
        loss_G, losses_G_list = loss_G.mean(), [loss.mean() if loss is not None else None for loss in losses_G_list]
        loss_G.backward()
        optimizerG.step()

        # --- discriminator update ---#
        model.zero_grad()
        loss_D, losses_D_list = model(**data, mode="losses_D", losses_computer=losses_computer)
        loss_D, losses_D_list = loss_D.mean(), [loss.mean() if loss is not None else None for loss in losses_D_list]
        loss_D.backward()
        optimizerD.step()
        t.set_description_str(f'G loss: {loss_G:.4f}, D loss: {loss_D:.4f}', refresh=False)

        # --- stats update ---#
        if not opt.no_EMA:
            utils.update_ema(model, cur_iter, dataloader, opt)
        if cur_iter % opt.freq_print == 0:
            im_saver.visualize_batch(model, data['image'], data['label'], cur_iter)
            timer(epoch, cur_iter)
        if cur_iter % opt.freq_save_ckpt == 0:
            utils.save_networks(opt, cur_iter, model)
        if cur_iter % opt.freq_save_latest == 0:
            utils.save_networks(opt, cur_iter, model, latest=True)
        if cur_iter % opt.freq_fid == 0 and cur_iter > 0:
            is_best = fid_computer.update(model, cur_iter)
            if is_best:
                utils.save_networks(opt, cur_iter, model, best=True)
        visualizer_losses(cur_iter, losses_G_list + losses_D_list)

# --- after training ---#
utils.update_ema(model, cur_iter, dataloader, opt, force_run_stats=True)
utils.save_networks(opt, cur_iter, model)
utils.save_networks(opt, cur_iter, model, latest=True)
is_best = fid_computer.update(model, cur_iter)
if is_best:
    utils.save_networks(opt, cur_iter, model, best=True)

print("The training has successfully finished")
