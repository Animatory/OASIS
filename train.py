import torch
import numpy as np
from apex import amp
from tqdm import tqdm

import config
import dataloaders.dataloaders as dataloaders
import models.losses as losses
import models.models_fast as models
import utils.utils as utils
from trainer import Trainer
from utils.fid_scores import FIDCalculator
from utils.miou import MetricManager
from utils.resampling import resample_dataset


def run():
    torch.multiprocessing.freeze_support()

    # --- read options ---#
    opt = config.read_arguments(train=True)

    # --- create utils ---#
    timer = utils.Timer(opt)
    # visualizer_losses = utils.LossesSaver(opt)
    losses_computer = losses.LossesComputer(opt)
    dataloader, dataloader_val, dataset_train, dataset_val = dataloaders.get_dataloaders(opt)
    im_saver = utils.ImageSaver(opt)
    fid_computer = FIDCalculator(opt, dataloader_val)
    # metric_meter = MetricManager(opt)

    # --- create models ---#
    model = models.OASIS(opt)
    if opt.gpu_ids != "-1":
        gpus = list(map(int, opt.gpu_ids.split(",")))
        model.to(gpus[0])

    # --- create optimizers ---#
    G_params = list(model.netG.parameters()) + list(model.to_feature.parameters())
    D_params = list(model.netD.parameters()) + list(model.to_logit.parameters())

    optimizerG = torch.optim.AdamW(G_params, lr=opt.lr_g, betas=(opt.beta1, opt.beta2))
    optimizerD = torch.optim.AdamW(D_params, lr=opt.lr_d, betas=(opt.beta1, opt.beta2))

    [model], [optimizerD, optimizerG] = amp.initialize(
        [model], [optimizerD, optimizerG], loss_scale="dynamic", max_loss_scale=0.1,
        opt_level=opt.opt_level, num_losses=2)
    optimizerD._lazy_init_maybe_master_weights()
    optimizerG._lazy_init_maybe_master_weights()

    model = models.put_on_multi_gpus(model, opt)

    trainer = Trainer(opt, model, optimizerD, optimizerG, losses_computer, None)

    # --- the training loop ---#
    already_started = False
    start_epoch, start_iter = utils.get_start_iters(opt.loaded_latest_iter, len(dataloader))
    for epoch in range(start_epoch, opt.num_epochs):
        print(f'Epoch {epoch} started')
        t = tqdm(dataloader, total=len(dataloader))
        for i, data_i in enumerate(t):
            if i > len(dataloader):
                break
            if not already_started and i < start_iter:
                continue
            already_started = True
            cur_iter = epoch * len(dataloader) + i
            data = models.preprocess_input(opt, data_i)

            loss_d, loss_g = trainer.train_step_fast_gan(data, cur_iter)

            if torch.isnan(loss_d):
                raise ValueError('Got Nan loss')
            t.set_description_str(f'G loss: {loss_g:.4f}, D loss: {loss_d:.4f}', refresh=False)

            # --- stats update ---#
            if not opt.no_EMA:
                model.update_ema(cur_iter, dataloader, opt)
            if cur_iter > 0:
                if cur_iter % opt.freq_print == 0:
                    torch.cuda.empty_cache()
                    im_saver.visualize_batch(model, data['image'], data['label'], cur_iter)
                    # timer(epoch, cur_iter)
                if cur_iter % opt.freq_save_ckpt == 0:
                    torch.cuda.empty_cache()
                    model.save_networks(cur_iter)
                if cur_iter % opt.freq_save_latest == 0:
                    torch.cuda.empty_cache()
                    model.save_networks(cur_iter, latest=True)
                if cur_iter % opt.freq_fid == 0:
                    torch.cuda.empty_cache()
                    is_best = fid_computer.update(model, cur_iter)
                    if is_best:
                        model.save_networks(cur_iter, best=True)
        t.close()

        # t = tqdm(dataloader_val, total=len(dataloader_val))
        # for i, data_i in enumerate(t):
        #     data = models.preprocess_input(opt, data_i)
        #     targets = data['label'].argmax(1)
        #     predictions, _ = model.forward(**data, mode='predict')
        #     metric_meter.update(targets, predictions)
        # t.close()
        # torch.cuda.empty_cache()
        # ious_per_class = metric_meter.on_epoch_end(epoch)
        # print(f'Epoch {epoch}, mIoU: {np.mean(ious_per_class)}')
        # dataloader = resample_dataset(opt, dataset_train, ious_per_class)

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
