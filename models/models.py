import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

from registry import DISCRIMINATORS
from . import OASIS_Generator, DataParallelWithCallback, VGGLoss


class OASIS(nn.Module):
    def __init__(self, opt):
        super(OASIS, self).__init__()
        self.opt = opt
        # --- generator and discriminator ---
        self.netG = OASIS_Generator(opt)
        self.netD = DISCRIMINATORS[opt.discriminator](opt=opt)
        self.label_smoother = LabelSmoother(kernel_size=9, sigma=3, channels=opt.semantic_nc)
        self.print_parameter_count()
        # --- EMA of generator weights ---
        if not self.opt.no_EMA:
            with torch.no_grad():
                self.netEMA = copy.deepcopy(self.netG) if not opt.no_EMA else None
        # --- load previous checkpoints if needed ---
        self.load_checkpoints()
        # --- perceptual loss ---#
        if opt.phase == "train" and opt.add_vgg_loss:
            self.VGG_loss = VGGLoss(self.opt.gpu_ids)

    def forward(self, image, label, mode, losses_computer=None, image_unsup=None, is_ema=None, noise=None):
        if mode == "losses_G":
            fake = self.netG(label)
            fake_pred_labels, fake_pred_features = self.forward_discriminator(fake)
            loss_G_adv = losses_computer.loss(fake_pred_labels, label, for_real=True)
            loss_G = loss_G_adv

            smoothed_label = self.label_smoother(label)
            fake_smoothed = self.netG(smoothed_label, noise=fake_pred_features.detach())
            smoothed_pred_labels, smoothed_pred_features = self.forward_discriminator(fake_smoothed)
            loss_G_adv_smoothed = losses_computer.loss(smoothed_pred_labels, label, for_real=True)
            loss_G += losses_computer.loss_ae(fake_smoothed, image)
            loss_G += loss_G_adv_smoothed

            if self.opt.add_vgg_loss:
                loss_G_vgg = self.opt.lambda_vgg * self.VGG_loss(fake, image)
                loss_G += loss_G_vgg
            else:
                loss_G_vgg = None
            return loss_G, [loss_G_adv, loss_G_vgg]

        if mode == "losses_D":
            loss_D = 0

            real_pred_labels, real_features = self.forward_discriminator(image)
            real_soft_labels = torch.softmax(real_pred_labels, 1)[:, 1:]
            fake_2 = self.netG(real_soft_labels, noise=real_features)
            loss_D += losses_computer.loss_ae(fake_2, image)
            loss_features = losses_computer.loss_features(real_features)
            loss_D_real = losses_computer.loss(real_pred_labels, label, for_real=True)

            with torch.no_grad():
                fake = self.netG(label)
            fake_pred_labels, fake_features = self.forward_discriminator(fake)
            loss_features += losses_computer.loss_features(fake_features)
            loss_D_fake = losses_computer.loss(fake_pred_labels, label, for_real=False)

            if image_unsup is not None:
                unsup_pred_labels, unsup_features = self.forward_discriminator(image_unsup)
                unsup_pred_labels = torch.softmax(unsup_pred_labels, 1)[:, 1:]
                fake_3 = self.netG(unsup_pred_labels, noise=unsup_features)
                loss_D += losses_computer.loss_ae(fake_3, image_unsup)
                loss_features += losses_computer.loss_features(unsup_features)
                loss_D_unsup = losses_computer.loss_unsup(unsup_pred_labels)
                loss_D += loss_D_unsup

            loss_D += loss_D_fake + loss_D_real
            if not self.opt.no_labelmix:
                mixed_inp, mask = self.generate_labelmix(label, fake, image)
                mixed_pred_labels, mixes_features = self.forward_discriminator(mixed_inp)
                loss_features += losses_computer.loss_features(mixes_features)

                loss_D_lm = self.opt.lambda_labelmix * losses_computer.loss_labelmix(mask, mixed_pred_labels,
                                                                                     fake_pred_labels,
                                                                                     real_pred_labels)
                loss_D += loss_D_lm
            else:
                loss_D_lm = None
            loss_D += loss_features
            return loss_D, [loss_D_fake, loss_D_real, loss_D_lm, loss_features]

        if mode == "generate":
            with torch.no_grad():
                if is_ema is None:
                    is_ema = not self.opt.no_EMA
                else:
                    is_ema = is_ema and not self.opt.no_EMA
                model = self.netEMA if is_ema else self.netG
                fake = model(label, noise=noise)
            return fake

        if mode == 'predict':
            with torch.no_grad():
                labels, features = self.forward_discriminator(image)
            return labels[:, 1:], features

    def forward_discriminator(self, input):
        labels, features = self.netD(input)
        return labels, features

    def load_checkpoints(self):
        if self.opt.phase == "test":
            which_iter = self.opt.ckpt_iter
            path = self.opt.checkpoints_dir / self.opt.name / "models"
            if self.opt.no_EMA:
                self.netG.load_state_dict(torch.load(path / f"{which_iter}_G.pth"))
            else:
                self.netEMA.load_state_dict(torch.load(path / f"{which_iter}_EMA.pth"))
            self.netD.load_state_dict(torch.load(path / f"{which_iter}_D.pth"))
        elif self.opt.continue_train:
            which_iter = self.opt.which_iter
            path = self.opt.checkpoints_dir / self.opt.name / "models"
            self.netG.load_state_dict(torch.load(path / f"{which_iter}_G.pth"))
            self.netD.load_state_dict(torch.load(path / f"{which_iter}_D.pth"))
            if not self.opt.no_EMA:
                self.netEMA.load_state_dict(torch.load(path / f"{which_iter}_EMA.pth"))

    def print_parameter_count(self):
        if self.opt.phase == "train":
            networks = [self.netG, self.netD]
        else:
            networks = [self.netG]
        for network in networks:
            param_count = 0
            for name, module in network.named_modules():
                if (isinstance(module, nn.Conv2d)
                        or isinstance(module, nn.Linear)
                        or isinstance(module, nn.Embedding)):
                    param_count += sum([p.data.nelement() for p in module.parameters()])
            print('Created', network.__class__.__name__, "with %d parameters" % param_count)

    def update_ema(self, cur_iter, dataloader, opt, force_run_stats=False):
        # update weights based on new generator weights
        with torch.no_grad():
            decay = opt.EMA_decay
            G_state_dict = self.netG.state_dict()
            EMA_state_dict = self.netEMA.state_dict()
            for key, value in EMA_state_dict.items():
                EMA_state_dict[key].data.copy_(value.data * decay +
                                               G_state_dict[key].data * (1 - decay))

        # collect running stats for batchnorm before FID computation, image or network saving
        condition_run_stats = (
                force_run_stats or
                cur_iter % opt.freq_print == 0 or
                cur_iter % opt.freq_fid == 0 or
                cur_iter % opt.freq_save_ckpt == 0 or
                cur_iter % opt.freq_save_latest == 0
        )
        if condition_run_stats:
            with torch.no_grad():
                for num_upd, data_i in enumerate(dataloader):
                    data = preprocess_input(opt, data_i)
                    self(**data, mode="generate", is_ema=True)
                    if num_upd > 50:
                        break

    def save_networks(self, cur_iter, latest=False, best=False):
        opt = self.opt
        path = opt.checkpoints_dir / opt.name / "models"
        path.mkdir(exist_ok=True)
        if latest:
            torch.save(self.netG.state_dict(), path / 'latest_G.pth')
            torch.save(self.netD.state_dict(), path / 'latest_D.pth')
            if not opt.no_EMA:
                torch.save(self.netEMA.state_dict(), path / 'latest_EMA.pth')
            file = opt.checkpoints_dir / opt.name / "latest_iter.txt"
            file.write_text(str(cur_iter))
        elif best:
            torch.save(self.netG.state_dict(), path / 'best_G.pth')
            torch.save(self.netD.state_dict(), path / 'best_D.pth')
            if not opt.no_EMA:
                torch.save(self.netEMA.state_dict(), path / 'best_EMA.pth')
            file = opt.checkpoints_dir / opt.name / "best_iter.txt"
            file.write_text(str(cur_iter))
        else:
            torch.save(self.netG.state_dict(), path / f'{cur_iter}_G.pth')
            torch.save(self.netD.state_dict(), path / f'{cur_iter}_D.pth')
            if not opt.no_EMA:
                torch.save(self.netEMA.state_dict(), path / f'{cur_iter}_EMA.pth')

    @staticmethod
    def generate_labelmix(label, fake_image, real_image):
        target_map = torch.argmax(label, dim=1, keepdim=True)
        all_classes = torch.unique(target_map)
        for c in all_classes:
            target_map[target_map == c] = torch.randint(2, (1,), device=fake_image.device)
        mixed_image = torch.where(target_map == 1, real_image, fake_image)
        return mixed_image, target_map


class LabelSmoother(nn.Module):
    def __init__(self, kernel_size, sigma, channels):
        super().__init__()
        self.kernel_size = kernel_size
        self.sigma = sigma
        self.channels = channels

        # Create a x, y coordinate grid of shape (kernel_size, kernel_size, 2)
        self.mean = (kernel_size - 1) // 2
        cords = torch.arange(kernel_size, requires_grad=False) - self.mean
        xy_grid = torch.stack(torch.meshgrid(cords, cords), dim=-1)

        # Calculate the 2-dimensional gaussian kernel which is
        # the product of two gaussian distributions for two different
        # variables (in this case called x and y)
        gaussian_kernel = torch.exp(torch.sum(xy_grid ** 2, dim=-1) / (-2 * sigma ** 2.))
        # Make sure sum of values in gaussian kernel equals 1.
        gaussian_kernel = gaussian_kernel / torch.sum(gaussian_kernel)

        # # Reshape to 2d depthwise convolutional weight
        gaussian_kernel = gaussian_kernel.view(1, 1, kernel_size, kernel_size)
        self.register_buffer('gaussian_kernel', gaussian_kernel.repeat(channels, 1, 1, 1))

    @torch.no_grad()
    def forward(self, labels):
        self.gaussian_kernel = self.gaussian_kernel.to(labels)
        smoothed_labels = F.conv2d(labels, self.gaussian_kernel, padding=self.mean, groups=self.channels)
        return F.normalize(smoothed_labels, p=1, dim=1)


def put_on_multi_gpus(model, opt):
    if opt.gpu_ids != "-1":
        gpus = list(map(int, opt.gpu_ids.split(",")))
        if len(gpus) > 1:
            model = DataParallelWithCallback(model, device_ids=gpus).cuda(gpus[0])
        else:
            model.to(gpus[0])
    assert len(opt.gpu_ids.split(",")) == 0 or opt.batch_size % len(opt.gpu_ids.split(",")) == 0
    return model


def preprocess_input(opt, data):
    if opt.opt_level == 'O0':
        dtype = torch.float32
    else:
        dtype = torch.float16
    if opt.gpu_ids != "-1":
        gpus = list(map(int, opt.gpu_ids.split(",")))
        data['label'] = data['label'].cuda(gpus[0])
        data['image'] = data['image'].to(dtype).cuda(gpus[0])
        if 'image_unsup' in data:
            data['image_unsup'] = data['image_unsup'].to(dtype).cuda(gpus[0])

    label_map = data['label']
    bs, h, w = label_map.shape
    nc = opt.semantic_nc
    input_label = torch.zeros(bs, nc, h, w, device=label_map.device, dtype=dtype)

    input_semantics = input_label.scatter_(1, label_map.unsqueeze_(1), 1)
    new_data = dict(image=data['image'], label=input_semantics)
    if 'image_unsup' in data:
        new_data['image_unsup'] = data['image_unsup']

    return new_data
