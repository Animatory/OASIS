import copy

import torch
import torch.nn as nn

from . import OASIS_Generator, OASIS_Discriminator, DataParallelWithCallback, VGGLoss


class OASIS(nn.Module):
    def __init__(self, opt):
        super(OASIS, self).__init__()
        self.opt = opt
        # --- generator and discriminator ---
        self.netG = OASIS_Generator(opt)
        self.netD = OASIS_Discriminator(opt)
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

    def forward(self, image, label, mode, losses_computer=None, is_ema=None, noise=None):
        # Branching is applied to be compatible with DataParallel
        if mode == "losses_G":
            loss_G = 0
            fake = self.netG(label)
            output_D = self.netD(fake)
            loss_G_adv = losses_computer.loss(output_D, label, for_real=True)
            loss_G += loss_G_adv
            if self.opt.add_vgg_loss:
                loss_G_vgg = self.opt.lambda_vgg * self.VGG_loss(fake, image)
                loss_G += loss_G_vgg
            else:
                loss_G_vgg = None
            return loss_G, [loss_G_adv, loss_G_vgg]

        if mode == "losses_D":
            loss_D = 0
            with torch.no_grad():
                fake = self.netG(label)
            output_D_fake = self.netD(fake)
            loss_D_fake = losses_computer.loss(output_D_fake, label, for_real=False)
            loss_D += loss_D_fake
            output_D_real = self.netD(image)
            loss_D_real = losses_computer.loss(output_D_real, label, for_real=True)
            loss_D += loss_D_real
            if not self.opt.no_labelmix:
                mixed_inp, mask = generate_labelmix(label, fake, image)
                output_D_mixed = self.netD(mixed_inp)
                loss_D_lm = self.opt.lambda_labelmix * losses_computer.loss_labelmix(mask, output_D_mixed,
                                                                                     output_D_fake,
                                                                                     output_D_real)
                loss_D += loss_D_lm
            else:
                loss_D_lm = None
            return loss_D, [loss_D_fake, loss_D_real, loss_D_lm]

        if mode == "generate":
            with torch.no_grad():
                is_ema = is_ema and not self.opt.no_EMA
                model = self.netEMA if is_ema else self.netG
                fake = model(label, noise=noise)
            return fake

        if mode == 'predict':
            with torch.no_grad():
                return self.netD(image)

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


def put_on_multi_gpus(model, opt):
    if opt.gpu_ids != "-1":
        gpus = list(map(int, opt.gpu_ids.split(",")))
        model = DataParallelWithCallback(model, device_ids=gpus).cuda(gpus[0])
    else:
        model.module = model
    assert len(opt.gpu_ids.split(",")) == 0 or opt.batch_size % len(opt.gpu_ids.split(",")) == 0
    return model


def preprocess_input(opt, data):
    if opt.gpu_ids != "-1":
        gpus = list(map(int, opt.gpu_ids.split(",")))
        data['label'] = data['label'].cuda(gpus[0])
        data['image'] = data['image'].cuda(gpus[0])
        if 'image_unsup' in data:
            data['image_unsup'] = data['image_unsup'].cuda(gpus[0])

    label_map = data['label']
    bs, h, w = label_map.shape
    nc = opt.semantic_nc
    input_label = torch.zeros(bs, nc, h, w, device=label_map.device)

    input_semantics = input_label.scatter_(1, label_map.unsqueeze_(1), 1)
    new_data = dict(image=data['image'], label=input_semantics)
    if 'image_unsup' in data:
        new_data['image_unsup'] = data['image_unsup']

    return new_data


def generate_labelmix(label, fake_image, real_image):
    target_map = torch.argmax(label, dim=1, keepdim=True)
    all_classes = torch.unique(target_map)
    for c in all_classes:
        target_map[target_map == c] = torch.randint(0, 2, (1,), device=fake_image.device)
    mixed_image = torch.where(target_map.bool(), real_image, fake_image)
    return mixed_image, target_map
