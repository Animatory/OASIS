import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

from registry import DISCRIMINATORS
from . import OASIS_Generator, DataParallelWithCallback


class OASIS(nn.Module):
    def __init__(self, opt):
        super(OASIS, self).__init__()
        self.opt = opt
        # --- generator and discriminator ---
        self.netG = OASIS_Generator(opt)
        self.netD = DISCRIMINATORS[opt.discriminator](opt=opt)
        self.label_smoother = LabelSmoother(kernel_size=9, sigma=3, channels=opt.semantic_nc)

        self.to_feature = StyleVectorizer(opt.z_dim, 6, is_discriminator=False)
        self.to_logit = StyleVectorizer(opt.z_dim, 3, is_discriminator=True)

        self.print_parameter_count()
        # --- EMA of generator weights ---
        if not self.opt.no_EMA:
            with torch.no_grad():
                self.netEMA = copy.deepcopy(self.netG) if not opt.no_EMA else None
        # --- load previous checkpoints if needed ---
        self.load_checkpoints()

    def forward(self, image=None, label=None, mode=None, is_ema=None, noise=None, image_unsup=None):
        if mode == "generate":
            with torch.no_grad():
                if is_ema is None:
                    is_ema = not self.opt.no_EMA
                else:
                    is_ema = is_ema and not self.opt.no_EMA
                model = self.netEMA if is_ema else self.netG
                if noise is None:
                    b, c, h, w = label.shape
                    noise = torch.zeros(b, self.opt.z_dim, dtype=label.dtype,
                                        device=label.device, requires_grad=False)
                    noise = self.to_feature(noise)
                fake = model(labels=label[:, -self.opt.semantic_nc:], features=noise)
            return fake

        if mode == 'predict':
            with torch.no_grad():
                features, labels = self.forward_discriminator(image)
            return labels[:, -self.opt.semantic_nc:], features

    def forward_discriminator(self, input):
        features, labels = self.netD(input)[:2]
        return features, labels

    def load_checkpoints(self, save_whole_network=True):
        if self.opt.phase == "test":
            which_iter = self.opt.ckpt_iter
            path = self.opt.checkpoints_dir / self.opt.name / "models"
            if save_whole_network:
                self.load_state_dict(torch.load(path / f"{which_iter}.pth"))
            else:
                if self.opt.no_EMA:
                    self.netG.load_state_dict(torch.load(path / f"{which_iter}_G.pth"))
                else:
                    self.netEMA.load_state_dict(torch.load(path / f"{which_iter}_EMA.pth"))
                self.netD.load_state_dict(torch.load(path / f"{which_iter}_D.pth"))
        elif self.opt.continue_train:
            which_iter = self.opt.which_iter
            path = self.opt.checkpoints_dir / self.opt.name / "models"
            if save_whole_network:
                self.load_state_dict(torch.load(path / f"{which_iter}.pth"))
            else:
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

    def save_networks(self, cur_iter, latest=False, best=False, save_whole_network=True):
        opt = self.opt
        path = opt.checkpoints_dir / opt.name / "models"
        path.mkdir(exist_ok=True)
        if save_whole_network:
            if latest:
                torch.save(self.state_dict(), path / 'latest.pth')
                file = opt.checkpoints_dir / opt.name / "latest_iter.txt"
                file.write_text(str(cur_iter))
            elif best:
                torch.save(self.state_dict(), path / 'best.pth')
                file = opt.checkpoints_dir / opt.name / "best_iter.txt"
                file.write_text(str(cur_iter))
            else:
                torch.save(self.state_dict(), path / f'{cur_iter}.pth')
        else:
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


class EqualLinear(nn.Module):
    def __init__(self, in_dim, out_dim, lr_mul=1, bias=True):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_dim, in_dim))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_dim))

        self.lr_mul = lr_mul

    def forward(self, x):
        return F.linear(x, self.weight * self.lr_mul, bias=self.bias * self.lr_mul)


class StyleVectorizer(nn.Module):
    def __init__(self, emb, depth, is_discriminator=False, lr_mul=0.1):
        super().__init__()
        self.is_discriminator = is_discriminator

        layers = []
        for i in range(depth - 1):
            layers.extend([EqualLinear(emb, emb, lr_mul), nn.LeakyReLU(0.2, inplace=True)])

        if is_discriminator:
            layers.append(EqualLinear(emb, 1, lr_mul))
        else:
            layers.append(EqualLinear(emb, emb, lr_mul))

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        if not self.is_discriminator:
            x = F.normalize(x, dim=1)
        return self.net(x)


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
