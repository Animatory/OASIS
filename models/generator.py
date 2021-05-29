import torch
import torch.nn as nn
import torch.nn.functional as F

import models.norms as norms


class OASIS_Generator(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        ch = opt.channels_G
        self.channels = [16 * ch, 16 * ch, 16 * ch, 8 * ch, 4 * ch, 2 * ch, 1 * ch]
        self.init_w = opt.image_size // (2 ** (opt.num_res_blocks - 1))
        self.init_h = round(self.init_w / opt.aspect_ratio)
        self.conv_img = nn.Conv2d(self.channels[-1], 3, 3, padding=1)

        self.up = nn.Upsample(scale_factor=2)
        self.body = nn.ModuleList([])
        for in_ch, out_ch in zip(self.channels[:-1], self.channels[1:]):
            self.body.append(ResnetBlock_with_SPADE(in_ch, out_ch, opt))

        last_nc = self.opt.semantic_nc + (0 if self.opt.no_3dnoise else self.opt.z_dim)
        self.fc = nn.Conv2d(last_nc, 16 * ch, 3, padding=1)

        self.init_weights()

    def forward(self, labels, features=None, zero_noise=False):
        b, c, h, w = labels.shape
        zeros = torch.zeros(b, self.opt.z_dim, dtype=labels.dtype,
                            device=labels.device, requires_grad=False)
        # if zero_noise:
        #     features = zeros
        # else:
        #     if features is None:
        #         log_var = zeros
        #         mu = zeros
        #     else:
        #         mu, log_var = torch.chunk(features, 2, dim=1)
        #     std = torch.exp(0.5 * log_var)
        #     features = torch.randn_like(std)
        #     features = features * std + mu
        if zero_noise:
            features = zeros
        elif features is None:
            features = torch.randn_like(zeros)

        if features.ndim == 2:
            features = features[:, :, None, None].expand(b, self.opt.z_dim, h, w)

        labels = torch.cat((features, labels), dim=1)
        x = F.interpolate(labels, size=(self.init_w, self.init_h), mode='bilinear', align_corners=True)
        x = self.fc(x)
        for i in range(self.opt.num_res_blocks):
            x = self.body[i](x, labels)
            if i < self.opt.num_res_blocks - 1:
                x = self.up(x)
        x = self.conv_img(F.leaky_relu(x, 2e-1))
        x = torch.tanh(x)
        return x

    def init_weights(self, gain=0.02):
        for m in self.modules():
            classname = m.__class__.__name__
            if classname.find('BatchNorm2d') != -1:
                if hasattr(m, 'weight') and m.weight is not None:
                    nn.init.normal_(m.weight.data, 1.0, gain)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias.data, 0.0)
            elif hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
                nn.init.xavier_normal_(m.weight.data, gain=gain)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias.data, 0.0)


class ResnetBlock_with_SPADE(nn.Module):
    def __init__(self, fin, fout, opt, z_dim=None):
        super().__init__()
        self.opt = opt
        self.learned_shortcut = (fin != fout)
        fmiddle = min(fin, fout)
        sp_norm = norms.get_spectral_norm(opt)
        self.conv_0 = sp_norm(nn.Conv2d(fin, fmiddle, kernel_size=3, padding=1))
        self.conv_1 = sp_norm(nn.Conv2d(fmiddle, fout, kernel_size=3, padding=1))
        if self.learned_shortcut:
            self.conv_s = sp_norm(nn.Conv2d(fin, fout, kernel_size=1, bias=False))

        spade_conditional_input_dims = opt.semantic_nc
        if not opt.no_3dnoise:
            spade_conditional_input_dims += opt.z_dim if z_dim is None else z_dim

        self.norm_0 = norms.SPADE(opt, fin, spade_conditional_input_dims)
        self.norm_1 = norms.SPADE(opt, fmiddle, spade_conditional_input_dims)
        if self.learned_shortcut:
            self.norm_s = norms.SPADE(opt, fin, spade_conditional_input_dims)
        self.activ = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x, seg):
        if self.learned_shortcut:
            x_s = self.conv_s(self.norm_s(x, seg))
        else:
            x_s = x
        dx = self.conv_0(self.activ(self.norm_0(x, seg)))
        dx = self.conv_1(self.activ(self.norm_1(dx, seg)))
        out = x_s + dx
        return out
