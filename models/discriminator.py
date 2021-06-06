import torch
import torch.nn as nn
import torch.nn.functional as F

import models.norms as norms
from registry import DISCRIMINATORS


class StyleVectorizer(nn.Module):
    def __init__(self, emb, depth, is_discriminator=False):
        super().__init__()
        self.is_discriminator = is_discriminator

        layers = []
        for i in range(depth - 1):
            layers.extend([nn.Linear(emb, emb), nn.LayerNorm(emb), nn.LeakyReLU(0.2, inplace=True)])

        if is_discriminator:
            layers.append(nn.Linear(emb, 1))
        else:
            layers.append(nn.Linear(emb, emb))

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        if not self.is_discriminator:
            x = F.normalize(x, dim=1)
        return self.net(x)


@DISCRIMINATORS.register_model
class OASIS_Discriminator(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        sp_norm = norms.get_spectral_norm(opt)
        output_channel = opt.semantic_nc + 1  # for N+1 loss
        self.channels = [3, 128, 128, 256, 256, 512, 512]
        self.body_up = nn.ModuleList([])
        self.body_down = nn.ModuleList([])
        # encoder part
        for i in range(opt.num_res_blocks):
            self.body_down.append(DiscriminatorResidualBlock(self.channels[i], self.channels[i + 1],
                                                             sp_norm, -1, first=not i))
        # decoder part
        self.body_up.append(DiscriminatorResidualBlock(self.channels[-1], self.channels[-2], sp_norm, 1))
        for i in range(1, opt.num_res_blocks - 1):
            self.body_up.append(DiscriminatorResidualBlock(2 * self.channels[-1 - i], self.channels[-2 - i],
                                                           sp_norm, 1))
        self.body_up.append(DiscriminatorResidualBlock(fin=2 * self.channels[1], fout=64,
                                                       norm_layer=sp_norm, up_or_down=1))
        self.layer_up_last = nn.Conv2d(64, output_channel, 1, 1, 0)

        self.pool = nn.AdaptiveAvgPool2d(1)
        # self.feature_linear = nn.Linear(self.num_features, self.opt.z_dim * 2)
        self.feature_mapper = nn.Linear(self.channels[-1], self.opt.z_dim)
        self.to_feature = StyleVectorizer(opt.z_dim, 3, is_discriminator=False)

        self.init_weights()

    def forward(self, x):
        # encoder
        encoder_res = list()
        for i in range(len(self.body_down)):
            x = self.body_down[i](x)
            encoder_res.append(x)

        features = self.to_feature(self.feature_mapper(x.mean((2, 3))))

        # decoder
        x = self.body_up[0](x)
        for i in range(1, len(self.body_down)):
            x = self.body_up[i](torch.cat((encoder_res[-i - 1], x), dim=1))
        ans = self.layer_up_last(x)
        return features, ans

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


class DiscriminatorResidualBlock(nn.Module):
    def __init__(self, fin, fout, norm_layer, up_or_down, first=False):
        super().__init__()
        # Attributes
        self.up_or_down = up_or_down
        self.first = first
        self.learned_shortcut = (fin != fout)
        fmiddle = fout
        if first:
            self.conv1 = nn.Sequential(norm_layer(nn.Conv2d(fin, fmiddle, 3, 1, 1)))
        else:
            if self.up_or_down > 0:
                self.conv1 = nn.Sequential(nn.LeakyReLU(0.2, False), nn.Upsample(scale_factor=2),
                                           norm_layer(nn.Conv2d(fin, fmiddle, 3, 1, 1)))
            else:
                self.conv1 = nn.Sequential(nn.LeakyReLU(0.2, False), norm_layer(nn.Conv2d(fin, fmiddle, 3, 1, 1)))
        self.conv2 = nn.Sequential(nn.LeakyReLU(0.2, False), norm_layer(nn.Conv2d(fmiddle, fout, 3, 1, 1)))
        if self.learned_shortcut:
            self.conv_s = norm_layer(nn.Conv2d(fin, fout, 1, 1, 0))
        if up_or_down > 0:
            self.sampling = nn.Upsample(scale_factor=2)
        elif up_or_down < 0:
            self.sampling = nn.AvgPool2d(2)
        else:
            self.sampling = nn.Sequential()

    def forward(self, x):
        x_s = self.shortcut(x)
        dx = self.conv1(x)
        dx = self.conv2(dx)
        if self.up_or_down < 0:
            dx = self.sampling(dx)
        out = x_s + dx
        return out

    def shortcut(self, x):
        if self.first:
            if self.up_or_down < 0:
                x = self.sampling(x)
            if self.learned_shortcut:
                x = self.conv_s(x)
            x_s = x
        else:
            if self.up_or_down > 0:
                x = self.sampling(x)
            if self.learned_shortcut:
                x = self.conv_s(x)
            if self.up_or_down < 0:
                x = self.sampling(x)
            x_s = x
        return x_s
