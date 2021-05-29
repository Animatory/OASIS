import torch.nn as nn
import torch.nn.functional as F

from .swin_transformer import SwinTransformer
from .upernet import UPerNet
from ..models_fast import StyleVectorizer
from registry import DISCRIMINATORS


@DISCRIMINATORS.register_model
class SwinDiscriminator(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.opt = opt

        self.backbone = SwinTransformer(
            embed_dim=96,
            depths=[2, 2, 6, 2],
            num_heads=[3, 6, 12, 24],
            window_size=7,
            ape=False,
            drop_path_rate=0.3,
            patch_norm=True,
            use_checkpoint=False
        )

        self.segm_head = UPerNet(
            in_channels=[96, 192, 384, 768],
            pool_scales=(1, 2, 3, 6),
            channels=768,
            dropout_ratio=0.1,
            num_classes=opt.semantic_nc + 1,
        )

        self.linear_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(self.backbone.num_features[-1], opt.z_dim)
        )
        self.to_feature = StyleVectorizer(opt.z_dim, 3, is_discriminator=False)

        # self.fake_extractor = nn.Sequential(
        #     nn.LeakyReLU(0.2), nn.Linear(opt.z_dim * 2, opt.z_dim * 2),
        #     nn.LeakyReLU(0.2), nn.Linear(opt.z_dim * 2, 1),
        # )

    def forward(self, x):
        # x_large = F.interpolate(x, scale_factor=(384, 384), mode='bilinear', align_corners=True)
        feature_maps = self.backbone(x)
        segm_prediction = self.segm_head(feature_maps)

        features = self.to_feature(self.linear_pool(feature_maps[-1]))

        segm_prediction = F.interpolate(segm_prediction, x.shape[2:], mode='bilinear', align_corners=True)
        return features, segm_prediction
