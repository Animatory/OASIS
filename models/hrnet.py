# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Ke Sun (sunk@mail.ustc.edu.cn), Jingyi Xie (hsfzxjy@gmail.com)
# ------------------------------------------------------------------------------

import logging
from typing import List

import torch
import torch._utils
import torch.nn as nn
import torch.nn.functional as F
from torch.hub import load_state_dict_from_url

from registry import DISCRIMINATORS
from .models_fast import StyleVectorizer

# BatchNorm2d = torch.nn.SyncBatchNorm
BatchNorm2d = torch.nn.InstanceNorm2d
ALIGN_CORNERS = True
_BN_MOMENTUM = 0.1
_logger = logging.getLogger(__name__)


url_weight_dir = 'https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-hrnet/'
model_urls = {
    'hrnet_w18_small': dict(url=url_weight_dir + 'hrnet_w18_small_v1-f460c6bc.pth'),
    'hrnet_w18_small_v2': dict(url=url_weight_dir + 'hrnet_w18_small_v2-4c50a8cb.pth'),
    'hrnet_w18': dict(url=url_weight_dir + 'hrnetv2_w18-8cb57bb9.pth'),
    'hrnet_w30': dict(url=url_weight_dir + 'hrnetv2_w30-8d7f8dab.pth'),
    'hrnet_w32': dict(url=url_weight_dir + 'hrnetv2_w32-90d8c5fb.pth'),
    'hrnet_w40': dict(url=url_weight_dir + 'hrnetv2_w40-7cd397a4.pth'),
    'hrnet_w44': dict(url=url_weight_dir + 'hrnetv2_w44-c9ac8c18.pth'),
    'hrnet_w48': dict(url=url_weight_dir + 'hrnetv2_w48-abd2e6ab.pth'),
    'hrnet_w64': dict(url=url_weight_dir + 'hrnetv2_w64-b47cc881.pth'),
}

cfg_cls = dict(
    hrnet_w18_small=dict(
        STEM_WIDTH=64,
        STAGE1=dict(
            NUM_MODULES=1,
            NUM_BRANCHES=1,
            BLOCK='BOTTLENECK',
            NUM_BLOCKS=(1,),
            NUM_CHANNELS=(32,),
            FUSE_METHOD='SUM',
        ),
        STAGE2=dict(
            NUM_MODULES=1,
            NUM_BRANCHES=2,
            BLOCK='BASIC',
            NUM_BLOCKS=(2, 2),
            NUM_CHANNELS=(16, 32),
            FUSE_METHOD='SUM'
        ),
        STAGE3=dict(
            NUM_MODULES=1,
            NUM_BRANCHES=3,
            BLOCK='BASIC',
            NUM_BLOCKS=(2, 2, 2),
            NUM_CHANNELS=(16, 32, 64),
            FUSE_METHOD='SUM'
        ),
        STAGE4=dict(
            NUM_MODULES=1,
            NUM_BRANCHES=4,
            BLOCK='BASIC',
            NUM_BLOCKS=(2, 2, 2, 2),
            NUM_CHANNELS=(16, 32, 64, 128),
            FUSE_METHOD='SUM',
        ),
    ),

    hrnet_w18_small_v2=dict(
        STEM_WIDTH=64,
        STAGE1=dict(
            NUM_MODULES=1,
            NUM_BRANCHES=1,
            BLOCK='BOTTLENECK',
            NUM_BLOCKS=(2,),
            NUM_CHANNELS=(64,),
            FUSE_METHOD='SUM',
        ),
        STAGE2=dict(
            NUM_MODULES=1,
            NUM_BRANCHES=2,
            BLOCK='BASIC',
            NUM_BLOCKS=(2, 2),
            NUM_CHANNELS=(18, 36),
            FUSE_METHOD='SUM'
        ),
        STAGE3=dict(
            NUM_MODULES=3,
            NUM_BRANCHES=3,
            BLOCK='BASIC',
            NUM_BLOCKS=(2, 2, 2),
            NUM_CHANNELS=(18, 36, 72),
            FUSE_METHOD='SUM'
        ),
        STAGE4=dict(
            NUM_MODULES=2,
            NUM_BRANCHES=4,
            BLOCK='BASIC',
            NUM_BLOCKS=(2, 2, 2, 2),
            NUM_CHANNELS=(18, 36, 72, 144),
            FUSE_METHOD='SUM',
        ),
    ),

    hrnet_w18=dict(
        STEM_WIDTH=64,
        STAGE1=dict(
            NUM_MODULES=1,
            NUM_BRANCHES=1,
            BLOCK='BOTTLENECK',
            NUM_BLOCKS=(4,),
            NUM_CHANNELS=(64,),
            FUSE_METHOD='SUM',
        ),
        STAGE2=dict(
            NUM_MODULES=1,
            NUM_BRANCHES=2,
            BLOCK='BASIC',
            NUM_BLOCKS=(4, 4),
            NUM_CHANNELS=(18, 36),
            FUSE_METHOD='SUM'
        ),
        STAGE3=dict(
            NUM_MODULES=4,
            NUM_BRANCHES=3,
            BLOCK='BASIC',
            NUM_BLOCKS=(4, 4, 4),
            NUM_CHANNELS=(18, 36, 72),
            FUSE_METHOD='SUM'
        ),
        STAGE4=dict(
            NUM_MODULES=3,
            NUM_BRANCHES=4,
            BLOCK='BASIC',
            NUM_BLOCKS=(4, 4, 4, 4),
            NUM_CHANNELS=(18, 36, 72, 144),
            FUSE_METHOD='SUM',
        ),
    ),

    hrnet_w30=dict(
        STEM_WIDTH=64,
        STAGE1=dict(
            NUM_MODULES=1,
            NUM_BRANCHES=1,
            BLOCK='BOTTLENECK',
            NUM_BLOCKS=(4,),
            NUM_CHANNELS=(64,),
            FUSE_METHOD='SUM',
        ),
        STAGE2=dict(
            NUM_MODULES=1,
            NUM_BRANCHES=2,
            BLOCK='BASIC',
            NUM_BLOCKS=(4, 4),
            NUM_CHANNELS=(30, 60),
            FUSE_METHOD='SUM'
        ),
        STAGE3=dict(
            NUM_MODULES=4,
            NUM_BRANCHES=3,
            BLOCK='BASIC',
            NUM_BLOCKS=(4, 4, 4),
            NUM_CHANNELS=(30, 60, 120),
            FUSE_METHOD='SUM'
        ),
        STAGE4=dict(
            NUM_MODULES=3,
            NUM_BRANCHES=4,
            BLOCK='BASIC',
            NUM_BLOCKS=(4, 4, 4, 4),
            NUM_CHANNELS=(30, 60, 120, 240),
            FUSE_METHOD='SUM',
        ),
    ),

    hrnet_w32=dict(
        STEM_WIDTH=64,
        STAGE1=dict(
            NUM_MODULES=1,
            NUM_BRANCHES=1,
            BLOCK='BOTTLENECK',
            NUM_BLOCKS=(4,),
            NUM_CHANNELS=(64,),
            FUSE_METHOD='SUM',
        ),
        STAGE2=dict(
            NUM_MODULES=1,
            NUM_BRANCHES=2,
            BLOCK='BASIC',
            NUM_BLOCKS=(4, 4),
            NUM_CHANNELS=(32, 64),
            FUSE_METHOD='SUM'
        ),
        STAGE3=dict(
            NUM_MODULES=4,
            NUM_BRANCHES=3,
            BLOCK='BASIC',
            NUM_BLOCKS=(4, 4, 4),
            NUM_CHANNELS=(32, 64, 128),
            FUSE_METHOD='SUM'
        ),
        STAGE4=dict(
            NUM_MODULES=3,
            NUM_BRANCHES=4,
            BLOCK='BASIC',
            NUM_BLOCKS=(4, 4, 4, 4),
            NUM_CHANNELS=(32, 64, 128, 256),
            FUSE_METHOD='SUM',
        ),
    ),

    hrnet_w40=dict(
        STEM_WIDTH=64,
        STAGE1=dict(
            NUM_MODULES=1,
            NUM_BRANCHES=1,
            BLOCK='BOTTLENECK',
            NUM_BLOCKS=(4,),
            NUM_CHANNELS=(64,),
            FUSE_METHOD='SUM',
        ),
        STAGE2=dict(
            NUM_MODULES=1,
            NUM_BRANCHES=2,
            BLOCK='BASIC',
            NUM_BLOCKS=(4, 4),
            NUM_CHANNELS=(40, 80),
            FUSE_METHOD='SUM'
        ),
        STAGE3=dict(
            NUM_MODULES=4,
            NUM_BRANCHES=3,
            BLOCK='BASIC',
            NUM_BLOCKS=(4, 4, 4),
            NUM_CHANNELS=(40, 80, 160),
            FUSE_METHOD='SUM'
        ),
        STAGE4=dict(
            NUM_MODULES=3,
            NUM_BRANCHES=4,
            BLOCK='BASIC',
            NUM_BLOCKS=(4, 4, 4, 4),
            NUM_CHANNELS=(40, 80, 160, 320),
            FUSE_METHOD='SUM',
        ),
    ),

    hrnet_w44=dict(
        STEM_WIDTH=64,
        STAGE1=dict(
            NUM_MODULES=1,
            NUM_BRANCHES=1,
            BLOCK='BOTTLENECK',
            NUM_BLOCKS=(4,),
            NUM_CHANNELS=(64,),
            FUSE_METHOD='SUM',
        ),
        STAGE2=dict(
            NUM_MODULES=1,
            NUM_BRANCHES=2,
            BLOCK='BASIC',
            NUM_BLOCKS=(4, 4),
            NUM_CHANNELS=(44, 88),
            FUSE_METHOD='SUM'
        ),
        STAGE3=dict(
            NUM_MODULES=4,
            NUM_BRANCHES=3,
            BLOCK='BASIC',
            NUM_BLOCKS=(4, 4, 4),
            NUM_CHANNELS=(44, 88, 176),
            FUSE_METHOD='SUM'
        ),
        STAGE4=dict(
            NUM_MODULES=3,
            NUM_BRANCHES=4,
            BLOCK='BASIC',
            NUM_BLOCKS=(4, 4, 4, 4),
            NUM_CHANNELS=(44, 88, 176, 352),
            FUSE_METHOD='SUM',
        ),
    ),

    hrnet_w48=dict(
        STEM_WIDTH=64,
        STAGE1=dict(
            NUM_MODULES=1,
            NUM_BRANCHES=1,
            BLOCK='BOTTLENECK',
            NUM_BLOCKS=(4,),
            NUM_CHANNELS=(64,),
            FUSE_METHOD='SUM',
        ),
        STAGE2=dict(
            NUM_MODULES=1,
            NUM_BRANCHES=2,
            BLOCK='BASIC',
            NUM_BLOCKS=(4, 4),
            NUM_CHANNELS=(48, 96),
            FUSE_METHOD='SUM'
        ),
        STAGE3=dict(
            NUM_MODULES=4,
            NUM_BRANCHES=3,
            BLOCK='BASIC',
            NUM_BLOCKS=(4, 4, 4),
            NUM_CHANNELS=(48, 96, 192),
            FUSE_METHOD='SUM'
        ),
        STAGE4=dict(
            NUM_MODULES=3,
            NUM_BRANCHES=4,
            BLOCK='BASIC',
            NUM_BLOCKS=(4, 4, 4, 4),
            NUM_CHANNELS=(48, 96, 192, 384),
            FUSE_METHOD='SUM',
        ),
    ),

    hrnet_w64=dict(
        STEM_WIDTH=64,
        STAGE1=dict(
            NUM_MODULES=1,
            NUM_BRANCHES=1,
            BLOCK='BOTTLENECK',
            NUM_BLOCKS=(4,),
            NUM_CHANNELS=(64,),
            FUSE_METHOD='SUM',
        ),
        STAGE2=dict(
            NUM_MODULES=1,
            NUM_BRANCHES=2,
            BLOCK='BASIC',
            NUM_BLOCKS=(4, 4),
            NUM_CHANNELS=(64, 128),
            FUSE_METHOD='SUM'
        ),
        STAGE3=dict(
            NUM_MODULES=4,
            NUM_BRANCHES=3,
            BLOCK='BASIC',
            NUM_BLOCKS=(4, 4, 4),
            NUM_CHANNELS=(64, 128, 256),
            FUSE_METHOD='SUM'
        ),
        STAGE4=dict(
            NUM_MODULES=3,
            NUM_BRANCHES=4,
            BLOCK='BASIC',
            NUM_BLOCKS=(4, 4, 4, 4),
            NUM_CHANNELS=(64, 128, 256, 512),
            FUSE_METHOD='SUM',
        ),
    )
)


class ModuleHelper:

    @staticmethod
    def BNReLU(num_features, bn_type=None, **kwargs):
        return nn.Sequential(
            BatchNorm2d(num_features, **kwargs),
            nn.LeakyReLU()
        )

    @staticmethod
    def BatchNorm2d(*args, **kwargs):
        return BatchNorm2d


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class SpatialGather_Module(nn.Module):
    """
        Aggregate the context features according to the initial
        predicted probability distribution.
        Employ the soft-weighted method to aggregate the context.
    """

    def __init__(self, cls_num=0, scale=1):
        super(SpatialGather_Module, self).__init__()
        self.cls_num = cls_num
        self.scale = scale

    def forward(self, feats, probs):
        batch_size, c, h, w = probs.shape
        probs = probs.view(batch_size, c, -1)
        feats = feats.view(batch_size, feats.size(1), -1)
        feats = feats.permute(0, 2, 1)  # batch x hw x c
        probs = F.softmax(self.scale * probs, dim=2)  # batch x k x hw
        ocr_context = torch.matmul(probs, feats).permute(0, 2, 1).unsqueeze(3)  # batch x k x c
        return ocr_context


class _ObjectAttentionBlock(nn.Module):
    """
    The basic implementation for object context block
    Input:
        N X C X H X W
    Parameters:
        in_channels       : the dimension of the input feature map
        key_channels      : the dimension after the key/query transform
        scale             : choose the scale to downsample the input feature maps (save memory cost)
        bn_type           : specify the bn type
    Return:
        N X C X H X W
    """

    def __init__(self,
                 in_channels,
                 key_channels,
                 scale=1,
                 bn_type=None):
        super(_ObjectAttentionBlock, self).__init__()
        self.scale = scale
        self.in_channels = in_channels
        self.key_channels = key_channels
        self.pool = nn.MaxPool2d(kernel_size=(scale, scale))
        self.f_pixel = nn.Sequential(
            nn.Conv2d(in_channels=self.in_channels, out_channels=self.key_channels,
                      kernel_size=1, stride=1, padding=0, bias=False),
            ModuleHelper.BNReLU(self.key_channels, bn_type=bn_type),
            nn.Conv2d(in_channels=self.key_channels, out_channels=self.key_channels,
                      kernel_size=1, stride=1, padding=0, bias=False),
            ModuleHelper.BNReLU(self.key_channels, bn_type=bn_type),
        )
        self.f_object = nn.Sequential(
            nn.Conv2d(in_channels=self.in_channels, out_channels=self.key_channels,
                      kernel_size=1, stride=1, padding=0, bias=False),
            ModuleHelper.BNReLU(self.key_channels, bn_type=bn_type),
            nn.Conv2d(in_channels=self.key_channels, out_channels=self.key_channels,
                      kernel_size=1, stride=1, padding=0, bias=False),
            ModuleHelper.BNReLU(self.key_channels, bn_type=bn_type),
        )
        self.f_down = nn.Sequential(
            nn.Conv2d(in_channels=self.in_channels, out_channels=self.key_channels,
                      kernel_size=1, stride=1, padding=0, bias=False),
            ModuleHelper.BNReLU(self.key_channels, bn_type=bn_type),
        )
        self.f_up = nn.Sequential(
            nn.Conv2d(in_channels=self.key_channels, out_channels=self.in_channels,
                      kernel_size=1, stride=1, padding=0, bias=False),
            ModuleHelper.BNReLU(self.in_channels, bn_type=bn_type),
        )

    def forward(self, x, proxy):
        batch_size, _, h, w = x.shape
        if self.scale > 1:
            x = self.pool(x)

        query = self.f_pixel(x).view(batch_size, self.key_channels, -1)
        query = query.permute(0, 2, 1)
        key = self.f_object(proxy).view(batch_size, self.key_channels, -1)
        value = self.f_down(proxy).view(batch_size, self.key_channels, -1)
        value = value.permute(0, 2, 1)

        sim_map = torch.matmul(query, key)
        sim_map = (self.key_channels ** -.5) * sim_map
        sim_map = F.softmax(sim_map, dim=-1)

        # add bg context ...
        context = torch.matmul(sim_map, value)
        context = context.permute(0, 2, 1).contiguous()
        context = context.view(batch_size, self.key_channels, *x.size()[2:])
        context = self.f_up(context)
        if self.scale > 1:
            context = F.interpolate(input=context, size=(h, w), mode='bilinear', align_corners=ALIGN_CORNERS)

        return context


class ObjectAttentionBlock2D(_ObjectAttentionBlock):
    def __init__(self,
                 in_channels,
                 key_channels,
                 scale=1,
                 bn_type=None):
        super(ObjectAttentionBlock2D, self).__init__(in_channels,
                                                     key_channels,
                                                     scale,
                                                     bn_type=bn_type)


class SpatialOCR(nn.Module):
    """
    Implementation of the OCR module:
    We aggregate the global object representation to update the representation for each pixel.
    """

    def __init__(self,
                 in_channels,
                 key_channels,
                 out_channels,
                 scale=1,
                 dropout=0.1,
                 bn_type=None):
        super(SpatialOCR, self).__init__()
        self.object_context_block = ObjectAttentionBlock2D(in_channels,
                                                           key_channels,
                                                           scale,
                                                           bn_type)
        _in_channels = 2 * in_channels

        self.conv_bn_dropout = nn.Sequential(
            nn.Conv2d(_in_channels, out_channels, kernel_size=1, padding=0, bias=False),
            ModuleHelper.BNReLU(out_channels, bn_type=bn_type),
            nn.Dropout2d(dropout)
        )

    def forward(self, feats, proxy_feats):
        context = self.object_context_block(feats, proxy_feats)
        output = self.conv_bn_dropout(torch.cat([context, feats], 1))
        return output


class HRNetOCR(nn.Module):
    def __init__(self, num_classes, encoder_channels, ocr_mid_channels=512,
                 ocr_key_channels=256, **kwargs):
        super(HRNetOCR, self).__init__()
        encoder_channels = sum(encoder_channels)
        self.out_channels = encoder_channels
        self.num_classes = num_classes

        self.conv3x3_ocr = nn.Sequential(
            nn.Conv2d(encoder_channels, ocr_mid_channels, kernel_size=3, padding=1),
            nn.InstanceNorm2d(ocr_mid_channels),
            nn.LeakyReLU()
        )

        self.ocr_gather_head = SpatialGather_Module(num_classes)

        self.ocr_distri_head = SpatialOCR(in_channels=ocr_mid_channels,
                                          key_channels=ocr_key_channels,
                                          out_channels=ocr_mid_channels,
                                          scale=1,
                                          dropout=0.05,
                                          )
        self.cls_head = nn.Conv2d(ocr_mid_channels, num_classes,
                                  kernel_size=1, stride=1, padding=0, bias=True)

        self.aux_head = nn.Sequential(
            nn.Conv2d(encoder_channels, encoder_channels,
                      kernel_size=1, stride=1, padding=0),
            nn.InstanceNorm2d(encoder_channels),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(encoder_channels, num_classes,
                      kernel_size=1, stride=1, padding=0, bias=True)
        )

    def forward(self, features):
        input_image, *features = features
        xs = [features[0]]
        for y in features[1:]:
            y = F.interpolate(y, size=features[0].shape[2:], mode='bilinear', align_corners=False)
            xs.append(y)

        feats = torch.cat(xs, 1)
        # ocr
        out_aux = self.aux_head(feats)
        # compute contrast feature
        feats = self.conv3x3_ocr(feats)

        context = self.ocr_gather_head(feats, out_aux)
        feats = self.ocr_distri_head(feats, context)

        out = self.cls_head(feats)
        out = F.interpolate(out, size=input_image.shape[2:], mode='bilinear', align_corners=False)
        out_aux = F.interpolate(out_aux, size=input_image.shape[2:], mode='bilinear', align_corners=False)

        # if self.training:
        #     out_aux_seg = [out_aux, out]
        # else:
        #     out_aux_seg = out

        return (out_aux + out) / 2


class HRNetHead(nn.Module):
    def __init__(self, num_classes, encoder_channels, **kwargs):
        super(HRNetHead, self).__init__()
        encoder_channels = sum(encoder_channels)
        self.out_channels = encoder_channels
        self.num_classes = num_classes

        self.aux_head = nn.Sequential(
            nn.Conv2d(encoder_channels, encoder_channels,
                      kernel_size=1, stride=1, padding=0),
            nn.InstanceNorm2d(encoder_channels),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(encoder_channels, num_classes,
                      kernel_size=1, stride=1, padding=0, bias=True)
        )

    def forward(self, features):
        input_image, *features = features
        xs = [features[0]]
        for y in features[1:]:
            y = F.interpolate(y, size=features[0].shape[2:], mode='bilinear', align_corners=False)
            xs.append(y)

        out_aux = self.aux_head(torch.cat(xs, 1))
        out_aux = F.interpolate(out_aux, size=input_image.shape[2:], mode='bilinear', align_corners=False)

        return out_aux


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=nn.Identity()):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = BatchNorm2d(planes, momentum=_BN_MOMENTUM)
        self.relu = nn.LeakyReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = BatchNorm2d(planes, momentum=_BN_MOMENTUM)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        residual = self.downsample(x)
        out = out + residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=nn.Identity()):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = BatchNorm2d(planes, momentum=_BN_MOMENTUM)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = BatchNorm2d(planes, momentum=_BN_MOMENTUM)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1,
                               bias=False)
        self.bn3 = BatchNorm2d(planes * self.expansion,
                               momentum=_BN_MOMENTUM)
        self.relu = nn.LeakyReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        residual = self.downsample(x)
        out = out + residual
        out = self.relu(out)

        return out


class HighResolutionModule(nn.Module):
    def __init__(self, num_branches, blocks, num_blocks, num_inchannels,
                 num_channels, fuse_method, multi_scale_output=True):
        super(HighResolutionModule, self).__init__()
        self._check_branches(
            num_branches, blocks, num_blocks, num_inchannels, num_channels)

        self.num_inchannels = num_inchannels
        self.fuse_method = fuse_method
        self.num_branches = num_branches

        self.multi_scale_output = multi_scale_output

        self.branches = self._make_branches(num_branches, blocks, num_blocks, num_channels)
        self.fuse_layers = self._make_fuse_layers()
        self.fuse_act = nn.LeakyReLU(False)

    def _check_branches(self, num_branches, blocks, num_blocks, num_inchannels, num_channels):
        error_msg = ''
        if num_branches != len(num_blocks):
            error_msg = 'NUM_BRANCHES({}) <> NUM_BLOCKS({})'.format(num_branches, len(num_blocks))
        elif num_branches != len(num_channels):
            error_msg = 'NUM_BRANCHES({}) <> NUM_CHANNELS({})'.format(num_branches, len(num_channels))
        elif num_branches != len(num_inchannels):
            error_msg = 'NUM_BRANCHES({}) <> NUM_INCHANNELS({})'.format(num_branches, len(num_inchannels))
        if error_msg:
            _logger.error(error_msg)
            raise ValueError(error_msg)

    def _make_one_branch(self, branch_index, block, num_blocks, num_channels, stride=1):
        downsample = nn.Identity()
        expanded_channels = num_channels[branch_index] * block.expansion
        if stride != 1 or self.num_inchannels[branch_index] != expanded_channels:
            downsample = nn.Sequential(
                nn.Conv2d(self.num_inchannels[branch_index], expanded_channels,
                          kernel_size=1, stride=stride, bias=False),
                nn.InstanceNorm2d(expanded_channels, momentum=_BN_MOMENTUM),
            )

        layers = [block(self.num_inchannels[branch_index], num_channels[branch_index], stride, downsample)]
        self.num_inchannels[branch_index] = expanded_channels
        for i in range(1, num_blocks[branch_index]):
            layers.append(block(self.num_inchannels[branch_index], num_channels[branch_index]))

        return nn.Sequential(*layers)

    def _make_branches(self, num_branches, block, num_blocks, num_channels):
        branches = []
        for i in range(num_branches):
            branches.append(self._make_one_branch(i, block, num_blocks, num_channels))

        return nn.ModuleList(branches)

    def _make_fuse_layers(self):
        if self.num_branches == 1:
            return nn.Identity()

        num_branches = self.num_branches
        num_inchannels = self.num_inchannels
        fuse_layers = []
        for i in range(num_branches if self.multi_scale_output else 1):
            fuse_layer = []
            for j in range(num_branches):
                if j > i:
                    fuse_layer.append(nn.Sequential(
                        nn.Conv2d(num_inchannels[j], num_inchannels[i], 1, 1, 0, bias=False),
                        nn.InstanceNorm2d(num_inchannels[i], momentum=_BN_MOMENTUM),
                        nn.Upsample(scale_factor=2 ** (j - i), mode='nearest')))
                elif j == i:
                    fuse_layer.append(nn.Identity())
                else:
                    conv3x3s = []
                    for k in range(i - j):
                        if k == i - j - 1:
                            num_outchannels_conv3x3 = num_inchannels[i]
                            conv3x3s.append(nn.Sequential(
                                nn.Conv2d(num_inchannels[j], num_outchannels_conv3x3, 3, 2, 1, bias=False),
                                nn.InstanceNorm2d(num_outchannels_conv3x3, momentum=_BN_MOMENTUM)))
                        else:
                            num_outchannels_conv3x3 = num_inchannels[j]
                            conv3x3s.append(nn.Sequential(
                                nn.Conv2d(num_inchannels[j], num_outchannels_conv3x3, 3, 2, 1, bias=False),
                                nn.InstanceNorm2d(num_outchannels_conv3x3, momentum=_BN_MOMENTUM),
                                nn.LeakyReLU(False)))
                    fuse_layer.append(nn.Sequential(*conv3x3s))
            fuse_layers.append(nn.ModuleList(fuse_layer))

        return nn.ModuleList(fuse_layers)

    def get_num_inchannels(self):
        return self.num_inchannels

    def forward(self, x: List[torch.Tensor]):
        if self.num_branches == 1:
            return [self.branches[0](x[0])]

        for i, branch in enumerate(self.branches):
            x[i] = branch(x[i])

        x_fuse = []
        for i, fuse_outer in enumerate(self.fuse_layers):
            y = x[0] if i == 0 else fuse_outer[0](x[0])
            for j in range(1, self.num_branches):
                if i == j:
                    y = y + x[j]
                else:
                    y = y + fuse_outer[j](x[j])
            x_fuse.append(self.fuse_act(y))

        return x_fuse


blocks_dict = {
    'BASIC': BasicBlock,
    'BOTTLENECK': Bottleneck
}


class HighResolutionNet(nn.Module):

    def __init__(self, cfg, opt, in_chans=3, **kwargs):
        super(HighResolutionNet, self).__init__()
        self.opt = opt

        self.out_channels = sum([cfg[f'STAGE4']['NUM_CHANNELS'][-1] // 2 ** i for i in range(4)])

        stem_width = cfg['STEM_WIDTH']
        self.conv1 = nn.Conv2d(in_chans, stem_width, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.InstanceNorm2d(stem_width, momentum=_BN_MOMENTUM)
        self.act1 = nn.LeakyReLU(inplace=True)
        self.conv2 = nn.Conv2d(stem_width, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn2 = nn.InstanceNorm2d(64, momentum=_BN_MOMENTUM)
        self.act2 = nn.LeakyReLU(inplace=True)

        self.stage1_cfg = cfg['STAGE1']
        num_channels = self.stage1_cfg['NUM_CHANNELS'][0]
        block = blocks_dict[self.stage1_cfg['BLOCK']]
        num_blocks = self.stage1_cfg['NUM_BLOCKS'][0]
        self.layer1 = self._make_layer(block, 64, num_channels, num_blocks)
        stage1_out_channel = block.expansion * num_channels

        self.stage2_cfg = cfg['STAGE2']
        num_channels = self.stage2_cfg['NUM_CHANNELS']
        block = blocks_dict[self.stage2_cfg['BLOCK']]
        num_channels = [num_channels[i] * block.expansion for i in range(len(num_channels))]
        self.transition1 = self._make_transition_layer([stage1_out_channel], num_channels)
        self.stage2, pre_stage_channels = self._make_stage(self.stage2_cfg, num_channels)

        self.stage3_cfg = cfg['STAGE3']
        num_channels = self.stage3_cfg['NUM_CHANNELS']
        block = blocks_dict[self.stage3_cfg['BLOCK']]
        num_channels = [num_channels[i] * block.expansion for i in range(len(num_channels))]
        self.transition2 = self._make_transition_layer(pre_stage_channels, num_channels)
        self.stage3, pre_stage_channels = self._make_stage(self.stage3_cfg, num_channels)

        self.stage4_cfg = cfg['STAGE4']
        num_channels = self.stage4_cfg['NUM_CHANNELS']
        block = blocks_dict[self.stage4_cfg['BLOCK']]
        num_channels = [num_channels[i] * block.expansion for i in range(len(num_channels))]
        self.transition3 = self._make_transition_layer(pre_stage_channels, num_channels)
        self.stage4, pre_stage_channels = self._make_stage(self.stage4_cfg, num_channels, multi_scale_output=True)

        self.num_features = 2048
        self.incre_modules, self.downsamp_modules, self.final_layer = self._make_neck(pre_stage_channels)
        self.pool = nn.AdaptiveAvgPool2d(1)
        # self.feature_linear = nn.Linear(self.num_features, self.opt.z_dim * 2)
        self.feature_mapper = nn.Linear(self.num_features, self.opt.z_dim)
        self.to_feature = StyleVectorizer(opt.z_dim, 3, is_discriminator=False)

        # self.head = HRNetOCR(1 + opt.semantic_nc + self.opt.z_dim, cfg[f'STAGE4']['NUM_CHANNELS'])
        self.head = HRNetOCR(1 + opt.semantic_nc, cfg[f'STAGE4']['NUM_CHANNELS'])
        # self.head_features = HRNetHead(self.opt.z_dim * 2, cfg[f'STAGE4']['NUM_CHANNELS'])

        self.init_weights()

    def init_weights(self):
        # #------- init weights --------
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def _make_neck(self, pre_stage_channels, incre_only=False):
        head_block = Bottleneck
        self.head_channels = [32, 64, 128, 256]

        # Increasing the #channels on each resolution
        # from C, 2C, 4C, 8C to 128, 256, 512, 1024
        incre_modules = []
        for i, channels in enumerate(pre_stage_channels):
            incre_modules.append(self._make_layer(head_block, channels, self.head_channels[i], 1, stride=1))
        incre_modules = nn.ModuleList(incre_modules)
        if incre_only:
            return incre_modules, None, None

        # downsampling modules
        downsamp_modules = []
        for i in range(len(pre_stage_channels) - 1):
            in_channels = self.head_channels[i] * head_block.expansion
            out_channels = self.head_channels[i + 1] * head_block.expansion
            downsamp_module = nn.Sequential(
                nn.Conv2d(
                    in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=2, padding=1),
                nn.InstanceNorm2d(out_channels, momentum=_BN_MOMENTUM),
                nn.LeakyReLU(inplace=True)
            )
            downsamp_modules.append(downsamp_module)
        downsamp_modules = nn.ModuleList(downsamp_modules)

        final_layer = nn.Sequential(
            nn.Conv2d(
                in_channels=self.head_channels[3] * head_block.expansion,
                out_channels=self.num_features, kernel_size=1, stride=1, padding=0
            ),
            nn.InstanceNorm2d(self.num_features, momentum=_BN_MOMENTUM),
            nn.LeakyReLU(inplace=True)
        )

        return incre_modules, downsamp_modules, final_layer

    def _make_transition_layer(self, num_channels_pre_layer, num_channels_cur_layer):
        num_branches_cur = len(num_channels_cur_layer)
        num_branches_pre = len(num_channels_pre_layer)

        transition_layers = []
        for i in range(num_branches_cur):
            if i < num_branches_pre:
                if num_channels_cur_layer[i] != num_channels_pre_layer[i]:
                    transition_layers.append(nn.Sequential(
                        nn.Conv2d(num_channels_pre_layer[i], num_channels_cur_layer[i], 3, 1, 1, bias=False),
                        nn.InstanceNorm2d(num_channels_cur_layer[i], momentum=_BN_MOMENTUM),
                        nn.LeakyReLU(inplace=True)))
                else:
                    transition_layers.append(nn.Identity())
            else:
                conv3x3s = []
                for j in range(i + 1 - num_branches_pre):
                    inchannels = num_channels_pre_layer[-1]
                    outchannels = num_channels_cur_layer[i] if j == i - num_branches_pre else inchannels
                    conv3x3s.append(nn.Sequential(
                        nn.Conv2d(inchannels, outchannels, 3, 2, 1, bias=False),
                        nn.InstanceNorm2d(outchannels, momentum=_BN_MOMENTUM),
                        nn.LeakyReLU(inplace=True)))
                transition_layers.append(nn.Sequential(*conv3x3s))

        return nn.ModuleList(transition_layers)

    def _make_layer(self, block, inplanes, planes, blocks, stride=1):
        downsample = nn.Identity()
        if stride != 1 or inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.InstanceNorm2d(planes * block.expansion, momentum=_BN_MOMENTUM),
            )

        layers = [block(inplanes, planes, stride, downsample)]
        inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(inplanes, planes))

        return nn.Sequential(*layers)

    def _make_stage(self, layer_config, num_inchannels, multi_scale_output=True):
        num_modules = layer_config['NUM_MODULES']
        num_branches = layer_config['NUM_BRANCHES']
        num_blocks = layer_config['NUM_BLOCKS']
        num_channels = layer_config['NUM_CHANNELS']
        block = blocks_dict[layer_config['BLOCK']]
        fuse_method = layer_config['FUSE_METHOD']

        modules = []
        for i in range(num_modules):
            # multi_scale_output is only used last module
            reset_multi_scale_output = multi_scale_output or i < num_modules - 1
            modules.append(HighResolutionModule(
                num_branches, block, num_blocks, num_inchannels, num_channels, fuse_method, reset_multi_scale_output)
            )
            num_inchannels = modules[-1].get_num_inchannels()

        return nn.Sequential(*modules), num_inchannels

    def stages(self, x) -> List[torch.Tensor]:
        x = self.layer1(x)

        yl = self.stage2([t(x) for i, t in enumerate(self.transition1)])

        xl = [t(yl[i if i < self.stage2_cfg['NUM_BRANCHES'] else -1]) for i, t in enumerate(self.transition2)]
        yl = self.stage3(xl)

        xl = [t(yl[i if i < self.stage3_cfg['NUM_BRANCHES'] else -1]) for i, t in enumerate(self.transition3)]
        yl = self.stage4(xl)
        return yl

    def forward(self, x):
        orig_x = x

        # Stem
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.act2(x)

        # Stages
        yl = self.stages(x)
        seg = self.head([orig_x] + yl)

        y = self.incre_modules[0](yl[0])
        for i in range(len(self.downsamp_modules)):
            y = self.downsamp_modules[i](y)
            if i + 1 < len(yl):
                y = self.incre_modules[i + 1](yl[i + 1])
        features = self.final_layer(y)
        features = self.to_feature(self.feature_mapper(features.mean((2, 3))))
        features = F.normalize(features, dim=1)

        return features, seg


def _create_hrnet(variant, pretrained, **model_kwargs):
    model = HighResolutionNet(cfg=cfg_cls[variant], **model_kwargs)

    if pretrained:
        pretrained_url = model_urls[variant].get('url', None)
        if not pretrained_url:
            _logger.warning("No pretrained weights exist for this model. Using random initialization.")
            return

        _logger.info(f'Loading pretrained weights from url ({pretrained_url})')
        state_dict = load_state_dict_from_url(pretrained_url, map_location='cpu')

        model.load_state_dict(state_dict, strict=False)

    return model


@DISCRIMINATORS.register_model
def hrnet_w18_small(pretrained=True, **kwargs):
    return _create_hrnet('hrnet_w18_small', pretrained, **kwargs)


@DISCRIMINATORS.register_model
def hrnet_w18_small_v2(pretrained=False, **kwargs):
    return _create_hrnet('hrnet_w18_small_v2', pretrained, **kwargs)


@DISCRIMINATORS.register_model
def hrnet_w18(pretrained=False, **kwargs):
    return _create_hrnet('hrnet_w18', pretrained, **kwargs)


@DISCRIMINATORS.register_model
def hrnet_w30(pretrained=False, **kwargs):
    return _create_hrnet('hrnet_w30', pretrained, **kwargs)


@DISCRIMINATORS.register_model
def hrnet_w32(pretrained=True, **kwargs):
    return _create_hrnet('hrnet_w32', pretrained, **kwargs)


@DISCRIMINATORS.register_model
def hrnet_w40(pretrained=True, **kwargs):
    return _create_hrnet('hrnet_w40', pretrained, **kwargs)


@DISCRIMINATORS.register_model
def hrnet_w44(pretrained=True, **kwargs):
    return _create_hrnet('hrnet_w44', pretrained, **kwargs)


@DISCRIMINATORS.register_model
def hrnet_w48(pretrained=True, **kwargs):
    return _create_hrnet('hrnet_w48', pretrained, **kwargs)


@DISCRIMINATORS.register_model
def hrnet_w64(pretrained=True, **kwargs):
    return _create_hrnet('hrnet_w64', pretrained, **kwargs)
