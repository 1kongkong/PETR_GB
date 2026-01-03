# Copyright (c) OpenMMLab. All rights reserved.
import torch
from torch import nn as nn

from mmdet3d.models import FUSION_LAYERS
import torch.nn.functional as F

@FUSION_LAYERS.register_module()
class BEVFusion(nn.Module):
    """Fuse Lidar bev features and camera bev features.

    Args:
        in_channels: int = 256
        out_channels: int = 128
    """

    def __init__(
        self,
        in_channels: int = 256,
        out_channels: int = 128,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.fusion_layer = nn.Sequential(
            nn.Conv2d(
                self.in_channels,
                self.out_channels,
                kernel_size=3,
                padding=1,
                stride=1,
                bias=False,
            ),
            nn.BatchNorm2d(self.out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, img_feats, pts_feats, pts_feat_weight=None, **kwargs):
        
        if pts_feat_weight is not None and pts_feat_weight < 0.99:
            lidar_feat_mask = torch.ones(img_feats.shape).to(pts_feats.device) * pts_feat_weight
            fusion_feat = pts_feats * lidar_feat_mask + img_feats
        else:
            fusion_feat = pts_feats + img_feats
        out_fusion_feat = self.fusion_layer(fusion_feat)

        return out_fusion_feat
    
    def forward_dummy(self, pts_feats, img_feats, **kwargs):
        if "pts_feat_weight" in kwargs.keys():
            return self.forward(pts_feats, img_feats, pts_feat_weight = kwargs["pts_feat_weight"])
        else:
            return self.forward(pts_feats, img_feats)
