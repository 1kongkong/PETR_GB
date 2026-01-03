# ------------------------------------------------------------------------
# Copyright (c) 2022 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from DETR3D (https://github.com/WangYueFt/detr3d)
# Copyright (c) 2021 Wang, Yue
# ------------------------------------------------------------------------
# Modified from mmdetection3d (https://github.com/open-mmlab/mmdetection3d)
# Copyright (c) OpenMMLab. All rights reserved.
# ------------------------------------------------------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import Conv2d, Linear
from mmcv.cnn.bricks.transformer import build_positional_encoding
from mmcv.runner import auto_fp16
from mmdet.models.utils import build_transformer
from mmdet.models.utils.transformer import inverse_sigmoid
import numpy as np
import math
from mmdet3d.models import MIDDLE_ENCODERS


def pos2posemb2d(pos, num_pos_feats=128, temperature=10000):
    """Convert 2D position coordinates to positional encoding.
    
    Args:
        pos: Position coordinates with shape [..., 2] (x, y)
        num_pos_feats: Number of positional feature dimensions
        temperature: Temperature parameter for positional encoding
    Returns:
        posemb: Positional encoding with shape [..., num_pos_feats * 2]
    """
    scale = 2 * math.pi
    pos = pos * scale
    dim_t = torch.arange(num_pos_feats, dtype=torch.float32, device=pos.device)
    dim_t = temperature ** (2 * (dim_t // 2) / num_pos_feats)
    pos_x = pos[..., 0, None] / dim_t
    pos_y = pos[..., 1, None] / dim_t
    pos_x = torch.stack((pos_x[..., 0::2].sin(), pos_x[..., 1::2].cos()), dim=-1).flatten(-2)
    pos_y = torch.stack((pos_y[..., 0::2].sin(), pos_y[..., 1::2].cos()), dim=-1).flatten(-2)
    posemb = torch.cat((pos_y, pos_x), dim=-1)
    return posemb


class SELayer(nn.Module):
    """Squeeze-and-Excitation attention layer."""
    def __init__(self, channels, act_layer=nn.ReLU, gate_layer=nn.Sigmoid):
        super().__init__()
        self.conv_reduce = nn.Conv2d(channels, channels, 1, bias=True)
        self.act1 = act_layer()
        self.conv_expand = nn.Conv2d(channels, channels, 1, bias=True)
        self.gate = gate_layer()

    def forward(self, x, x_se):
        x_se = self.conv_reduce(x_se)
        x_se = self.act1(x_se)
        x_se = self.conv_expand(x_se)
        return x * self.gate(x_se)


@MIDDLE_ENCODERS.register_module()
class PETR_ViewTrans(nn.Module):
    """PETR View Transformation Middle Encoder.
    
    This module transforms multi-view image features into BEV (Bird's Eye View)
    feature map using PETR-style position encoding and Transformer.
    
    Args:
        in_channels (int): Number of channels in the input feature map.
        embed_dims (int): Embedding dimensions. Default: 256.
        bev_h_query (int): Height of BEV query grid. Default: 25.
        bev_w_query (int): Width of BEV query grid. Default: 25.
        bev_h (int): Height of output BEV feature map. Default: 200.
        bev_w (int): Width of output BEV feature map. Default: 200.
        transformer (dict): Config for transformer.
        positional_encoding (dict): Config for position encoding.
        with_position (bool): Whether to use 3D position encoding. Default: True.
        with_multiview (bool): Whether to use multi-view position encoding. Default: False.
        with_se (bool): Whether to use SE attention. Default: False.
        with_time (bool): Whether to use time information. Default: False.
        with_detach (bool): Whether to detach past frames. Default: False.
        depth_step (float): Depth sampling step. Default: 0.8.
        depth_num (int): Number of depth bins. Default: 64.
        LID (bool): Whether to use linear increasing depth sampling. Default: False.
        depth_start (float): Start depth. Default: 1.0.
        position_level (int): Feature level for position encoding. Default: 0.
        position_range (list): 3D position range [x_min, y_min, z_min, x_max, y_max, z_max].
            Default: [-65, -65, -8.0, 65, 65, 8.0].
    """
    def __init__(self,
                 in_channels,
                 embed_dims=256,
                 bev_h_query=25,
                 bev_w_query=25,
                 bev_h=200,
                 bev_w=200,
                 transformer=None,
                 positional_encoding=dict(
                     type='SinePositionalEncoding',
                     num_feats=128,
                     normalize=True),
                 with_position=True,
                 with_multiview=False,
                 with_se=False,
                 with_time=False,
                 with_detach=False,
                 depth_step=0.8,
                 depth_num=64,
                 LID=False,
                 depth_start=1.0,
                 position_level=0,
                 position_range=[-65, -65, -8.0, 65, 65, 8.0],
                 init_cfg=None):
        super().__init__()
        
        self.in_channels = in_channels
        self.embed_dims = embed_dims
        self.bev_h_query = bev_h_query
        self.bev_w_query = bev_w_query
        self.bev_h = bev_h
        self.bev_w = bev_w
        self.depth_step = depth_step
        self.depth_num = depth_num
        self.position_dim = 3 * self.depth_num
        self.position_range = position_range
        self.LID = LID
        self.depth_start = depth_start
        self.position_level = position_level
        self.with_position = with_position
        self.with_multiview = with_multiview
        self.with_se = with_se
        self.with_time = with_time
        self.with_detach = with_detach
        self.fp16_enabled = False
        
        # Validate positional encoding config
        assert 'num_feats' in positional_encoding
        num_feats = positional_encoding['num_feats']
        assert num_feats * 2 == self.embed_dims, 'embed_dims should' \
            f' be exactly 2 times of num_feats. Found {self.embed_dims}' \
            f' and {num_feats}.'
        
        # Build components
        self.positional_encoding = build_positional_encoding(positional_encoding)
        self.transformer = build_transformer(transformer)
        
        self._init_layers()
        
    def _init_layers(self):
        """Initialize layers."""
        # Input projection
        self.input_proj = Conv2d(
            self.in_channels, self.embed_dims, kernel_size=1)
        
        # Position encoding adapters
        if self.with_multiview:
            self.adapt_pos3d = nn.Sequential(
                nn.Conv2d(self.embed_dims*3//2, self.embed_dims*4, kernel_size=1, stride=1, padding=0),
                nn.ReLU(),
                nn.Conv2d(self.embed_dims*4, self.embed_dims, kernel_size=1, stride=1, padding=0),
            )
        else:
            self.adapt_pos3d = nn.Sequential(
                nn.Conv2d(self.embed_dims, self.embed_dims, kernel_size=1, stride=1, padding=0),
                nn.ReLU(),
                nn.Conv2d(self.embed_dims, self.embed_dims, kernel_size=1, stride=1, padding=0),
            )
        
        # 3D position encoder
        if self.with_position:
            self.position_encoder = nn.Sequential(
                nn.Conv2d(self.position_dim, self.embed_dims*4, kernel_size=1, stride=1, padding=0),
                nn.ReLU(),
                nn.Conv2d(self.embed_dims*4, self.embed_dims, kernel_size=1, stride=1, padding=0),
            )
        
        # SE attention layer
        if self.with_se:
            self.se = SELayer(self.embed_dims)
        
        # BEV query points (regular grid) - independent x and y dimensions
        x = (torch.arange(self.bev_w_query) + 0.5) / self.bev_w_query
        y = (torch.arange(self.bev_h_query) + 0.5) / self.bev_h_query
        xy = torch.meshgrid(x, y)  # Returns (x_grid, y_grid) with shape (H, W)
        # xy[0] has shape (bev_h_query, bev_w_query), values are x coordinates
        # xy[1] has shape (bev_h_query, bev_w_query), values are y coordinates
        self.register_buffer('reference_points_bev', 
                           torch.cat([xy[0].reshape(-1)[..., None], 
                                     xy[1].reshape(-1)[..., None]], -1))
        
        # Query embedding
        self.query_embedding = nn.Sequential(
            nn.Linear(self.embed_dims, self.embed_dims),
            nn.ReLU(),
            nn.Linear(self.embed_dims, self.embed_dims),
        )
        
        # Build multi-stage upsampling layers (4x upsampling each stage: 2x2)
        self.upsample_bev = self._build_upsample_layers()
    
    def _build_upsample_layers(self):
        """Build multi-stage upsampling layers.
        
        Each stage upsamples by 2x2 (4x area increase).
        Calculates how many stages are needed to reach target resolution.
        """
        if self.bev_h == self.bev_h_query and self.bev_w == self.bev_w_query:
            return None
        
        # Calculate how many 2x upsampling stages are needed for height and width
        h_current, w_current = self.bev_h_query, self.bev_w_query
        h_target, w_target = self.bev_h, self.bev_w
        
        # Calculate number of stages needed for each dimension
        h_stages = 0
        w_stages = 0
        while h_current < h_target:
            h_stages += 1
            h_current *= 2
        while w_current < w_target:
            w_stages += 1
            w_current *= 2
        
        # Use the maximum number of stages needed
        num_stages = max(h_stages, w_stages)
        
        if num_stages == 0:
            return None
        
        # Build upsampling stages
        layers = []
        h_curr, w_curr = self.bev_h_query, self.bev_w_query
        
        for stage in range(num_stages):
            # Determine target size for this stage
            # For the last stage, use the target resolution
            if stage == num_stages - 1:
                h_next = self.bev_h
                w_next = self.bev_w
            else:
                # For intermediate stages, upsample by 2x
                h_next = min(h_curr * 2, self.bev_h)
                w_next = min(w_curr * 2, self.bev_w)
            
            # Create upsampling layer for this stage
            layers.append(nn.Conv2d(self.embed_dims, self.embed_dims, 3, padding=1, bias=False))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Upsample(size=(h_next, w_next), mode='bilinear', align_corners=True))
            
            h_curr, w_curr = h_next, w_next
        
        return nn.Sequential(*layers)
    
    def init_weights(self):
        """Initialize weights."""
        self.transformer.init_weights()
    
    def position_embeding(self, img_feats, img_metas, masks=None):
        """Generate 3D position encoding.
        
        Args:
            img_feats: Multi-level image features
            img_metas: Image meta information
            masks: Image masks
        Returns:
            coords_position_embeding: Position encoding [B, N, embed_dims, H, W]
            coords_mask: Coordinate mask
        """
        eps = 1e-5
        pad_h, pad_w, _ = img_metas[0]['pad_shape'][0]
        
        B, N, C, H, W = img_feats[self.position_level].shape
        coords_h = torch.arange(H, device=img_feats[0].device).float() * pad_h / H
        coords_w = torch.arange(W, device=img_feats[0].device).float() * pad_w / W
        
        # Depth sampling
        if self.LID:
            index = torch.arange(start=0, end=self.depth_num, step=1, device=img_feats[0].device).float()
            index_1 = index + 1
            bin_size = (self.position_range[3] - self.depth_start) / (self.depth_num * (1 + self.depth_num))
            coords_d = self.depth_start + bin_size * index * index_1
        else:
            index = torch.arange(start=0, end=self.depth_num, step=1, device=img_feats[0].device).float()
            bin_size = (self.position_range[3] - self.depth_start) / self.depth_num
            coords_d = self.depth_start + bin_size * index
        
        D = coords_d.shape[0]
        coords = torch.stack(torch.meshgrid([coords_w, coords_h, coords_d])).permute(1, 2, 3, 0)  # W, H, D, 3
        coords = torch.cat((coords, torch.ones_like(coords[..., :1])), -1)
        coords[..., :2] = coords[..., :2] * torch.maximum(coords[..., 2:3], torch.ones_like(coords[..., 2:3])*eps)
        
        # Transform to 3D world coordinates
        img2lidars = []
        for img_meta in img_metas:
            img2lidar = []
            for i in range(len(img_meta['lidar2img'])):
                img2lidar.append(np.linalg.inv(img_meta['lidar2img'][i]))
            img2lidars.append(np.asarray(img2lidar))
        img2lidars = np.asarray(img2lidars)
        img2lidars = coords.new_tensor(img2lidars)  # (B, N, 4, 4)
        
        coords = coords.view(1, 1, W, H, D, 4, 1).repeat(B, N, 1, 1, 1, 1, 1)
        img2lidars = img2lidars.view(B, N, 1, 1, 1, 4, 4).repeat(1, 1, W, H, D, 1, 1)
        coords3d = torch.matmul(img2lidars, coords).squeeze(-1)[..., :3]
        
        # Normalize to [0, 1]
        coords3d[..., 0:1] = (coords3d[..., 0:1] - self.position_range[0]) / (self.position_range[3] - self.position_range[0])
        coords3d[..., 1:2] = (coords3d[..., 1:2] - self.position_range[1]) / (self.position_range[4] - self.position_range[1])
        coords3d[..., 2:3] = (coords3d[..., 2:3] - self.position_range[2]) / (self.position_range[5] - self.position_range[2])
        
        # Generate mask
        coords_mask = (coords3d > 1.0) | (coords3d < 0.0)
        coords_mask = coords_mask.flatten(-2).sum(-1) > (D * 0.5)
        coords_mask = masks | coords_mask.permute(0, 1, 3, 2)
        
        # Encode position
        coords3d = coords3d.permute(0, 1, 4, 5, 3, 2).contiguous().view(B*N, -1, H, W)
        coords3d = inverse_sigmoid(coords3d)
        coords_position_embeding = self.position_encoder(coords3d)
        
        return coords_position_embeding.view(B, N, self.embed_dims, H, W), coords_mask
    
    @auto_fp16(apply_to=('mlvl_feats',))
    def forward(self, mlvl_feats, img_metas):
        """Forward function.
        
        Args:
            mlvl_feats (tuple[Tensor]): Features from the upstream network,
                each is a 5D-tensor with shape (B, N, C, H, W).
            img_metas (list[dict]): Image meta information.
        Returns:
            bev_feature (Tensor): BEV feature map with shape (B, C, H, W).
        """
        x = mlvl_feats[self.position_level]
        batch_size, num_cams = x.size(0), x.size(1)
        
        # Optional: detach past frames
        if self.with_detach:
            current_frame = x[:, :6]
            past_frame = x[:, 6:]
            x = torch.cat([current_frame, past_frame.detach()], 1)
        
        # Generate masks
        input_img_h, input_img_w, _ = img_metas[0]['pad_shape'][0]
        masks = x.new_ones((batch_size, num_cams, input_img_h, input_img_w))
        for img_id in range(batch_size):
            for cam_id in range(num_cams):
                img_h, img_w, _ = img_metas[img_id]['img_shape'][cam_id]
                masks[img_id, cam_id, :img_h, :img_w] = 0
        
        # Project input features
        x = self.input_proj(x.flatten(0, 1))
        x = x.view(batch_size, num_cams, *x.shape[-3:])
        
        # Interpolate masks to match feature spatial size
        masks = F.interpolate(masks, size=x.shape[-2:]).to(torch.bool)
        
        # Generate position encoding
        if self.with_position:
            coords_position_embeding, _ = self.position_embeding(mlvl_feats, img_metas, masks)
            
            if self.with_se:
                coords_position_embeding = self.se(
                    coords_position_embeding.flatten(0, 1), 
                    x.flatten(0, 1)
                ).view(x.size())
            
            pos_embed = coords_position_embeding
            
            if self.with_multiview:
                sin_embed = self.positional_encoding(masks)
                sin_embed = self.adapt_pos3d(sin_embed.flatten(0, 1)).view(x.size())
                pos_embed = pos_embed + sin_embed
            else:
                pos_embeds = []
                for i in range(num_cams):
                    xy_embed = self.positional_encoding(masks[:, i, :, :])
                    pos_embeds.append(xy_embed.unsqueeze(1))
                sin_embed = torch.cat(pos_embeds, 1)
                sin_embed = self.adapt_pos3d(sin_embed.flatten(0, 1)).view(x.size())
                pos_embed = pos_embed + sin_embed
        else:
            if self.with_multiview:
                pos_embed = self.positional_encoding(masks)
                pos_embed = self.adapt_pos3d(pos_embed.flatten(0, 1)).view(x.size())
            else:
                pos_embeds = []
                for i in range(num_cams):
                    pos_embed = self.positional_encoding(masks[:, i, :, :])
                    pos_embeds.append(pos_embed.unsqueeze(1))
                pos_embed = torch.cat(pos_embeds, 1)
        
        # Generate BEV queries
        query_bev = self.query_embedding(pos2posemb2d(self.reference_points_bev))
        
        # Transformer forward
        # Use the last decoder layer output
        outs_dec, _ = self.transformer(x, masks, query_bev, pos_embed, reg_branch=None)
        outs_dec = torch.nan_to_num(outs_dec)
        
        # Take the last decoder layer output
        # outs_dec shape: [num_layers, B, num_query, embed_dims]
        bev_queries = outs_dec[-1]  # [B, num_query, embed_dims]
        
        # Reshape to spatial feature map
        # bev_queries shape: [B, num_query, embed_dims]
        num_query = bev_queries.shape[1]
        assert num_query == self.bev_h_query * self.bev_w_query, \
            f'Number of queries ({num_query}) must equal bev_h_query * bev_w_query ({self.bev_h_query * self.bev_w_query})'
        
        bev_feature = bev_queries.view(batch_size, self.bev_h_query, self.bev_w_query, self.embed_dims)
        bev_feature = bev_feature.permute(0, 3, 1, 2)  # [B, C, H_query, W_query]
        
        # Optional: upsample to target BEV resolution
        if self.upsample_bev is not None:
            bev_feature = self.upsample_bev(bev_feature)
        
        return bev_feature
