from mmcv.cnn import build_conv_layer, build_norm_layer
from torch import nn

from mmdet.models.builder import BACKBONES


def same_padding(kernel_size):
    return (kernel_size - 1) // 2


@BACKBONES.register_module()
class RES2Ddilation(nn.Module):
    """2D Backbone network in msd.

    Args:
        in_channels (int): Input channels.
        out_channels (list[int]): Output channels for multi-scale feature maps.
        layer_nums (list[int]): Number of layers in each stage.
        layer_strides (list[int]): Strides of each stage.
        norm_cfg (dict): Config dict of normalization layers.
        conv_cfg (dict): Config dict of convolutional layers.
    """

    def __init__(self,
                 in_channels=128,
                 out_channels=[96, 128, 128],
                 kernel_sizes=[(3, 3), (5, 3), (5, 3)],
                 block_strides=[(1, 1), (2, 1), (2, 1)],
                 block_nums=[3, 3, 3],
                 block_dilations=[2, 4, 6],
                 cfg = {},
                 norm_cfg=dict(type='BN', eps=1e-5, momentum=0.01),
                 conv_cfg=dict(type='Conv2d', bias=False)):
        super(RES2Ddilation, self).__init__()
        assert len(out_channels) == len(block_nums)
        assert len(kernel_sizes) == len(block_nums)
        assert len(block_strides) == len(block_nums)
        assert len(block_dilations) == block_nums[-1]
        self.cfg = cfg
        blocks = []
        in_channels = [in_channels, *out_channels[:-1]]
        for i, block_num in enumerate(block_nums):
            block = [
                stride_block(in_channels[i], out_channels[i],
                             kernel_sizes[i][0], block_strides[i][0], cfg, norm_cfg,
                             conv_cfg)
            ]
            if i == len(block_nums) - 1:
                for k in range(block_num):
                    block.append(
                        res_block(out_channels[i], kernel_sizes[i][1],
                                  block_strides[i][1], block_dilations[k],
                                  cfg, norm_cfg, conv_cfg))
            else:
                for k in range(block_num):
                    block.append(
                        res_block(out_channels[i], kernel_sizes[i][1],
                                  block_strides[i][1], 1, cfg, norm_cfg, conv_cfg))
            block = nn.Sequential(*block)
            blocks.append(block)

        self.blocks = nn.ModuleList(blocks)

    def forward(self, x):
        """Forward function.

        Args:
            x (torch.Tensor): Input with shape (N, C, H, W).

        Returns:
            tuple[torch.Tensor]: Multi-scale features.
        """
        features = []
        for i, block in enumerate(self.blocks):
            x = block(x)
            features.append(x)
        return features


class stride_block(nn.Module):

    def __init__(self,
                 in_channel,
                 out_channel,
                 kernel_size=3,
                 stride=2,
                 cfg = {},
                 norm_cfg=dict(type='BN', eps=1e-5, momentum=0.01),
                 conv_cfg=dict(type='Conv2d', bias=False)):
        super(stride_block, self).__init__()

        self.block = nn.Sequential(*[
            build_conv_layer(
                conv_cfg,
                in_channel,
                out_channel,
                kernel_size=kernel_size,
                stride=stride,
                padding=same_padding(kernel_size)),
            build_norm_layer(norm_cfg, out_channel)[1],
            nn.ReLU(inplace=True)
        ])

    def forward(self, x):
        out = self.block(x)
        return out


class res_block(nn.Module):

    def __init__(self,
                 channel,
                 kernel_size=3,
                 stride=1,
                 dilation_rate=1,
                 cfg = {},
                 norm_cfg=dict(type='BN', eps=1e-5, momentum=0.01),
                 conv_cfg=dict(type='Conv2d', bias=False)):
        super(res_block, self).__init__()

        self.block1 = nn.Sequential(*[
            build_conv_layer(
                conv_cfg,
                channel,
                channel,
                kernel_size=kernel_size,
                stride=stride,
                dilation=dilation_rate,
                padding=dilation_rate),
            build_norm_layer(norm_cfg, channel)[1],
            nn.ReLU(inplace=True)
        ])

        self.block2 = nn.Sequential(*[
            build_conv_layer(
                conv_cfg,
                channel,
                channel,
                kernel_size,
                stride,
                # dilation=dilation_rate,
                padding=same_padding(kernel_size)),
            build_norm_layer(norm_cfg, channel)[1]
        ])

        self.block3 = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x
        out = self.block1(x)
        out = self.block2(out)
        out = out + identity
        out = self.block3(out)
        return out
