from torch import nn
from mmcv.cnn import build_conv_layer, build_norm_layer, build_upsample_layer

from mmdet.models import NECKS


@NECKS.register_module()
class TUMFPNLEVEL2RED8(nn.Module):
    """TUM FPN Neck with 2-level features.

    Args:
        conv_cfg (dict): Config dict of convolutional layers.
        norm_cfg (dict): Config dict of normalization layers
        upsample_cfg (dict): Config dict of upsample layers in u-net
    """

    def __init__(
        self,
        in_channel_0=96,
        in_channel_1_2=128,
        out_channel=256,
        f2_shrink=False,
        cfg={},
        conv_cfg=dict(type="Conv2d", bias=False),
        norm_cfg=dict(type="BN", eps=1e-5, momentum=0.01),
        upsample_cfg=dict(type="nearest", scale_factor=2),
    ):
        super(TUMFPNLEVEL2RED8, self).__init__()
        self.f2_shrink = f2_shrink
        self.nearest_up_sampling = build_upsample_layer(upsample_cfg)
        self.cfg = cfg
        
        block_connect_1 = [
            build_conv_layer(
                conv_cfg, in_channel_1_2, in_channel_1_2, 3, stride=1, padding=1
            ),
            build_norm_layer(norm_cfg, in_channel_1_2)[1],
            nn.ReLU(inplace=True),
        ]
        self.block_connect_1 = nn.Sequential(*block_connect_1)

        # output features block
        block_connect_d4o = [
            build_conv_layer(
                conv_cfg, in_channel_1_2, out_channel, 3, stride=1, padding=1
            ),
            build_norm_layer(norm_cfg, out_channel)[1],
            nn.ReLU(inplace=True),
        ]
        self.block_connect_d4o = nn.Sequential(*block_connect_d4o)

        block_connect_d2o = [
            build_conv_layer(
                conv_cfg, in_channel_1_2, out_channel, 3, stride=1, padding=1
            ),
            build_norm_layer(norm_cfg, out_channel)[1],
            nn.ReLU(inplace=True),
        ]
        self.block_connect_d2o = nn.Sequential(*block_connect_d2o)

        block_connect_d8o = [
            build_conv_layer(
                conv_cfg, in_channel_1_2, out_channel, 3, stride=1, padding=1
            ),
            build_norm_layer(norm_cfg, out_channel)[1],
            nn.ReLU(inplace=True),
        ]
        self.block_connect_d8o = nn.Sequential(*block_connect_d8o)

        proj_block_81 = [
            build_conv_layer(
                conv_cfg, in_channel_1_2, in_channel_1_2, 1, stride=1, padding=0
            ),
            build_norm_layer(norm_cfg, in_channel_1_2)[1],
        ]
        self.proj_block_81 = nn.Sequential(*proj_block_81)

        proj_block_41 = [
            build_conv_layer(
                conv_cfg, in_channel_0, in_channel_1_2, 1, stride=1, padding=0
            ),
            build_norm_layer(norm_cfg, in_channel_1_2)[1],
        ]
        self.proj_block_41 = nn.Sequential(*proj_block_41)

    
    def forward(self, features):
        """Forward function.

        Args:
            features (List[torch.Tensor]): Input with shape [(N, C, H, W)].

        Returns:
            tuple[torch.Tensor]: Multi-scale features.
        """

        x = self.nearest_up_sampling(features[-1])

        x += self.proj_block_81(features[-2])
        d8_x = self.block_connect_1(x)

        x = self.nearest_up_sampling(d8_x)

        x += self.proj_block_41(features[-3])

        x_d8 = self.block_connect_d8o(d8_x)
        x_d4 = self.block_connect_d4o(x)

        xx = self.nearest_up_sampling(x)

        x_d2 = self.block_connect_d2o(xx)
        return [x_d4, x_d2, x_d8]
