# ------------------------------------------------------------------------
# Copyright (c) 2022 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from DETR3D (https://github.com/WangYueFt/detr3d)
# Copyright (c) 2021 Wang, Yue
# ------------------------------------------------------------------------
from .vovnet import VoVNet
from .vovnetcp import VoVNetCP
from .res2d_dilation import RES2Ddilation
from .resnet import ResNetS64
__all__ = ['VoVNet', 'VoVNetCP', 'RES2Ddilation', 'ResNetS64']

