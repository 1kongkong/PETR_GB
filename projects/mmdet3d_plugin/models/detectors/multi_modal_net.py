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
from torch.nn import ModuleDict
import mmcv
import numpy as np
from mmcv.parallel import DataContainer as DC
from os import path as osp
from mmcv.runner import force_fp32, auto_fp16
from mmdet.models import DETECTORS
from mmdet3d.core import bbox3d2result
from mmdet3d.core import (CameraInstance3DBoxes,LiDARInstance3DBoxes, bbox3d2result,
                          show_multi_modality_result)
from mmdet3d.models.detectors.mvx_two_stage import MVXTwoStageDetector
from mmdet3d.models.detectors.base import Base3DDetector
from projects.mmdet3d_plugin.models.utils.grid_mask import GridMask
from projects.mmdet3d_plugin.models.utils.bin_voxelization import BinaryVoxelization
import cv2
import copy
# from .dct import ProcessorDCT
from einops import rearrange
from mmdet3d.models import builder


def IOU (intputs, targets, eps=1e-6):
    intputs = intputs.bool()
    targets = targets.bool()
    inter = (intputs & targets).sum(-1)
    union = (intputs | targets).sum(-1)
    # iou = (numerator + eps) / (denominator + eps - numerator)
    return inter.cpu(),union.cpu()

@DETECTORS.register_module()
class MultiModalNet(Base3DDetector):
    """
                   -----------               ---------------
    Task_A data -- |         | -- feature -- | Task_A head | -- Task_A pred
                   |         |               ---------------
    Task_B data -- | encoder | -- feature -- | Task_B head | -- Task_B pred
                   |         |               ---------------
    Task_C data -- |         | -- feature -- | Tash_C head | -- Task_C pred
                   -----------               ---------------
    """
    def __init__(self,
                 use_grid_mask=False,
                 pts_voxel_layer=None,
                 pts_backbone=None,
                 pts_neck=None,
                 img_backbone=None,
                 img_view_trans=None,
                 fusion_layer=None,
                 heads_cfg=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 ):
        super(MultiModalNet, self).__init__()
        
        self.pts_voxel_layer = None
        self.pts_backbone = None
        self.pts_neck = None
        self.img_backbone = None
        self.img_view_trans = None
        self.fusion_layer = None
        
        if pts_voxel_layer:
            self.pts_voxel_layer = BinaryVoxelization(**pts_voxel_layer)
        
        if pts_backbone:
            self.pts_backbone = builder.build_backbone(pts_backbone)
        if pts_neck is not None:
            self.pts_neck = builder.build_neck(pts_neck)

        if img_backbone:
            self.img_backbone = builder.build_backbone(img_backbone)
        if img_view_trans:
            self.img_view_trans = builder.build_middle_encoder(img_view_trans)
        
        if fusion_layer:
            self.fusion_layer = builder.build_fusion_layer(fusion_layer)
        
        self.heads = ModuleDict()
        if heads_cfg:
            for task_name in heads_cfg.keys():
                self.heads[task_name] = builder.build_head(heads_cfg[task_name])

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        if pretrained is None:
            img_pretrained = None
            pts_pretrained = None
        elif isinstance(pretrained, dict):
            img_pretrained = pretrained.get('img', None)
            pts_pretrained = pretrained.get('pts', None)
        else:
            raise ValueError(
                f'pretrained should be a dict, got {type(pretrained)}')

        self.grid_mask = GridMask(True, True, rotate=1, offset=False, ratio=0.5, mode=1, prob=0.7)
        self.use_grid_mask = use_grid_mask

    @property
    def with_img_neck(self):
        """bool: Whether the detector has a neck in image branch."""
        return hasattr(self, 'img_neck') and self.img_neck is not None

    @property
    def with_pts_neck(self):
        """bool: Whether the detector has a neck in pts branch."""
        return hasattr(self, 'pts_neck') and self.pts_neck is not None

    def extract_img_feat(self, img, img_metas):
        """Extract features of images."""
        if img is None:
            return None
            
        if isinstance(img, list):
            img = torch.stack(img, dim=0)

        B = img.size(0)
        if img is not None:
            input_shape = img.shape[-2:]
            # update real input shape of each single img
            for img_meta in img_metas:
                img_meta.update(input_shape=input_shape)
            if img.dim() == 5:
                if img.size(0) == 1 and img.size(1) != 1:
                    img.squeeze_()
                else:
                    B, N, C, H, W = img.size()
                    img = img.view(B * N, C, H, W)
            if self.use_grid_mask:
                img = self.grid_mask(img)
            img_feats = self.img_backbone(img)
            if isinstance(img_feats, dict):
                img_feats = list(img_feats.values())
        else:
            return None
        if self.with_img_neck:
            img_feats = self.img_neck(img_feats)
        img_feats_reshaped = []
        for img_feat in img_feats:
            BN, C, H, W = img_feat.size()
            img_feats_reshaped.append(img_feat.view(B, int(BN / B), C, H, W))
        
        img_bev_feats = self.img_view_trans(img_feats_reshaped, img_metas)

        return img_bev_feats

    @auto_fp16(apply_to=('img', 'pts'), out_fp32=True)
    def extract_feat(self, img, pts, img_metas):
        """Extract features from images and points."""
        img_feats, pts_feats = None, None
        if img is not None:
            img_feats = self.extract_img_feat(img, img_metas)
        if pts is not None:
            pts_feats = self.extract_pts_feat(pts)
        return img_feats, pts_feats

    def forward_pts_train(self,
                          pts_feats,
                          gt_bboxes_3d,
                          gt_labels_3d,
                          maps,
                          img_metas,
                          gt_bboxes_ignore=None):
        """Forward function for point cloud branch.
        Args:
            pts_feats (list[torch.Tensor]): Features of point cloud branch
            gt_bboxes_3d (list[:obj:`BaseInstance3DBoxes`]): Ground truth
                boxes for each sample.
            gt_labels_3d (list[torch.Tensor]): Ground truth labels for
                boxes of each sampole
            img_metas (list[dict]): Meta information of samples.
            gt_bboxes_ignore (list[torch.Tensor], optional): Ground truth
                boxes to be ignored. Defaults to None.
        Returns:
            dict: Losses of each branch.
        """
        outs = self.pts_bbox_head(pts_feats, img_metas)
        
        loss_inputs = [gt_bboxes_3d, gt_labels_3d, outs,maps]
        losses = self.pts_bbox_head.loss(*loss_inputs)

        return losses

    def create_fake_data(self):
        """创建伪造的输入数据和真值数据"""
        
        # 基本参数
        batch_size = 1
        num_cams = 6  # 6个相机
        img_h, img_w = 640, 1600  # 图像尺寸
        num_points = 10000  # 点云数量
        bev_h, bev_w = 80, 256  # BEV特征图尺寸 (修正为正确的尺寸)
        
        # 1. 创建图像数据 [B, N, C, H, W]
        img = torch.randn(batch_size, num_cams, 3, img_h, img_w).cuda()
        
        # 2. 创建点云数据 [B, N, 5] (x, y, z, intensity, timestamp)
        # 确保点云在正确的范围内：[-10.0, -14.4, -1.0, 82.16, 14.4, 7.0]
        pts = torch.zeros(batch_size, num_points, 5).cuda()
        pts[:, :, 0] = torch.rand(batch_size, num_points).cuda() * (82.16 - (-10.0)) + (-10.0)  # x
        pts[:, :, 1] = torch.rand(batch_size, num_points).cuda() * (14.4 - (-14.4)) + (-14.4)   # y
        pts[:, :, 2] = torch.rand(batch_size, num_points).cuda() * (7.0 - (-1.0)) + (-1.0)      # z
        pts[:, :, 3] = torch.rand(batch_size, num_points).cuda()  # intensity
        pts[:, :, 4] = torch.rand(batch_size, num_points).cuda()  # timestamp
        
        # 3. 创建图像元数据
        img_metas = []
        for b in range(batch_size):
            meta = {
                'filename': [f'cam_{i}.jpg' for i in range(num_cams)],
                'ori_shape': [(img_h, img_w, 3) for _ in range(num_cams)],
                'img_shape': [(img_h, img_w, 3) for _ in range(num_cams)],
                'pad_shape': [(img_h, img_w, 3) for _ in range(num_cams)],
                'scale_factor': [1.0 for _ in range(num_cams)],
                'flip': [False for _ in range(num_cams)],
                'box_mode_3d': 'LiDAR',
                'box_type_3d': 'LiDAR',
                'img_norm_cfg': {
                    'mean': [103.53, 116.28, 123.675],
                    'std': [57.375, 57.12, 58.395],
                    'to_rgb': False
                },
                'sample_idx': f'sample_{b}',
                'timestamp': 1234567890.0,
                'data_source': 'GB',  # 重要：匹配compute_type
                # 相机内参和外参
                'lidar2img': [np.eye(4).astype(np.float32) for _ in range(num_cams)],
                'intrinsics': [np.eye(3).astype(np.float32) for _ in range(num_cams)],
                'extrinsics': [np.eye(4).astype(np.float32) for _ in range(num_cams)],
                'bda': np.eye(4).astype(np.float32)
            }
            img_metas.append(meta)
        
        # 4. 创建真值数据 - maps字典
        maps = {
            'gb_roadmap_gt': torch.randint(0, 2, (batch_size, 1, bev_h, bev_w)).long().cuda(),  # 道路图真值 [B, 1, H, W] - FocalLoss需要long类型
            'gb_offset_gt': torch.randn(batch_size, 2, bev_h, bev_w).cuda() * 0.1,  # 偏移真值 [B, 2, H, W] - 缩小数值范围
        }
        
        # 5. 创建3D边界框真值（虽然这个模型可能不用，但保持兼容性）
        gt_bboxes_3d = [torch.randn(5, 7).cuda() for _ in range(batch_size)]  # 每个batch 5个框，7个参数
        gt_labels_3d = [torch.randint(0, 10, (5,)).cuda() for _ in range(batch_size)]  # 对应的标签
        
        return {
            'img': img,
            'pts': pts,
            'img_metas': img_metas,
            'maps': maps,
            'gt_bboxes_3d': gt_bboxes_3d,
            'gt_labels_3d': gt_labels_3d
        }
        
    def create_one_fake_data(self):
        if self.data is None:
            self.data = self.create_fake_data()
        return self.data

    # @force_fp32(apply_to=('img', 'points'))
    def forward(self, return_loss=True, **kwargs):
        """Calls either forward_train or forward_test depending on whether
        return_loss=True.
        Note this setting will change the expected inputs. When
        `return_loss=True`, img and img_metas are single-nested (i.e.
        torch.Tensor and list[dict]), and when `resturn_loss=False`, img and
        img_metas should be double nested (i.e.  list[torch.Tensor],
        list[list[dict]]), with the outer list indicating test time
        augmentations.
        """
        ## hack data 
        # data = self.create_fake_data()
        # kwargs.update(data)


        if return_loss:
            return self.forward_train(**kwargs)
        else:
            return self.forward_test(**kwargs)

    def forward_train(self, img=None, pts=None, img_metas=None, gt_bboxes_3d=None,
                     gt_labels_3d=None, maps=None, **kwargs):
        """Forward training function."""
        # 提取图像和点云特征
        img_feats, pts_feats = self.extract_feat(img=img, pts=pts, img_metas=img_metas)

        # 融合特征
        if img_feats is not None and pts_feats is not None:
            if isinstance(img_feats, list):
                img_feat = img_feats[-1]
            else:
                img_feat = img_feats
            if isinstance(pts_feats, list):
                pts_feat = pts_feats[0] # 4倍降采样, 2倍降采样，8倍降采样
            else:
                pts_feat = pts_feats
            fusion_bev_feat = self.fusion_layer(img_feat, pts_feat)
        elif img_feats is not None:
            fusion_bev_feat = img_feats[-1] if isinstance(img_feats, list) else img_feats
        elif pts_feats is not None:
            fusion_bev_feat = pts_feats[-1] if isinstance(pts_feats, list) else pts_feats
        else:
            raise ValueError("Both img_feats and pts_feats are None")

        # 计算损失
        losses = dict()
        for task_name in self.heads:
            task_loss = self.heads[task_name](fusion_bev_feat,
                                            gt_bboxes_3d=gt_bboxes_3d,
                                            gt_labels_3d=gt_labels_3d,
                                            maps=maps,
                                            img_metas=img_metas)
            losses.update(task_loss)

        return losses
    
    def extract_pts_feat(self, pts):
        """Extract features from point cloud."""
        if pts is None:
            return None
        
        # 二值体素化
        pts_voxel = self.pts_voxel_layer(pts)
        
        # 使用backbone提取特征
        pts_feats = self.pts_backbone(pts_voxel)
        
        # 使用neck进一步处理特征
        if self.with_pts_neck:
            pts_feats = self.pts_neck(pts_feats)
        return pts_feats
    
    def img_show(self, imgs):
        import os
        import cv2
        import random
        import numpy as np
        mean= np.array([103.530, 116.280, 123.675])
        mean = mean.reshape(1,1,3)
        if not os.path.exists("./imgs"):
            os.makedirs("./imgs")
        name = str(random.randint(1,20))
        for i in range(imgs.size(1)):
            img = imgs[0][i]
            img = img.permute(1, 2, 0).detach().cpu().numpy()
            img = img + mean
            # print(img)

            cv2.imwrite("./imgs/"+name+"_"+str(i)+".png", img.astype(np.uint8))
            print(img.shape)

    
    def forward_test(self, img_metas,gt_map, maps,img=None, **kwargs):
        for var, name in [(img_metas, 'img_metas')]:
            if not isinstance(var, list):
                raise TypeError('{} must be a list, but got {}'.format(
                    name, type(var)))
        img = [img] if img is None else img
        return self.simple_test(img_metas[0], gt_map,img[0],maps, **kwargs)

    def simple_test_pts(self, x, img_metas, gt_map,maps,rescale=False):
        """Test function of point cloud branch."""
        outs = self.pts_bbox_head(x, img_metas)
        
        with torch.no_grad():
            
            lane_preds=outs['all_lane_preds'][5].squeeze(0)    #[B,N,H,W]
            
            n,w=lane_preds.size()
            
            f_lane=lane_preds.sigmoid()
            f_lane[f_lane>=0.43]=1
            f_lane[f_lane<0.43]=0
            f_lane_show=copy.deepcopy(f_lane).reshape(3,200,200)
            gt_map_show=copy.deepcopy(gt_map[0])
            
            f_lane=f_lane.view(3,-1)
            gt_map=gt_map[0].view(3,-1) 
            
                     
            inter,union=IOU(f_lane,gt_map)
            ret_iou=inter/union
            ret_iou=[ret_iou[0].item(),ret_iou[1].item(),ret_iou[2].item()]
            ret_ious=[inter,union]


            show_res=False
            if show_res:
            
                pres=gt_map_show[0]
                
                gt=torch.zeros(200,200,3)
                gt+=255
                label=[[71,130,255],[255,255,0],[255,144,30]]
                
                gt[...,0][pres[0]==1]=label[0][0]
                gt[...,1][pres[0]==1]=label[0][1]
                gt[...,2][pres[0]==1]=label[0][2]
                gt[...,0][pres[2]==1]=label[2][0]
                gt[...,1][pres[2]==1]=label[2][1]
                gt[...,2][pres[2]==1]=label[2][2]
                gt[...,0][pres[1]==1]=label[1][0]
                gt[...,1][pres[1]==1]=label[1][1]
                gt[...,2][pres[1]==1]=label[1][2]
                gt=gt.cpu().numpy()
                
                pres=f_lane_show
                pre=torch.zeros(200,200,3)
                pre+=255
                label=[[71,130,255],[255,255,0],[255,144,30]]
                
                pre[...,0][pres[0]==1]=label[0][0]
                pre[...,1][pres[0]==1]=label[0][1]
                pre[...,2][pres[0]==1]=label[0][2]
                pre[...,0][pres[2]==1]=label[2][0]
                pre[...,1][pres[2]==1]=label[2][1]
                pre[...,2][pres[2]==1]=label[2][2]
                pre[...,0][pres[1]==1]=label[1][0]
                pre[...,1][pres[1]==1]=label[1][1]
                pre[...,2][pres[1]==1]=label[1][2]
                pre=pre.cpu().numpy()
                imgss=np.concatenate((pre, gt),axis=1)
                
                cv2.imwrite('./res/'+img_metas[0]['filename'][0].split('/')[-1].split('.')[0]+'.png',imgss)
                
            
        return ret_ious


    
    def simple_test(self, img_metas,gt_map=None, img=None,maps=None, rescale=False):
        """Test function without augmentaiton."""
        img_feats = self.extract_feat(img=img, img_metas=img_metas)

        bbox_list = [dict() for i in range(len(img_metas))]
        ret_iou = self.simple_test_pts(
            img_feats, img_metas, gt_map,maps,rescale=rescale)
        for result_dict in bbox_list:
            result_dict['ret_iou']=ret_iou
        return bbox_list

    def aug_test_pts(self, feats, img_metas, rescale=False):
        feats_list = []
        for j in range(len(feats[0])):
            feats_list_level = []
            for i in range(len(feats)):
                feats_list_level.append(feats[i][j])
            feats_list.append(torch.stack(feats_list_level, -1).mean(-1))
        outs = self.pts_bbox_head(feats_list, img_metas)
        bbox_list = self.pts_bbox_head.get_bboxes(
            outs, img_metas, rescale=rescale)
        bbox_results = [
            bbox3d2result(bboxes, scores, labels)
            for bboxes, scores, labels in bbox_list
        ]
        return bbox_results

    def aug_test(self, img_metas, imgs=None, rescale=False):
        """Test function with augmentaiton."""
        img_feats = self.extract_feats(img_metas, imgs)
        img_metas = img_metas[0]
        bbox_list = [dict() for i in range(len(img_metas))]
        bbox_pts = self.aug_test_pts(img_feats, img_metas, rescale)
        for result_dict, pts_bbox in zip(bbox_list, bbox_pts):
            result_dict['pts_bbox'] = pts_bbox
        return bbox_list
    
    
    def show_results(self, data, result, out_dir, score_thr=0.1):
        """Results visualization.

        Args:
            data (list[dict]): Input images and the information of the sample.
            result (list[dict]): Prediction results.
            out_dir (str): Output directory of visualization result.
        """

        for batch_id in range(len(result)):
            if isinstance(data['img_metas'][0], DC):
                img_filename = data['img_metas'][0]._data[0][batch_id][
                    'filename']
                cam2img = data['img_metas'][0]._data[0][batch_id]['lidar2img']
            elif mmcv.is_list_of(data['img_metas'][0], dict):
                img_filename = data['img_metas'][0][batch_id]['filename']
                cam2img = data['img_metas'][0][batch_id]['lidar2img']
            else:
                ValueError(
                    f"Unsupported data type {type(data['img_metas'][0])} "
                    f'for visualization!')

            for i in range(len(img_filename)):
                if "once" in img_filename[i]:
                    img_path = img_filename[i].replace("/home/sunjianjian/workspace/temp/once_benchmark/data/","/data/Dataset/")
                    img = mmcv.imread(img_path)
                    
                    file_name =  img_path.split("/")[-2] + osp.split(img_path)[-1].split('.')[0]
                else:
                    img_path = img_filename[i]
                    img = mmcv.imread(img_path)
                    file_name =  osp.split(img_path)[-1].split('.')[0]
                print(file_name)
                assert out_dir is not None, 'Expect out_dir, got none.'

                pred_bboxes = result[batch_id]['pts_bbox']['boxes_3d']
                pred_scores = result[batch_id]['pts_bbox']['scores_3d']
                pred_labels = result[batch_id]['pts_bbox']['labels_3d']

                mask = pred_scores> score_thr

                pred_bboxes = pred_bboxes[mask]
                pred_scores = pred_scores[mask]
                pred_labels = pred_labels[mask]

                assert isinstance(pred_bboxes, LiDARInstance3DBoxes), \
                    f'unsupported predicted bbox type {type(pred_bboxes)}'
                

                show_multi_modality_result(
                    img,
                    None,
                    pred_bboxes,
                    cam2img[i],
                    out_dir,
                    file_name,
                    'lidar',
                    show=False)

