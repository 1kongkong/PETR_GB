import os
import copy
import json
import torch
import torch.nn.functional as F
from torch import nn
import mmcv
import cv2
import numpy as np
from mmcv.runner import get_dist_info
from mmdet.models.builder import build_loss
from mmdet.models import HEADS

def _sigmoid(x):
    y = torch.clamp(x.sigmoid_(), min=1e-4, max=1 - 1e-4)
    return y

def _sigmoid_without_clamp(x):
    y = x.sigmoid_()
    return y

@HEADS.register_module()
class KeyPointHeadGeneralBarrier(nn.Module):

    def __init__(
        self,
        train_cfg=None,
        test_cfg=None,
        num_classes=1,
        class_names=["GeneralBarrier"],
        offset_dims=2,
        feat_levels=1,
        in_channels=128,
        feat_channels=64,
        class_weight=[1.0],
        loss_hm=dict(type="FocalLoss", loss_weight=1.0),
        loss_reg=dict(type="SmoothL1Loss", beta=1.0, loss_weight=2.0),
        compute_type="GB",
        ignore_mask=False,
        add_gb_roadmap_conv=False,
        gd_map_infer_roi=[48, 24, 256, 56],
        gt_key='gb_gt_dict_roadline',
        **kwargs
    ):
        super(KeyPointHeadGeneralBarrier, self).__init__()
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.num_class = num_classes
        self.offset_dims = offset_dims
        self.in_channels = in_channels
        self.feat_channels = feat_channels
        self.compute_type = compute_type
        self.show_debug = kwargs.get('show_debug', False)
        self.ignore_mask = ignore_mask
        self.gd_map_infer_roi = gd_map_infer_roi
        self.class_names = class_names
        self.gt_key = gt_key

        if class_weight is not None:
            self.class_weight = torch.tensor(class_weight, device="cuda")
        else:
            self.class_weight = torch.ones(num_classes, device="cuda")


        # init head output channel
        self.out_channels = {
            "gb_roadmap_preds": self.num_class,
            "gb_offset_preds": self.offset_dims,
        }
    

        # roadmap configs
        hm_loss_cfg = loss_hm
        self.hm_loss_func = build_loss(hm_loss_cfg)

        # offset confis
        reg_loss_cfg = loss_reg
        self.reg_loss_func = build_loss(reg_loss_cfg)
        
        if self.compute_type == "GB":
            for head in self.out_channels:
                output_channel = self.out_channels[head]
                if head == "gb_roadmap_preds":
                    head_modules = [
                        nn.Conv2d(
                            in_channels,
                            feat_channels,
                            kernel_size=3,
                            padding=1,
                            bias=False,
                        ),
                        nn.ReLU(inplace=True),
                        nn.Conv2d(
                            feat_channels,
                            feat_channels//2,
                            kernel_size=3,
                            padding=1,
                            bias=False,
                        ),
                        nn.ReLU(inplace=True),
                            nn.Conv2d(
                            feat_channels//2,
                            feat_channels//4,
                            kernel_size=3,
                            padding=1,
                            bias=False,
                        ),
                        nn.ReLU(inplace=True),
                        nn.Conv2d(
                            feat_channels//4,
                            output_channel,
                            kernel_size=1,
                            stride=1,
                        ),
                    ]
                elif head == "gb_offset_preds":
                    head_modules = [
                        nn.Conv2d(
                            in_channels,
                            feat_channels,
                            kernel_size=3,
                            padding=1,
                            bias=False,
                        ),
                        nn.ReLU(inplace=True),
                        nn.Conv2d(
                            feat_channels,
                            output_channel,
                            kernel_size=1,
                            stride=1,
                        ),
                    ]
                else:
                    raise NotImplementedError
                fc = nn.Sequential(*head_modules)
                self.__setattr__(head, fc)

            if self.gt_key == "gb_gt_dict_roadline":
                self.loss_suffix = ""
            else:
                raise ValueError("gt_key not in ['gb_gt_dict_roadline', 'gb_far_gt_dict_roadline']")
        self.log_vars = {}

    def init_weights(self):
        """Initialize the weights of heads."""
        pass

    def forward(self, feats, gt_bboxes_3d=None, gt_labels_3d=None, maps=None, img_metas=None, inference=False, **kwargs):
        """Forward function."""
        pred_dict = {}
        for head in self.out_channels:
            pred_dict[head] = self.__getattr__(head)(feats)
            if head in ["gb_roadmap_preds"]:
                pred_dict[head] = _sigmoid(pred_dict[head])

        # 构建数据字典
        data = {
            'img_metas': img_metas,
            self.gt_key: maps  # 假设maps包含了gt数据
        }
        
        if not inference:
            return self.forward_train(pred_dict, data, **kwargs)
        else:
            return self.forward_inference(pred_dict, data, **kwargs)

    def forward_dummy(self, feats, bev_feat=None):
        pred_dict = {}
        for head in self.out_channels:
            pred_dict[head] = self.__getattr__(head)(feats)
            if head in ["gb_roadmap_preds"]:
                pred_dict[head] = _sigmoid_without_clamp(pred_dict[head])
                if self.ignore_mask:
                    print("ignore mask")
                    ignore_mask = torch.ones(pred_dict[head].shape, device=pred_dict[head].device)

                    x1, y1, x2, y2 = self.gd_map_infer_roi
                    ignore_mask[0,0,...,:int(x1)] = 0
                    ignore_mask[0,0,:int(y1),...] = 0
                    ignore_mask[0,0,int(y2):,...] = 0

                    pred_dict[head] = pred_dict[head] * ignore_mask

        gb_mask = pred_dict['gb_roadmap_preds']
        gb_downsample_mask = self.max_pooling2(gb_mask)
        pred_dict['gb_downsample_mask'] = gb_downsample_mask

        return pred_dict

    def forward_train(self, pred_dict, data, **kwargs):
        gb_gt_dict_roadline = data[self.gt_key]
        img_metas = data['img_metas']

        losses_dict = self.loss(
            pred_dict,
            gt_dict = gb_gt_dict_roadline,
            img_metas = img_metas,
            data=data,
        )
        return losses_dict

    def forward_inference(self, pred_dict, data, **kwargs):
        from mdet.models.task_modules.coders.gd_decoder import GDDecode

        if self.ignore_mask:
            ignore_mask = torch.ones(pred_dict["gb_roadmap_preds"].shape, device=pred_dict["gb_roadmap_preds"].device)
            ignore_mask[0,0,...,:48] = 0
            pred_dict["gb_roadmap_preds"] = pred_dict["gb_roadmap_preds"] * ignore_mask
            if self.use_od_auxiliary:
                pred_dict["od_roadmap_preds"] = pred_dict["od_roadmap_preds"] * ignore_mask

        test_cfg = kwargs.get("test_cfg", None)
        if self.gt_key == "gb_gt_dict_roadline":
            voxel_size = test_cfg.get("voxel_size", [])
            point_cloud_range = test_cfg.get("point_cloud_range", [])
            test_cfg["far_mode"] = False
        elif self.gt_key == "gb_far_gt_dict_roadline":
            voxel_size = test_cfg.get("voxel_size_far", [])
            point_cloud_range = test_cfg.get("point_cloud_range_far", [])
            test_cfg["far_mode"] = True
        else:
            raise ValueError("gt_keys not in ['gb_gt_dict_roadline', 'gb_far_gt_dict_roadline']")
        top_loss = test_cfg.get("top_loss", 0)
        save_pred = test_cfg.get("save_pred", False)
        roadmap_loss_map = None
        if top_loss != 0:
            roadmap_preds = pred_dict["gb_roadmap_preds"]
            roadmap_gts = data[self.gt_key]['gb_roadmap_gt']
            roadins_gts = data[self.gt_key]["gb_roadins_gt"]
            pos_neg_mask = (roadins_gts >= 0).float()

            # 适配不同的loss
            roadmap_loss, roadmap_loss_map = self.get_roadmap_loss(pos_neg_mask, roadins_gts, roadmap_preds, roadmap_gts, return_loss_map=True)
            roadmap_loss = roadmap_loss.cpu().numpy().tolist()
        else:
            roadmap_loss = 0

        if 'img' in data:
            images = [data['img'][0,i,...].permute(1,2,0).cpu().numpy() for i in range(data['img'].shape[1])]
        else:
            images = None

        voxels = kwargs.get("voxels", None)
        img_metas = data['img_metas']
        if not isinstance(img_metas, list):
            img_metas = img_metas.data[0]

        gd_decoder = GDDecode(voxel_size, point_cloud_range, test_cfg, self.compute_type)
        gd_rst = gd_decoder(
            pred_dict,
            img_metas,
            voxels,
            images=images,
            loss=roadmap_loss,
            loss_map=roadmap_loss_map,
            vis_loss=(top_loss != 0),
            gt=data.get(self.gt_key, None)
        )
        if save_pred:
            file_dir = os.path.join(test_cfg.get('out_dir', '.'), 'pred')
            if not os.path.exists(file_dir):
                os.makedirs(file_dir)
            
            gb_mask = pred_dict['gb_roadmap_preds']
            gb_downsample_mask = self.max_pooling2(gb_mask)
            pred_dict['gb_downsample_mask'] = gb_downsample_mask

            for key, value in pred_dict.items():
                filename = f"{file_dir}/{img_metas[0]['sample_idx']}_{key}.npy"
                np.save(filename, value.cpu().numpy())

        if self.show_debug:
            print('************save debug data [gd heatmap]')
            save_root = os.path.join(os.getcwd(), "work_dirs/debug_data/input_output_inference")
            os.makedirs(save_root, exist_ok=True)

            roadmap_preds = pred_dict["gb_roadmap_preds"]
            b, c, h, w = roadmap_preds.shape
            for b_id in range(b):
                batch_data_source = img_metas[b_id]['data_source']
                md5 = img_metas[b_id]['sample_idx']
                save_dir = os.path.join(save_root, f'data_source_{batch_data_source}', md5)
                os.makedirs(save_dir, exist_ok=True)
                for c_id in range(c):
                    roadmap_preds_map = roadmap_preds[b_id][c_id].detach().cpu().numpy() * 255
                    roadmap_preds_map = cv2.applyColorMap(roadmap_preds_map.astype(np.uint8), cv2.COLORMAP_JET) 
                    save_path = os.path.join(save_dir, f"output_gb-head_roadmap_preds_batch-{b_id}_channel-{c_id}_loss-{roadmap_loss}.png")
                    cv2.imwrite(save_path, roadmap_preds_map)
            
        return gd_rst

    def get_roadmap_loss(self, pos_neg_mask, roadins_gts, roadmap_preds, roadmap_gts,
            loss_weight_map=None, gt_weight_map=None, fp_weight_map=None,
            distance_weight_map=None, distance_ignore_mask=None,
            return_loss_map=False, score_list=[0.35,0.4,0.45]):
        
        # 将4D张量重塑为2D以适配FocalLoss
        # roadmap_preds: [B, C, H, W] -> [B*H*W, C]
        # roadmap_gts: [B, C, H, W] -> [B*H*W]
        B, C, H, W = roadmap_preds.shape
        roadmap_preds_flat = roadmap_preds.permute(0, 2, 3, 1).contiguous().view(-1, C)  # [B*H*W, C]
        roadmap_gts_flat = roadmap_gts.view(-1)  # [B*H*W]
        pos_neg_mask_flat = pos_neg_mask.view(-1)  # [B*H*W]
        
        # 使用FocalLoss
        roadmap_loss = self.hm_loss_func(roadmap_preds_flat, roadmap_gts_flat, weight=pos_neg_mask_flat)
        
        # 简化的log变量
        self.log_vars[f"pos_loss{self.loss_suffix}"] = 0.0
        self.log_vars[f"neg_loss{self.loss_suffix}"] = 0.0
        self.log_vars[f"ave_gt_num{self.loss_suffix}"] = roadmap_gts.sum().item()

        loss_map = None
        return roadmap_loss, loss_map

    def loss(self, pred_dict, gt_dict, img_metas=None, data=None):
        roadmap_preds = pred_dict["gb_roadmap_preds"]
        offset_preds = pred_dict["gb_offset_preds"]

        roadmap_gts = copy.deepcopy(gt_dict["gb_roadmap_gt"])
        offset_gts = copy.deepcopy(gt_dict["gb_offset_gt"])
        
        loss_weight_map = gt_dict.get('loss_weight_map', None)
        gt_weight_map = gt_dict.get('gt_weight_map', None)
        fp_weight_map = gt_dict.get('fp_weight_map', None)
        distance_weight_map = gt_dict.get('distance_weight_map', None)
        distance_ignore_mask = gt_dict.get('distance_ignore_mask', None)

        # 根据compute_type过滤数据
        if self.compute_type is not None and img_metas is not None:
            ignore_mask = torch.ones(roadmap_gts.shape, device=roadmap_gts.device) * -1.0
            data_sources = [m.get("data_source", self.compute_type) for m in img_metas]
            ignore_idxes = []
            for i, source in enumerate(data_sources):
                if source != self.compute_type:
                    ignore_idxes.append(i)
            for i in ignore_idxes:
                roadmap_gts[i] = ignore_mask[i]
        
        # 使用roadmap_gts来生成mask
        pos_neg_mask = (roadmap_gts >= 0).float()  # pos+neg
        pos_mask = (roadmap_gts > 0).float()  # pos

        loss_dict = {}

        assert roadmap_preds.shape == roadmap_gts.shape

        # 计算seg loss (roadmap loss)
        roadmap_loss, roadmap_loss_map = self.get_roadmap_loss(
                pos_neg_mask, roadmap_gts, roadmap_preds, roadmap_gts,
                loss_weight_map=loss_weight_map,
                gt_weight_map=gt_weight_map,
                fp_weight_map=fp_weight_map,
                distance_weight_map=distance_weight_map,
                distance_ignore_mask=distance_ignore_mask,
                return_loss_map=True,
            )
        loss_dict.update({f"gb_roadmap_loss{self.loss_suffix}": roadmap_loss,})

        # 计算reg loss (offset loss)
        pos_weight = pos_mask.expand_as(offset_preds).float()
        assert offset_preds.shape == offset_gts.shape
        reg_loss = self.reg_loss_func(
            offset_preds,
            offset_gts,
            weight=pos_weight,
        )
        reg_loss = reg_loss.sum() / max(pos_weight.sum(), 1)
        loss_dict.update({f"gb_roadreg_loss{self.loss_suffix}": reg_loss,})

        if self.show_debug and img_metas is not None:
            print('************save debug data [gd heatmap]')
            save_root = os.path.join(os.getcwd(), "work_dirs/debug_data/input_output_train")
            os.makedirs(save_root, exist_ok=True)

            b, c, h, w = roadmap_gts.shape
            for b_id in range(b):
                batch_data_source = img_metas[b_id].get('data_source', 'unknown')
                md5 = img_metas[b_id].get('sample_idx', f'sample_{b_id}')
                save_dir = os.path.join(save_root, f'data_source_{batch_data_source}', str(md5))
                os.makedirs(save_dir, exist_ok=True)
                for c_id in range(c):
                    roadmap_gts_map = roadmap_gts[b_id][c_id].detach().cpu().numpy() * 255
                    save_path = os.path.join(save_dir, f"output_gb-head_roadmap_gts_batch-{b_id}_channel-{c_id}_loss-{roadmap_loss}.png")
                    cv2.imwrite(save_path, roadmap_gts_map)

                    pos_neg_mask_map = pos_neg_mask[b_id][0].detach().cpu().numpy() * 255
                    save_path = os.path.join(save_dir, f"output_gb-head_pos_neg_mask_batch-{b_id}_channel-{c_id}_loss-{roadmap_loss}.png")
                    cv2.imwrite(save_path, pos_neg_mask_map)

                    roadmap_gts_map_roi = roadmap_gts_map * pos_neg_mask_map
                    save_path = os.path.join(save_dir, f"output_gb-head_roadmap_gts_roi_batch-{b_id}_channel-{c_id}_loss-{roadmap_loss}.png")
                    cv2.imwrite(save_path, roadmap_gts_map_roi)
                
                    roadmap_preds_map = roadmap_preds[b_id][c_id].detach().cpu().numpy() * 255
                    roadmap_preds_map = cv2.applyColorMap(roadmap_preds_map.astype(np.uint8), cv2.COLORMAP_JET)
                    save_path = os.path.join(save_dir, f"output_gb-head_roadmap_preds_batch-{b_id}_channel-{c_id}_loss-{roadmap_loss}.png")
                    cv2.imwrite(save_path, roadmap_preds_map)

        return loss_dict


