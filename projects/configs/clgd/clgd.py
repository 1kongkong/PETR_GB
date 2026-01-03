from pickle import FALSE


_base_ = [
    '../../../mmdetection3d/configs/_base_/datasets/nus-3d.py',
    '../../../mmdetection3d/configs/_base_/default_runtime.py'
]
backbone_norm_cfg = dict(type='LN', requires_grad=True)
plugin=True
plugin_dir='projects/mmdet3d_plugin/'

# 添加缺失的变量定义
freeze_img_encoder = False
train_cfg = None
test_cfg = None

# If point cloud range is changed, the models should also change their point
# cloud range accordingly
point_cloud_range = [-10.0, -14.4, -1.0, 82.16, 14.4, 7.0]
# 确保 point_cloud_range 是 list 类型，避免类型转换问题
point_cloud_range = [float(x) for x in point_cloud_range]
voxel_size = [0.09, 0.09, 0.1]
img_norm_cfg = dict(
    mean=[103.530, 116.280, 123.675], std=[57.375, 57.120, 58.395], to_rgb=False)
# For nuScenes we usually do 10-class detection
class_names = [
    'car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
    'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
]
input_modality = dict(
    use_lidar=True,
    use_camera=True,
    use_radar=False,
    use_map=False,
    use_external=False
)

model = dict(
    type="MultiModalNet",
    use_grid_mask=False,
    train_cfg=None,
    test_cfg=None,
    pretrained=None,
    pts_voxel_layer=dict(
        voxel_size=voxel_size,
        point_cloud_range=point_cloud_range,
        device='cuda',
    ),
    pts_backbone=dict(
        type="RES2Ddilation",
        cfg=train_cfg,
        in_channels=80,
        out_channels=[32, 96, 128, 128],
        kernel_sizes=[(3, 3), (5, 3), (5, 3), (5, 3)],
        block_strides=[(2, 1), (2, 1), (2, 1), (2, 1)],
        block_nums=[2, 4, 5, 5],
        block_dilations=[2, 4, 6, 2, 4],
        norm_cfg=dict(type="BN", eps=1e-05, momentum=0.01),
        conv_cfg=dict(type="Conv2d", bias=False),
    ),
    pts_neck=dict(
        type="TUMFPNLEVEL2RED8",
            cfg=train_cfg,
            out_channel=256,
            norm_cfg=dict(type="BN", eps=1e-05, momentum=0.01),
            conv_cfg=dict(type="Conv2d", bias=False),
            upsample_cfg=dict(type="nearest", scale_factor=2),
    ),
    img_backbone=dict(
        type="ResNetS64",
        depth=34,
        num_stages=4,
        strides=(1, 2, 2, 2),
        dilations=(1, 1, 1, 1),
        stage_planes=(64, 128, 256, 512),
        out_indices=(3,),
        frozen_stages=4 if freeze_img_encoder else -1,
        norm_cfg=dict(type="BN", requires_grad=False),
        stage_with_dcn=(False, False, False, False),
        norm_eval=False,
        style="pytorch",
    ),
    img_view_trans=dict(
        type="PETR_ViewTrans",
        in_channels=512,
        embed_dims=256,
        bev_h_query=32,
        bev_w_query=10,
        bev_h=80,
        bev_w=256,
        transformer=dict(
            type='PETRTransformer',
            decoder=dict(
                type='PETRTransformerDecoder',
                return_intermediate=True,
                num_layers=1,
                transformerlayers=dict(
                    type='PETRTransformerDecoderLayer',
                    attn_cfgs=[
                        dict(
                            type='MultiheadAttention',
                            embed_dims=256,
                            num_heads=4,
                            dropout=0.1
                        ),
                        dict(
                            type='PETRMultiheadAttention',
                            embed_dims=256,
                            num_heads=4,
                            dropout=0.1
                        ),
                    ],
                    feedforward_channels=1024,
                    ffn_dropout=0.1,
                    operation_order=('self_attn', 'norm', 'cross_attn', 'norm',
                                     'ffn', 'norm')),
            )
        ),
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
        position_range=point_cloud_range,
    ),
    fusion_layer=dict(
        type="BEVFusion",
        in_channels=256,
        out_channels=128,
    ),
    heads_cfg=dict(
        gb_head=dict(
            type="KeyPointHeadGeneralBarrier",
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            num_classes=1,
            class_names=["GeneralBarrier"],
            offset_dims=2,
            feat_levels=1,
            in_channels=128,
            feat_channels=64,
            class_weight=[1.0],
            compute_type="GB",
            ignore_mask=False,
            add_gb_roadmap_conv=False,
            gd_map_infer_roi=[48, 24, 256, 56],
            gt_key='gb_gt_dict_roadline',
            loss_hm=dict(
                type="FocalLoss",
                reduction="none",
            ),
            loss_reg=dict(
                type="SmoothL1Loss", beta=1.0, reduction="none"
            ),
        ),
    ),
)

dataset_type = 'MultiCustomNuScenesDataset'
data_root = 'data/nuscenes/'

file_client_args = dict(backend='disk')

db_sampler = dict(
    data_root=data_root,
    info_path=data_root + 'nuscenes_infos_train.pkl',
    rate=1.0,
    prepare=dict(
        filter_by_difficulty=[-1],
        filter_by_min_points=dict(
            car=5,
            truck=5,
            bus=5,
            trailer=5,
            construction_vehicle=5,
            traffic_cone=5,
            barrier=5,
            motorcycle=5,
            bicycle=5,
            pedestrian=5)),
    classes=class_names,
    sample_groups=dict(
        car=2,
        truck=3,
        construction_vehicle=7,
        bus=4,
        trailer=6,
        barrier=2,
        motorcycle=6,
        bicycle=6,
        pedestrian=2,
        traffic_cone=2),
    points_loader=dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=5,
        use_dim=[0, 1, 2, 3, 4],
        file_client_args=file_client_args))
# ida_aug_conf = {
#         "resize_lim": (0.47, 0.625),
#         "final_dim": (320, 800),
#         "bot_pct_lim": (0.0, 0.0),
#         "rot_lim": (0.0, 0.0),
#         "H": 900,
#         "W": 1600,
#         # "rand_flip": False,
#         "rand_flip": True,
#     }
ida_aug_conf = {
        "resize_lim": (0.94, 1.25),
        "final_dim": (640, 1600),
        "bot_pct_lim": (0.0, 0.0),
        "rot_lim": (0.0, 0.0),
        "H": 900,
        "W": 1600,
        # "rand_flip": False,
        "rand_flip": True,
    }
train_pipeline = [
    dict(type='LoadMultiViewImageFromFiles', to_float32=True),
    dict(type='LoadMapsFromFiles_flattenf200f3'),
    dict(type='LoadMultiViewImageFromMultiSweepsFiles', sweeps_num=1, to_float32=True, pad_empty_sweeps=True, test_mode=False, sweep_range=[3,27]),
    # dict(type='LoadMultiViewImageFromSweepsFiles', sweeps_num=1, to_float32=True, pad_empty_sweeps=True, is_nori_read=True),
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True, with_attr_label=False),
    dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectNameFilter', classes=class_names),
    # dict(type='ResizeCropFlipImage', data_aug_conf = ida_aug_conf, training=True),
    # dict(type='ResizeMultiview3D', img_scale=[(1800, 640), (1800, 900)], multiscale_mode='range', keep_ratio=True),
    # dict(type='GlobalRotScaleTransImage',
    #         rot_range=[-0.3925, 0.3925],
    #         translation_std=[0, 0, 0],
    #         scale_ratio_range=[0.95, 1.05],
    #         reverse_angle=True,
    #         training=True
    #         ),
    dict(type='NormalizeMultiviewImage', **img_norm_cfg),
    dict(type='PadMultiViewImage', size_divisor=32),
    dict(type='DefaultFormatBundle3D', class_names=class_names),
    dict(type='Collect3D', keys=['gt_bboxes_3d', 'gt_labels_3d', 'img','maps'],
            meta_keys=('filename', 'ori_shape', 'img_shape', 'lidar2img', 'intrinsics', 'extrinsics','bda',
                'pad_shape', 'scale_factor', 'flip', 'box_mode_3d', 'box_type_3d',
                'img_norm_cfg', 'sample_idx', 'timestamp'))
]
test_pipeline = [
    dict(type='LoadMultiViewImageFromFiles', to_float32=True),
    dict(type='LoadMapsFromFiles_flattenf200f3'),
    dict(type='LoadMultiViewImageFromMultiSweepsFiles', sweeps_num=1, to_float32=True, pad_empty_sweeps=True, test_mode=False, sweep_range=[3,27]),
    # dict(type='ResizeCropFlipImage', data_aug_conf = ida_aug_conf, training=False),
    # dict(type='ResizeMultiview3D', img_scale= (1600, 800), keep_ratio=True),
    dict(type='NormalizeMultiviewImage', **img_norm_cfg),
    dict(type='PadMultiViewImage', size_divisor=32),
    dict(
        type='MultiScaleFlipAug3D',
        img_scale=(1333, 800),
        pts_scale_ratio=1,
        flip=False,
        transforms=[
            dict(
                type='DefaultFormatBundle3D',
                class_names=class_names,
                with_label=False),
            dict(type='Collect3D', keys=['img','gt_map','maps'],
            meta_keys=('filename', 'ori_shape', 'img_shape', 'lidar2img', 'intrinsics', 'extrinsics','bda',
                'pad_shape', 'scale_factor', 'flip', 'box_mode_3d', 'box_type_3d',
                'img_norm_cfg', 'sample_idx', 'timestamp'))
        ])
]

data = dict(
    samples_per_gpu=1,
    workers_per_gpu=0,
    train=dict(
        type=dataset_type,
        data_root=data_root,
        
        ann_file=data_root + 'nuscenes_infos_train.pkl',
        lane_ann_file=data_root + 'nuscenes_infos_train.pkl',
        pipeline=train_pipeline,
        classes=class_names,
        modality=input_modality,
        test_mode=False,
        use_valid_flag=True,
        # we use box_type_3d='LiDAR' in kitti and nuscenes dataset
        # and box_type_3d='Depth' in sunrgbd and scannet dataset.
        box_type_3d='LiDAR'),
    val=dict(type=dataset_type, pipeline=test_pipeline, ann_file=data_root + 'nuscenes_infos_val.pkl',lane_ann_file=data_root + 'nuscenes_infos_val.pkl', classes=class_names, modality=input_modality),
    test=dict(type=dataset_type, pipeline=test_pipeline, ann_file=data_root + 'nuscenes_infos_val.pkl',lane_ann_file=data_root + 'nuscenes_infos_val.pkl', classes=class_names, modality=input_modality))


optimizer = dict(
    type='AdamW',
    lr=1e-5, 
    paramwise_cfg=dict(
        custom_keys={
            'img_backbone': dict(lr_mult=0.1),
        }),
    weight_decay=0.01)

# 暂时禁用FP16训练，使用普通的优化器钩子
optimizer_config = dict(grad_clip=dict(max_norm=10, norm_type=2))

# learning policy
lr_config = dict(
    policy='CosineAnnealing',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=1.0 / 3,
    min_lr_ratio=1e-3,
    # by_epoch=False
    )
total_epochs = 50
evaluation = dict(interval=50, pipeline=test_pipeline)
find_unused_parameters=True
ddp_cfg = dict(static_graph=True)
checkpoint_config = dict(interval=1, max_keep_ckpts=1)
runner = dict(type='EpochBasedRunner', max_epochs=total_epochs)
# load_from='ckpts/fcos3d_vovnet_imgbackbone-remapped.pth'
resume_from=None

