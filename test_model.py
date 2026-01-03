#!/usr/bin/env python3
"""
测试脚本：伪造数据让模型完成一次反向传播
"""

import torch
import torch.nn as nn
import numpy as np
import sys
import os

# 添加项目路径
sys.path.append('.')
sys.path.append('projects')

from mmcv import Config
from mmdet3d.models import build_model
from projects.mmdet3d_plugin import *

def create_fake_data():
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
        'gb_roadmap_gt': torch.randint(0, 2, (batch_size, 1, bev_h, bev_w)).long().cuda(),  # 道路图真值 [B, 1, H, W] - 使用long类型
        'gb_offset_gt': torch.randn(batch_size, 2, bev_h, bev_w).cuda(),  # 偏移真值 [B, 2, H, W]
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

def test_model():
    """测试模型前向和反向传播"""
    
    print("🚀 开始测试模型...")
    
    # 1. 加载配置
    config_path = 'projects/configs/clgd/clgd.py'
    cfg = Config.fromfile(config_path)
    
    print("✅ 配置加载成功")
    
    # 2. 构建模型
    model = build_model(cfg.model, train_cfg=cfg.get('train_cfg'), test_cfg=cfg.get('test_cfg'))
    model = model.cuda()
    model.train()
    
    print("✅ 模型构建成功")
    print(f"模型参数数量: {sum(p.numel() for p in model.parameters()):,}")
    
    # 3. 创建伪造数据
    fake_data = create_fake_data()
    
    print("✅ 伪造数据创建成功")
    print(f"图像数据形状: {fake_data['img'].shape}")
    print(f"点云数据形状: {fake_data['pts'].shape}")
    print(f"真值道路图形状: {fake_data['maps']['gb_roadmap_gt'].shape}")
    print(f"真值偏移形状: {fake_data['maps']['gb_offset_gt'].shape}")
    
    # 4. 创建优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    
    print("✅ 优化器创建成功")
    
    # 5. 前向传播
    print("\n🔄 开始前向传播...")
    
    try:
        losses = model(**fake_data)

        print("✅ 前向传播成功!")
        
        # 打印损失信息
        total_loss = 0
        print("\n📊 损失信息:")
        for key, value in losses.items():
            if isinstance(value, torch.Tensor):
                if value.numel() == 1:  # 只有单个元素的张量才能转换为标量
                    loss_val = value.item()
                    total_loss += loss_val
                    print(f"  {key}: {loss_val:.6f}")
                else:
                    # 对于多元素张量，计算平均值
                    loss_val = value.mean().item()
                    total_loss += loss_val
                    print(f"  {key}: {loss_val:.6f} (mean of {value.numel()} elements)")
            else:
                print(f"  {key}: {value}")
        
        print(f"\n总损失: {total_loss:.6f}")
        
        # 6. 反向传播
        print("\n🔄 开始反向传播...")
        
        optimizer.zero_grad()
        
        # 计算总损失 - 确保每个损失都是标量
        loss_tensor = 0
        for v in losses.values():
            if isinstance(v, torch.Tensor):
                if v.numel() == 1:
                    loss_tensor += v
                else:
                    loss_tensor += v.mean()  # 对多元素张量取平均值
        
        # 反向传播
        loss_tensor.backward()
        
        print("✅ 反向传播成功!")
        
        # 检查梯度
        grad_norm = 0
        param_count = 0
        for name, param in model.named_parameters():
            if param.grad is not None:
                grad_norm += param.grad.data.norm(2).item() ** 2
                param_count += 1
        
        grad_norm = grad_norm ** 0.5
        print(f"梯度范数: {grad_norm:.6f}")
        print(f"有梯度的参数数量: {param_count}")
        
        # 7. 优化器步骤
        print("\n🔄 执行优化器步骤...")
        optimizer.step()
        print("✅ 优化器步骤完成!")
        
        print("\n🎉 测试完成! 模型成功完成了一次完整的前向和反向传播!")
        
        return True
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    # 设置随机种子
    torch.manual_seed(42)
    np.random.seed(42)
    
    # 检查CUDA
    if not torch.cuda.is_available():
        print("❌ CUDA不可用，请在GPU环境中运行")
        sys.exit(1)
    
    print(f"🔧 使用GPU: {torch.cuda.get_device_name()}")
    print(f"🔧 CUDA版本: {torch.version.cuda}")
    print(f"🔧 PyTorch版本: {torch.__version__}")
    
    # 运行测试
    success = test_model()
    
    if success:
        print("\n✅ 所有测试通过!")
        sys.exit(0)
    else:
        print("\n❌ 测试失败!")
        sys.exit(1)