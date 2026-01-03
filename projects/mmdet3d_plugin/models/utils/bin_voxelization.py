import torch

class BinaryVoxelization:
    """
    二值体素化占据网格（纯 PyTorch 实现）
    
    Args:
        voxel_size: [x, y, z] 体素尺寸
        point_cloud_range: [x_min, y_min, z_min, x_max, y_max, z_max]
        device: 'cuda' or 'cpu'
    
    Returns:
        occupancy_grid: (C, H, W) 二值张量，C对应Z轴，H对应Y轴，W对应X轴
    """
    
    def __init__(self, voxel_size, point_cloud_range, device='cpu'):
        self.device = device
        self.voxel_size = torch.tensor(voxel_size, dtype=torch.float32, device=device)
        self.point_cloud_range = torch.tensor(point_cloud_range, dtype=torch.float32, device=device)
        
        # 计算网格尺寸
        self.grid_size = torch.round(
            (self.point_cloud_range[3:] - self.point_cloud_range[:3]) / self.voxel_size
        ).long()
        
        print(f"网格尺寸 (X, Y, Z): {self.grid_size.cpu().tolist()}")
    
    def __call__(self, points):
        """使对象可调用"""
        return self.voxelize_batch(points)
    
    @torch.no_grad()
    def voxelize_batch(self, points):
        """
        支持batch维度的体素化
        Args:
            points: (B, N, 3+) tensor 或 (N, 3+) tensor
        Returns:
            occupancy_grid: (B, C, H, W) torch.Tensor
        """
        if points.dim() == 2:
            # 单个样本，添加batch维度
            points = points.unsqueeze(0)
        
        batch_size = points.shape[0]
        occupancy_grids = []
        
        for b in range(batch_size):
            occupancy_grid = self.voxelize(points[b])
            occupancy_grids.append(occupancy_grid)
        
        # 堆叠成batch
        return torch.stack(occupancy_grids, dim=0)
    
    @torch.no_grad()
    def voxelize(self, points):
        """
        Args:
            points: (N, 3+) tensor 或 numpy array, xyz坐标在前3列
        
        Returns:
            occupancy_grid: (C, H, W) torch.Tensor, dtype=torch.uint8
        """
        # 转换为 tensor
        if not isinstance(points, torch.Tensor):
            import numpy as np
            if isinstance(points, np.ndarray):
                points = torch.from_numpy(points)
            else:
                points = torch.tensor(points)
        
        # 移动到指定设备
        if points.device != self.device:
            points = points.to(self.device)
        
        points = points.float()
        
        # 过滤范围外的点
        mask = (points[:, :3] >= self.point_cloud_range[:3]) & \
               (points[:, :3] < self.point_cloud_range[3:])
        mask = mask.all(dim=1)
        points = points[mask]
        
        if points.shape[0] == 0:
            print("警告: 没有点在指定范围内！")
            return torch.zeros(
                self.grid_size[2], self.grid_size[1], self.grid_size[0],
                dtype=torch.float32, device=self.device
            )
        
        # 计算体素索引
        voxel_indices = torch.floor(
            (points[:, :3] - self.point_cloud_range[:3]) / self.voxel_size
        ).long()
        
        # 限制在网格范围内（防止边界误差）
        voxel_indices = torch.clamp(voxel_indices, 
                                     min=torch.zeros(3, device=self.device, dtype=torch.long),
                                     max=self.grid_size - 1)
        
        # 创建占据网格 (Z, Y, X)
        occupancy_grid = torch.zeros(
            self.grid_size[2], self.grid_size[1], self.grid_size[0],
            dtype=torch.float32, device=self.device
        )
        
        # 使用线性索引去重（更高效）
        linear_indices = (
            voxel_indices[:, 2] * self.grid_size[0] * self.grid_size[1] +
            voxel_indices[:, 1] * self.grid_size[0] +
            voxel_indices[:, 0]
        )
        
        # 去重
        unique_linear_indices = torch.unique(linear_indices)
        
        # 转回 3D 索引
        z = unique_linear_indices // (self.grid_size[0] * self.grid_size[1])
        remainder = unique_linear_indices % (self.grid_size[0] * self.grid_size[1])
        y = remainder // self.grid_size[0]
        x = remainder % self.grid_size[0]
        
        # 填充占据信息
        occupancy_grid[z, y, x] = 1
        
        # 返回 (C, H, W) 其中 C=Z=80, H=Y=320, W=X=1024
        return occupancy_grid  # (80, 320, 1024)
    
    def visualize_slice(self, occupancy_grid, z_index=None):
        """可视化某个高度切片"""
        try:
            import matplotlib.pyplot as plt
            
            if z_index is None:
                z_index = occupancy_grid.shape[0] // 2
            
            # 转到 CPU
            slice_data = occupancy_grid[z_index].cpu()
            
            plt.figure(figsize=(10, 8))
            plt.imshow(slice_data.numpy(), cmap='binary', origin='lower')
            plt.title(f'Occupancy Grid at Z={z_index}')
            plt.xlabel('X axis')
            plt.ylabel('Y axis')
            plt.colorbar(label='Occupancy (0=empty, 1=occupied)')
            plt.tight_layout()
            plt.savefig(f'occupancy_slice_z{z_index}.png', dpi=150)
            print(f"可视化结果已保存到 occupancy_slice_z{z_index}.png")
            plt.close()
        except ImportError:
            print("matplotlib 未安装，跳过可视化")
        except Exception as e:
            print(f"可视化失败: {e}")


# 使用示例
if __name__ == '__main__':
    # 配置参数（KITTI数据集风格）
    voxel_size = [0.2, 0.2, 0.2]  # 每个体素 20cm
    point_cloud_range = [0, -40, -3, 70.4, 40, 1]  # [x_min, y_min, z_min, x_max, y_max, z_max]
    
    # 检测可用设备
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"使用设备: {device}")
    
    # 创建体素化器
    voxelizer = BinaryVoxelization(voxel_size, point_cloud_range, device=device)
    
    # 模拟点云数据 (N, 4) - xyz + intensity
    # 使用 torch 生成，避免 numpy
    points = torch.rand(10000, 4, device=device)
    points[:, 0] = points[:, 0] * 70  # x: 0-70
    points[:, 1] = points[:, 1] * 80 - 40  # y: -40 to 40
    points[:, 2] = points[:, 2] * 4 - 3  # z: -3 to 1
    
    print(f"点云形状: {points.shape}")
    print(f"点云范围: X[{points[:, 0].min():.2f}, {points[:, 0].max():.2f}], "
          f"Y[{points[:, 1].min():.2f}, {points[:, 1].max():.2f}], "
          f"Z[{points[:, 2].min():.2f}, {points[:, 2].max():.2f}]")
    
    # 体素化
    occupancy_grid = voxelizer.voxelize(points)
    print(f"占据网格形状: {occupancy_grid.shape}")  # (C=20, H=400, W=352)
    print(f"占据体素数量: {occupancy_grid.sum().item()}")
    print(f"占据率: {occupancy_grid.sum() / occupancy_grid.numel() * 100:.2f}%")
    
    # 可视化
    voxelizer.visualize_slice(occupancy_grid)
    
    # 额外统计信息
    print("\n=== 每层统计 ===")
    for z in range(occupancy_grid.shape[0]):
        occupied = occupancy_grid[z].sum().item()
        if occupied > 0:
            print(f"Z层 {z}: {occupied} 个占据体素")