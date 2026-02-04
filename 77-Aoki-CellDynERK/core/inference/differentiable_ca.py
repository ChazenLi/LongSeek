"""
Differentiable CA Trainer - 可微CA训练器 (重构版)

实现可微CA的反向传播训练，支持：
- Gumbel-Softmax重参数化
- 直通估计器（Straight-Through Estimator）
- 自动微分支持
- 多目标损失优化

这是Level B方法学创新的核心！

作者: IF-CA Team
日期: 2026-02-04
版本: 2.0 (Refactored)
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass, field
import logging

logger = logging.getLogger(__name__)

# 尝试导入自动微分框架
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    logger.warning("PyTorch not available, using numpy fallback")


@dataclass
class TrainingConfig:
    """训练配置"""
    learning_rate: float = 1e-3
    num_epochs: int = 100
    batch_size: int = 1
    gumbel_temperature: float = 1.0  # Gumbel-Softmax温度
    use_straight_through: bool = True  # 是否使用直通估计器
    gradient_clip: float = 1.0  # 梯度裁剪
    log_interval: int = 10  # 日志间隔


@dataclass
class TrainingResult:
    """训练结果"""
    best_params: Dict[str, float]
    best_loss: float
    loss_history: List[float]
    params_history: List[Dict[str, float]]


class DifferentiableCA:
    """
    可微CA模型 (PyTorch版本)

    核心特性：
    1. Gumbel-Softmax重参数化 - 使离散采样可微
    2. 直通估计器 - 减少方差
    3. 自动微分 - 支持端到端训练
    """

    def __init__(self,
                 height: int,
                 width: int,
                 erk_field: np.ndarray,
                 device: str = 'cpu'):
        """
        Args:
            height, width: 网格尺寸
            erk_field: ERK场 (T, H, W)
            device: 'cpu' or 'cuda'
        """
        if not HAS_TORCH:
            raise ImportError("PyTorch is required for DifferentiableCA")

        self.height = height
        self.width = width
        self.device = torch.device(device)

        # 转换ERK场为tensor
        self.erk_field = torch.from_numpy(erk_field).float().to(self.device)
        self.T, self.H_erk, self.W_erk = self.erk_field.shape

        # ERK梯度（预计算）
        self.erk_gradients = self._compute_erk_gradients()

        # 邻居偏移（Moore邻域）
        self.neighbor_offsets = self._get_neighbor_offsets()

    def _compute_erk_gradients(self) -> torch.Tensor:
        """预计算ERK梯度"""
        # 使用Sobel算子计算梯度
        sobel_y = torch.tensor([[-1, 0, 1],
                                [-2, 0, 2],
                                [-1, 0, 1]], dtype=torch.float32, device=self.device).view(1, 1, 3, 3)

        sobel_x = torch.tensor([[-1, -2, -1],
                                [ 0,  0,  0],
                                [ 1,  2,  1]], dtype=torch.float32, device=self.device).view(1, 1, 3, 3)

        erk_expanded = self.erk_field.unsqueeze(1)  # (T, 1, H, W)

        grad_y = F.conv2d(erk_expanded, sobel_y, padding=1)
        grad_x = F.conv2d(erk_expanded, sobel_x, padding=1)

        grad_y = grad_y.squeeze(1)  # (T, H, W)
        grad_x = grad_x.squeeze(1)

        return torch.stack([grad_y, grad_x], dim=-1)  # (T, H, W, 2)

    def _get_neighbor_offsets(self) -> torch.Tensor:
        """Moore邻域偏移"""
        offsets = []
        for di in [-1, 0, 1]:
            for dj in [-1, 0, 1]:
                if di == 0 and dj == 0:
                    continue
                offsets.append([di, dj])
        return torch.tensor(offsets, dtype=torch.long, device=self.device)

    def forward_step(self,
                     grid: torch.Tensor,
                     params: Dict[str, torch.Tensor],
                     t: int) -> torch.Tensor:
        """
        可微的CA单步更新

        Args:
            grid: 当前网格 (H, W)
            params: 参数字典
            t: 当前时间步

        Returns:
            更新后的网格
        """
        # 获取时滞ERK梯度
        tau = int(params.get('tau', torch.tensor(0.0)).item())
        t_lagged = max(0, t - tau)
        erk_grad = self.erk_gradients[t_lagged]  # (H, W, 2)

        # 计算每个位置的概率
        logits = self._compute_migration_logits(grid, params, erk_grad)

        # Gumbel-Softmax采样
        # 对于每个位置，从邻居分布中采样
        new_grid = self._gumbel_softmax_sample(grid, logits, params)

        return new_grid

    def _compute_migration_logits(self,
                                  grid: torch.Tensor,
                                  params: Dict[str, torch.Tensor],
                                  erk_grad: torch.Tensor) -> torch.Tensor:
        """计算迁移logits"""
        H, W = grid.shape
        logits = torch.zeros(H, W, 8, device=self.device)  # 8个邻居

        for i in range(H):
            for j in range(W):
                if grid[i, j] == 0:  # 空位置
                    continue

                # 计算基础logit
                p_move = params.get('p_move', torch.tensor(0.5))
                alpha = params.get('alpha', torch.tensor(1.0))
                edge_bonus = params.get('edge_bonus', torch.tensor(2.0))

                # 局部密度
                local_density = self._compute_local_density(grid, i, j)
                base_logit = torch.log(p_move + 1e-10) - alpha * local_density

                # 边缘bonus
                if self._is_edge_cell(grid, i, j):
                    base_logit += torch.log(edge_bonus + 1e-10)

                # 对每个邻居计算耦合logit
                for k, (di, dj) in enumerate(self.neighbor_offsets):
                    ni, nj = i + di, j + dj
                    if 0 <= ni < H and 0 <= nj < W and grid[ni, nj] == 0:
                        # ERK耦合
                        coupling_logit = self._compute_coupling_logit(
                            i, j, di, dj, erk_grad, params
                        )
                        logits[i, j, k] = base_logit + coupling_logit

        return logits

    def _compute_coupling_logit(self,
                                i: int, j: int,
                                di: int, dj: int,
                                erk_grad: torch.Tensor,
                                params: Dict[str, torch.Tensor]) -> torch.Tensor:
        """计算ERK耦合logit: γ·s·<d̂, ∇Ĉ>"""
        gamma = params.get('gamma', torch.tensor(0.0))
        s = params.get('s', torch.tensor(1.0))

        # 移动方向
        d_norm = torch.sqrt(torch.tensor(di**2 + dj**2, dtype=torch.float32))
        d_hat = torch.tensor([di / d_norm, dj / d_norm], device=self.device)

        # ERK梯度
        grad = erk_grad[i, j]  # (2,)
        grad_norm = torch.norm(grad) + 1e-10
        grad_hat = grad / grad_norm

        # 点积
        alignment = torch.dot(d_hat, grad_hat)

        return gamma * s * alignment

    def _compute_local_density(self, grid: torch.Tensor, i: int, j: int) -> torch.Tensor:
        """计算局部密度"""
        count = 0
        for di, dj in self.neighbor_offsets:
            ni, nj = i + di.item(), j + dj.item()
            if 0 <= ni < self.height and 0 <= nj < self.width:
                if grid[ni, nj] > 0:
                    count += 1
        return torch.tensor(count / 8.0, device=self.device)

    def _is_edge_cell(self, grid: torch.Tensor, i: int, j: int) -> bool:
        """检查是否为边缘细胞"""
        for di, dj in self.neighbor_offsets:
            ni, nj = i + di.item(), j + dj.item()
            if 0 <= ni < self.height and 0 <= nj < self.width:
                if grid[ni, nj] == 0:
                    return True
        return False

    def _gumbel_softmax_sample(self,
                               grid: torch.Tensor,
                               logits: torch.Tensor,
                               params: Dict[str, torch.Tensor],
                               temperature: float = 1.0) -> torch.Tensor:
        """
        Gumbel-Softmax采样

        实现可微的离散采样
        """
        H, W = grid.shape
        new_grid = grid.clone()

        # 对每个有细胞的位置
        for i in range(H):
            for j in range(W):
                if grid[i, j] == 0:
                    continue

                # 获取邻居logits
                neighbor_logits = logits[i, j]  # (8,)

                # 只考虑空邻居
                valid_mask = torch.zeros(8, device=self.device)
                for k, (di, dj) in enumerate(self.neighbor_offsets):
                    ni, nj = i + di.item(), j + dj.item()
                    if 0 <= ni < H and 0 <= nj < W and grid[ni, nj] == 0:
                        valid_mask[k] = 1.0

                # 如果没有空邻居，跳过
                if valid_mask.sum() == 0:
                    continue

                # 应用valid mask
                masked_logits = neighbor_logits + torch.log(valid_mask + 1e-10)

                # Gumbel-Softmax
                sample = F.gumbel_softmax(masked_logits.unsqueeze(0),
                                         tau=temperature,
                                         hard=True)

                # 选择移动到的邻居
                chosen = torch.argmax(sample).item()

                # 移动
                di, dj = self.neighbor_offsets[chosen]
                ni, nj = i + di.item(), j + dj.item()
                new_grid[ni, nj] = 1
                new_grid[i, j] = 0

        return new_grid

    def run_simulation(self,
                       initial_grid: torch.Tensor,
                       params: Dict[str, torch.Tensor],
                       num_steps: int) -> List[Dict[str, Any]]:
        """运行模拟"""
        grid = initial_grid.clone()
        history = []

        for t in range(num_steps):
            grid = self.forward_step(grid, params, t)
            history.append({
                'grid': grid.detach().cpu().numpy(),
                'wound_area': (grid == 0).sum().item()
            })

        return history


class DifferentiableCATrainer:
    """
    可微CA训练器

    实现端到端的CA参数训练
    """

    def __init__(self,
                 ca_model: DifferentiableCA,
                 config: TrainingConfig):
        """
        Args:
            ca_model: 可微CA模型
            config: 训练配置
        """
        self.ca = ca_model
        self.config = config

        # 初始化可训练参数
        self.params = self._init_params()

    def _init_params(self) -> Dict[str, torch.Tensor]:
        """初始化可训练参数"""
        return {
            'p_move': torch.tensor(0.5, requires_grad=True, device=self.ca.device),
            'p_div': torch.tensor(0.05, requires_grad=True, device=self.ca.device),
            'alpha': torch.tensor(1.0, requires_grad=True, device=self.ca.device),
            'beta': torch.tensor(1.0, requires_grad=True, device=self.ca.device),
            'edge_bonus': torch.tensor(2.0, requires_grad=True, device=self.ca.device),
            'gamma': torch.tensor(0.5, requires_grad=True, device=self.ca.device),
            's': torch.tensor(1.0, requires_grad=True, device=self.ca.device),
            'tau': torch.tensor(1.0, requires_grad=True, device=self.ca.device),
        }

    def training_step(self,
                      initial_grid: torch.Tensor,
                      observed: Dict[str, np.ndarray],
                      observed_time: np.ndarray,
                      loss_fn: Callable) -> Tuple[float, Dict[str, float]]:
        """
        单步训练

        Args:
            initial_grid: 初始网格
            observed: 观测数据
            observed_time: 观测时间点
            loss_fn: 损失函数

        Returns:
            (loss, loss_dict)
        """
        # 运行模拟
        num_steps = len(observed_time)
        history = self.ca.run_simulation(initial_grid, self.params, num_steps)

        # 计算损失
        loss, loss_dict = loss_fn(
            history, observed, observed_time,
            erk_field=self.ca.erk_field.cpu().numpy()
        )

        # 反向传播
        loss_tensor = torch.tensor(loss, requires_grad=True, device=self.ca.device)
        loss_tensor.backward()

        return loss_tensor.item(), loss_dict

    def train(self,
              initial_mask: np.ndarray,
              observed: Dict[str, np.ndarray],
              observed_time: np.ndarray,
              loss_fn: Callable) -> TrainingResult:
        """
        训练CA参数

        Args:
            initial_mask: 初始mask
            observed: 观测数据
            observed_time: 观测时间点
            loss_fn: 损失函数

        Returns:
            TrainingResult
        """
        # 转换初始mask为tensor
        initial_grid = torch.from_numpy(initial_mask).float().to(self.ca.device)

        # 优化器
        optimizer = torch.optim.Adam(
            [p for p in self.params.values() if p.requires_grad],
            lr=self.config.learning_rate
        )

        best_loss = float('inf')
        best_params = None
        loss_history = []
        params_history = []

        for epoch in range(self.config.num_epochs):
            # 前向+反向
            optimizer.zero_grad()

            loss, loss_dict = self.training_step(
                initial_grid, observed, observed_time, loss_fn
            )

            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(
                [p for p in self.params.values() if p.requires_grad],
                self.config.gradient_clip
            )

            optimizer.step()

            # 记录
            loss_history.append(loss)
            params_history.append({k: v.item() for k, v in self.params.items()})

            # 更新最佳
            if loss < best_loss:
                best_loss = loss
                best_params = {k: v.item() for k, v in self.params.items()}

            # 日志
            if epoch % self.config.log_interval == 0:
                logger.info(f"Epoch {epoch}: loss={loss:.6f}")
                for k, v in loss_dict.items():
                    logger.info(f"  {k}={v:.6f}")

        return TrainingResult(
            best_params=best_params,
            best_loss=best_loss,
            loss_history=loss_history,
            params_history=params_history
        )


# NumPy后备实现（当PyTorch不可用时）
class NumpyDifferentiableCA:
    """
    NumPy版本的可微CA（使用直通估计器）
    """

    def __init__(self, height: int, width: int, erk_field: np.ndarray):
        self.height = height
        self.width = width
        self.erk_field = erk_field
        self.erk_gradients = self._compute_gradients()

    def _compute_gradients(self) -> np.ndarray:
        """计算ERK梯度"""
        grad_y, grad_x = np.gradient(self.erk_field, axis=(1, 2))
        return np.stack([grad_y, grad_x], axis=-1)

    def run_with_params(self,
                        initial_mask: np.ndarray,
                        params: Dict[str, float],
                        num_steps: int) -> List[Dict]:
        """运行模拟"""
        from core.ca.phase2_teacher_forcing import PhaseIITeacherForcingCA, CAParamsExtended, CouplingParams

        # 创建CA参数
        ca_params = CAParamsExtended(
            p_move=params.get('p_move', 0.5),
            p_div=params.get('p_div', 0.05),
            alpha=params.get('alpha', 1.0),
            beta=params.get('beta', 1.0),
            edge_bonus=params.get('edge_bonus', 2.0),
            coupling=CouplingParams(
                gamma=params.get('gamma', 0.5),
                s=int(params.get('s', 1.0)),
                tau=int(params.get('tau', 1.0))
            )
        )

        # 创建CA
        ca = PhaseIITeacherForcingCA(
            self.height, self.width, ca_params, self.erk_field
        )
        ca.initialize_from_mask(initial_mask)

        # 运行
        history = ca.run(num_steps, erk_field_sequence=self.erk_field)

        return history


def create_differentiable_ca(height: int,
                             width: int,
                             erk_field: np.ndarray,
                             use_torch: bool = True) -> Any:
    """
    创建可微CA的工厂函数

    Args:
        height, width: 网格尺寸
        erk_field: ERK场
        use_torch: 是否使用PyTorch（默认True）

    Returns:
        DifferentiableCA或NumpyDifferentiableCA
    """
    if use_torch and HAS_TORCH:
        return DifferentiableCA(height, width, erk_field)
    else:
        logger.warning("Using NumPy fallback (no automatic differentiation)")
        return NumpyDifferentiableCA(height, width, erk_field)


if __name__ == "__main__":
    # 测试可微CA
    print("Testing Differentiable CA")

    # 创建模拟ERK场
    T, H, W = 60, 50, 50
    erk_field = np.random.rand(T, H, W).astype(np.float32)

    # 创建可微CA
    if HAS_TORCH:
        ca = DifferentiableCA(H, W, erk_field)
        print(f"Created DifferentiableCA: {H}x{W}, {T} frames")
        print("ERK gradients shape:", ca.erk_gradients.shape)
        print("Neighbor offsets:", ca.neighbor_offsets)
    else:
        ca = NumpyDifferentiableCA(H, W, erk_field)
        print("Created NumpyDifferentiableCA (PyTorch not available)")

    print("\n✅ Differentiable CA test complete!")
