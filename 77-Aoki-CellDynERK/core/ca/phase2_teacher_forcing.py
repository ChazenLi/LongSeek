"""
Phase II: Teacher-Forcing CA with Molecular Field
使用观测ERK场作为外场，实现方向偏置与时滞机制
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Deque
from collections import deque
from dataclasses import dataclass, field
import random
import logging

logger = logging.getLogger(__name__)


@dataclass
class CouplingParams:
    """分子场耦合参数"""
    gamma: float = 0.5        # 耦合强度 [0, 2.0]
    s: int = +1               # 符号开关 {-1, +1}：+1=沿梯度，-1=逆梯度
    tau: int = 0              # 时滞（时间步）[0, 5]
    D_coupling: float = 1.0   # 耦合扩散系数 [0.1, 2.0]


@dataclass
class CAParamsExtended:
    """扩展的CA参数（包含耦合参数）"""
    # 基础参数
    p_move: float = 0.5
    p_div: float = 0.05
    alpha: float = 1.0
    beta: float = 1.0
    edge_bonus: float = 2.0
    k_time: float = 1.0
    gamma: float = 0.0  # Phase I的gamma（仅用于距离场偏置）
    
    # Phase II耦合参数
    coupling: CouplingParams = field(default_factory=CouplingParams)


class PhaseIITeacherForcingCA:
    """
    Phase II: Teacher-Forcing CA
    
    核心机制：
    1. 使用观测ERK场 C_obs(t) 作为外场
    2. 实现时滞 τ：使用 C_obs(t-τ)
    3. 方向偏置：P(move to d) ∝ exp(u_base + γ·s·<d̂, ∇Ĉ(t-τ)>)
    """
    
    def __init__(self, 
                 height: int, 
                 width: int, 
                 params: CAParamsExtended,
                 erk_field: Optional[np.ndarray] = None):
        """
        Args:
            height, width: 网格尺寸
            params: 扩展CA参数
            erk_field: 观测ERK场 (T, H, W)
        """
        self.height = height
        self.width = width
        self.params = params
        self.erk_field = erk_field
        
        # CA grid
        self.grid = np.zeros((height, width), dtype=np.int32)
        
        # 时滞缓冲区：存储过去τ个时间步的ERK场
        self.field_buffer: Deque[np.ndarray] = deque(maxlen=params.coupling.tau + 1)
        
        # 当前时间步
        self.current_step = 0
        
        # 缓存梯度场
        self.grad_y: Optional[np.ndarray] = None
        self.grad_x: Optional[np.ndarray] = None
        self.grad_mag: Optional[np.ndarray] = None
        
        # 距离场（用于fallback）
        self.distance_field: Optional[np.ndarray] = None
        
        # 数值稳定性常量（修复2：避免exp溢出）
        self.MAX_LOGIT = 20.0  # Clip range for exp (exp(20) ≈ 4.85e8, 足够大但不会溢出)
    
    def initialize_from_mask(self, mask: np.ndarray, k: int = 1):
        """从初始mask初始化"""
        from .model import CellOnlyCA  # Import for downsample

        # 下采样
        mask_down = mask[::k, ::k] if k > 1 else mask
        self.grid = mask_down.astype(np.int32)
        self.height, self.width = self.grid.shape

        # 计算距离场（fallback）
        self.compute_distance_field()

        # 初始化场缓冲区
        if self.erk_field is not None:
            # 检查erk_field的维度
            if self.erk_field.ndim == 3:
                # 时间序列 (T, H, W) - 使用第一帧初始化
                erk_first_frame = self.erk_field[0]
            else:
                # 单帧 (H, W)
                erk_first_frame = self.erk_field

            # 下采样ERK场到CA网格尺寸
            if erk_first_frame.shape != (self.height, self.width):
                from skimage.transform import resize
                erk_down = resize(erk_first_frame, (self.height, self.width),
                                 preserve_range=True, anti_aliasing=True).astype(np.float32)
            else:
                erk_down = erk_first_frame.astype(np.float32)

            # 填充缓冲区（用第一帧）
            for _ in range(self.params.coupling.tau + 1):
                self.field_buffer.append(erk_down.copy())

            # 计算初始梯度
            self._compute_gradient_field(erk_down)

        logger.info(f"Initialized PhaseII CA: size=({self.height}, {self.width}), "
                   f"tau={self.params.coupling.tau}, gamma={self.params.coupling.gamma}")
    
    def compute_distance_field(self):
        """计算距离场（fallback用）"""
        from scipy.ndimage import distance_transform_edt
        
        wound_mask = (self.grid == 0).astype(np.uint8)
        self.distance_field = distance_transform_edt(wound_mask)
    
    def _compute_gradient_field(self, field: np.ndarray):
        """计算场梯度"""
        # 高斯平滑
        from scipy.ndimage import gaussian_filter
        field_smooth = gaussian_filter(field, sigma=self.params.coupling.D_coupling)
        
        # 计算梯度（numpy.gradient返回列表）
        gradients = np.gradient(field_smooth)
        self.grad_y = gradients[0]
        self.grad_x = gradients[1]
        self.grad_mag = np.sqrt(self.grad_y**2 + self.grad_x**2)
    
    def get_neighbors(self, i: int, j: int) -> List[Tuple[int, int]]:
        """获取Moore邻域"""
        neighbors = []
        for di in [-1, 0, 1]:
            for dj in [-1, 0, 1]:
                if di == 0 and dj == 0:
                    continue
                ni, nj = i + di, j + dj
                if 0 <= ni < self.height and 0 <= nj < self.width:
                    neighbors.append((ni, nj))
        return neighbors
    
    def get_local_density(self, i: int, j: int) -> float:
        """计算局部密度"""
        neighbors = self.get_neighbors(i, j)
        occupied = sum(1 for ni, nj in neighbors if self.grid[ni, nj] > 0)
        return occupied / len(neighbors) if neighbors else 0
    
    def is_edge_cell(self, i: int, j: int) -> bool:
        """检查是否为边缘细胞"""
        neighbors = self.get_neighbors(i, j)
        return any(self.grid[ni, nj] == 0 for ni, nj in neighbors)
    
    def migration_probability(self, i: int, j: int) -> float:
        """计算迁移概率（基础部分）"""
        density = self.get_local_density(i, j)
        p = self.params.p_move * np.exp(-self.params.alpha * density)
        
        if self.is_edge_cell(i, j):
            p *= self.params.edge_bonus
        
        return min(p, 1.0)
    
    def division_probability(self, i: int, j: int) -> float:
        """计算分裂概率"""
        density = self.get_local_density(i, j)
        p = self.params.p_div * np.exp(-self.params.beta * density)
        return min(p, 1.0)
    
    def _get_coupling_bias(self, 
                           i: int, 
                           j: int,
                           neighbor_i: int,
                           neighbor_j: int) -> float:
        """
        计算分子场耦合偏置
        
        核心公式：
        bias = γ · s · <d̂, ∇Ĉ(t-τ)>
        
        Args:
            i, j: 当前细胞位置
            neighbor_i, neighbor_j: 目标邻居位置
            
        Returns:
            耦合偏置（加到logits上）
        """
        # 如果没有ERK场，fallback到距离场
        if self.erk_field is None or len(self.field_buffer) == 0:
            if self.distance_field is not None:
                # Phase I的gamma机制（距离场）
                new_dist = self.distance_field[neighbor_i, neighbor_j]
                return -self.params.gamma * new_dist  # 偏向小距离
            else:
                return 0.0
        
        # Phase II机制：使用时滞ERK场梯度
        tau = self.params.coupling.tau
        
        # 获取时滞场（如果没有足够的历史，用最早的）
        if len(self.field_buffer) > tau:
            field_t_tau = self.field_buffer[-tau - 1]
        else:
            field_t_tau = self.field_buffer[0]
        
        # 在目标位置插值梯度
        grad_y_at_target = self.grad_y[neighbor_i, neighbor_j] if self.grad_y is not None else 0
        grad_x_at_target = self.grad_x[neighbor_i, neighbor_j] if self.grad_x is not None else 0
        
        # 计算移动方向
        d_y = neighbor_i - i
        d_x = neighbor_j - j
        d_norm = np.sqrt(d_y**2 + d_x**2) + 1e-10
        d_y_norm = d_y / d_norm
        d_x_norm = d_x / d_norm
        
        # 计算点积：<d̂, ∇Ĉ>
        dot_product = d_y_norm * grad_y_at_target + d_x_norm * grad_x_at_target
        
        # 耦合偏置
        bias = self.params.coupling.gamma * \
                self.params.coupling.s * \
                dot_product
        
        return bias
    
    def weighted_choose_neighbor(self, 
                                i: int, 
                                j: int, 
                                empty_neighbors: List[Tuple[int, int]]) -> Tuple[int, int]:
        """
        带方向偏置的邻居选择（数值稳定版本）
    
        P(choose d) ∝ exp(u_base(d) + γ·s·<d̂, ∇Ĉ(t-τ)>)
        """
        if len(empty_neighbors) == 1:
            return empty_neighbors[0]
        
        # 基础迁移概率（所有邻居相同）
        p_base = self.migration_probability(i, j)
        log_p_base = np.log(p_base + 1e-10)
        
        # 计算每个邻居的权重（修复2：数值稳定性）
        logits = []
        for ni, nj in empty_neighbors:
            # 耦合偏置
            bias = self._get_coupling_bias(i, j, ni, nj)
            
            # 修复2：总logit（clip到安全范围）
            MAX_LOGIT = 20.0  # exp(20) ≈ 4.85e8，足够大但不会溢出
            logit = np.clip(log_p_base + bias, -MAX_LOGIT, MAX_LOGIT)
            
            logits.append(logit)
        
        # 使用log-sum-exp技巧计算softmax（更稳定）
        logits = np.array(logits, dtype=np.float64)
        max_logit = np.max(logits)
        log_weights = logits - max_logit
        weights = np.exp(log_weights)
        weight_sum = weights.sum()
        
        # 归一化
        if weight_sum == 0 or not np.isfinite(weight_sum):
            return random.choice(empty_neighbors)
        
        weights = weights / weight_sum
        
        # 检查weights有效性
        if not np.all(np.isfinite(weights)) or np.any(weights < 0):
            logger.warning(f"Invalid weights: {weights}")
            return random.choice(empty_neighbors)
        
        # 加权随机选择
        try:
            idx = np.random.choice(len(empty_neighbors), p=weights)
            return empty_neighbors[idx]
        except (ValueError, RuntimeError):
            # Fallback: 等概率选择
            return random.choice(empty_neighbors)
    
    def update_field_buffer(self, new_field: Optional[np.ndarray] = None):
        """
        更新场缓冲区
        
        Args:
            new_field: 新的ERK场（如果为None，用当前缓冲区的最后）
        """
        if self.erk_field is None:
            return
        
        if new_field is not None:
            self.field_buffer.append(new_field.copy())
        
        # 重新计算梯度（基于时滞场）
        if len(self.field_buffer) > 0:
            # 使用最旧的场（对应最大时滞）
            field_for_gradient = self.field_buffer[0]
            self._compute_gradient_field(field_for_gradient)
    
    def step(self) -> Dict[str, float]:
        """执行一步CA更新"""
        # 注意：不在这里调用update_field_buffer()！
        # 场场更新只在run()方法中进行

        # 获取所有细胞位置
        cell_positions = [(i, j) for i in range(self.height)
                          for j in range(self.width) if self.grid[i, j] > 0]

        # 随机顺序异步更新
        random.shuffle(cell_positions)

        migration_count = 0
        division_count = 0

        for i, j in cell_positions:
            action = random.random()

            # 尝试迁移
            if action < self.migration_probability(i, j):
                neighbors = [(ni, nj) for ni, nj in self.get_neighbors(i, j)
                            if self.grid[ni, nj] == 0]

                if neighbors:
                    # 使用带方向偏置的选择
                    ni, nj = self.weighted_choose_neighbor(i, j, neighbors)
                    self.grid[ni, nj] = 1
                    self.grid[i, j] = 0
                    migration_count += 1

            # 尝试分裂
            elif action < self.migration_probability(i, j) + self.division_probability(i, j):
                neighbors = [(ni, nj) for ni, nj in self.get_neighbors(i, j)
                            if self.grid[ni, nj] == 0]

                if neighbors:
                    ni, nj = random.choice(neighbors)
                    self.grid[ni, nj] = 1
                    division_count += 1

            self.current_step += 1
            
        return {
            "migrations": migration_count,
            "divisions": division_count,
            "wound_area": float(np.sum(1 - self.grid)),
        }
    
    def run(self,
            num_steps: int,
            erk_field_sequence: Optional[np.ndarray] = None) -> List[Dict[str, float]]:
        """
        运行模拟
        
        Args:
            num_steps: 模拟步数
            erk_field_sequence: ERK场序列 (T, H, W)，如果提供则每步更新
        
        Returns:
            历史记录
        """
        history = []
        
        for step in range(num_steps):
            # 如果有ERK场序列，更新场（修复：只在有新数据时才更新）
            if erk_field_sequence is not None and step < erk_field_sequence.shape[0]:
                # 下采样ERK场到CA网格尺寸
                erk_step = erk_field_sequence[step]
                
                # 如果尺寸不匹配，进行下采样
                if erk_step.shape != (self.height, self.width):
                    from skimage.transform import resize
                    # 使用resize进行平滑下采样
                    erk_step = resize(erk_step, (self.height, self.width),
                                     preserve_range=True, anti_aliasing=True).astype(np.float32)
                
                # 只在这里更新场缓冲区（修复：step()中不再调用）
                self.update_field_buffer(erk_step)
            
            # 执行CA步
            stats = self.step()
            stats["step"] = step
            history.append(stats)
        
        return history
    
    def get_grid(self) -> np.ndarray:
        """获取当前网格"""
        return self.grid.copy()
    
    def get_current_field(self) -> Optional[np.ndarray]:
        """获取当前使用的场（时滞场）"""
        if len(self.field_buffer) > 0:
            tau = self.params.coupling.tau
            if len(self.field_buffer) > tau:
                return self.field_buffer[-tau - 1]
            else:
                return self.field_buffer[0]
        return None


if __name__ == "__main__":
    # 测试Phase II CA
    print("Testing Phase II Teacher-Forcing CA")
    
    # 创建参数
    params = CAParamsExtended(
        p_move=0.5,
        p_div=0.05,
        alpha=1.0,
        beta=1.0,
        edge_bonus=2.0,
        k_time=1.0,
        gamma=0.0,  # Phase I的gamma
        coupling=CouplingParams(
            gamma=1.0,
            s=+1,
            tau=1,
            D_coupling=1.0
        )
    )
    
    # 创建合成ERK场
    H, W = 50, 50
    T = 10
    erk_field = np.zeros((T, H, W))
    for t in range(T):
        x = np.linspace(-3, 3, W)
        y = np.linspace(-3, 3, H)
        X, Y = np.meshgrid(x, y)
        # 径向梯度（从中心向外）
        erk_field[t] = 1.0 / (1 + 0.5 * (X**2 + Y**2))
    
    # 创建CA
    ca = PhaseIITeacherForcingCA(H, W, params, erk_field)
    
    # 初始化
    initial_mask = np.ones((H, W), dtype=np.float32)
    initial_mask[20:30, 20:30] = 0  # 伤口
    ca.initialize_from_mask(initial_mask)
    
    # 运行模拟
    history = ca.run(num_steps=10, erk_field_sequence=erk_field)
    
    print(f"Simulation complete!")
    print(f"Initial wound area: {history[0]['wound_area']:.0f}")
    print(f"Final wound area: {history[-1]['wound_area']:.0f}")
    print(f"Total migrations: {sum(h['migrations'] for h in history)}")
    print(f"Total divisions: {sum(h['divisions'] for h in history)}")
    print(f"Parameters: gamma={params.coupling.gamma}, s={params.coupling.s}, tau={params.coupling.tau}")
    
    print("\nTest complete!")
