"""
改进版 CA 模型
根据建议.md实施 P0-P1.5 优先级改进

改进内容：
1. P0-1: p_div=0 硬证伪实验支持
2. P0-1 E2: 修正 k_time 与步数的精确对齐
3. P0-2: 增加 L_activity 正则项（记录总迁移次数）
4. P1: Phase II ERK 动性门控参数 η
5. P1.5: 参数语义去歧义（gamma_geo vs gamma_erk）
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Deque
from collections import deque
from dataclasses import dataclass, field
import random
import logging

logger = logging.getLogger(__name__)


@dataclass
class CAParamsImproved:
    """
    改进版 CA 参数（去歧义命名）

    参数重命名：
    - gamma_geo: 向伤口/距离场偏置（原 Phase I 的 gamma）
    - p_move, p_div, alpha, beta, edge_bonus, k_time 保持不变
    """
    # 基础参数
    p_move: float = 0.5
    p_div: float = 0.05
    alpha: float = 1.0  # 密度抑制（迁移）
    beta: float = 1.0   # 密度抑制（分裂）
    edge_bonus: float = 2.0
    k_time: float = 20.0  # 时间缩放（每小时 CA 步数）- 修复：与优化器默认值一致

    # P1.5: 去歧义参数命名
    gamma_geo: float = 0.0  # 几何方向偏置（向伤口），原名 gamma

    # P0-1: 低闭合 regime 参数
    force_no_division: bool = False  # 强制 p_div=0（硬证伪实验用）

    def __post_init__(self):
        """初始化后处理：force_no_division 时自动将 p_div 设为 0"""
        if self.force_no_division:
            self.p_div = 0.0


@dataclass
class CouplingParamsImproved:
    """
    改进版 Phase II 耦合参数

    P1 新增：
    - eta: ERK 动性门控参数（>0 稳定化，<0 促动）
    """
    # 方向偏置参数
    gamma_erk: float = 0.5   # ERK 耦合强度（原名 gamma，去歧义）
    s: int = +1              # 符号开关：+1=沿梯度，-1=逆梯度
    tau: int = 0             # 时滞（时间步）

    # P1 新增：动性门控参数
    eta: float = 0.0         # ERK 动性门控（>0 抑制迁移，<0 促进迁移）

    # 其他参数
    D_coupling: float = 1.0  # 耦合扩散系数

    def __post_init__(self):
        """确保 tau 是整数"""
        self.tau = int(self.tau)
        self.s = int(self.s)


@dataclass
class CAParamsExtendedImproved:
    """扩展参数（包含耦合参数）"""
    # 基础参数
    p_move: float = 0.5
    p_div: float = 0.05
    alpha: float = 1.0
    beta: float = 1.0
    edge_bonus: float = 2.0
    k_time: float = 20.0  # 修复：与优化器默认值一致
    gamma_geo: float = 0.0  # P1.5: 几何偏置
    force_no_division: bool = False  # P0-1: 强制无分裂

    # Phase II 耦合参数
    coupling: CouplingParamsImproved = field(default_factory=CouplingParamsImproved)

    def __post_init__(self):
        """初始化后处理：force_no_division 时自动将 p_div 设为 0"""
        if self.force_no_division:
            self.p_div = 0.0


class CellOnlyCAImproved:
    """
    改进版 Phase I CA

    改进：
    1. 支持 p_div=0 硬证伪实验（force_no_division）
    2. 精确的时间-步数对齐
    3. 记录总活动次数（用于 L_activity 正则）
    4. 去歧义参数命名（gamma_geo）
    """

    def __init__(self, height: int, width: int, params: CAParamsImproved):
        self.height = height
        self.width = width
        self.params = params
        self.grid = np.zeros((height, width), dtype=np.int32)
        self.distance_field: Optional[np.ndarray] = None

        # P0-2: 累计活动统计（用于 L_activity 正则）
        self.total_migrations = 0
        self.total_divisions = 0
        self.total_activity = []  # 每步的活动计数

    def initialize_from_mask(self, mask: np.ndarray, k: int = 1):
        """从初始 mask 初始化"""
        from skimage.transform import resize

        if k > 1:
            mask_down = resize(mask, (mask.shape[0]//k, mask.shape[1]//k),
                              preserve_range=True, anti_aliasing=True).astype(np.float32)
        else:
            mask_down = mask

        self.grid = (mask_down > 0.5).astype(np.int32)
        self.height, self.width = self.grid.shape
        self.compute_distance_field()

    def compute_distance_field(self):
        """计算距离场（用于 gamma_geo 方向偏置）"""
        from scipy.ndimage import distance_transform_edt

        wound_mask = (self.grid == 0).astype(np.uint8)
        self.distance_field = distance_transform_edt(wound_mask)

    def get_neighbors(self, i: int, j: int) -> List[Tuple[int, int]]:
        """获取 Moore 邻域"""
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
        """计算迁移概率"""
        density = self.get_local_density(i, j)
        p = self.params.p_move * np.exp(-self.params.alpha * density)

        if self.is_edge_cell(i, j):
            p *= self.params.edge_bonus

        return min(p, 1.0)

    def division_probability(self, i: int, j: int) -> float:
        """
        计算分裂概率

        P0-1: 支持 force_no_division 硬证伪实验
        """
        if self.params.force_no_division:
            return 0.0

        density = self.get_local_density(i, j)
        p = self.params.p_div * np.exp(-self.params.beta * density)
        return min(p, 1.0)

    def weighted_choose_neighbor(self, i: int, j: int,
                                  empty_neighbors: List[Tuple[int, int]]) -> Tuple[int, int]:
        """
        带 gamma_geo 方向偏置的邻居选择

        P1.5: 使用 gamma_geo 替代原 gamma
        """
        if self.params.gamma_geo <= 0 or len(empty_neighbors) == 1:
            return random.choice(empty_neighbors)

        if self.distance_field is None:
            self.compute_distance_field()

        # 计算基于距离场的权重
        weights = []
        for ni, nj in empty_neighbors:
            new_dist = self.distance_field[ni, nj] if self.distance_field is not None else 0
            # 偏向小距离（向伤口）
            weight = np.exp(-self.params.gamma_geo * new_dist)
            weights.append(weight)

        weights = np.array(weights, dtype=np.float64)
        weight_sum = weights.sum()

        if weight_sum == 0 or not np.isfinite(weight_sum):
            return random.choice(empty_neighbors)

        weights = weights / weight_sum

        if not np.all(np.isfinite(weights)):
            return random.choice(empty_neighbors)

        try:
            idx = np.random.choice(len(empty_neighbors), p=weights)
            return empty_neighbors[idx]
        except (ValueError, RuntimeError):
            return random.choice(empty_neighbors)

    def step(self) -> Dict[str, float]:
        """
        执行一步 CA 更新

        P0-2: 记录活动统计（用于 L_activity 正则）
        """
        cell_positions = [(i, j) for i in range(self.height)
                          for j in range(self.width) if self.grid[i, j] > 0]

        random.shuffle(cell_positions)

        migration_count = 0
        division_count = 0

        for i, j in cell_positions:
            action = random.random()

            if action < self.migration_probability(i, j):
                neighbors = [(ni, nj) for ni, nj in self.get_neighbors(i, j)
                            if self.grid[ni, nj] == 0]

                if neighbors:
                    ni, nj = self.weighted_choose_neighbor(i, j, neighbors)
                    self.grid[ni, nj] = 1
                    self.grid[i, j] = 0
                    migration_count += 1

            elif action < self.migration_probability(i, j) + self.division_probability(i, j):
                neighbors = [(ni, nj) for ni, nj in self.get_neighbors(i, j)
                            if self.grid[ni, nj] == 0]

                if neighbors:
                    ni, nj = random.choice(neighbors)
                    self.grid[ni, nj] = 1
                    division_count += 1

        # P0-2: 累计活动统计
        self.total_migrations += migration_count
        self.total_divisions += division_count
        self.total_activity.append(migration_count + division_count)

        wound_area = float(np.sum(1 - self.grid))

        return {
            "migrations": migration_count,
            "divisions": division_count,
            "activity": migration_count + division_count,  # P0-2
            "wound_area": wound_area,
        }

    def run(self, num_steps: int) -> List[Dict[str, float]]:
        """运行模拟"""
        history = []

        for step in range(num_steps):
            stats = self.step()
            stats["step"] = step
            history.append(stats)

        return history

    def get_total_activity(self) -> int:
        """P0-2: 获取总活动次数（用于 L_activity 正则）"""
        return self.total_migrations + self.total_divisions


class PhaseIITeacherForcingCAImproved:
    """
    改进版 Phase II Teacher-Forcing CA

    改进：
    1. P1: 新增 eta 动性门控参数
    2. P1.5: gamma_erk 去歧义命名
    3. P0-1: 支持 p_div=0 硬证伪实验
    4. P0-2: 活动统计
    """

    def __init__(self,
                 height: int,
                 width: int,
                 params: CAParamsExtendedImproved,
                 erk_field: Optional[np.ndarray] = None):
        self.height = height
        self.width = width
        self.params = params
        self.erk_field = erk_field

        self.grid = np.zeros((height, width), dtype=np.int32)
        self.field_buffer: Deque[np.ndarray] = deque(maxlen=int(params.coupling.tau) + 1)
        self.current_step = 0

        # 梯度场
        self.grad_y: Optional[np.ndarray] = None
        self.grad_x: Optional[np.ndarray] = None
        self.grad_mag: Optional[np.ndarray] = None

        # 距离场（fallback）
        self.distance_field: Optional[np.ndarray] = None

        # P0-2: 活动统计
        self.total_migrations = 0
        self.total_divisions = 0
        self.total_activity = []

        self.MAX_LOGIT = 20.0

    def initialize_from_mask(self, mask: np.ndarray, k: int = 1):
        """从初始 mask 初始化"""
        from skimage.transform import resize

        if k > 1:
            mask_down = resize(mask, (mask.shape[0]//k, mask.shape[1]//k),
                              preserve_range=True, anti_aliasing=True).astype(np.float32)
        else:
            mask_down = mask

        self.grid = (mask_down > 0.5).astype(np.int32)
        self.height, self.width = self.grid.shape
        self.compute_distance_field()

        # 初始化 ERK 场缓冲区
        if self.erk_field is not None:
            if self.erk_field.ndim == 3:
                erk_first_frame = self.erk_field[0]
            else:
                erk_first_frame = self.erk_field

            if erk_first_frame.shape != (self.height, self.width):
                erk_down = resize(erk_first_frame, (self.height, self.width),
                                 preserve_range=True, anti_aliasing=True).astype(np.float32)
            else:
                erk_down = erk_first_frame.astype(np.float32)

            for _ in range(self.params.coupling.tau + 1):
                self.field_buffer.append(erk_down.copy())

            self._compute_gradient_field(erk_down)

        logger.info(f"Initialized PhaseII CA: tau={self.params.coupling.tau}, "
                   f"gamma_erk={self.params.coupling.gamma_erk}, "
                   f"eta={self.params.coupling.eta}")

    def compute_distance_field(self):
        """计算距离场"""
        from scipy.ndimage import distance_transform_edt
        wound_mask = (self.grid == 0).astype(np.uint8)
        self.distance_field = distance_transform_edt(wound_mask)

    def _compute_gradient_field(self, field: np.ndarray):
        """计算场梯度"""
        from scipy.ndimage import gaussian_filter
        field_smooth = gaussian_filter(field, sigma=self.params.coupling.D_coupling)
        gradients = np.gradient(field_smooth)
        self.grad_y = gradients[0]
        self.grad_x = gradients[1]
        self.grad_mag = np.sqrt(self.grad_y**2 + self.grad_x**2)

    def get_neighbors(self, i: int, j: int) -> List[Tuple[int, int]]:
        """获取 Moore 邻域"""
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
        """
        计算迁移概率（基础部分）

        P1: 增加 eta 动性门控
        """
        density = self.get_local_density(i, j)
        p = self.params.p_move * np.exp(-self.params.alpha * density)

        if self.is_edge_cell(i, j):
            p *= self.params.edge_bonus

        # P1: ERK 动性门控（eta）
        # eta > 0: ERK 梯度大时抑制迁移（稳定化）
        # eta < 0: ERK 梯度大时促进迁移（促动）
        if self.erk_field is not None and len(self.field_buffer) > 0:
            if self.grad_mag is not None:
                grad_at_cell = self.grad_mag[i, j]
                # 归一化梯度强度（避免数值问题）
                grad_norm = np.tanh(grad_at_cell / 2.0)  # 归一化到 [0, 1]
                p *= np.exp(-self.params.coupling.eta * grad_norm)

        return min(p, 1.0)

    def division_probability(self, i: int, j: int) -> float:
        """
        计算分裂概率

        P0-1: 支持 force_no_division
        """
        if self.params.force_no_division:
            return 0.0

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

        P1.5: 使用 gamma_erk 替代原 gamma

        公式：bias = γ_erk · s · <d̂, ∇Ĉ(t-τ)>
        """
        if self.erk_field is None or len(self.field_buffer) == 0:
            if self.distance_field is not None:
                # Fallback 到 gamma_geo 机制
                new_dist = self.distance_field[neighbor_i, neighbor_j]
                return -self.params.gamma_geo * new_dist
            else:
                return 0.0

        tau = self.params.coupling.tau

        if len(self.field_buffer) > tau:
            field_t_tau = self.field_buffer[-tau - 1]
        else:
            field_t_tau = self.field_buffer[0]

        grad_y_at_target = self.grad_y[neighbor_i, neighbor_j] if self.grad_y is not None else 0
        grad_x_at_target = self.grad_x[neighbor_i, neighbor_j] if self.grad_x is not None else 0

        d_y = neighbor_i - i
        d_x = neighbor_j - j
        d_norm = np.sqrt(d_y**2 + d_x**2) + 1e-10
        d_y_norm = d_y / d_norm
        d_x_norm = d_x / d_norm

        dot_product = d_y_norm * grad_y_at_target + d_x_norm * grad_x_at_target

        # P1.5: 使用 gamma_erk
        bias = self.params.coupling.gamma_erk * \
                self.params.coupling.s * \
                dot_product

        return bias

    def weighted_choose_neighbor(self,
                                i: int,
                                j: int,
                                empty_neighbors: List[Tuple[int, int]]) -> Tuple[int, int]:
        """
        带方向偏置的邻居选择
        """
        if len(empty_neighbors) == 1:
            return empty_neighbors[0]

        p_base = self.migration_probability(i, j)
        log_p_base = np.log(p_base + 1e-10)

        logits = []
        for ni, nj in empty_neighbors:
            bias = self._get_coupling_bias(i, j, ni, nj)
            logit = np.clip(log_p_base + bias, -self.MAX_LOGIT, self.MAX_LOGIT)
            logits.append(logit)

        logits = np.array(logits, dtype=np.float64)
        max_logit = np.max(logits)
        log_weights = logits - max_logit
        weights = np.exp(log_weights)
        weight_sum = weights.sum()

        if weight_sum == 0 or not np.isfinite(weight_sum):
            return random.choice(empty_neighbors)

        weights = weights / weight_sum

        if not np.all(np.isfinite(weights)) or np.any(weights < 0):
            return random.choice(empty_neighbors)

        try:
            idx = np.random.choice(len(empty_neighbors), p=weights)
            return empty_neighbors[idx]
        except (ValueError, RuntimeError):
            return random.choice(empty_neighbors)

    def update_field_buffer(self, new_field: Optional[np.ndarray] = None):
        """更新场缓冲区"""
        if self.erk_field is None:
            return

        if new_field is not None:
            self.field_buffer.append(new_field.copy())

        if len(self.field_buffer) > 0:
            field_for_gradient = self.field_buffer[0]
            self._compute_gradient_field(field_for_gradient)

    def step(self) -> Dict[str, float]:
        """
        执行一步 CA 更新

        P0-2: 记录活动统计
        """
        cell_positions = [(i, j) for i in range(self.height)
                          for j in range(self.width) if self.grid[i, j] > 0]

        random.shuffle(cell_positions)

        migration_count = 0
        division_count = 0

        for i, j in cell_positions:
            action = random.random()

            if action < self.migration_probability(i, j):
                neighbors = [(ni, nj) for ni, nj in self.get_neighbors(i, j)
                            if self.grid[ni, nj] == 0]

                if neighbors:
                    ni, nj = self.weighted_choose_neighbor(i, j, neighbors)
                    self.grid[ni, nj] = 1
                    self.grid[i, j] = 0
                    migration_count += 1

            elif action < self.migration_probability(i, j) + self.division_probability(i, j):
                neighbors = [(ni, nj) for ni, nj in self.get_neighbors(i, j)
                            if self.grid[ni, nj] == 0]

                if neighbors:
                    ni, nj = random.choice(neighbors)
                    self.grid[ni, nj] = 1
                    division_count += 1

        self.total_migrations += migration_count
        self.total_divisions += division_count
        self.total_activity.append(migration_count + division_count)

        wound_area = float(np.sum(1 - self.grid))

        return {
            "migrations": migration_count,
            "divisions": division_count,
            "activity": migration_count + division_count,
            "wound_area": wound_area,
        }

    def run(self,
            num_steps: int,
            erk_field_sequence: Optional[np.ndarray] = None) -> List[Dict[str, float]]:
        """
        运行模拟

        P0-1 E2: 精确的时间-步数对齐
        num_steps 应根据 k_time 计算：num_steps = round((t_end - t0) * k_time)
        """
        history = []

        for step in range(num_steps):
            if erk_field_sequence is not None and step < erk_field_sequence.shape[0]:
                erk_step = erk_field_sequence[step]

                if erk_step.shape != (self.height, self.width):
                    from skimage.transform import resize
                    erk_step = resize(erk_step, (self.height, self.width),
                                     preserve_range=True, anti_aliasing=True).astype(np.float32)

                self.update_field_buffer(erk_step)

            stats = self.step()
            stats["step"] = step
            history.append(stats)

        return history

    def get_total_activity(self) -> int:
        """P0-2: 获取总活动次数"""
        return self.total_migrations + self.total_divisions


def calculate_steps_from_time(t_end: float, k_time: float, buffer: int = 5) -> int:
    """
    P0-1 E2: 根据 k_time 计算精确的模拟步数

    公式：num_steps = round(t_end * k_time) + buffer

    Args:
        t_end: 结束时间（小时）
        k_time: 时间缩放（每小时 CA 步数）
        buffer: 缓冲步数

    Returns:
        模拟步数
    """
    return int(np.round(t_end * k_time)) + buffer


if __name__ == "__main__":
    # 测试改进版模型
    print("Testing improved CA models...")

    # 测试 Phase I
    print("\n=== Testing Phase I Improved ===")
    params = CAParamsImproved(
        p_move=0.5,
        p_div=0.05,
        alpha=1.0,
        beta=1.0,
        edge_bonus=2.0,
        k_time=20.0,  # 20 steps per hour
        gamma_geo=1.0,  # P1.5: 使用新命名
        force_no_division=False,  # P0-1: 可以设为 True 进行硬证伪实验
    )

    ca = CellOnlyCAImproved(50, 50, params)
    initial_mask = np.ones((50, 50), dtype=np.float32)
    initial_mask[15:35, 15:35] = 0
    ca.initialize_from_mask(initial_mask)

    # P0-1 E2: 使用精确的时间-步数计算
    t_end = 3.0  # 3 小时
    num_steps = calculate_steps_from_time(t_end, params.k_time)
    print(f"Simulation: {t_end} hours, k_time={params.k_time} -> {num_steps} steps")

    history = ca.run(num_steps)

    print(f"Initial wound area: {history[0]['wound_area']:.0f}")
    print(f"Final wound area: {history[-1]['wound_area']:.0f}")
    print(f"Closure rate: {100*(1-history[-1]['wound_area']/history[0]['wound_area']):.2f}%")
    print(f"Total activity: {ca.get_total_activity()}")

    # 测试 Phase II
    print("\n=== Testing Phase II Improved ===")
    erk_field = np.zeros((10, 50, 50))
    for t in range(10):
        x = np.linspace(-3, 3, 50)
        y = np.linspace(-3, 3, 50)
        X, Y = np.meshgrid(x, y)
        erk_field[t] = 1.0 / (1 + 0.5 * (X**2 + Y**2))

    params_ext = CAParamsExtendedImproved(
        p_move=0.5,
        p_div=0.05,
        alpha=1.0,
        beta=1.0,
        edge_bonus=2.0,
        k_time=20.0,
        gamma_geo=0.0,
        force_no_division=False,
        coupling=CouplingParamsImproved(
            gamma_erk=1.0,  # P1.5: 使用新命名
            s=+1,
            tau=1,
            eta=0.5,  # P1: 动性门控参数
            D_coupling=1.0
        )
    )

    ca2 = PhaseIITeacherForcingCAImproved(50, 50, params_ext, erk_field)
    ca2.initialize_from_mask(initial_mask)

    history2 = ca2.run(num_steps, erk_field_sequence=erk_field)

    print(f"Initial wound area: {history2[0]['wound_area']:.0f}")
    print(f"Final wound area: {history2[-1]['wound_area']:.0f}")
    print(f"Closure rate: {100*(1-history2[-1]['wound_area']/history2[0]['wound_area']):.2f}%")
    print(f"Total activity: {ca2.get_total_activity()}")
    print(f"Parameters: gamma_erk={params_ext.coupling.gamma_erk}, eta={params_ext.coupling.eta}")

    print("\n=== All tests passed ===")
