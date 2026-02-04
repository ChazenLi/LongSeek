"""
细胞自动机模型（CA）- Phase I 和 Phase II

参考 plan.md 第 7 节：
- Phase I: Cell-only CA（几何偏置）
- Phase II: Teacher-forcing（ERK 耦合）
"""

import numpy as np
from numba import njit
from typing import Tuple, Dict, Optional, List
from dataclasses import dataclass


# =============================================================================
# Numba 加速的核心 CA 更新规则
# =============================================================================

@njit
def _count_neighbors(state: np.ndarray) -> np.ndarray:
    """
    计算 8 邻居中占据的细胞数

    Args:
        state: (H, W) 二值状态，0=empty, 1=cell

    Returns:
        neighbor_count: (H, W) 邻居计数 [0-8]
    """
    H, W = state.shape
    neighbor_count = np.zeros((H, W), dtype=np.int32)

    for i in range(H):
        for j in range(W):
            count = 0
            for di in (-1, 0, 1):
                for dj in (-1, 0, 1):
                    if di == 0 and dj == 0:
                        continue
                    ni, nj = i + di, j + dj
                    if 0 <= ni < H and 0 <= nj < W:
                        count += state[ni, nj]
            neighbor_count[i, j] = count

    return neighbor_count


@njit
def _distance_transform_gradient(state: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    计算到 empty 区域的距离变换梯度（几何方向场）

    Args:
        state: (H, W) 二值状态，1=cell, 0=empty

    Returns:
        grad_y: (H, W) y 方向梯度（指向 empty）
        grad_x: (H, W) x 方向梯度
    """
    H, W = state.shape

    # 简化的距离变换（迭代法）
    distance = np.zeros((H, W), dtype=np.float32)
    max_iter = 100

    # 初始化：cell 位置距离为 0
    for i in range(H):
        for j in range(W):
            if state[i, j] == 1:
                distance[i, j] = 0.0
            else:
                distance[i, j] = 1e6

    # 迭代传播
    for _ in range(max_iter):
        updated = False
        for i in range(H):
            for j in range(W):
                if state[i, j] == 0:  # 只处理 empty 区域
                    min_dist = distance[i, j]
                    for di in (-1, 0, 1):
                        for dj in (-1, 0, 1):
                            ni, nj = i + di, j + dj
                            if 0 <= ni < H and 0 <= nj < W:
                                d = distance[ni, nj] + np.sqrt(di*di + dj*dj)
                                if d < min_dist:
                                    min_dist = d
                                    updated = True
                    distance[i, j] = min_dist
        if not updated:
            break

    # 计算梯度
    grad_y = np.zeros((H, W), dtype=np.float32)
    grad_x = np.zeros((H, W), dtype=np.float32)

    for i in range(H):
        for j in range(W):
            if state[i, j] == 1:  # cell 位置
                # 梯度指向距离增加的方向（即指向 empty）
                min_d = distance[i, j]
                best_di, best_dj = 0.0, 0.0

                for di in (-1, 0, 1):
                    for dj in (-1, 0, 1):
                        ni, nj = i + di, j + dj
                        if 0 <= ni < H and 0 <= nj < W and state[ni, nj] == 0:
                            if distance[ni, nj] > min_d:
                                min_d = distance[ni, nj]
                                best_di, best_dj = float(di), float(dj)

                # 归一化
                norm = np.sqrt(best_di**2 + best_dj**2)
                if norm > 0:
                    grad_y[i, j] = best_di / norm
                    grad_x[i, j] = best_dj / norm

    return grad_y, grad_x


@njit
def _update_phase1(
    state: np.ndarray,
    p_move: float,
    p_split: float,
    kappa_geo: float,
    edge_bonus: float,
    max_density: float
) -> np.ndarray:
    """
    Phase I 更新：Cell-only CA（几何偏置）

    参考 plan.md 7.1:
    1. 是否移动/分裂：由拥挤抑制 + 边缘加成控制
    2. 往哪移动：由几何方向场偏置

    Args:
        state: (H, W) 当前状态，0=empty, 1=cell
        p_move: 基础移动概率
        p_split: 分裂概率
        kappa_geo: 几何偏置强度
        edge_bonus: 边缘加成（前沿细胞的额外移动概率）
        max_density: 最大局部密度（拥挤抑制）

    Returns:
        new_state: (H, W) 更新后的状态
    """
    H, W = state.shape
    new_state = state.copy()

    # 计算邻居计数
    neighbor_count = _count_neighbors(state)

    # 计算几何方向场
    geo_grad_y, geo_grad_x = _distance_transform_gradient(state)

    # 定义 8 个方向
    directions = [(-1, -1), (-1, 0), (-1, 1),
                  (0, -1),           (0, 1),
                  (1, -1),  (1, 0),  (1, 1)]

    # 随机数生成器（每个 cell 决策一次）
    np.random.seed(None)  # 使用随机种子

    for i in range(H):
        for j in range(W):
            if state[i, j] == 0:  # empty，可能被占据
                continue

            # 检查是否是边缘细胞
            is_edge = neighbor_count[i, j] < 8

            # 1. 决定是否移动/分裂
            # 拥挤抑制
            density = neighbor_count[i, j] / 8.0
            crowded = density > max_density

            # 移动概率
            if is_edge:
                move_prob = p_move * (1 + edge_bonus)
            else:
                move_prob = p_move * 0.1  # 内部细胞很少移动

            if crowded:
                move_prob *= 0.5

            # 决定是否移动
            if np.random.random() < move_prob:
                # 2. 决定移动方向
                # 计算每个方向的概率
                log_probs = np.zeros(8, dtype=np.float32)

                for d_idx, (di, dj) in enumerate(directions):
                    ni, nj = i + di, j + dj

                    # 检查边界
                    if not (0 <= ni < H and 0 <= nj < W):
                        log_probs[d_idx] = -10.0  # 不允许移出边界
                        continue

                    # 倾向于移动到 empty
                    if state[ni, nj] == 1:
                        log_probs[d_idx] = -5.0  # 已占据，低概率
                    else:
                        log_probs[d_idx] = 1.0  # empty，基准概率

                        # 几何偏置
                        alignment = geo_grad_y[i, j] * di + geo_grad_x[i, j] * dj
                        log_probs[d_idx] += kappa_geo * alignment

                # Softmax
                max_log = np.max(log_probs)
                log_probs -= max_log
                probs = np.exp(log_probs)
                probs /= np.sum(probs)

                # 采样方向
                cumsum = 0.0
                r = np.random.random()
                for d_idx, (di, dj) in enumerate(directions):
                    cumsum += probs[d_idx]
                    if r < cumsum:
                        ni, nj = i + di, j + dj
                        if 0 <= ni < H and 0 <= nj < W and state[ni, nj] == 0:
                            new_state[ni, nj] = 1
                        break

            # 3. 分裂（在原位置产生新细胞，如果空间允许）
            if np.random.random() < p_split:
                # 找一个空邻居
                empty_neighbors = []
                for di, dj in directions:
                    ni, nj = i + di, j + dj
                    if 0 <= ni < H and 0 <= nj < W and state[ni, nj] == 0:
                        empty_neighbors.append((ni, nj))

                if empty_neighbors:
                    ni, nj = empty_neighbors[np.random.randint(len(empty_neighbors))]
                    new_state[ni, nj] = 1

    return new_state


@njit
def _update_phase2(
    state: np.ndarray,
    C_obs_lagged: np.ndarray,
    grad_y: np.ndarray,
    grad_x: np.ndarray,
    p_move: float,
    p_split: float,
    kappa_geo: float,
    gamma: float,
    sign: int,
    edge_bonus: float,
    max_density: float
) -> np.ndarray:
    """
    Phase II 更新：Teacher-forcing（ERK 耦合）

    参考 plan.md 7.2:
    P(d|move) = softmax_d(u_geo(d) + γ·s·<d, ∇C(t-τ)>)

    Args:
        state: (H, W) 当前状态
        C_obs_lagged: (H, W) 时滞后的 ERK 活性
        grad_y: (H, W) ERK 梯度 y 分量
        grad_x: (H, W) ERK 梯度 x 分量
        p_move, p_split, kappa_geo, edge_bonus, max_density: 同 Phase I
        gamma: ERK 耦合强度
        sign: 符号开关（+1 或 -1）

    Returns:
        new_state: (H, W) 更新后的状态
    """
    H, W = state.shape
    new_state = state.copy()

    neighbor_count = _count_neighbors(state)
    geo_grad_y, geo_grad_x = _distance_transform_gradient(state)

    directions = [(-1, -1), (-1, 0), (-1, 1),
                  (0, -1),           (0, 1),
                  (1, -1),  (1, 0),  (1, 1)]

    for i in range(H):
        for j in range(W):
            if state[i, j] == 0:
                continue

            is_edge = neighbor_count[i, j] < 8
            density = neighbor_count[i, j] / 8.0
            crowded = density > max_density

            if is_edge:
                move_prob = p_move * (1 + edge_bonus)
            else:
                move_prob = p_move * 0.1

            if crowded:
                move_prob *= 0.5

            if np.random.random() < move_prob:
                # 方向选择：几何 + ERK
                log_probs = np.zeros(8, dtype=np.float32)

                # 归一化 ERK 梯度（使其在同一尺度）
                erk_mag = np.sqrt(grad_y[i, j]**2 + grad_x[i, j]**2) + 1e-6
                erk_norm_y = grad_y[i, j] / erk_mag
                erk_norm_x = grad_x[i, j] / erk_mag

                for d_idx, (di, dj) in enumerate(directions):
                    ni, nj = i + di, j + dj

                    if not (0 <= ni < H and 0 <= nj < W):
                        log_probs[d_idx] = -10.0
                        continue

                    if state[ni, nj] == 1:
                        log_probs[d_idx] = -5.0
                    else:
                        log_probs[d_idx] = 1.0

                        # 几何偏置
                        geo_alignment = geo_grad_y[i, j] * di + geo_grad_x[i, j] * dj
                        log_probs[d_idx] += kappa_geo * geo_alignment

                        # ERK 偏置（核心！）
                        # sign 决定沿/逆梯度方向
                        erk_alignment = erk_norm_y * di + erk_norm_x * dj
                        log_probs[d_idx] += gamma * sign * erk_alignment

                # Softmax
                max_log = np.max(log_probs)
                log_probs -= max_log
                probs = np.exp(log_probs)
                probs /= np.sum(probs)

                cumsum = 0.0
                r = np.random.random()
                for d_idx, (di, dj) in enumerate(directions):
                    cumsum += probs[d_idx]
                    if r < cumsum:
                        ni, nj = i + di, j + dj
                        if 0 <= ni < H and 0 <= nj < W and state[ni, nj] == 0:
                            new_state[ni, nj] = 1
                        break

            if np.random.random() < p_split:
                empty_neighbors = []
                for di, dj in directions:
                    ni, nj = i + di, j + dj
                    if 0 <= ni < H and 0 <= nj < W and state[ni, nj] == 0:
                        empty_neighbors.append((ni, nj))

                if empty_neighbors:
                    ni, nj = empty_neighbors[np.random.randint(len(empty_neighbors))]
                    new_state[ni, nj] = 1

    return new_state


# =============================================================================
# Python 封装的 CA 模型
# =============================================================================

@dataclass
class Phase1Params:
    """Phase I 模型参数"""
    p_move: float = 0.1      # 基础移动概率
    p_split: float = 0.01    # 分裂概率
    kappa_geo: float = 1.0   # 几何偏置强度
    edge_bonus: float = 0.5  # 边缘加成
    max_density: float = 0.8 # 最大局部密度


@dataclass
class Phase2Params(Phase1Params):
    """Phase II 模型参数（继承 Phase I + ERK 耦合）"""
    gamma: float = 1.0       # ERK 耦合强度
    sign: int = 1            # 符号开关（+1 或 -1）
    tau_sec: float = 600.0   # 时滞（秒）


class CellularAutomaton:
    """
    细胞自动机基类
    """

    def __init__(self, H: int, W: int, dt: float):
        self.H = H
        self.W = W
        self.dt = dt
        self.state: Optional[np.ndarray] = None
        self.history: List[np.ndarray] = []

    def initialize(self, mask: np.ndarray):
        """从掩膜初始化状态"""
        self.state = mask.astype(np.int32).copy()
        self.history = [self.state.copy()]

    def step(self) -> np.ndarray:
        """执行一步更新（子类实现）"""
        raise NotImplementedError

    def simulate(self, n_steps: int, record_every: int = 1) -> np.ndarray:
        """
        运行模拟

        Args:
            n_steps: 步数
            record_every: 记录间隔

        Returns:
            history: (T, H, W) 状态历史
        """
        for step in range(n_steps):
            self.state = self.step()
            if step % record_every == 0:
                self.history.append(self.state.copy())

        return np.array(self.history)

    def get_wound_area(self) -> np.ndarray:
        """获取 wound 面积时间序列"""
        history = np.array(self.history)
        return (1 - history).sum(axis=(1, 2))

    def reset(self):
        """重置状态"""
        self.history = []
        if self.state is not None:
            self.history = [self.state.copy()]


class Phase1CA(CellularAutomaton):
    """
    Phase I: Cell-only CA

    参考 plan.md 7.1
    """

    def __init__(self, H: int, W: int, dt: float, params: Phase1Params):
        super().__init__(H, W, dt)
        self.params = params

    def step(self) -> np.ndarray:
        """执行一步 Phase I 更新"""
        return _update_phase1(
            self.state,
            self.params.p_move,
            self.params.p_split,
            self.params.kappa_geo,
            self.params.edge_bonus,
            self.params.max_density
        )


class Phase2CA(CellularAutomaton):
    """
    Phase II: Teacher-forcing CA

    参考 plan.md 7.2
    """

    def __init__(
        self,
        H: int,
        W: int,
        dt: float,
        params: Phase2Params,
        C_obs: np.ndarray,
        erk_grad_y: Optional[np.ndarray] = None,
        erk_grad_x: Optional[np.ndarray] = None
    ):
        super().__init__(H, W, dt)
        self.params = params
        self.C_obs = C_obs  # (T, H, W) 完整 ERK 序列
        self.erk_grad_y = erk_grad_y
        self.erk_grad_x = erk_grad_x

        # 计算 tau_frames
        self.tau_frames = max(1, int(round(params.tau_sec / dt)))
        self.current_step = 0

    def step(self) -> np.ndarray:
        """执行一步 Phase II 更新"""
        # 获取时滞后的 ERK 梯度
        t_lagged = min(self.current_step, self.C_obs.shape[0] - 1)

        if self.erk_grad_y is None:
            # 动态计算梯度
            from src.preprocess.normalize import compute_gradient
            C_lagged = self.C_obs[t_lagged]
            _, grad_x, grad_y = compute_gradient(C_lagged[np.newaxis, ...], sigma=2.0)
            erk_grad_y = grad_y[0]
            erk_grad_x = grad_x[0]
        else:
            erk_grad_y = self.erk_grad_y[t_lagged]
            erk_grad_x = self.erk_grad_x[t_lagged]

        self.current_step += 1

        return _update_phase2(
            self.state,
            self.C_obs[t_lagged],
            erk_grad_y,
            erk_grad_x,
            self.params.p_move,
            self.params.p_split,
            self.params.kappa_geo,
            self.params.gamma,
            self.params.sign,
            self.params.edge_bonus,
            self.params.max_density
        )

    def reset(self):
        """重置状态"""
        super().reset()
        self.current_step = 0


def create_ca(
    phase: str,
    H: int,
    W: int,
    dt: float,
    params: dict,
    C_obs: Optional[np.ndarray] = None,
    erk_grad_y: Optional[np.ndarray] = None,
    erk_grad_x: Optional[np.ndarray] = None
) -> CellularAutomaton:
    """
    工厂函数：创建 CA 模型

    Args:
        phase: 'phase1' 或 'phase2'
        H, W: 空间维度
        dt: 时间步长（秒）
        params: 参数字典
        C_obs: ERK 观测序列（Phase II 必需）
        erk_grad_y, erk_grad_x: ERK 梯度（可选）

    Returns:
        CA 模型实例
    """
    if phase == "phase1":
        params_obj = Phase1Params(**params)
        return Phase1CA(H, W, dt, params_obj)
    elif phase == "phase2":
        params_obj = Phase2Params(**params)
        if C_obs is None:
            raise ValueError("C_obs required for Phase II")
        return Phase2CA(H, W, dt, params_obj, C_obs, erk_grad_y, erk_grad_x)
    else:
        raise ValueError(f"Unknown phase: {phase}")
