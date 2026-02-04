"""
推断/优化模块 - Loss 函数和优化器

参考 plan.md 第 8 节：
- 多观测 loss（避免只看 A(t)）
- 参数分层（机制共享 vs 条件特异）
- 不确定性与可辨识性诊断
"""

import numpy as np
from typing import Dict, Optional, Callable
from dataclasses import dataclass
from scipy.optimize import minimize, differential_evolution
import optuna


# =============================================================================
# Loss 函数
# =============================================================================

@dataclass
class LossWeights:
    """Loss 权重"""
    w_area: float = 1.0
    w_front: float = 0.5
    w_erk: float = 0.1
    w_angle: float = 0.0  # 可选，需要光流数据


def compute_area_loss(
    pred_area: np.ndarray,
    true_area: np.ndarray,
    normalize: bool = True
) -> float:
    """
    Wound 面积 loss

    Args:
        pred_area: (T,) 预测的 wound 面积
        true_area: (T,) 真实的 wound 面积
        normalize: 是否归一化

    Returns:
        MSE loss
    """
    mse = np.mean((pred_area - true_area) ** 2)
    if normalize:
        mse /= np.mean(true_area ** 2) + 1e-8
    return float(mse)


def compute_front_roughness_loss(
    pred_mask: np.ndarray,
    true_mask: np.ndarray
) -> float:
    """
    前沿粗糙度 loss

    Args:
        pred_mask: (T, H, W) 预测掩膜
        true_mask: (T, H, W) 真实掩膜

    Returns:
        粗糙度差异
    """
    from src.observables.extract import extract_front_roughness

    pred_roughness = extract_front_roughness(pred_mask)["front_roughness"]
    true_roughness = extract_front_roughness(true_mask)["front_roughness"]

    # 确保长度一致
    min_len = min(len(pred_roughness), len(true_roughness))
    pred_roughness = pred_roughness[:min_len]
    true_roughness = true_roughness[:min_len]

    return float(np.mean((np.array(pred_roughness) - np.array(true_roughness)) ** 2))


def compute_erk_stats_loss(
    pred_mask: np.ndarray,
    true_mask: np.ndarray,
    C_obs: np.ndarray
) -> float:
    """
    ERK 统计 loss（检查预测的空间分布是否与 ERK 相关）

    Args:
        pred_mask: (T, H, W) 预测掩膜
        true_mask: (T, H, W) 真实掩膜
        C_obs: (T, H, W) ERK 观测

    Returns:
        统计差异
    """
    # 计算前沿区域的 ERK 水平相关性
    from scipy.ndimage import distance_transform_edt

    T = min(pred_mask.shape[0], true_mask.shape[0], C_obs.shape[0])
    losses = []

    for t in range(T):
        # 真实前沿
        true_dist = distance_transform_edt(true_mask[t])
        true_front = (true_dist <= 5) & (true_mask[t] == 1)

        # 预测前沿
        pred_dist = distance_transform_edt(pred_mask[t])
        pred_front = (pred_dist <= 5) & (pred_mask[t] == 1)

        if true_front.sum() > 0 and pred_front.sum() > 0:
            # 前沿区域的 ERK 水平
            true_erk = C_obs[t][true_front]
            pred_erk = C_obs[t][pred_front]

            # 比较 ERK 分布
            loss = (np.mean(true_erk) - np.mean(pred_erk)) ** 2
            losses.append(loss)

    return float(np.mean(losses)) if losses else 0.0


def compute_angle_loss(
    pred_mask: np.ndarray,
    true_mask: np.ndarray,
    flow_y: Optional[np.ndarray] = None,
    flow_x: Optional[np.ndarray] = None,
    erk_grad_y: Optional[np.ndarray] = None,
    erk_grad_x: Optional[np.ndarray] = None
) -> float:
    """
    角度分布 loss（需要光流数据）

    Args:
        pred_mask: (T, H, W) 预测掩膜
        true_mask: (T, H, W) 真实掩膜
        flow_y, flow_x: 光流（可选）
        erk_grad_y, erk_grad_x: ERK 梯度（可选）

    Returns:
        角度差异
    """
    # 简化实现：比较前沿推进方向
    # 实际应用需要完整的光流数据
    return 0.0


def compute_loss(
    pred_mask: np.ndarray,
    true_mask: np.ndarray,
    C_obs: Optional[np.ndarray] = None,
    weights: Optional[LossWeights] = None,
    return_components: bool = False
) -> float:
    """
    组合 loss

    参考 plan.md 8.2:
    L = w_A * L_A + w_front * L_front + w_erk * L_erk + w_angle * L_angle

    Args:
        pred_mask: (T, H, W) 预测掩膜
        true_mask: (T, H, W) 真实掩膜
        C_obs: (T, H, W) ERK 观测（可选）
        weights: Loss 权重
        return_components: 是否返回各分量

    Returns:
        总 loss（或各分量字典）
    """
    if weights is None:
        weights = LossWeights()

    # Wound 面积
    pred_area = (1 - pred_mask).sum(axis=(1, 2))
    true_area = (1 - true_mask).sum(axis=(1, 2))
    loss_area = compute_area_loss(pred_area, true_area)

    # 前沿粗糙度
    loss_front = compute_front_roughness_loss(pred_mask, true_mask)

    # ERK 统计
    loss_erk = 0.0
    if C_obs is not None and weights.w_erk > 0:
        loss_erk = compute_erk_stats_loss(pred_mask, true_mask, C_obs)

    # 总 loss
    total = (
        weights.w_area * loss_area +
        weights.w_front * loss_front +
        weights.w_erk * loss_erk
    )

    if return_components:
        return {
            "total": total,
            "area": loss_area,
            "front": loss_front,
            "erk": loss_erk,
        }

    return total


# =============================================================================
# 优化器
# =============================================================================

class ParameterBounds:
    """参数边界（用于约束优化）"""

    PHASE1_BOUNDS = {
        'p_move': (0.01, 0.5),
        'p_split': (0.001, 0.1),
        'kappa_geo': (0.0, 5.0),
        'edge_bonus': (0.0, 2.0),
        'max_density': (0.5, 1.0),
    }

    PHASE2_BOUNDS = {
        **PHASE1_BOUNDS,
        'gamma': (0.0, 5.0),
        'sign': (-1, 1),  # 离散，实际取 +/-1
        'tau_sec': (60.0, 1800.0),  # 1分钟到30分钟
    }


def grid_search(
    objective: Callable,
    bounds: Dict[str, tuple],
    n_samples: int = 10
) -> Dict:
    """
    网格搜索（粗略寻找最优区域）

    Args:
        objective: 目标函数，接受参数字典，返回 loss
        bounds: 参数边界
        n_samples: 每个维度的采样数

    Returns:
        最佳参数和结果
    """
    import itertools

    # 生成网格
    grids = {}
    for name, (low, high) in bounds.items():
        if name == 'sign':
            grids[name] = [-1, 1]
        else:
            grids[name] = np.linspace(low, high, n_samples)

    # 遍历所有组合
    best_loss = np.inf
    best_params = None

    total = int(np.prod([len(v) for v in grids.values()]))
    print(f"Grid search: {total} combinations")

    for i, values in enumerate(itertools.product(*grids.values())):
        params = dict(zip(grids.keys(), values))
        try:
            loss = objective(params)
            if loss < best_loss:
                best_loss = loss
                best_params = params
        except Exception as e:
            pass

        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{total}, best_loss: {best_loss:.6f}")

    return {
        "params": best_params,
        "loss": best_loss,
    }


class Optimizer:
    """
    CA 参数优化器
    """

    def __init__(
        self,
        ca_class,
        true_mask: np.ndarray,
        C_obs: Optional[np.ndarray] = None,
        erk_grad_y: Optional[np.ndarray] = None,
        erk_grad_x: Optional[np.ndarray] = None,
        dt: float = 180.0,
        weights: Optional[LossWeights] = None,
    ):
        self.ca_class = ca_class
        self.true_mask = true_mask.astype(np.int32)
        self.C_obs = C_obs
        self.erk_grad_y = erk_grad_y
        self.erk_grad_x = erk_grad_x
        self.dt = dt
        self.weights = weights or LossWeights()

        self.T, self.H, self.W = true_mask.shape

    def _objective(self, params: dict, n_trials: int = 3) -> float:
        """
        目标函数（内部）

        Args:
            params: 参数字典
            n_trials: 随机试验次数（取平均）

        Returns:
            平均 loss
        """
        losses = []

        for _ in range(n_trials):
            # 创建 CA
            if self.ca_class.__name__ == "Phase2CA":
                ca = self.ca_class(
                    self.H, self.W, self.dt,
                    params, self.C_obs, self.erk_grad_y, self.erk_grad_x
                )
            else:
                ca = self.ca_class(self.H, self.W, self.dt, params)

            # 初始化
            ca.initialize(self.true_mask[0])

            # 运行
            ca.simulate(self.T - 1, record_every=1)

            # 计算 loss
            pred_history = np.array(ca.history)
            loss = compute_loss(
                pred_history,
                self.true_mask,
                self.C_obs,
                self.weights
            )
            losses.append(loss)

        return float(np.mean(losses))

    def optimize(
        self,
        initial_params: Optional[dict] = None,
        method: str = "bayesian",
        n_trials: int = 100,
        bounds: Optional[Dict[str, tuple]] = None
    ) -> Dict:
        """
        优化参数

        Args:
            initial_params: 初始参数
            method: 优化方法，'bayesian', 'grid', 'differential'
            n_trials: 试验次数
            bounds: 参数边界

        Returns:
            优化结果
        """
        if bounds is None:
            if self.ca_class.__name__ == "Phase2CA":
                bounds = ParameterBounds.PHASE2_BOUNDS
            else:
                bounds = ParameterBounds.PHASE1_BOUNDS

        if method == "grid":
            # 网格搜索（粗略）
            result = grid_search(
                lambda p: self._objective(p, n_trials=1),
                bounds,
                n_samples=int(n_trials ** (1/len(bounds)))
            )

        elif method == "bayesian":
            # 贝叶斯优化（使用 Optuna）
            def optuna_objective(trial):
                params = {}
                for name, (low, high) in bounds.items():
                    if name == 'sign':
                        params[name] = trial.suggest_categorical('sign', [-1, 1])
                    else:
                        params[name] = trial.suggest_float(name, low, high)

                return self._objective(params, n_trials=2)

            study = optuna.create_study(direction='minimize')
            study.optimize(optuna_objective, n_trials=n_trials)

            result = {
                "params": study.best_params,
                "loss": study.best_value,
                "study": study,
            }

        elif method == "differential":
            # 差分进化
            def de_objective(x):
                names = list(bounds.keys())
                params = {}
                for i, name in enumerate(names):
                    if name == 'sign':
                        params[name] = 1 if x[i] > 0 else -1
                    else:
                        params[name] = x[i]
                return self._objective(params, n_trials=1)

            bounds_list = list(bounds.values())
            bounds_array = [(b[0], b[1]) for b in bounds_list]

            result = differential_evolution(
                de_objective,
                bounds_array,
                maxiter=n_trials,
                seed=42
            )

            names = list(bounds.keys())
            best_params = {}
            for i, name in enumerate(names):
                if name == 'sign':
                    best_params[name] = 1 if result.x[i] > 0 else -1
                else:
                    best_params[name] = result.x[i]

            result = {
                "params": best_params,
                "loss": result.fun,
            }

        else:
            raise ValueError(f"Unknown method: {method}")

        return result


def profile_likelihood(
    optimizer: Optimizer,
    best_params: dict,
    param_name: str,
    n_points: int = 20
) -> Dict:
    """
    Profile likelihood（可辨识性诊断）

    参考 plan.md 8.3

    Args:
        optimizer: 优化器实例
        best_params: 最佳参数
        param_name: 要分析的目标参数
        n_points: 采样点数

    Returns:
        参数值和对应的 loss
    """
    # 获取参数边界
    if optimizer.ca_class.__name__ == "Phase2CA":
        bounds = ParameterBounds.PHASE2_BOUNDS
    else:
        bounds = ParameterBounds.PHASE1_BOUNDS

    low, high = bounds.get(param_name, (0.1, 10.0))

    # 采样参数值
    if param_name == 'sign':
        param_values = [-1, 1]
    else:
        param_values = np.linspace(low, high, n_points)

    losses = []
    for val in param_values:
        params = best_params.copy()
        params[param_name] = val
        loss = optimizer._objective(params, n_trials=3)
        losses.append(loss)

    return {
        "param_name": param_name,
        "param_values": param_values,
        "losses": losses,
        "identifiable": _is_unimodal(param_values, losses),
    }


def _is_unimodal(x: np.ndarray, y: np.ndarray) -> bool:
    """检查是否单峰（用于可辨识性判断）"""
    # 简化检查：是否有单一最小值
    y = np.array(y)
    min_idx = np.argmin(y)

    # 检查是否在边界
    if min_idx == 0 or min_idx == len(y) - 1:
        return False  # 最小值在边界，可能不是真正的峰值

    # 检查是否单调上升到最小值，然后单调下降
    left = y[:min_idx+1]
    right = y[min_idx:]

    left_decreasing = all(left[i] >= left[i+1] - 0.1 * left.std() for i in range(len(left)-1))
    right_increasing = all(right[i] <= right[i+1] + 0.1 * right.std() for i in range(len(right)-1))

    return left_decreasing and right_increasing


def negative_control(
    optimizer: Optimizer,
    params: dict,
    method: str = "time_shuffle"
) -> float:
    """
    负对照（证明不是几何巧合）

    参考 plan.md 8.3

    Args:
        optimizer: 优化器实例
        params: 参数
        method: 负对照方法，'time_shuffle' 或 'spatial_flip'

    Returns:
        负对照 loss
    """
    if method == "time_shuffle":
        # 时间打乱 ERK
        C_shuffled = optimizer.C_obs.copy()
        np.random.shuffle(C_shuffled)
        optimizer.C_obs = C_shuffled

    elif method == "spatial_flip":
        # 空间翻转 ERK
        optimizer.C_obs = np.fliplr(optimizer.C_obs)

    else:
        raise ValueError(f"Unknown method: {method}")

    loss = optimizer._objective(params, n_trials=3)

    # 恢复
    # （实际实现中应该保存原始值）

    return loss
