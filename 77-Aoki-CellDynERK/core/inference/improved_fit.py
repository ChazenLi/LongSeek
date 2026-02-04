"""
改进版贝叶斯优化器
根据建议.md实施 P0-P1.5 优先级改进

改进内容：
1. P0-2: L_activity 正则项（总迁移次数/单位时间）
2. P0-1 E2: 精确的时间-步数对齐
3. 支持改进版参数（gamma_geo, gamma_erk, eta）
4. 扩展参数搜索空间
"""

import numpy as np
from typing import Dict, List, Callable, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class ImprovedLossCalculator:
    """
    改进版损失计算器

    P0-2: 新增 L_activity 正则项
    """

    def __init__(self,
                 lambda_activity: float = 0.001,
                 target_activity: Optional[float] = None):
        """
        Args:
            lambda_activity: 活动正则权重
            target_activity: 目标活动水平（如果为None，则使用相对正则）
        """
        self.lambda_activity = lambda_activity
        self.target_activity = target_activity

    def calculate_loss(
        self,
        simulated: List[Dict[str, float]],
        observed: Dict[str, np.ndarray],
        observed_time: np.ndarray,
        k_time: float = 1.0,
        weights: Optional[Dict[str, float]] = None,
        frame_weights: Optional[np.ndarray] = None,
        noise_levels: Optional[np.ndarray] = None,
    ) -> Tuple[float, Dict[str, float]]:
        """
        计算总损失（包含 L_activity 正则）

        Returns:
            (total_loss, loss_components) 元组
        """
        if weights is None:
            weights = {
                "A_t": 1.0,
                "roughness_perimeter": 0.5,
                "roughness_height_field": 0.5,
                "width_mean": 0.3,
                "velocity": 0.2,
            }

        loss = 0.0
        loss_components = {}
        num_steps = len(simulated)

        # P0-1 E2: 转换 CA 步数到真实时间
        sim_time_real = np.arange(num_steps) / k_time

        # === 伤口面积损失 ===
        if "A_t" in observed:
            sim_A = np.array([s["wound_area"] for s in simulated])
            obs_A = observed["A_t"]

            # 归一化到初始值
            sim_A_norm = sim_A / sim_A[0] if sim_A[0] > 0 else sim_A
            obs_A_norm = obs_A / obs_A[0] if obs_A[0] > 0 else obs_A

            # 插值到观测时间点
            from scipy.interpolate import interp1d

            if num_steps > 1:
                f_sim = interp1d(sim_time_real, sim_A_norm, kind='linear',
                               bounds_error=False, fill_value="extrapolate")
                sim_at_obs_time = f_sim(observed_time)
            else:
                sim_at_obs_time = np.full_like(obs_A_norm, sim_A_norm[0])

            # 计算残差
            residuals = (sim_at_obs_time - obs_A_norm) ** 2

            if frame_weights is not None:
                residuals = residuals * frame_weights

            if noise_levels is not None and len(noise_levels) == len(residuals):
                sigma_sq = noise_levels ** 2 + 1e-10
                residuals = residuals / sigma_sq

            mse_A = np.mean(residuals)
            loss += weights.get("A_t", 1.0) * mse_A
            loss_components["mse_area"] = mse_A

        # === P0-2: L_activity 正则项 ===
        # 计算总活动（迁移 + 分裂）
        if "activity" in simulated[0]:
            activities = np.array([s["activity"] for s in simulated])
            total_activity = np.sum(activities)
            avg_activity_per_step = np.mean(activities)

            # 方法1: 绝对活动正则（如果提供了目标）
            if self.target_activity is not None:
                activity_loss = (avg_activity_per_step - self.target_activity) ** 2
            else:
                # 方法2: 相对活动正则（惩罚过高活动）
                # 真实数据几乎不闭合 → 活动应该很低
                # 使用 Huber-like 损失：低活动时惩罚小，高活动时惩罚大
                threshold = 10.0  # 每步10次活动被认为是"低活动"
                if avg_activity_per_step < threshold:
                    activity_loss = 0.1 * avg_activity_per_step  # 轻微惩罚
                else:
                    activity_loss = (avg_activity_per_step - threshold) ** 2

            loss += self.lambda_activity * activity_loss
            loss_components["activity_loss"] = activity_loss
            loss_components["avg_activity"] = avg_activity_per_step
            loss_components["total_activity"] = total_activity

        # === 速度损失 ===
        if len(observed["A_t"]) >= 2 and num_steps >= 2:
            obs_velocities = []
            for i in range(len(observed["A_t"]) - 1):
                delta_A = observed["A_t"][i+1] - observed["A_t"][i]
                delta_t = observed_time[i+1] - observed_time[i]
                if delta_t > 0:
                    obs_velocities.append(delta_A / delta_t)

            if len(obs_velocities) > 0:
                sim_A = np.array([s["wound_area"] for s in simulated])
                sim_A_norm = sim_A / sim_A[0] if sim_A[0] > 0 else sim_A

                if num_steps > 1:
                    from scipy.interpolate import interp1d
                    f_sim = interp1d(sim_time_real, sim_A_norm, kind='linear',
                                   fill_value='extrapolate')
                    sim_at_obs_time = f_sim(observed_time)

                    sim_velocities = []
                    for i in range(len(sim_at_obs_time) - 1):
                        delta_A = sim_at_obs_time[i+1] - sim_at_obs_time[i]
                        delta_t = observed_time[i+1] - observed_time[i]
                        if delta_t > 0:
                            sim_velocities.append(delta_A / delta_t)

                    if len(sim_velocities) == len(obs_velocities):
                        vel_loss = np.mean((np.array(sim_velocities) - np.array(obs_velocities)) ** 2)
                        loss += weights.get("velocity", 0.2) * vel_loss
                        loss_components["velocity_loss"] = vel_loss

        loss_components["total_loss"] = loss
        return loss, loss_components


def run_simulation_improved(
    params: dict,
    initial_mask: np.ndarray,
    num_steps: int,
    downsample_k: int = 4,
    seed: Optional[int] = None,
    erk_field: Optional[np.ndarray] = None,
    erk_field_sequence: Optional[np.ndarray] = None,
    model_type: str = "phase1",
) -> List[Dict[str, float]]:
    """
    运行改进版模拟

    Args:
        params: 参数字典（包含改进版参数）
        model_type: "phase1" 或 "phase2"
        erk_field: ERK 场（Phase II 需要）
        erk_field_sequence: ERK 场序列
    """
    from core.ca.improved_model import (
        CellOnlyCAImproved,
        PhaseIITeacherForcingCAImproved,
        CAParamsImproved,
        CAParamsExtendedImproved,
        CouplingParamsImproved,
    )

    if seed is not None:
        np.random.seed(seed)
        import random
        random.seed(seed)

    H, W = initial_mask.shape

    if model_type == "phase1":
        ca_params = CAParamsImproved(
            p_move=params.get("p_move", 0.5),
            p_div=params.get("p_div", 0.05),
            alpha=params.get("alpha", 1.0),
            beta=params.get("beta", 1.0),
            edge_bonus=params.get("edge_bonus", 2.0),
            k_time=params.get("k_time", 20.0),
            gamma_geo=params.get("gamma_geo", 0.0),
            force_no_division=params.get("force_no_division", False),
        )
        ca = CellOnlyCAImproved(H // downsample_k, W // downsample_k, ca_params)
        ca.initialize_from_mask(initial_mask, k=downsample_k)
        history = ca.run(num_steps)

    elif model_type == "phase2":
        coupling = CouplingParamsImproved(
            gamma_erk=params.get("gamma_erk", 0.5),
            s=params.get("s", 1),
            tau=params.get("tau", 0),
            eta=params.get("eta", 0.0),
            D_coupling=params.get("D_coupling", 1.0),
        )
        ca_params = CAParamsExtendedImproved(
            p_move=params.get("p_move", 0.5),
            p_div=params.get("p_div", 0.05),
            alpha=params.get("alpha", 1.0),
            beta=params.get("beta", 1.0),
            edge_bonus=params.get("edge_bonus", 2.0),
            k_time=params.get("k_time", 20.0),
            gamma_geo=params.get("gamma_geo", 0.0),
            force_no_division=params.get("force_no_division", False),
            coupling=coupling,
        )
        ca = PhaseIITeacherForcingCAImproved(H // downsample_k, W // downsample_k,
                                            ca_params, erk_field)
        ca.initialize_from_mask(initial_mask, k=downsample_k)
        history = ca.run(num_steps, erk_field_sequence=erk_field_sequence)

    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    return history


def objective_function_improved(
    params_array: np.ndarray,
    initial_mask: np.ndarray,
    observed: Dict[str, np.ndarray],
    observed_time: np.ndarray,
    downsample_k: int = 4,
    seed: Optional[int] = None,
    frame_weights: Optional[np.ndarray] = None,
    noise_levels: Optional[np.ndarray] = None,
    lambda_activity: float = 0.001,
    erk_field: Optional[np.ndarray] = None,
    erk_field_sequence: Optional[np.ndarray] = None,
    model_type: str = "phase1",
    param_names: List[str] = None,
    force_no_division: bool = False,  # 修复：添加此参数
) -> float:
    """
    改进版目标函数

    支持：
    - P0-2: L_activity 正则
    - P0-1 E2: 精确时间-步数对齐
    - 改进版参数（gamma_geo, gamma_erk, eta）
    - P0-1 E1: force_no_division 硬证伪实验
    """
    if param_names is None:
        # 默认 Phase I 参数顺序
        param_names = ["p_move", "p_div", "alpha", "beta", "edge_bonus", "k_time", "gamma_geo"]

    # 构建参数字典
    params = {}
    for i, name in enumerate(param_names):
        params[name] = float(params_array[i])

    # 修复：P0-1 E1: force_no_division 时强制设 p_div=0
    if force_no_division:
        params["p_div"] = 0.0
        params["force_no_division"] = True

    # P0-1 E2: 根据 k_time 计算精确步数
    t_end = observed_time[-1] if len(observed_time) > 0 else 3.0
    k_time = params.get("k_time", 20.0)
    num_steps = int(np.round(t_end * k_time)) + 5

    # 修复：Phase II 需要完整的 ERK 场序列
    if erk_field is not None and erk_field_sequence is None:
        # 如果只给了 erk_field，用它创建序列
        if erk_field.ndim == 3:
            erk_field_sequence = erk_field[:num_steps] if num_steps <= erk_field.shape[0] else erk_field
        else:
            # 单帧，重复
            erk_field_sequence = np.repeat(erk_field[np.newaxis, ...], num_steps, axis=0)

    # 运行模拟
    history = run_simulation_improved(
        params=params,
        initial_mask=initial_mask,
        num_steps=num_steps,
        downsample_k=downsample_k,
        seed=seed,
        erk_field=erk_field,
        erk_field_sequence=erk_field_sequence,
        model_type=model_type,
    )

    # 计算损失（含 L_activity 正则）
    loss_calc = ImprovedLossCalculator(lambda_activity=lambda_activity)
    loss, components = loss_calc.calculate_loss(
        history, observed, observed_time, k_time=k_time,
        frame_weights=frame_weights, noise_levels=noise_levels,
    )

    return loss


def fit_improved_phase1(
    initial_mask: np.ndarray,
    observed: Dict[str, np.ndarray],
    observed_time: np.ndarray,
    n_calls: int = 100,
    n_random_starts: int = 20,
    random_state: Optional[int] = None,
    downsample_k: int = 4,
    lambda_activity: float = 0.001,
    force_no_division: bool = False,
) -> Dict:
    """
    改进版 Phase I 贝叶斯优化

    P0-1 E1: 支持 force_no_division 硬证伪实验
    P0-2: L_activity 正则
    P0-1 E2: 精确时间-步数对齐
    """
    try:
        from skopt import gp_minimize
        from skopt.space import Real
    except ImportError:
        logger.warning("scikit-optimize not installed, using random search")
        return fit_improved_random_search(
            initial_mask, observed, observed_time,
            downsample_k, lambda_activity, force_no_division,
            model_type="phase1"
        )

    # 修复：force_no_division 时固定 p_div=0
    if force_no_division:
        space = [
            Real(0.01, 0.99, name='p_move'),
            Real(0.0, 0.0, name='p_div'),  # 固定为 0
            Real(0.1, 5.0, name='alpha'),
            Real(0.1, 5.0, name='beta'),
            Real(0.5, 5.0, name='edge_bonus'),
            Real(5.0, 50.0, name='k_time'),
            Real(0.0, 2.0, name='gamma_geo'),
        ]
    else:
        space = [
            Real(0.01, 0.99, name='p_move'),
            Real(0.0, 0.1, name='p_div'),
            Real(0.1, 5.0, name='alpha'),
            Real(0.1, 5.0, name='beta'),
            Real(0.5, 5.0, name='edge_bonus'),
            Real(5.0, 50.0, name='k_time'),
            Real(0.0, 2.0, name='gamma_geo'),
        ]

    param_names = ['p_move', 'p_div', 'alpha', 'beta', 'edge_bonus', 'k_time', 'gamma_geo']

    def objective(params_array):
        return objective_function_improved(
            params_array=params_array,
            initial_mask=initial_mask,
            observed=observed,
            observed_time=observed_time,
            downsample_k=downsample_k,
            seed=random_state,
            lambda_activity=lambda_activity,
            model_type="phase1",
            param_names=param_names,
            force_no_division=force_no_division,  # 修复：传递参数
        )

    result = gp_minimize(
        objective,
        space,
        n_calls=n_calls,
        n_initial_points=n_random_starts,
        random_state=random_state,
        verbose=True,
        acq_func='EI',  # Expected Improvement
    )

    # 构建结果
    best_params = {}
    for i, name in enumerate(param_names):
        best_params[name] = float(result.x[i])
    best_params["force_no_division"] = force_no_division

    return {
        "best_params": best_params,
        "best_loss": float(result.fun),
        "result": result,
    }


def fit_improved_phase2(
    initial_mask: np.ndarray,
    observed: Dict[str, np.ndarray],
    observed_time: np.ndarray,
    erk_field: np.ndarray,
    erk_field_sequence: Optional[np.ndarray] = None,
    n_calls: int = 100,
    n_random_starts: int = 20,
    random_state: Optional[int] = None,
    downsample_k: int = 4,
    lambda_activity: float = 0.001,
    force_no_division: bool = False,
) -> Dict:
    """
    改进版 Phase II 贝叶斯优化

    P1: 新增 eta 参数
    P1.5: gamma_erk 去歧义

    修复：正确传递 erk_field_sequence
    """
    try:
        from skopt import gp_minimize
        from skopt.space import Real, Categorical
    except ImportError:
        logger.warning("scikit-optimize not installed, using random search")
        return fit_improved_random_search(
            initial_mask, observed, observed_time,
            downsample_k, lambda_activity, force_no_division,
            erk_field, erk_field_sequence, "phase2"
        )

    # 修复：force_no_division 时固定 p_div=0
    p_div_bounds = (0.0, 0.0) if force_no_division else (0.0, 0.1)

    # 定义搜索空间（Phase II）
    space = [
        Real(0.01, 0.99, name='p_move'),
        Real(p_div_bounds[0], p_div_bounds[1], name='p_div'),  # P0-1: 降低 p_div 上限
        Real(0.1, 5.0, name='alpha'),
        Real(0.1, 5.0, name='beta'),
        Real(0.5, 5.0, name='edge_bonus'),
        Real(5.0, 50.0, name='k_time'),
        Real(0.0, 2.0, name='gamma_geo'),
        # Phase II 耦合参数
        Real(0.0, 2.0, name='gamma_erk'),  # P1.5: 去歧义命名
        Categorical([-1, 1], name='s'),
        Real(0, 3, name='tau'),
        Real(-1.0, 1.0, name='eta'),  # P1: 动性门控参数
        Real(0.5, 2.0, name='D_coupling'),
    ]

    param_names = ['p_move', 'p_div', 'alpha', 'beta', 'edge_bonus', 'k_time', 'gamma_geo',
                   'gamma_erk', 's', 'tau', 'eta', 'D_coupling']

    def objective(params_array):
        # 处理 Categorical 参数
        processed_params = []
        for i, (val, name) in enumerate(zip(params_array, param_names)):
            if name == 's':
                processed_params.append(int(val))
            else:
                processed_params.append(float(val))

        return objective_function_improved(
            params_array=np.array(processed_params),
            initial_mask=initial_mask,
            observed=observed,
            observed_time=observed_time,
            downsample_k=downsample_k,
            seed=random_state,
            lambda_activity=lambda_activity,
            erk_field=erk_field,
            erk_field_sequence=erk_field_sequence,  # 修复：传递序列
            model_type="phase2",
            param_names=param_names,
            force_no_division=force_no_division,  # 修复：传递参数
        )

    result = gp_minimize(
        objective,
        space,
        n_calls=n_calls,
        n_initial_points=n_random_starts,
        random_state=random_state,
        verbose=True,
        acq_func='EI',
    )

    best_params = {}
    for i, name in enumerate(param_names):
        if name == 's':
            best_params[name] = int(result.x[i])
        else:
            best_params[name] = float(result.x[i])
    best_params["force_no_division"] = force_no_division

    return {
        "best_params": best_params,
        "best_loss": float(result.fun),
        "result": result,
    }


def fit_improved_random_search(
    initial_mask: np.ndarray,
    observed: Dict[str, np.ndarray],
    observed_time: np.ndarray,
    downsample_k: int = 4,
    lambda_activity: float = 0.001,
    force_no_division: bool = False,
    erk_field: Optional[np.ndarray] = None,
    erk_field_sequence: Optional[np.ndarray] = None,
    model_type: str = "phase1",
    n_samples: int = 50,
) -> Dict:
    """随机搜索后备方案"""
    best_loss = float('inf')
    best_params = None

    if model_type == "phase1":
        param_names = ['p_move', 'p_div', 'alpha', 'beta', 'edge_bonus', 'k_time', 'gamma_geo']
        bounds = [
            (0.01, 0.99),
            (0.0, 0.1),
            (0.1, 5.0),
            (0.1, 5.0),
            (0.5, 5.0),
            (5.0, 50.0),
            (0.0, 2.0),
        ]
    else:
        param_names = ['p_move', 'p_div', 'alpha', 'beta', 'edge_bonus', 'k_time', 'gamma_geo',
                       'gamma_erk', 's', 'tau', 'eta', 'D_coupling']
        bounds = [
            (0.01, 0.99),
            (0.0, 0.1),
            (0.1, 5.0),
            (0.1, 5.0),
            (0.5, 5.0),
            (5.0, 50.0),
            (0.0, 2.0),
            (0.0, 2.0),
            (-1, 1),
            (0, 3),
            (-1.0, 1.0),
            (0.5, 2.0),
        ]

    for i in range(n_samples):
        params = {}
        for name, bound in zip(param_names, bounds):
            if name == 's':
                params[name] = np.random.choice([-1, 1])
            else:
                params[name] = float(np.random.uniform(bound[0], bound[1]))
        params["force_no_division"] = force_no_division

        # 构建参数数组
        params_array = np.array([params[name] for name in param_names])

        loss = objective_function_improved(
            params_array=params_array,
            initial_mask=initial_mask,
            observed=observed,
            observed_time=observed_time,
            downsample_k=downsample_k,
            lambda_activity=lambda_activity,
            erk_field=erk_field,
            erk_field_sequence=erk_field_sequence,
            model_type=model_type,
            param_names=param_names,
        )

        if loss < best_loss:
            best_loss = loss
            best_params = params

        if (i + 1) % 10 == 0:
            logger.info(f"Random search {i+1}/{n_samples}: best_loss={best_loss:.4f}")

    return {
        "best_params": best_params,
        "best_loss": best_loss,
    }


if __name__ == "__main__":
    # 测试
    print("Testing improved fit module...")

    # 创建合成数据
    initial_mask = np.ones((100, 100), dtype=np.float32)
    initial_mask[30:70, 30:70] = 0.0

    observed_time = np.linspace(0, 3, 61)  # 0-3小时，61个时间点
    observed_A = np.full(61, 20000)  # 几乎不闭合
    observed_A[0] = 20234  # 初始面积

    observed = {"A_t": observed_A}

    # 测试 Phase I 优化
    print("\n=== Testing Phase I optimization ===")
    result = fit_improved_phase1(
        initial_mask=initial_mask,
        observed=observed,
        observed_time=observed_time,
        n_calls=20,
        n_random_starts=10,
        lambda_activity=0.01,
        force_no_division=False,
    )

    print(f"Best loss: {result['best_loss']:.4f}")
    print(f"Best parameters:")
    for k, v in result['best_params'].items():
        print(f"  {k}: {v}")

    print("\n=== Test passed ===")
