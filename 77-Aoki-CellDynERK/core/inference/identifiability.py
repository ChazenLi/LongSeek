"""
Identifiability Diagnostics for CA Models (Week 6)

This module provides tools for assessing parameter identifiability:
1. Posterior correlation matrix - reveals parameter correlations
2. Profile likelihood analysis - assesses single-parameter identifiability
3. Fisher information matrix - quantifies parameter precision
4. Sensitivity analysis - evaluates parameter influence on outputs

Key Insight: A parameter is "identifiable" if:
- Profile likelihood has a unique peak (not flat)
- Posterior correlation with other params is low (|corr| < 0.5)
- Fisher information is high (I_ii > 0.01)
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Callable
from dataclasses import dataclass
import logging
from scipy.optimize import minimize
from scipy.stats import pearsonr
from scipy.ndimage import gaussian_filter

logger = logging.getLogger(__name__)


@dataclass
class IdentifiabilityResult:
    """可辨识性分析结果"""
    parameter_name: str
    identifiable: bool
    profile_likelihood: np.ndarray
    param_range: np.ndarray
    mle: float
    confidence_interval: Tuple[float, float]
    fisher_info: float
    posterior_corr: Dict[str, float]


def compute_posterior_correlation(
    params_history: List[Dict],
    param_names: Optional[List[str]] = None
) -> Tuple[np.ndarray, List[str]]:
    """
    计算后验相关矩阵

    Args:
        params_history: 参数历史记录 (每次迭代的参数值)
        param_names: 参数名称列表

    Returns:
        correlation_matrix: 相关矩阵 (n_params, n_params)
        param_names: 参数名称
    """
    if param_names is None:
        param_names = list(params_history[0].keys())

    # 提取参数值
    n_samples = len(params_history)
    n_params = len(param_names)

    param_matrix = np.zeros((n_samples, n_params))
    for i, params in enumerate(params_history):
        for j, name in enumerate(param_names):
            param_matrix[i, j] = params.get(name, 0.0)

    # 计算相关矩阵
    correlation_matrix = np.corrcoef(param_matrix.T)

    return correlation_matrix, param_names


def compute_profile_likelihood(
    param_name: str,
    param_range: np.ndarray,
    fixed_params: Dict[str, float],
    objective_fn: Callable,
    baseline_loss: float,
    n_points: int = 20
) -> Tuple[np.ndarray, np.ndarray, float, Tuple[float, float]]:
    """
    计算单参数剖面似然

    原理：固定其他参数在MLE值，扫描目标参数，观察loss变化

    Args:
        param_name: 目标参数名
        param_range: 参数扫描范围
        fixed_params: 固定的其他参数
        objective_fn: 目标函数
        baseline_loss: MLE处的loss值
        n_points: 扫描点数

    Returns:
        param_values: 扫描的参数值
        likelihood_values: 对应的似然值 (-loss)
        mle: 最大似然估计值
        confidence_interval: 95%置信区间
    """
    param_values = np.linspace(param_range[0], param_range[1], n_points)
    likelihood_values = np.zeros(n_points)

    for i, val in enumerate(param_name, param_values):
        # 构造参数
        params = fixed_params.copy()
        params[param_name] = val

        # 计算loss
        try:
            loss = objective_fn(params)
            likelihood_values[i] = -loss
        except Exception as e:
            logger.warning(f"Failed to evaluate at {param_name}={val}: {e}")
            likelihood_values[i] = -np.inf

    # 归一化似然
    likelihood_values = likelihood_values - np.max(likelihood_values)

    # 找到MLE
    mle_idx = np.argmax(likelihood_values)
    mle = param_values[mle_idx]

    # 计算置信区间 (likelihood threshold = -1.92 for 95% CI)
    threshold = -1.92
    valid_mask = likelihood_values >= threshold

    if np.any(valid_mask):
        valid_params = param_values[valid_mask]
        confidence_interval = (float(valid_params.min()), float(valid_params.max()))
    else:
        confidence_interval = (mle, mle)

    return param_values, likelihood_values, mle, confidence_interval


def compute_fisher_information(
    objective_fn: Callable,
    params: Dict[str, float],
    param_names: List[str],
    epsilon: float = 1e-6
) -> Tuple[np.ndarray, np.ndarray]:
    """
    计算Fisher信息矩阵

    原理：Fisher信息 = Hessian矩阵的负逆
    I_ij = E[∂²L/∂θ_i∂θ_j]

    Args:
        objective_fn: 目标函数
        params: 参数字典
        param_names: 参数名称列表
        epsilon: 有限差分步长

    Returns:
        fisher_info: Fisher信息矩阵 (n_params, n_params)
        eigenvalues: 特征值（参数精度指标）
    """
    n_params = len(param_names)
    hessian = np.zeros((n_params, n_params))

    # 转换为向量
    param_vector = np.array([params[name] for name in param_names])

    # 计算Hessian（二阶导数）
    base_loss = objective_fn(params)

    for i in range(n_params):
        for j in range(n_params):
            # 二阶中心差分
            params_ij = params.copy()

            if i == j:
                # 对角元素：f(x+h) - 2f(x) + f(x-h)
                params_ij[param_names[i]] = param_vector[i] + epsilon
                f_plus = objective_fn(params_ij)

                params_ij[param_names[i]] = param_vector[i] - epsilon
                f_minus = objective_fn(params_ij)

                hessian[i, j] = (f_plus - 2 * base_loss + f_minus) / (epsilon ** 2)
            else:
                # 非对角元素：混合导数
                params_ij[param_names[i]] = param_vector[i] + epsilon
                params_ij[param_names[j]] = param_vector[j] + epsilon
                f_pp = objective_fn(params_ij)

                params_ij[param_names[i]] = param_vector[i] - epsilon
                f_mp = objective_fn(params_ij)

                params_ij[param_names[i]] = param_vector[i] + epsilon
                params_ij[param_names[j]] = param_vector[j] - epsilon
                f_pm = objective_fn(params_ij)

                params_ij[param_names[i]] = param_vector[i] - epsilon
                params_ij[param_names[j]] = param_vector[j] - epsilon
                f_mm = objective_fn(params_ij)

                hessian[i, j] = (f_pp - f_mp - f_pm + f_mm) / (4 * epsilon ** 2)

    # Fisher信息 = Hessian的负逆（在最优解附近）
    try:
        fisher_info = -np.linalg.inv(hessian)
        eigenvalues, _ = np.linalg.eig(fisher_info)
    except np.linalg.LinAlgError:
        logger.warning("Hessian is singular, using pseudo-inverse")
        fisher_info = -np.linalg.pinv(hessian)
        eigenvalues, _ = np.linalg.eig(fisher_info)

    return fisher_info, eigenvalues


def compute_sensitivity_indices(
    params: Dict[str, float],
    param_ranges: Dict[str, Tuple[float, float]],
    model_fn: Callable,
    output_names: List[str],
    n_samples: int = 100
) -> Dict[str, Dict[str, float]]:
    """
    计算敏感性指数（Sobol指数近似）

    原理：参数扰动对输出的影响程度

    Args:
        params: 基准参数
        param_ranges: 参数范围
        model_fn: 模型函数
        output_names: 输出名称
        n_samples: 采样数

    Returns:
        sensitivity_indices: {param: {output: sensitivity}}
    """
    param_names = list(param_ranges.keys())
    n_params = len(param_names)

    # 获取基准输出
    baseline_outputs = model_fn(params)

    # 计算敏感性
    sensitivity = {name: {} for name in param_names}

    for param_name in param_names:
        base_val = params[param_name]
        min_val, max_val = param_ranges[param_name]
        delta = (max_val - min_val) * 0.1  # 10%扰动

        # 正向扰动
        params_plus = params.copy()
        params_plus[param_name] = base_val + delta
        outputs_plus = model_fn(params_plus)

        # 负向扰动
        params_minus = params.copy()
        params_minus[param_name] = max(base_val - delta, min_val)
        outputs_minus = model_fn(params_minus)

        # 计算敏感性指数 (归一化)
        for output_name in output_names:
            baseline_val = baseline_outputs.get(output_name, 0.0)
            output_plus = outputs_plus.get(output_name, 0.0)
            output_minus = outputs_minus.get(output_name, 0.0)

            # 中心差分
            diff = (output_plus - output_minus) / 2
            normalized = abs(diff / (abs(baseline_val) + 1e-10))

            sensitivity[param_name][output_name] = float(normalized)

    return sensitivity


class IdentifiabilityDiagnostics:
    """
    可辨识性诊断工具类

    使用方法：
        diagnostics = IdentifiabilityDiagnostics(
            objective_fn=loss_function,
            param_names=['p_move', 'p_div', 'alpha', 'beta', 'gamma', 'tau'],
            param_ranges={...}
        )

        results = diagnostics.analyze(params_history, best_params)
        diagnostics.plot_results(results)
        diagnostics.save_report(results, 'identifiability_report.json')
    """

    def __init__(self,
                 objective_fn: Callable,
                 param_names: List[str],
                 param_ranges: Dict[str, Tuple[float, float]]):
        """
        Args:
            objective_fn: 目标函数
            param_names: 参数名称列表
            param_ranges: 参数搜索范围
        """
        self.objective_fn = objective_fn
        self.param_names = param_names
        self.param_ranges = param_ranges

    def analyze_correlation(self,
                          params_history: List[Dict]) -> Tuple[np.ndarray, Dict]:
        """
        分析参数相关性

        Returns:
            correlation_matrix: 相关矩阵
            high_correlations: 高相关参数对 (|corr| > 0.5)
        """
        corr_matrix, names = compute_posterior_correlation(params_history, self.param_names)

        # 找出高相关参数对
        high_correlations = {}
        for i in range(len(names)):
            for j in range(i+1, len(names)):
                corr_val = corr_matrix[i, j]
                if abs(corr_val) > 0.5:
                    high_correlations[f"{names[i]}_{names[j]}"] = corr_val

        return corr_matrix, high_correlations

    def analyze_profile_likelihood(self,
                                   best_params: Dict,
                                   baseline_loss: float,
                                   n_points: int = 20) -> Dict[str, IdentifiabilityResult]:
        """
        分析每个参数的剖面似然

        Returns:
            results: {param_name: IdentifiabilityResult}
        """
        results = {}

        for param_name in self.param_names:
            param_range = self.param_ranges[param_name]

            # 固定其他参数
            fixed_params = best_params.copy()

            # 计算剖面似然
            param_vals, likelihood, mle, ci = compute_profile_likelihood(
                param_name,
                param_range,
                fixed_params,
                self._param_wrapper(fixed_params),
                baseline_loss,
                n_points
            )

            # 判断可辨识性：是否有清晰峰值
            # 使用likelihood变化范围判断
            likelihood_range = likelihood.max() - likelihood.min()
            identifiable = likelihood_range > 2.0  # 阈值：2个log单位

            results[param_name] = IdentifiabilityResult(
                parameter_name=param_name,
                identifiable=identifiable,
                profile_likelihood=likelihood,
                param_range=param_vals,
                mle=mle,
                confidence_interval=ci,
                fisher_info=0.0,  # 后续填充
                posterior_corr={}  # 后续填充
            )

        return results

    def analyze_fisher_information(self,
                                   best_params: Dict) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """
        计算Fisher信息矩阵

        Returns:
            fisher_info: Fisher信息矩阵
            eigenvalues: 特征值
            precision: 参数精度指标
        """
        fisher_info, eigenvalues = compute_fisher_information(
            self._param_wrapper(best_params),
            best_params,
            self.param_names
        )

        # 提取对角线元素作为参数精度
        precision = {}
        for i, name in enumerate(self.param_names):
            precision[name] = fisher_info[i, i]

        return fisher_info, eigenvalues, precision

    def analyze_sensitivity(self,
                          best_params: Dict,
                          model_fn: Callable,
                          n_samples: int = 50) -> Dict[str, Dict[str, float]]:
        """
        参数敏感性分析

        Returns:
            sensitivity: {param: {output: sensitivity_index}}
        """
        return compute_sensitivity_indices(
            best_params,
            self.param_ranges,
            model_fn,
            ['wound_area', 'migrations', 'divisions'],
            n_samples
        )

    def full_analysis(self,
                     params_history: List[Dict],
                     best_params: Dict,
                     baseline_loss: float,
                     model_fn: Optional[Callable] = None) -> Dict:
        """
        完整的可辨识性分析

        Returns:
            analysis_results: 包含所有分析结果
        """
        results = {
            'correlation': {},
            'profile_likelihood': {},
            'fisher_info': {},
            'sensitivity': {},
            'summary': {}
        }

        # 1. 相关性分析
        logger.info("Computing posterior correlation matrix...")
        corr_matrix, high_corr = self.analyze_correlation(params_history)
        results['correlation']['matrix'] = corr_matrix
        results['correlation']['high_correlations'] = high_corr

        # 2. 剖面似然分析
        logger.info("Computing profile likelihood...")
        profile_results = self.analyze_profile_likelihood(best_params, baseline_loss)
        results['profile_likelihood']['results'] = profile_results

        # 3. Fisher信息
        logger.info("Computing Fisher information...")
        fisher_info, eigenvalues, precision = self.analyze_fisher_information(best_params)
        results['fisher_info']['matrix'] = fisher_info
        results['fisher_info']['eigenvalues'] = eigenvalues
        results['fisher_info']['precision'] = precision

        # 4. 敏感性分析
        if model_fn is not None:
            logger.info("Computing sensitivity indices...")
            sensitivity = self.analyze_sensitivity(best_params, model_fn)
            results['sensitivity']['indices'] = sensitivity

        # 5. 汇总
        results['summary'] = self._generate_summary(
            corr_matrix, profile_results, precision, high_corr
        )

        return results

    def _generate_summary(self,
                         corr_matrix: np.ndarray,
                         profile_results: Dict[str, IdentifiabilityResult],
                         precision: Dict[str, float],
                         high_corr: Dict) -> Dict:
        """生成可辨识性汇总"""
        summary = {
            'identifiable_params': [],
            'weakly_identifiable_params': [],
            'unidentifiable_params': [],
            'highly_correlated_pairs': high_corr,
            'recommendations': []
        }

        for param_name in self.param_names:
            result = profile_results[param_name]
            prec = precision.get(param_name, 0.0)

            # 判断标准
            if result.identifiable and prec > 0.01:
                summary['identifiable_params'].append(param_name)
            elif result.identifiable or prec > 0.001:
                summary['weakly_identifiable_params'].append(param_name)
            else:
                summary['unidentifiable_params'].append(param_name)

        # 生成建议
        if len(summary['unidentifiable_params']) > 0:
            summary['recommendations'].append(
                f"Unidentifiable parameters: {summary['unidentifiable_params']}. "
                "Consider fixing these or collecting more data."
            )

        if len(high_corr) > 0:
            summary['recommendations'].append(
                f"High correlations detected: {list(high_corr.keys())}. "
                "Consider reparameterization or adding constraints."
            )

        return summary

    def _param_wrapper(self, fixed_params: Dict) -> Callable:
        """包装目标函数，接受参数字典"""
        def wrapper(params_array: np.ndarray) -> float:
            params = fixed_params.copy()
            for i, name in enumerate(self.param_names):
                params[name] = params_array[i]
            return self.objective_fn(params)
        return wrapper

    def save_report(self, results: Dict, filepath: str):
        """保存分析报告为JSON"""
        import json
        from datetime import datetime

        # 转换numpy类型为Python原生类型
        def convert(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.integer, np.floating)):
                return float(obj)
            elif isinstance(obj, dict):
                return {k: convert(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert(v) for v in obj]
            elif isinstance(obj, IdentifiabilityResult):
                return {
                    'parameter_name': obj.parameter_name,
                    'identifiable': obj.identifiable,
                    'mle': obj.mle,
                    'confidence_interval': obj.confidence_interval,
                    'fisher_info': obj.fisher_info,
                }
            return obj

        report = {
            'timestamp': datetime.now().isoformat(),
            'results': convert(results),
            'param_names': self.param_names,
        }

        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2)

        logger.info(f"Identifiability report saved to {filepath}")


def plot_identifiability_results(results: Dict, save_path: Optional[str] = None):
    """
    可视化可辨识性分析结果

    Args:
        results: full_analysis返回的结果
        save_path: 保存路径（可选）
    """
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    param_names = results.get('param_names', [])
    corr_matrix = results['correlation']['matrix']
    profile_results = results['profile_likelihood'].get('results', {})

    fig = plt.figure(figsize=(16, 10))
    gs = GridSpec(2, 3, figure=fig)

    # 1. 后验相关矩阵热力图
    ax1 = fig.add_subplot(gs[0, 0])
    im = ax1.imshow(corr_matrix, cmap='RdBu_r', vmin=-1, vmax=1)
    ax1.set_xticks(range(len(param_names)))
    ax1.set_yticks(range(len(param_names)))
    ax1.set_xticklabels(param_names, rotation=45, ha='right')
    ax1.set_yticklabels(param_names)
    ax1.set_title('Posterior Correlation Matrix')
    plt.colorbar(im, ax=ax1)

    # 添加相关系数标注
    for i in range(len(param_names)):
        for j in range(len(param_names)):
            text = ax1.text(j, i, f'{corr_matrix[i, j]:.2f}',
                          ha='center', va='center', fontsize=8)

    # 2. 剖面似然曲线
    ax2 = fig.add_subplot(gs[0, 1:])
    for param_name, result in profile_results.items():
        if hasattr(result, 'param_range') and hasattr(result, 'profile_likelihood'):
            ax2.plot(result.param_range, result.profile_likelihood,
                    label=param_name, marker='o', markersize=3)

    ax2.axhline(y=-1.92, color='r', linestyle='--', label='95% CI threshold')
    ax2.set_xlabel('Parameter Value')
    ax2.set_ylabel('Log-Likelihood')
    ax2.set_title('Profile Likelihood')
    ax2.legend(ncol=3, fontsize=8)
    ax2.grid(True, alpha=0.3)

    # 3. Fisher信息（条形图）
    ax3 = fig.add_subplot(gs[1, 0])
    fisher_info = results['fisher_info'].get('precision', {})
    if fisher_info:
        names = list(fisher_info.keys())
        values = [fisher_info[n] for n in names]
        colors = ['green' if v > 0.01 else 'orange' if v > 0.001 else 'red' for v in values]
        ax3.barh(names, values, color=colors)
        ax3.axvline(x=0.01, color='r', linestyle='--', label='High precision threshold')
        ax3.set_xlabel('Fisher Information (Diagonal)')
        ax3.set_title('Parameter Precision')
        ax3.legend()

    # 4. 可辨识性汇总
    ax4 = fig.add_subplot(gs[1, 1:])
    ax4.axis('off')

    summary = results.get('summary', {})
    text = "IDENTIFIABILITY SUMMARY\n\n"

    text += "✅ Identifiable:\n"
    for p in summary.get('identifiable_params', []):
        text += f"  - {p}\n"

    text += "\n⚠️ Weakly Identifiable:\n"
    for p in summary.get('weakly_identifiable_params', []):
        text += f"  - {p}\n"

    text += "\n❌ Unidentifiable:\n"
    for p in summary.get('unidentifiable_params', []):
        text += f"  - {p}\n"

    text += "\n📗 Recommendations:\n"
    for rec in summary.get('recommendations', []):
        text += f"  - {rec}\n"

    ax4.text(0.05, 0.95, text, transform=ax4.transAxes,
            verticalalignment='top', fontsize=10,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"Identifiability plot saved to {save_path}")

    return fig


if __name__ == "__main__":
    # 测试可辨识性诊断
    print("Testing Identifiability Diagnostics...")

    # 模拟参数历史
    np.random.seed(42)
    n_samples = 100

    param_names = ['p_move', 'p_div', 'alpha', 'beta', 'gamma', 'tau']
    params_history = []

    for _ in range(n_samples):
        params = {
            'p_move': np.random.uniform(0.3, 0.7),
            'p_div': np.random.uniform(0.01, 0.1),
            'alpha': np.random.uniform(0.5, 2.0),
            'beta': np.random.uniform(0.5, 2.0),
            'gamma': np.random.uniform(0.5, 1.5),
            'tau': np.random.randint(0, 3),
        }
        params_history.append(params)

    # 计算相关矩阵
    corr_matrix, names = compute_posterior_correlation(params_history, param_names)
    print("\nPosterior Correlation Matrix:")
    print(corr_matrix)

    print("\nHigh correlations (|corr| > 0.5):")
    for i in range(len(names)):
        for j in range(i+1, len(names)):
            if abs(corr_matrix[i, j]) > 0.5:
                print(f"  {names[i]} - {names[j]}: {corr_matrix[i, j]:.3f}")

    print("\nIdentifiability diagnostics test complete!")
