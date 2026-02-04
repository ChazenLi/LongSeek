"""
ERK-Coupling Visualization - ERK耦合专用可视化 (重构版)

提供ERK波传播与细胞迁移耦合分析的可视化：
1. ERK场时空演化图
2. ∠(v, ∇C) 角度耦合分布直方图
3. τ敏感性曲线
4. 波前传播图
5. 多条件对比图
6. 论文级Figure生成

作者: IF-CA Team
日期: 2026-02-04
版本: 2.0 (Refactored)
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import matplotlib.colors as mcolors
from matplotlib.colors import LinearSegmentedColormap
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from pathlib import Path
import logging
import sys
import io

# Set UTF-8 encoding
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

logger = logging.getLogger(__name__)


# ============================================================================
# ERK专用配色方案
# ============================================================================

class ERKColors:
    """ERK可视化配色"""
    # ERK场颜色
    ERK_LOW = '#1a237e'      # 深蓝 - 低ERK
    ERK_HIGH = '#fdd835'     # 黄色 - 高ERK

    # 角度耦合颜色
    ANGLE_ALIGNED = '#27ae60'   # 绿色 - 对齐 (角度小)
    ANGLE_PERP = '#e74c3c'     # 红色 - 垂直 (角度大)
    ANGLE_NEUTRAL = '#95a5a6'  # 灰色 - 中性

    # 条件颜色
    SPONTANEOUS = '#3498db'    # 蓝色
    INHIBITOR = '#9b59b6'      # 紫色
    OPTO = '#e67e22'           # 橙色

    # 图表背景
    FIGURE_BG = '#ffffff'
    PANEL_BG = '#f8f9fa'


# ============================================================================
# ERK场可视化
# ============================================================================

def plot_erk_field_evolution(erk_field: np.ndarray,
                             time_points: np.ndarray,
                             mask: Optional[np.ndarray] = None,
                             save_path: Optional[str] = None,
                             dpi: int = 300) -> plt.Figure:
    """
    绘制ERK场时空演化图

    Args:
        erk_field: (T, H, W) ERK场序列
        time_points: (T,) 时间点
        mask: (H, W) 细胞mask
        save_path: 保存路径
        dpi: 分辨率

    Returns:
        matplotlib Figure对象
    """
    T, H, W = erk_field.shape

    # 选择代表性时间点
    n_frames = min(6, T)
    frame_indices = np.linspace(0, T-1, n_frames, dtype=int)

    fig, axes = plt.subplots(2, n_frames, figsize=(3*n_frames, 6),
                            facecolor=ERKColors.FIGURE_BG)

    # 自定义colormap
    colors = [ERKColors.ERK_LOW, ERKColors.ERK_HIGH]
    cmap_erk = LinearSegmentedColormap.from_list('erk', colors)

    for i, idx in enumerate(frame_indices):
        # ERK场
        ax = axes[0, i]
        im = ax.imshow(erk_field[idx], cmap=cmap_erk, origin='lower')
        ax.set_title(f't={time_points[idx]:.1f}h', fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])

        # 叠加mask边界
        if mask is not None:
            ax.contour(mask, levels=[0.5], colors='white', linewidths=1, alpha=0.5)

        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        # ERK梯度方向
        ax = axes[1, i]
        grad_y, grad_x = np.gradient(erk_field[idx])
        grad_mag = np.sqrt(grad_x**2 + grad_y**2)

        # 显示梯度幅值
        im = ax.imshow(grad_mag, cmap='viridis', origin='lower')

        # 叠加梯度方向箭头
        step = max(H, W) // 10
        Y, X = np.mgrid[0:H:step, 0:W:step]
        ax.quiver(X, Y, grad_x[::step, ::step], grad_y[::step, ::step],
                color='white', alpha=0.6, pivot='mid')

        ax.set_title(f'∇C (t={time_points[idx]:.1f}h)', fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle('ERK Field Evolution', fontsize=14, fontweight='bold')
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
        logger.info(f"Saved ERK field evolution to {save_path}")

    return fig


# ============================================================================
# 角度耦合可视化
# ============================================================================

def plot_angle_coupling(angle_v_grad_C: np.ndarray,
                        time_points: np.ndarray,
                        save_path: Optional[str] = None,
                        dpi: int = 300) -> plt.Figure:
    """
    绘制∠(v, ∇C)角度耦合分析图

    Args:
        angle_v_grad_C: (T,) 角度序列 (弧度)
        time_points: (T,) 时间点
        save_path: 保存路径
        dpi: 分辨率

    Returns:
        matplotlib Figure对象
    """
    fig = plt.figure(figsize=(12, 4), facecolor=ERKColors.FIGURE_BG)
    gs = GridSpec(1, 3, figure=fig, wspace=0.3)

    # 1. 时间序列
    ax1 = fig.add_subplot(gs[0])
    ax1.plot(time_points, np.degrees(angle_v_grad_C), '-o',
             color=ERKColors.SPONTANEOUS, markersize=4)
    ax1.set_xlabel('Time (hours)', fontsize=10)
    ax1.set_ylabel('∠(v, ∇C) (degrees)', fontsize=10)
    ax1.set_title('Angle Coupling Over Time', fontsize=12)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 180)

    # 添加参考线
    ax1.axhline(45, color='gray', linestyle='--', alpha=0.5, label='45°')
    ax1.axhline(90, color='gray', linestyle='--', alpha=0.5, label='90°')
    ax1.legend(fontsize=8)

    # 2. 直方图
    ax2 = fig.add_subplot(gs[1])
    angles_deg = np.degrees(angle_v_grad_C)
    ax2.hist(angles_deg, bins=20, range=(0, 180),
             color=ERKColors.SPONTANEOUS, alpha=0.7, edgecolor='black')
    ax2.set_xlabel('∠(v, ∇C) (degrees)', fontsize=10)
    ax2.set_ylabel('Frequency', fontsize=10)
    ax2.set_title('Angle Distribution', fontsize=12)
    ax2.axvline(45, color='gray', linestyle='--', alpha=0.5)
    ax2.axvline(90, color='gray', linestyle='--', alpha=0.5)

    # 统计信息
    mean_angle = np.mean(angles_deg)
    std_angle = np.std(angles_deg)
    ax2.text(0.95, 0.95, f'Mean: {mean_angle:.1f}°\nStd: {std_angle:.1f}°',
             transform=ax2.transAxes, fontsize=9, verticalalignment='top',
             horizontalalignment='right', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # 3. 玫形图 (Polar)
    ax3 = fig.add_subplot(gs[2], projection='polar')

    # 转换为弧度
    theta = np.deg2rad(angles_deg)
    # 创建直方图
    bins = 18
    hist, bin_edges = np.histogram(theta, bins=bins, range=(0, np.pi))
    # 宽度
    width = (2 * np.pi) / bins

    bars = ax3.bar(bin_edges[:-1], hist, width=width, bottom=0.0,
                  color=ERKColors.SPONTANEOUS, alpha=0.7, edgecolor='black')

    ax3.set_theta_zero_location('N')
    ax3.set_theta_direction(-1)
    ax3.set_title('Polar Angle Distribution', fontsize=12, pad=20)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
        logger.info(f"Saved angle coupling plot to {save_path}")

    return fig


# ============================================================================
# τ敏感性可视化
# ============================================================================

def plot_tau_sensitivity(tau_values: np.ndarray,
                         correlations: np.ndarray,
                         optimal_tau: float,
                         save_path: Optional[str] = None,
                         dpi: int = 300) -> plt.Figure:
    """
    绘制τ敏感性曲线

    Args:
        tau_values: τ值数组
        correlations: 对应的相关系数
        optimal_tau: 最佳τ值
        save_path: 保存路径
        dpi: 分辨率

    Returns:
        matplotlib Figure对象
    """
    fig, ax = plt.subplots(figsize=(8, 5), facecolor=ERKColors.FIGURE_BG)

    # 绘制曲线
    ax.plot(tau_values, correlations, 'o-',
            color=ERKColors.SPONTANEOUS, markersize=8, linewidth=2)

    # 标记最佳τ
    optimal_idx = np.argmin(np.abs(tau_values - optimal_tau))
    ax.scatter([optimal_tau], [correlations[optimal_idx]],
              s=200, color=ERKColors.ANGLE_ALIGNED,
              edgecolors='black', linewidths=2, zorder=5)
    ax.annotate(f'Optimal: τ={optimal_tau:.0f}',
               xy=(optimal_tau, correlations[optimal_idx]),
               xytext=(10, 10), textcoords='offset points',
               fontsize=10, arrowprops=dict(arrowstyle='->', lw=1))

    ax.set_xlabel('Time Lag τ (steps)', fontsize=12)
    ax.set_ylabel('Correlation with Migration Direction', fontsize=12)
    ax.set_title('τ Sensitivity Analysis', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)

    # 设置整数x轴
    ax.set_xticks(tau_values)
    ax.set_xlim(min(tau_values) - 0.5, max(tau_values) + 0.5)

    # 添加参考线
    ax.axhline(0, color='gray', linestyle='--', alpha=0.5)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
        logger.info(f"Saved tau sensitivity plot to {save_path}")

    return fig


# ============================================================================
# 多条件对比可视化
# ============================================================================

def plot_multi_condition_comparison(results: Dict[str, Dict[str, Any]],
                                    save_path: Optional[str] = None,
                                    dpi: int = 300) -> plt.Figure:
    """
    绘制多条件对比图

    Args:
        results: {condition: {metrics}}
        save_path: 保存路径
        dpi: 分辨率

    Returns:
        matplotlib Figure对象
    """
    conditions = list(results.keys())
    n_conditions = len(conditions)

    fig, axes = plt.subplots(2, 2, figsize=(12, 10),
                            facecolor=ERKColors.FIGURE_BG)

    # 颜色映射
    condition_colors = {
        'spontaneous': ERKColors.SPONTANEOUS,
        'inhibitor': ERKColors.INHIBITOR,
        'opto': ERKColors.OPTO,
    }

    # 1. 角度耦合对比
    ax = axes[0, 0]
    for cond in conditions:
        if 'angle_v_grad_C' in results[cond]:
            angles = results[cond]['angle_v_grad_C']
            time_points = results[cond].get('time_points', np.arange(len(angles)))
            ax.plot(time_points, np.degrees(angles), '-o',
                    label=cond, color=condition_colors.get(cond, 'gray'),
                    markersize=4)

    ax.set_xlabel('Time (hours)', fontsize=10)
    ax.set_ylabel('∠(v, ∇C) (degrees)', fontsize=10)
    ax.set_title('Angle Coupling Comparison', fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 180)

    # 2. 波速对比
    ax = axes[0, 1]
    wave_speeds = []
    wave_speeds_err = []
    cond_labels = []

    for cond in conditions:
        if 'wave_speed' in results[cond]:
            speed = results[cond]['wave_speed']
            wave_speeds.append(speed)
            cond_labels.append(cond)

    if wave_speeds:
        colors = [condition_colors.get(c, 'gray') for c in cond_labels]
        bars = ax.bar(cond_labels, wave_speeds, color=colors, alpha=0.7, edgecolor='black')
        ax.set_ylabel('Wave Speed (μm/min)', fontsize=10)
        ax.set_title('ERK Wave Speed Comparison', fontsize=12)
        ax.grid(axis='y', alpha=0.3)

    # 3. 损失对比
    ax = axes[1, 0]
    losses = []
    for cond in conditions:
        if 'loss' in results[cond]:
            losses.append(results[cond]['loss'])
            cond_labels.append(cond)

    if losses:
        colors = [condition_colors.get(c, 'gray') for c in cond_labels]
        bars = ax.bar(cond_labels, losses, color=colors, alpha=0.7, edgecolor='black')
        ax.set_ylabel('Loss', fontsize=10)
        ax.set_title('Loss Comparison', fontsize=12)
        ax.grid(axis='y', alpha=0.3)

    # 4. 参数对比
    ax = axes[1, 1]
    params_to_plot = ['gamma', 'tau']
    x = np.arange(len(params_to_plot))
    width = 0.25

    for i, cond in enumerate(conditions):
        if 'params' in results[cond]:
            param_values = [results[cond]['params'].get(p, 0) for p in params_to_plot]
            ax.bar(x + i * width, param_values, width,
                  label=cond, color=condition_colors.get(cond, 'gray'), alpha=0.7)

    ax.set_xticks(x + width * (n_conditions - 1) / 2)
    ax.set_xticklabels([f'γ ({p})' if p == 'gamma' else f'τ ({p})' for p in params_to_plot])
    ax.set_ylabel('Parameter Value', fontsize=10)
    ax.set_title('Parameter Comparison', fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(axis='y', alpha=0.3)

    fig.suptitle('Multi-Condition Comparison', fontsize=16, fontweight='bold')
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
        logger.info(f"Saved multi-condition comparison to {save_path}")

    return fig


# ============================================================================
# 论文级Figure生成
# ============================================================================

def create_paper_figure(erk_field: np.ndarray,
                        angle_v_grad_C: np.ndarray,
                        tau_results: Dict[str, Any],
                        multi_condition_results: Dict[str, Any],
                        save_path: Optional[str] = None,
                        dpi: int = 300) -> plt.Figure:
    """
    创建论文级Figure (4面板)

    Panel A: ERK场演化
    Panel B: 角度耦合分析
    Panel C: τ敏感性
    Panel D: 多条件对比

    Args:
        erk_field: ERK场
        angle_v_grad_C: 角度序列
        tau_results: τ敏感性结果
        multi_condition_results: 多条件结果
        save_path: 保存路径
        dpi: 分辨率

    Returns:
        matplotlib Figure对象
    """
    fig = plt.figure(figsize=(16, 10), facecolor='white')
    gs = GridSpec(2, 3, figure=fig, hspace=0.3, wspace=0.3)

    # Panel A: ERK场演化 (占2列)
    ax_a = fig.add_subplot(gs[0, :2])
    T, H, W = erk_field.shape

    # 显示最后一帧ERK场
    colors = [ERKColors.ERK_LOW, ERKColors.ERK_HIGH]
    cmap_erk = LinearSegmentedColormap.from_list('erk', colors)

    im = ax_a.imshow(erk_field[-1], cmap=cmap_erk, origin='lower')
    ax_a.set_title('A) ERK Field (Final Frame)', fontsize=12, fontweight='bold', loc='left')
    ax_a.set_xticks([])
    ax_a.set_yticks([])

    # 添加colorbar
    cbar = plt.colorbar(im, ax=ax_a, fraction=0.046, pad=0.04)
    cbar.set_label('ERK Activity (a.u.)', fontsize=10)

    # 叠加梯度方向
    grad_y, grad_x = np.gradient(erk_field[-1])
    step = max(H, W) // 15
    Y, X = np.mgrid[0:H:step, 0:W:step]
    ax_a.quiver(X, Y, grad_x[::step, ::step], grad_y[::step, ::step],
               color='white', alpha=0.5, pivot='mid')

    # Panel B: 角度耦合
    ax_b = fig.add_subplot(gs[0, 2])
    ax_b.hist(np.degrees(angle_v_grad_C), bins=20, range=(0, 180),
             color=ERKColors.SPONTANEOUS, alpha=0.7, edgecolor='black')
    ax_b.set_xlabel('∠(v, ∇C) (degrees)', fontsize=10)
    ax_b.set_ylabel('Frequency', fontsize=10)
    ax_b.set_title('B) Angle Coupling Distribution', fontsize=12, fontweight='bold', loc='left')
    ax_b.axvline(45, color='gray', linestyle='--', alpha=0.5, label='45°')
    ax_b.axvline(90, color='gray', linestyle='--', alpha=0.5, label='90°')
    ax_b.legend(fontsize=8)

    # Panel C: τ敏感性
    ax_c = fig.add_subplot(gs[1, 0])
    tau_values = tau_results['tau_values']
    correlations = tau_results['correlations']
    optimal_tau = tau_results['optimal_tau']

    ax_c.plot(tau_values, correlations, 'o-',
              color=ERKColors.SPONTANEOUS, markersize=6, linewidth=2)
    ax_c.scatter([optimal_tau], [correlations[np.argmin(np.abs(tau_values - optimal_tau))]],
                s=150, color=ERKColors.ANGLE_ALIGNED, edgecolors='black', linewidths=1.5, zorder=5)
    ax_c.set_xlabel('Time Lag τ (steps)', fontsize=10)
    ax_c.set_ylabel('Correlation', fontsize=10)
    ax_c.set_title('C) τ Sensitivity', fontsize=12, fontweight='bold', loc='left')
    ax_c.set_xticks(tau_values)
    ax_c.grid(True, alpha=0.3)

    # Panel D: 多条件对比
    ax_d = fig.add_subplot(gs[1, 1:])
    conditions = list(multi_condition_results.keys())
    condition_colors = {
        'spontaneous': ERKColors.SPONTANEOUS,
        'inhibitor': ERKColors.INHIBITOR,
        'opto': ERKColors.OPTO,
    }

    x = np.arange(len(conditions))
    width = 0.35

    # 绘制子图
    if all('loss' in multi_condition_results[c] for c in conditions):
        losses = [multi_condition_results[c]['loss'] for c in conditions]
        colors = [condition_colors.get(c, 'gray') for c in conditions]
        bars = ax_d.bar(x, losses, color=colors, alpha=0.7, edgecolor='black')
        ax_d.set_ylabel('Loss', fontsize=10)
        ax_d.set_title('D) Cross-Condition Loss', fontsize=12, fontweight='bold', loc='left')
        ax_d.set_xticks(x)
        ax_d.set_xticklabels(conditions, fontsize=10)
        ax_d.grid(axis='y', alpha=0.3)

    fig.suptitle('ERK-Coupling Analysis', fontsize=16, fontweight='bold', y=0.98)

    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
        logger.info(f"Saved paper figure to {save_path}")

    return fig


# ============================================================================
# 便捷函数
# ============================================================================

def plot_erk_analysis(erk_field: np.ndarray,
                      results: Dict[str, Any],
                      save_dir: Optional[str] = None):
    """
    一键生成所有ERK分析图表

    Args:
        erk_field: ERK场
        results: 分析结果字典
        save_dir: 保存目录
    """
    if save_dir is None:
        save_dir = 'results/erk_analysis'
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    # 生成各图表
    if 'time_points' in results:
        plot_erk_field_evolution(
            erk_field, results['time_points'],
            save_path=f'{save_dir}/erk_evolution.png'
        )

    if 'angle_v_grad_C' in results:
        plot_angle_coupling(
            results['angle_v_grad_C'], results['time_points'],
            save_path=f'{save_dir}/angle_coupling.png'
        )

    if 'tau_sensitivity' in results:
        tau_results = results['tau_sensitivity']
        plot_tau_sensitivity(
            tau_results['tau_values'],
            tau_results['correlations'],
            tau_results['optimal_tau'],
            save_path=f'{save_dir}/tau_sensitivity.png'
        )

    if 'multi_condition' in results:
        plot_multi_condition_comparison(
            results['multi_condition'],
            save_path=f'{save_dir}/multi_condition.png'
        )

    logger.info(f"All ERK analysis plots saved to {save_dir}")


if __name__ == "__main__":
    # 测试可视化
    print("Testing ERK Visualization")

    # 创建模拟数据
    T, H, W = 60, 50, 50
    erk_field = np.random.rand(T, H, W).astype(np.float32)
    time_points = np.linspace(0, 3, T)

    # 创建有梯度的ERK场
    for t in range(T):
        x = np.linspace(-1, 1, W)
        y = np.linspace(-1, 1, H)
        X, Y = np.meshgrid(x, y)
        erk_field[t] = 0.5 + 0.3 * np.sin(2 * np.pi * (X + t * 0.05))

    angle_v_grad_C = np.random.uniform(0, np.pi/2, T)

    # 测试ERK场演化图
    fig1 = plot_erk_field_evolution(erk_field, time_points)
    plt.close(fig1)

    # 测试角度耦合图
    fig2 = plot_angle_coupling(angle_v_grad_C, time_points)
    plt.close(fig2)

    # 测试τ敏感性
    tau_values = np.arange(6)
    correlations = np.array([0.3, 0.5, 0.7, 0.6, 0.4, 0.2])
    fig3 = plot_tau_sensitivity(tau_values, correlations, 2)
    plt.close(fig3)

    print("✅ ERK Visualization test complete!")
