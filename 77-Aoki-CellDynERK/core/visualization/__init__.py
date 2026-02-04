"""
IF-CA Visualization Module

提供全面的可视化方案，包括：
1. 参数搜索结果可视化 (Parameter Search Visualization)
2. 消融实验可视化 (Ablation Study Visualization)
3. 可辨识性诊断可视化 (Identifiability Visualization)
4. 多条件训练可视化 (Multi-Condition Visualization)
5. 因果验证可视化 (Causal Validation Visualization)
6. 综合面板可视化 (Summary Dashboard)
"""

from .ablation_viz import (
    # 样式配置
    setup_paper_style,
    setup_presentation_style,
    IFCAColors,

    # 数据结构
    ParameterSearchResult,
    AblationResult,
    MultiConditionResult,
    CausalValidationResult,

    # 可视化函数
    plot_parameter_search,
    plot_ablation_study,
    plot_identifiability_diagnostics,
    plot_multi_condition_training,
    plot_causal_validation,
    plot_summary_dashboard,

    # 工具函数
    create_paper_figure,
    save_all_figures,
)

__all__ = [
    # 样式配置
    'setup_paper_style',
    'setup_presentation_style',
    'IFCAColors',

    # 数据结构
    'ParameterSearchResult',
    'AblationResult',
    'MultiConditionResult',
    'CausalValidationResult',

    # 可视化函数
    'plot_parameter_search',
    'plot_ablation_study',
    'plot_identifiability_diagnostics',
    'plot_multi_condition_training',
    'plot_causal_validation',
    'plot_summary_dashboard',

    # 工具函数
    'create_paper_figure',
    'save_all_figures',
]
