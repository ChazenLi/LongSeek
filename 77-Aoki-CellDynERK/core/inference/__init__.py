"""Parameter optimization and inference methods."""

from .fit_bo import (
    run_simulation,
    calculate_loss,
    fit_simple_grid_search,
    fit_bayesian_optimization,
    objective_function
)

from .identifiability import (
    IdentifiabilityDiagnostics,
    compute_posterior_correlation,
    plot_identifiability_results,
)

from .multi_condition import (
    MultiConditionTrainer,
    ConditionDataset,
    ParameterTransferAnalyzer,
)

from .causal_validation import (
    CausalValidator,
    plot_causal_validation,
)

# 改进版优化器（根据建议.md P0-P1.5 优先级改进）
try:
    from .improved_fit import (
        ImprovedLossCalculator,
        run_simulation_improved,
        objective_function_improved,
        fit_improved_phase1,
        fit_improved_phase2,
        fit_improved_random_search,
    )
    _improved_available = True
except ImportError:
    _improved_available = False

__all__ = [
    # Week 1-5: Basic inference
    'run_simulation',
    'calculate_loss',
    'fit_simple_grid_search',
    'fit_bayesian_optimization',
    'objective_function',
    # Week 6: Identifiability
    'IdentifiabilityDiagnostics',
    'compute_posterior_correlation',
    'plot_identifiability_results',
    # Week 7: Multi-condition
    'MultiConditionTrainer',
    'ConditionDataset',
    'ParameterTransferAnalyzer',
    # Week 8: Causal validation
    'CausalValidator',
    'plot_causal_validation',
]

# 添加改进版模块（如果可用）
if _improved_available:
    __all__.extend([
        'ImprovedLossCalculator',
        'run_simulation_improved',
        'objective_function_improved',
        'fit_improved_phase1',
        'fit_improved_phase2',
        'fit_improved_random_search',
    ])
