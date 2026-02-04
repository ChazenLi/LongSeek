"""CA models for wound healing simulation."""

from .model import CellOnlyCA, CAParams, CAMolParams, CAMolField
from .phase2_teacher_forcing import PhaseIITeacherForcingCA, CAParamsExtended, CouplingParams
# 改进版模型（根据建议.md P0-P1.5 优先级改进）
from .improved_model import (
    CellOnlyCAImproved,
    CAParamsImproved,
    PhaseIITeacherForcingCAImproved,
    CAParamsExtendedImproved,
    CouplingParamsImproved,
    calculate_steps_from_time,
)

__all__ = [
    # 原版模型
    'CellOnlyCA', 'CAParams', 'CAMolParams', 'CAMolField',
    'PhaseIITeacherForcingCA', 'CAParamsExtended', 'CouplingParams',
    # 改进版模型
    'CellOnlyCAImproved', 'CAParamsImproved',
    'PhaseIITeacherForcingCAImproved', 'CAParamsExtendedImproved', 'CouplingParamsImproved',
    'calculate_steps_from_time',
]
