#!/usr/bin/env python3
"""
验证：CA_project核心文件完整性检查
"""

import sys
import os
from pathlib import Path
import traceback

print("="*70)
print("CA_project核心文件完整性检查")
print("="*70)

# 检查核心模块
core_modules = [
    "core/ca/model.py",
    "core/ca/__init__.py",
    "core/inference/fit_bo.py",
    "core/inference/fit_multiobjective.py",
    "core/inference/__init__.py",
    "core/preprocess/data_loader.py",
    "core/preprocess/analyze_time_series.py",
    "core/preprocess/extract_observations.py",
    "core/preprocess/process_data.py",
    "core/preprocess/__init__.py",
    "core/metrics/__init__.py",
    "core/metrics/morphological.py",
    "core/metrics/field_consistency.py",
    "core/metrics/uncertainty.py",
    "core/utils/io_mat.py",
    "core/utils/__init__.py",
    "core/__init__.py",
]

all_ok = True
missing_files = []

for module_path in core_modules:
    full_path = Path("E:/Python/funny/CA_project") / module_path
    if not full_path.exists():
        all_ok = False
        missing_files.append(module_path)
        print(f"❌ 缺失: {module_path}")
    else:
        print(f"✅ 存在: {module_path}")

# 检查数据集配置
dataset_files = [
    "datasets/sn15/config.py",
    "datasets/sn15/data_manifest.py",
]

for file_path in dataset_files:
    full_path = Path("E:/Python/funny/CA_project") / file_path
    if not full_path.exists():
        all_ok = False
        missing_files.append(file_path)
        print(f"❌ 缺失: {file_path}")
    else:
        print(f"✅ 存在: {file_path}")

# 检查根目录脚本（应该保留）
root_scripts = [
    "examples/minimal_validation.py",
    "generate_visualization_en.py",
    "run_optimized_simulation.py",
    "run_grid_search_optimization.py",
    "run_bayesian_optimization.py",
    "run_real_data_analysis.py",
]

for script_path in root_scripts:
    full_path = Path("E:/Python/funny/CA_project") / script_path
    if not full_path.exists():
        all_ok = False
        missing_files.append(script_path)
        print(f"❌ 缺失: {script_path}")
    else:
        print(f"✅ 存在: {script_path}")

# 检查文档
docs = [
    "README.md",
    "USER_GUIDE.md",
    "DEVELOPMENT.md",
    "IDENTIFIABILITY_SOLUTION.md",
    "METHODOLOGY.md",
    "CA_rules.md",
    "ARCHIVE_SUMMARY.md",
]

for doc_path in docs:
    full_path = Path("E:/Python/funny/CA_project") / doc_path
    if not full_path.exists():
        all_ok = False
        missing_files.append(doc_path)
        print(f"❌ 缺失: {doc_path}")
    else:
        print(f"✅ 存在: {doc_path}")

# 验证trial目录已清空
trial_path = Path("E:/Python/funny/CA_project") / "trial"
if trial_path.exists():
    files_in_trial = list(trial_path.rglob("*.*"))
    if len(files_in_trial) > 0:
        print(f"⚠️ 警告：trial目录仍有文件: {len(files_in_trial)}个")
        all_ok = False
    else:
        print(f"✅ trial目录已清空")

print("\n" + "="*70)
if all_ok:
    print("✅ CA_project核心文件完整")
    print("✅ 所有核心模块和文档存在")
    print("✅ Trial目录已清空")
else:
    print("❌ 有文件缺失:")
    for f in missing_files:
        print(f"   - {f}")

print("="*70)

# 尝试导入核心模块测试
print("\n尝试导入核心模块...")
try:
    sys.path.insert(0, str(Path("E:/Python/funny/CA_project")))
    from core import CellOnlyCA, CAParams
    print("✅ 核心模块导入成功")
except Exception as e:
    print(f"❌ 核心模块导入失败: {e}")
    all_ok = False

print("\n总结:")
print("-" * 70)
if all_ok:
    print("状态: ✅ 所有文件完整，可以开始新数据集项目")
else:
    print("状态: ❌ 有文件缺失，请检查")
print("-" * 70)

sys.exit(0 if all_ok else 1)
