#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
示例脚本：展示如何使用向量分析工具的非交互模式

此脚本演示了如何在代码中直接指定mol2文件路径，而不是通过交互方式输入。
"""

import os
import sys

# 添加父目录到系统路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 导入向量分析工具
from standard_vector.vector_utils import analyze_vectors


def main():
    # 直接指定mol2文件路径
    mol2_file = r"E:\APTAMER-GEN\mol2\1AW4-1.mol2"  # 这里替换为您的mol2文件路径
    
    # 检查文件是否存在
    if not os.path.exists(mol2_file):
        print(f"文件 {mol2_file} 不存在")
        return
    
    # 示例1：非交互模式，查看第一个子结构的backbone向量模长
    print("\n示例1：查看第一个子结构的backbone向量模长")
    result1 = analyze_vectors(
        mol2_file,
        interactive=False,
        substruct_idx=0,  # 第一个子结构
        vector_type='backbone',
        operation='magnitude'
    )
    
    # 示例2：非交互模式，可视化第一个子结构的所有向量
    print("\n示例2：可视化第一个子结构的所有向量")
    result2 = analyze_vectors(
        mol2_file,
        interactive=False,
        substruct_idx=0,  # 第一个子结构
        vector_type='all',
        operation='visualize'
    )
    
    # 示例3：获取第一个子结构的向量信息，不执行特定操作
    print("\n示例3：获取第一个子结构的向量信息")
    result3 = analyze_vectors(
        mol2_file,
        interactive=False,
        substruct_idx=0,  # 第一个子结构
        vector_type='all'
    )
    
    if result3:
        print(f"获取到 {len(result3['vectors'])} 个向量信息")
        print(f"Backbone原子数量: {len(result3['backbone_atoms'])}")
        print(f"非Backbone原子数量: {len(result3['non_backbone_atoms'])}")
    
    # 示例4：保存第一个子结构的所有向量数据为CSV格式
    print("\n示例4：保存第一个子结构的所有向量数据为CSV格式")
    csv_output = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output", "vectors.csv")
    # 确保输出目录存在
    os.makedirs(os.path.dirname(csv_output), exist_ok=True)
    result4 = analyze_vectors(
        mol2_file,
        interactive=False,
        substruct_idx=0,  # 第一个子结构
        vector_type='all',
        operation='save',
        save_to_file=csv_output,
        file_format='csv'
    )
    
    # 示例5：保存第一个子结构的backbone向量数据为JSON格式
    print("\n示例5：保存第一个子结构的backbone向量数据为JSON格式")
    json_output = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output", "backbone_vectors.json")
    result5 = analyze_vectors(
        mol2_file,
        interactive=False,
        substruct_idx=0,  # 第一个子结构
        vector_type='backbone',
        save_to_file=json_output,
        file_format='json'
    )


if __name__ == "__main__":
    main()