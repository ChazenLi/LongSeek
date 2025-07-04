import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import networkx as nx
from collections import defaultdict
import logging
import csv
import json

# 添加父目录到系统路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 导入现有的mol2解析和可视化函数
from mol2_pymol_visualizer.mol2_pymol_visualizer import (
    parse_mol2, group_by_substruct, create_networkx_graph, 
    plot_3d_structure_matplotlib, logger
)

# 导入backbone定义
from standard_vector.vector import back_bone


class MoleculeVectorAnalyzer:
    """分子向量分析器，用于分析分子结构中的原子向量"""
    
    def __init__(self, mol2_file):
        """初始化分子向量分析器
        
        Args:
            mol2_file: mol2文件路径
        """
        # 解析mol2文件
        self.atoms, self.bonds, self.molecule_info = parse_mol2(mol2_file)
        if not self.atoms:
            logger.error(f"文件 {mol2_file} 中没有找到原子数据")
            return
        
        # 按子结构分组
        self.substruct_groups, self.substruct_bonds = group_by_substruct(self.atoms, self.bonds)
        
        # 存储向量信息
        self.vectors = {}
        self.backbone_atoms = {}
        self.non_backbone_atoms = {}
        
        # 初始化处理
        self._process_atoms()
        self._build_vectors()
    
    def _process_atoms(self):
        """处理原子，将非氢原子按照是否存在于backbone中进行分组"""
        for substruct_name, atoms in self.substruct_groups.items():
            # 初始化该子结构的原子分组
            self.backbone_atoms[substruct_name] = []
            self.non_backbone_atoms[substruct_name] = []
            
            # 分组非氢原子
            for atom in atoms:
                # 跳过氢原子
                if atom['atom_name'].startswith('H'):
                    continue
                
                # 判断是否为backbone原子
                if atom['atom_name'] in back_bone:
                    self.backbone_atoms[substruct_name].append(atom)
                else:
                    self.non_backbone_atoms[substruct_name].append(atom)
            
            logger.info(f"子结构 {substruct_name} 中的backbone原子数量: {len(self.backbone_atoms[substruct_name])}")
            logger.info(f"子结构 {substruct_name} 中的非backbone原子数量: {len(self.non_backbone_atoms[substruct_name])}")
    
    def _build_vectors(self):
        """构建所有非氢原子之间的向量"""
        for substruct_name, atoms in self.substruct_groups.items():
            # 初始化该子结构的向量字典
            self.vectors[substruct_name] = []
            
            # 获取该子结构的所有非氢原子
            non_h_atoms = [atom for atom in atoms if not atom['atom_name'].startswith('H')]
            
            # 构建向量（起始原子为序号在前的atom_id，结束原子为序号在后的atom_id）
            for i in range(len(non_h_atoms)):
                for j in range(i+1, len(non_h_atoms)):
                    atom1 = non_h_atoms[i]
                    atom2 = non_h_atoms[j]
                    
                    # 确保atom1的ID小于atom2的ID
                    if atom1['atom_id'] > atom2['atom_id']:
                        atom1, atom2 = atom2, atom1
                    
                    # 计算向量
                    vector = np.array([atom2['x'] - atom1['x'], 
                                      atom2['y'] - atom1['y'], 
                                      atom2['z'] - atom1['z']])
                    
                    # 计算向量模长
                    magnitude = np.linalg.norm(vector)
                    
                    # 存储向量信息
                    vector_info = {
                        'atom1': atom1,
                        'atom2': atom2,
                        'vector': vector,
                        'magnitude': magnitude,
                        'is_backbone_vector': atom1['atom_name'] in back_bone and atom2['atom_name'] in back_bone,
                        'is_mixed_vector': (atom1['atom_name'] in back_bone) != (atom2['atom_name'] in back_bone)
                    }
                    
                    self.vectors[substruct_name].append(vector_info)
            
            logger.info(f"子结构 {substruct_name} 中构建了 {len(self.vectors[substruct_name])} 个向量")
    
    def get_vectors_by_type(self, substruct_name, vector_type='all'):
        """获取指定类型的向量
        
        Args:
            substruct_name: 子结构名称
            vector_type: 向量类型，可选值为'all'、'backbone'、'non_backbone'、'mixed'
        
        Returns:
            符合条件的向量列表
        """
        if substruct_name not in self.vectors:
            logger.error(f"未找到子结构 {substruct_name} 的向量信息")
            return []
        
        if vector_type == 'all':
            return self.vectors[substruct_name]
        elif vector_type == 'backbone':
            return [v for v in self.vectors[substruct_name] if v['is_backbone_vector']]
        elif vector_type == 'non_backbone':
            return [v for v in self.vectors[substruct_name] if not v['is_backbone_vector'] and not v['is_mixed_vector']]
        elif vector_type == 'mixed':
            return [v for v in self.vectors[substruct_name] if v['is_mixed_vector']]
        else:
            logger.error(f"未知的向量类型: {vector_type}")
            return []
    
    def get_vector_magnitude(self, vector_info):
        """获取向量的模长
        
        Args:
            vector_info: 向量信息字典
        
        Returns:
            向量的模长
        """
        return vector_info['magnitude']
    
    def calculate_vector_similarity(self, vector1, vector2):
        """计算两个向量之间的相似度（夹角余弦值）
        
        Args:
            vector1: 第一个向量信息字典
            vector2: 第二个向量信息字典
        
        Returns:
            向量之间的夹角余弦值
        """
        v1 = vector1['vector']
        v2 = vector2['vector']
        
        # 计算夹角余弦值
        cos_sim = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        return cos_sim
    
    def visualize_vectors(self, substruct_name, vector_indices=None, vector_type='all'):
        """可视化指定的向量
        
        Args:
            substruct_name: 子结构名称
            vector_indices: 要可视化的向量索引列表，如果为None则可视化所有向量
            vector_type: 向量类型，可选值为'all'、'backbone'、'non_backbone'、'mixed'
        """
        if substruct_name not in self.vectors:
            logger.error(f"未找到子结构 {substruct_name} 的向量信息")
            return
        
        # 获取指定类型的向量
        vectors = self.get_vectors_by_type(substruct_name, vector_type)
        
        # 如果指定了向量索引，则只可视化这些向量
        if vector_indices is not None:
            vectors = [vectors[i] for i in vector_indices if i < len(vectors)]
        
        # 创建图形
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # 绘制原子
        for vector in vectors:
            atom1 = vector['atom1']
            atom2 = vector['atom2']
            
            # 绘制起始原子
            ax.scatter(atom1['x'], atom1['y'], atom1['z'], 
                      color='blue' if atom1['atom_name'] in back_bone else 'green', 
                      s=100, alpha=0.8)
            ax.text(atom1['x'], atom1['y'], atom1['z'], atom1['atom_name'], size=8)
            
            # 绘制结束原子
            ax.scatter(atom2['x'], atom2['y'], atom2['z'], 
                      color='blue' if atom2['atom_name'] in back_bone else 'green', 
                      s=100, alpha=0.8)
            ax.text(atom2['x'], atom2['y'], atom2['z'], atom2['atom_name'], size=8)
            
            # 绘制向量
            ax.plot([atom1['x'], atom2['x']], 
                    [atom1['y'], atom2['y']], 
                    [atom1['z'], atom2['z']], 
                    color='red' if vector['is_backbone_vector'] else 
                          'purple' if vector['is_mixed_vector'] else 'orange', 
                    linewidth=2, alpha=0.6)
        
        # 设置图表属性
        ax.set_title(f"{substruct_name} - 向量可视化 ({vector_type})")
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        
        # 添加图例
        ax.scatter([], [], [], color='blue', s=100, label='Backbone原子')
        ax.scatter([], [], [], color='green', s=100, label='非Backbone原子')
        ax.plot([], [], [], color='red', linewidth=2, label='Backbone向量')
        ax.plot([], [], [], color='purple', linewidth=2, label='混合向量')
        ax.plot([], [], [], color='orange', linewidth=2, label='非Backbone向量')
        plt.legend()
        
        plt.show()


def save_vectors(vectors, output_file, file_format='csv'):
    """将向量数据保存到指定文件
    
    Args:
        vectors: 向量数据列表
        output_file: 输出文件路径
        file_format: 文件格式，可选值为'csv'、'json'
    
    Returns:
        保存是否成功
    """
    try:
        # 确保输出目录存在
        output_dir = os.path.dirname(output_file)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        if file_format.lower() == 'csv':
            # 保存为CSV格式
            with open(output_file, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                # 写入表头
                writer.writerow(['起始atom_id', '结束atom_id', '起始atom_name', '结束atom_name', 
                               'vector_x', 'vector_y', 'vector_z', '向量模长', 
                               'is_backbone_vector', 'is_mixed_vector'])
                
                # 写入数据
                for v in vectors:
                    writer.writerow([
                        v['atom1']['atom_id'], v['atom2']['atom_id'],
                        v['atom1']['atom_name'], v['atom2']['atom_name'],
                        v['vector'][0], v['vector'][1], v['vector'][2],
                        v['magnitude'],
                        v['is_backbone_vector'], v['is_mixed_vector']
                    ])
            
            logger.info(f"向量数据已保存到 {output_file} (CSV格式)")
            return True
            
        elif file_format.lower() == 'json':
            # 保存为JSON格式
            data = []
            for v in vectors:
                data.append({
                    'start_atom_id': v['atom1']['atom_id'],
                    'end_atom_id': v['atom2']['atom_id'],
                    'start_atom_name': v['atom1']['atom_name'],
                    'end_atom_name': v['atom2']['atom_name'],
                    'vector': [float(v['vector'][0]), float(v['vector'][1]), float(v['vector'][2])],
                    'magnitude': float(v['magnitude']),
                    'is_backbone_vector': v['is_backbone_vector'],
                    'is_mixed_vector': v['is_mixed_vector']
                })
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"向量数据已保存到 {output_file} (JSON格式)")
            return True
        
        else:
            logger.error(f"不支持的文件格式: {file_format}，支持的格式为: csv, json")
            return False
    
    except Exception as e:
        logger.error(f"保存向量数据时出错: {str(e)}")
        return False


def analyze_vectors(mol2_file, interactive=True, substruct_idx=None, vector_type='all', operation=None, save_to_file=None, file_format='csv'):
    """向量分析函数，支持交互式和非交互式模式
    
    Args:
        mol2_file: mol2文件路径
        interactive: 是否使用交互模式，默认为True
        substruct_idx: 非交互模式下要分析的子结构索引（从0开始）
        vector_type: 非交互模式下要分析的向量类型，可选值为'all'、'backbone'、'non_backbone'、'mixed'
        operation: 非交互模式下要执行的操作，可选值为'magnitude'、'similarity'、'visualize'
    
    Returns:
        非交互模式下返回分析结果，交互模式下返回None
    """
    # 创建分子向量分析器
    analyzer = MoleculeVectorAnalyzer(mol2_file)
    
    # 如果没有找到原子数据，直接返回
    if not analyzer.atoms:
        return None
    
    # 获取子结构列表
    substruct_names = list(analyzer.substruct_groups.keys())
    
    # 显示可用的子结构列表
    print("\n可用的子结构：")
    for i, name in enumerate(substruct_names, 1):
        print(f"{i}. {name}")
    
    # 非交互模式
    if not interactive:
        if substruct_idx is not None and 0 <= substruct_idx < len(substruct_names):
            substruct_name = substruct_names[substruct_idx]
            print(f"\n分析子结构: {substruct_name}")
            print(f"Backbone原子数量: {len(analyzer.backbone_atoms[substruct_name])}")
            print(f"非Backbone原子数量: {len(analyzer.non_backbone_atoms[substruct_name])}")
            
            # 获取向量信息
            vectors = analyzer.get_vectors_by_type(substruct_name, vector_type)
            print(f"{vector_type}类型向量数量: {len(vectors)}")
            
            # 如果指定了输出文件，则保存向量数据
            if save_to_file:
                save_success = save_vectors(vectors, save_to_file, file_format)
                if save_success:
                    print(f"向量数据已保存到 {save_to_file} ({file_format}格式)")
                else:
                    print(f"保存向量数据到 {save_to_file} 失败")
            
            # 执行指定操作
            if operation == 'magnitude':
                # 返回向量模长信息
                magnitude_info = []
                for i, v in enumerate(vectors):
                    atom1 = v['atom1']['atom_name']
                    atom2 = v['atom2']['atom_name']
                    magnitude = v['magnitude']
                    print(f"{i+1}. {atom1} -> {atom2}: {magnitude:.4f} Å")
                    magnitude_info.append({
                        'index': i,
                        'atom1': atom1,
                        'atom2': atom2,
                        'magnitude': magnitude
                    })
                return magnitude_info
            
            elif operation == 'visualize':
                # 可视化向量
                analyzer.visualize_vectors(substruct_name, None, vector_type)
                return {'substruct': substruct_name, 'vector_type': vector_type, 'visualized': True}
            
            elif operation == 'save':
                # 仅保存向量数据，不执行其他操作
                return {'substruct': substruct_name, 'vector_type': vector_type, 'saved': save_to_file is not None}
            
            # 默认返回向量信息
            return {
                'substruct': substruct_name,
                'vectors': vectors,
                'backbone_atoms': analyzer.backbone_atoms[substruct_name],
                'non_backbone_atoms': analyzer.non_backbone_atoms[substruct_name],
                'saved': save_to_file is not None
            }
        else:
            print("无效的子结构索引")
            return None
    
    # 交互模式
    while True:
        choice = input("\n请输入要分析的子结构编号（输入q退出）: ")
        if choice.lower() == 'q':
            break
        
        try:
            idx = int(choice) - 1
            substruct_name = list(analyzer.substruct_groups.keys())[idx]
            
            # 显示子结构信息
            print(f"\n子结构 {substruct_name} 信息:")
            print(f"Backbone杂原子数量: {len(analyzer.backbone_atoms[substruct_name])}")
            print(f"非Backbone杂原子数量: {len(analyzer.non_backbone_atoms[substruct_name])}")
            
            # 获取向量信息
            all_vectors = analyzer.get_vectors_by_type(substruct_name, 'all')
            backbone_vectors = analyzer.get_vectors_by_type(substruct_name, 'backbone')
            non_backbone_vectors = analyzer.get_vectors_by_type(substruct_name, 'non_backbone')
            mixed_vectors = analyzer.get_vectors_by_type(substruct_name, 'mixed')
            
            print(f"总向量数量: {len(all_vectors)}")
            print(f"Backbone向量数量: {len(backbone_vectors)}")
            print(f"非Backbone向量数量: {len(non_backbone_vectors)}")
            print(f"混合向量数量: {len(mixed_vectors)}")
            
            # 显示向量操作菜单
            while True:
                print("\n向量操作:")
                print("1. 查看向量模长")
                print("2. 计算向量相似度")
                print("3. 可视化向量")
                print("4. 返回子结构选择")
                
                op_choice = input("请选择操作: ")
                
                if op_choice == '1':
                    # 查看向量模长
                    vector_type = input("请选择向量类型(all/backbone/non_backbone/mixed): ")
                    if vector_type not in ['all', 'backbone', 'non_backbone', 'mixed']:
                        vector_type = 'all'
                    
                    vectors = analyzer.get_vectors_by_type(substruct_name, vector_type)
                    print(f"\n{vector_type}类型向量的模长:")
                    for i, v in enumerate(vectors):
                        atom1 = v['atom1']['atom_name']
                        atom2 = v['atom2']['atom_name']
                        magnitude = v['magnitude']
                        print(f"{i+1}. {atom1} -> {atom2}: {magnitude:.4f} Å")
                
                elif op_choice == '2':
                    # 计算向量相似度
                    vector_type = input("请选择向量类型(all/backbone/non_backbone/mixed): ")
                    if vector_type not in ['all', 'backbone', 'non_backbone', 'mixed']:
                        vector_type = 'all'
                    
                    vectors = analyzer.get_vectors_by_type(substruct_name, vector_type)
                    
                    # 显示向量列表
                    print(f"\n{vector_type}类型向量列表:")
                    for i, v in enumerate(vectors):
                        atom1 = v['atom1']['atom_name']
                        atom2 = v['atom2']['atom_name']
                        print(f"{i+1}. {atom1} -> {atom2}")
                    
                    # 选择两个向量计算相似度
                    try:
                        idx1 = int(input("请输入第一个向量的编号: ")) - 1
                        idx2 = int(input("请输入第二个向量的编号: ")) - 1
                        
                        if 0 <= idx1 < len(vectors) and 0 <= idx2 < len(vectors):
                            similarity = analyzer.calculate_vector_similarity(vectors[idx1], vectors[idx2])
                            print(f"向量相似度(夹角余弦值): {similarity:.4f}")
                            angle = np.arccos(np.clip(similarity, -1.0, 1.0)) * 180 / np.pi
                            print(f"夹角: {angle:.2f}°")
                        else:
                            print("无效的向量编号")
                    except ValueError:
                        print("请输入有效的数字")
                
                elif op_choice == '3':
                    # 可视化向量
                    vector_type = input("请选择向量类型(all/backbone/non_backbone/mixed): ")
                    if vector_type not in ['all', 'backbone', 'non_backbone', 'mixed']:
                        vector_type = 'all'
                    
                    vectors = analyzer.get_vectors_by_type(substruct_name, vector_type)
                    
                    # 选择是否可视化特定向量
                    specific = input("是否可视化特定向量? (y/n): ").lower()
                    if specific == 'y':
                        # 显示向量列表
                        print(f"\n{vector_type}类型向量列表:")
                        for i, v in enumerate(vectors):
                            atom1 = v['atom1']['atom_name']
                            atom2 = v['atom2']['atom_name']
                            print(f"{i+1}. {atom1} -> {atom2}")
                        
                        # 选择向量
                        try:
                            indices_str = input("请输入要可视化的向量编号(用逗号分隔): ")
                            indices = [int(idx.strip()) - 1 for idx in indices_str.split(',')]
                            analyzer.visualize_vectors(substruct_name, indices, vector_type)
                        except ValueError:
                            print("请输入有效的数字")
                    else:
                        # 可视化所有向量
                        analyzer.visualize_vectors(substruct_name, None, vector_type)
                
                elif op_choice == '4':
                    break
                
                else:
                    print("无效的选择")
        
        except (ValueError, IndexError):
            print("无效的输入，请重试")


def interactive_vector_analysis(mol2_file):
    """交互式向量分析（保留此函数以保持向后兼容性）
    
    Args:
        mol2_file: mol2文件路径
    """
    return analyze_vectors(mol2_file, interactive=True)


if __name__ == "__main__":
    # 设置输入文件
    import argparse
    
    parser = argparse.ArgumentParser(description='分子结构向量分析工具')
    parser.add_argument('--mol2', type=str, help='mol2文件路径')
    parser.add_argument('--interactive', action='store_true', help='是否使用交互模式')
    parser.add_argument('--substruct', type=int, help='要分析的子结构索引（从0开始）')
    parser.add_argument('--vector-type', type=str, default='all', 
                        choices=['all', 'backbone', 'non_backbone', 'mixed'],
                        help='要分析的向量类型')
    parser.add_argument('--operation', type=str, 
                        choices=['magnitude', 'similarity', 'visualize', 'save'],
                        help='要执行的操作')
    parser.add_argument('--save-to', type=str, help='保存向量数据的输出文件路径')
    parser.add_argument('--format', type=str, default='csv', choices=['csv', 'json'],
                        help='输出文件格式，可选值为csv、json')
    
    args = parser.parse_args()
    
    # 如果提供了mol2文件路径，则使用该路径
    if args.mol2 and os.path.exists(args.mol2):
        mol2_file = args.mol2
    else:
        # 否则提示用户输入
        mol2_file = input("请输入mol2文件路径: ")
        if not os.path.exists(mol2_file):
            print(f"文件 {mol2_file} 不存在")
            sys.exit(1)
    
    # 启动向量分析
    analyze_vectors(
        mol2_file, 
        interactive=args.interactive, 
        substruct_idx=args.substruct,
        vector_type=args.vector_type,
        operation=args.operation,
        save_to_file=args.save_to,
        file_format=args.format
    )