import os
import sys
import numpy as np
from scipy.spatial import ConvexHull
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
matplotlib.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# 添加父目录到系统路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mol2_pymol_visualizer.mol2_pymol_visualizer import (
    parse_mol2, group_by_substruct, create_networkx_graph, 
    plot_3d_structure_matplotlib, logger
)

def calculate_minimum_bounding_sphere(points):
    """计算最小外接球"""
    hull = ConvexHull(points)
    center_estimate = np.mean(points[hull.vertices], axis=0)
    
    # 计算到最远点的距离作为半径
    distances = np.linalg.norm(points - center_estimate, axis=1)
    radius = np.max(distances)
    
    # 使用迭代方法优化球心位置
    max_iterations = 100
    tolerance = 1e-6
    min_surface_points = 4  # 至少需要4个点确定球
    
    for _ in range(max_iterations):
        distances = np.linalg.norm(points - center_estimate, axis=1)
        surface_points = points[np.abs(distances - radius) < tolerance]
        
        if len(surface_points) < min_surface_points:
            # 如果表面点太少，增加容差直到找到足够的点
            while len(surface_points) < min_surface_points and tolerance < 0.5:
                tolerance *= 2
                surface_points = points[np.abs(distances - radius) < tolerance]
        
        if len(surface_points) >= min_surface_points:
            new_center = np.mean(surface_points, axis=0)
            if np.linalg.norm(new_center - center_estimate) < tolerance:
                center_estimate = new_center
                break
            center_estimate = new_center
        
        distances = np.linalg.norm(points - center_estimate, axis=1)
        radius = np.max(distances)
    
    return center_estimate, radius

def identify_surface_atoms(atoms, center, radius, tolerance=0.1):
    """识别位于球面上的原子
    
    Args:
        atoms: 原子列表
        center: 球心坐标
        radius: 球半径
        tolerance: 判断原子是否在球面上的容差（单位：埃）
        
    Returns:
        surface_atoms: 位于球面上的原子列表
    """
    surface_atoms = []
    
    for atom in atoms:
        point = np.array([atom['x'], atom['y'], atom['z']])
        distance = np.linalg.norm(point - center)
        
        # 如果原子到球心的距离接近球半径，认为它在球面上
        if abs(distance - radius) < tolerance:
            surface_atoms.append(atom)
    
    return surface_atoms

def analyze_substruct_spheres(mol2_file):
    """分析分子的每个子结构的最小外接球
    
    Args:
        mol2_file: mol2文件路径
    
    Returns:
        sphere_info: 包含每个子结构球信息的字典
    """
    # 解析mol2文件
    atoms, bonds, molecule_info = parse_mol2(mol2_file)
    if not atoms:
        logger.error(f"文件 {mol2_file} 中没有找到原子数据")
        return {}
    
    # 按子结构分组
    substruct_groups, substruct_bonds = group_by_substruct(atoms, bonds)
    sphere_info = {}
    
    # 分析每个子结构
    for substruct_name, substruct_atoms in substruct_groups.items():
        # 提取原子坐标
        points = np.array([[atom['x'], atom['y'], atom['z']] for atom in substruct_atoms])
        
        # 计算最小外接球
        center, radius = calculate_minimum_bounding_sphere(points)
        
        # 识别球面上的原子
        surface_atoms = identify_surface_atoms(substruct_atoms, center, radius)
        
        # 存储信息
        sphere_info[substruct_name] = {
            'center': center,
            'radius': radius,
            'surface_atoms': surface_atoms,
            'all_atoms': substruct_atoms,
            'bonds': substruct_bonds.get(substruct_name, [])
        }
        
        logger.info(f"子结构 {substruct_name} 的球心坐标: {center}, 半径: {radius:.2f}")
        logger.info(f"球面上的原子数量: {len(surface_atoms)}")
    
    return sphere_info

def _add_sphere_surface(ax, center, radius):
    """添加球面网格（内部辅助函数）"""
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 50)
    x = radius * np.outer(np.cos(u), np.sin(v)) + center[0]
    y = radius * np.outer(np.sin(u), np.sin(v)) + center[1]
    z = radius * np.outer(np.ones(np.size(u)), np.cos(v)) + center[2]
    ax.plot_surface(x, y, z, color='b', alpha=0.1)

def _mark_surface_atoms(ax, surface_atoms):
    """标记球面上的原子（内部辅助函数）"""
    for atom in surface_atoms:
        ax.scatter(atom['x'], atom['y'], atom['z'], 
                  color='red', s=150, alpha=0.5, 
                  label='Surface Atom')

def visualize_sphere_structure(sphere_info, substruct_name):
    """可视化带有最小外接球的子结构
    
    Args:
        sphere_info: 子结构的球信息
        substruct_name: 要可视化的子结构名称
    """
    if substruct_name not in sphere_info:
        logger.error(f"未找到子结构 {substruct_name} 的信息")
        return
    
    info = sphere_info[substruct_name]
    
    # 创建NetworkX图
    G = create_networkx_graph(info['all_atoms'], info['bonds'])
    
    # 添加球心作为特殊节点
    center = info['center']
    G.add_node('sphere_center', 
               pos=tuple(center),
               x=center[0],
               y=center[1],
               z=center[2],
               atom_name='*',  # 使用特殊标记表示球心
               atom_type='virtual',
               charge=0.0,
               substruct_name=substruct_name,
               substruct_num=0)
    
    # 绘制3D结构
    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_subplot(111, projection='3d')
    plot_3d_structure_matplotlib(G, f"{substruct_name} - 带最小外接球")
    
    # 添加球面和标记原子
    _add_sphere_surface(ax, center, info['radius'])
    _mark_surface_atoms(ax, info['surface_atoms'])
    
    plt.show()

def print_sphere_info(sphere_info, substruct_name):
    """打印球的详细参数信息"""
    if substruct_name not in sphere_info:
        logger.error(f"未找到子结构 {substruct_name} 的信息")
        return
    
    info = sphere_info[substruct_name]
    center = info['center']
    radius = info['radius']
    # 转换为埃(Å)单位
    radius_angstrom = radius * 0.1
    volume = (4/3) * np.pi * (radius_angstrom**3)
    surface_area = 4 * np.pi * (radius_angstrom**2)
    
    print("\n" + "="*50)
    print(f"子结构 {substruct_name} 的球参数信息：")
    print("="*50)
    print(f"球心坐标: ({center[0]:.3f}, {center[1]:.3f}, {center[2]:.3f}) Å")
    print(f"球半径: {radius_angstrom:.3f} Å")
    print(f"球体积: {volume:.3f} Å³")
    print(f"表面积: {surface_area:.3f} Å²")
    print(f"球面原子数量: {len(info['surface_atoms'])}")
    if len(info['surface_atoms']) < 4:
        print("\n警告：球面原子数量少于4个，可能需要调整容差值")
    print("\n球面原子列表：")
    for atom in info['surface_atoms']:
        print(f"原子ID: {atom['atom_id']}, 类型: {atom['atom_type']}, "
              f"坐标: ({atom['x']:.3f}, {atom['y']:.3f}, {atom['z']:.3f}) Å")
    print("="*50 + "\n")

def query_and_visualize_substruct(mol2_file):
    """交互式查询和可视化子结构的最小外接球"""
    # 分析所有子结构的球
    sphere_info = analyze_substruct_spheres(mol2_file)
    
    if not sphere_info:
        print("未找到任何子结构")
        return
    
    # 显示可用的子结构列表
    print("\n可用的子结构：")
    for i, name in enumerate(sphere_info.keys(), 1):
        print(f"{i}. {name}")
    
    while True:
        choice = input("\n请输入要查看的子结构编号（输入q退出）: ")
        if choice.lower() == 'q':
            break
        
        try:
            idx = int(choice) - 1
            substruct_name = list(sphere_info.keys())[idx]
            
            # 打印球的参数信息
            print_sphere_info(sphere_info, substruct_name)
            
            # 可视化带球的结构
            visualize_sphere_structure(sphere_info, substruct_name)
            
        except (ValueError, IndexError):
            print("无效的输入，请重试")

if __name__ == "__main__":
    # 设置输入文件
    mol2_file = r"E:\APTAMER-GEN\mol2\1AW4-1.mol2"
    
    # 启动交互式查询
    query_and_visualize_substruct(mol2_file)