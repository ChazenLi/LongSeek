import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import networkx as nx
from collections import defaultdict
import logging
import subprocess
import tempfile
from matplotlib.colors import to_rgba
import tkinter as tk
from tkinter import ttk, messagebox

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("mol2_visualizer.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# 定义原子颜色映射（符合化学标准）
ATOM_COLORS = {
    'C': '#00FF00',  # 碳原子 - 绿色
    'H': '#FFFFFF',  # 氢原子 - 白色
    'O': '#FF0000',  # 氧原子 - 红色
    'N': '#0000FF',  # 氮原子 - 蓝色
    'P': '#FFA500',  # 磷原子 - 橙色
    'S': '#FFFF00',  # 硫原子 - 黄色
    'F': '#90EE90',  # 氟原子 - 浅绿色
    'Cl': '#A0A0A0', # 氯原子 - 灰色
    'Br': '#A52A2A', # 溴原子 - 棕色
    'I': '#800080',  # 碘原子 - 紫色
    'default': '#808080'  # 默认颜色 - 灰色
}

# 定义键类型的线型映射
BOND_STYLES = {
    '1': '-',    # 单键 - 实线
    '2': '--',   # 双键 - 虚线
    '3': '-.',   # 三键 - 点划线
    'ar': ':',   # 芳香键 - 点线
    'am': '-',   # 酰胺键 - 实线
    'du': '-',   # 虚拟键 - 实线
    'un': '-',   # 未知键 - 实线
    'nc': '-',   # 非共价键 - 实线
    'default': '-'  # 默认 - 实线
}

# PyMOL颜色映射
PYMOL_COLORS = {
    'C': 'green',
    'H': 'white',
    'O': 'red',
    'N': 'blue',
    'P': 'orange',
    'S': 'yellow',
    'F': 'palegreen',
    'Cl': 'grey',
    'Br': 'brown',
    'I': 'purple',
    'default': 'grey'
}

def parse_mol2(file_path):
    """解析 mol2 文件，返回原子和键信息"""
    atoms = []
    bonds = []
    molecule_info = {}
    current_section = None
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                
                if line.startswith('@<TRIPOS>MOLECULE'):
                    current_section = 'MOLECULE'
                    continue
                elif line.startswith('@<TRIPOS>ATOM'):
                    current_section = 'ATOM'
                    continue
                elif line.startswith('@<TRIPOS>BOND'):
                    current_section = 'BOND'
                    continue
                elif line.startswith('@<TRIPOS>'):
                    current_section = None
                    continue
                
                if current_section == 'MOLECULE':
                    # 解析分子信息
                    if 'name' not in molecule_info:
                        molecule_info['name'] = line
                    elif 'counts' not in molecule_info:
                        parts = line.split()
                        if len(parts) >= 2:
                            molecule_info['num_atoms'] = int(parts[0])
                            molecule_info['num_bonds'] = int(parts[1])
                
                elif current_section == 'ATOM':
                    # 使用更灵活的方式解析原子行
                    try:
                        parts = line.split()
                        if len(parts) >= 9:
                            # 确保原子名称完整保留，包括可能的特殊字符
                            atom_name = parts[1]
                            atom_type = parts[5]
                            
                            # 添加子结构编号信息
                            substruct_num = parts[6]  # 子结构编号
                            substruct_name = parts[7]  # 子结构名称
                            
                            # 尝试解析数值字段
                            try:
                                atom_id = int(parts[0])
                                x = float(parts[2])
                                y = float(parts[3])
                                z = float(parts[4])
                                charge = float(parts[8])
                                
                                atom_info = {
                                    'atom_id': atom_id,
                                    'atom_name': atom_name,
                                    'atom_type': atom_type,
                                    'substruct_num': substruct_num,
                                    'substruct_name': substruct_name,
                                    'x': x,
                                    'y': y,
                                    'z': z,
                                    'charge': charge
                                }
                                atoms.append(atom_info)
                            except (ValueError, IndexError) as e:
                                logger.warning(f"解析原子数值字段出错 '{line}': {str(e)}")
                        else:
                            logger.warning(f"原子行格式不正确 '{line}'")
                    except Exception as e:
                        logger.warning(f"解析原子行出错 '{line}': {str(e)}")
                
                # 键的解析
                elif current_section == 'BOND':
                    parts = line.split()
                    if len(parts) >= 4:
                        try:
                            bond_info = {
                                'bond_id': int(parts[0]),
                                'atom1_id': int(parts[1]),
                                'atom2_id': int(parts[2]),
                                'bond_type': parts[3].lower()
                            }
                            bonds.append(bond_info)
                        except (ValueError, IndexError) as e:
                            logger.warning(f"解析键行出错 '{line}': {str(e)}")
    except Exception as e:
        logger.error(f"解析文件 {file_path} 时出错: {str(e)}")
        return [], [], {}
    
    return atoms, bonds, molecule_info

def group_by_substruct(atoms, bonds):
    """按照substruct_name对原子进行分组，并找出每个子结构相关的键"""
    # 创建原子ID到原子信息的映射，用于快速查找
    atom_dict = {atom['atom_id']: atom for atom in atoms}
    
    # 按子结构名称分组原子
    substruct_groups = defaultdict(list)
    for atom in atoms:
        substruct_groups[atom['substruct_name']].append(atom)
    
    # 为每个子结构找出相关的键
    substruct_bonds = defaultdict(list)
    atom_to_substruct = {atom['atom_id']: atom['substruct_name'] for atom in atoms}
    
    for bond in bonds:
        atom1_substruct = atom_to_substruct.get(bond['atom1_id'])
        atom2_substruct = atom_to_substruct.get(bond['atom2_id'])
        
        # 如果键连接的两个原子属于同一子结构，则将该键添加到该子结构
        if atom1_substruct and atom2_substruct and atom1_substruct == atom2_substruct:
            # 确保两个原子的完整信息都被包含
            substruct_bonds[atom1_substruct].append({
                'bond_id': bond['bond_id'],
                'atom1_id': bond['atom1_id'],
                'atom2_id': bond['atom2_id'],
                'bond_type': bond['bond_type']
            })
        # 如果键连接不同子结构
        elif atom1_substruct and atom2_substruct:
            # 为每个相关的子结构添加键和完整的原子信息
            for substruct in [atom1_substruct, atom2_substruct]:
                substruct_bonds[substruct].append({
                    'bond_id': bond['bond_id'],
                    'atom1_id': bond['atom1_id'],
                    'atom2_id': bond['atom2_id'],
                    'bond_type': bond['bond_type']
                })
                
                # 确保两端原子的信息都被添加到相应的子结构中
                if substruct == atom1_substruct and atom_dict[bond['atom2_id']] not in substruct_groups[substruct]:
                    substruct_groups[substruct].append(atom_dict[bond['atom2_id']])
                if substruct == atom2_substruct and atom_dict[bond['atom1_id']] not in substruct_groups[substruct]:
                    substruct_groups[substruct].append(atom_dict[bond['atom1_id']])
    
    return substruct_groups, substruct_bonds

def get_atom_color(atom_name):
    """根据原子名称获取颜色"""
    # 提取原子元素符号（通常是名称的第一个字符，除非有特殊情况）
    element = atom_name[0]
    if len(atom_name) > 1 and atom_name[1].isalpha() and atom_name[1].islower():
        element += atom_name[1]
    
    return ATOM_COLORS.get(element, ATOM_COLORS['default'])

def get_pymol_color(atom_name):
    """根据原子名称获取PyMOL颜色名称"""
    element = atom_name[0]
    if len(atom_name) > 1 and atom_name[1].isalpha() and atom_name[1].islower():
        element += atom_name[1]
    
    return PYMOL_COLORS.get(element, PYMOL_COLORS['default'])

def get_bond_style(bond_type):
    """根据键类型获取线型"""
    return BOND_STYLES.get(bond_type, BOND_STYLES['default'])

def create_networkx_graph(atoms, bonds):
    """创建NetworkX图"""
    G = nx.Graph()
    
    # 添加节点
    for atom in atoms:
        # 确保所有必需的坐标信息都存在
        if all(k in atom for k in ['x', 'y', 'z']):
            G.add_node(
                atom['atom_id'],
                pos=(atom['x'], atom['y'], atom['z']),  # 3D位置信息
                x=atom['x'],                            # 单独的坐标
                y=atom['y'],
                z=atom['z'],
                atom_name=atom['atom_name'],
                atom_type=atom['atom_type'],
                charge=atom['charge'],
                substruct_name=atom['substruct_name'],
                substruct_num=atom['substruct_num']
            )
        else:
            logger.warning(f"原子 {atom['atom_id']} 缺少坐标信息")
    
    # 添加边
    for bond in bonds:
        G.add_edge(
            bond['atom1_id'],
            bond['atom2_id'],
            bond_type=bond['bond_type'],
            bond_id=bond['bond_id']
        )
    
    return G

def plot_3d_structure_matplotlib(G, title, output_path=None, show=True):
    """使用Matplotlib绘制三维分子结构"""
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # 获取节点位置和属性
    for node, data in G.nodes(data=True):
        try:
            # 首先尝试使用pos属性
            if 'pos' in data:
                x, y, z = data['pos']
            # 然后尝试使用单独的坐标
            elif all(k in data for k in ['x', 'y', 'z']):
                x, y, z = data['x'], data['y'], data['z']
            else:
                logger.error(f"节点 {node} 没有有效的坐标信息")
                continue
                
            atom_name = data['atom_name']
            color = get_atom_color(atom_name)
            ax.scatter(x, y, z, color=color, s=100, edgecolor='black', alpha=0.8)
            ax.text(x, y, z, atom_name, size=8, zorder=1, color='black')
        except Exception as e:
            logger.error(f"处理节点 {node} 时出错: {str(e)}")
            continue
    
    # 绘制边（键）
    for u, v, data in G.edges(data=True):
        try:
            bond_type = data['bond_type']
            
            # 获取节点位置
            u_data = G.nodes[u]
            v_data = G.nodes[v]
            
            # 获取起点坐标
            if 'pos' in u_data:
                u_pos = u_data['pos']
            elif all(k in u_data for k in ['x', 'y', 'z']):
                u_pos = (u_data['x'], u_data['y'], u_data['z'])
            else:
                logger.warning(f"边 {u}-{v} 的起点 {u} 没有有效的坐标信息")
                continue
                
            # 获取终点坐标
            if 'pos' in v_data:
                v_pos = v_data['pos']
            elif all(k in v_data for k in ['x', 'y', 'z']):
                v_pos = (v_data['x'], v_data['y'], v_data['z'])
            else:
                logger.warning(f"边 {u}-{v} 的终点 {v} 没有有效的坐标信息")
                continue
            
            # 根据键类型选择线型
            linestyle = get_bond_style(bond_type)
            
            # 绘制键
            ax.plot([u_pos[0], v_pos[0]], 
                    [u_pos[1], v_pos[1]], 
                    [u_pos[2], v_pos[2]], 
                    color='black', linestyle=linestyle, linewidth=1.5)
        except Exception as e:
            logger.error(f"处理边 {u}-{v} 时出错: {str(e)}")
            continue
    
    # 设置图表属性
    ax.set_title(title)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    
    # 调整视角
    ax.view_init(elev=20, azim=30)
    
    # 保存图像
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
    
    # 显示图像
    if show:
        plt.show()
    else:
        plt.close()

def generate_pymol_script(atoms, bonds, output_path, title, is_substruct=False, substruct_name=None):
    """生成PyMOL脚本以可视化分子结构"""
    script_path = output_path.replace('.png', '.pml')
    
    with open(script_path, 'w') as f:
        f.write("from pymol import cmd\n")
        f.write("cmd.reinitialize()\n")
        
        # 创建对象
        obj_name = "molecule"
        if is_substruct and substruct_name:
            obj_name = f"substruct_{substruct_name}"
        
        f.write(f"cmd.set('bg_color', 'white')\n")
        f.write(f"cmd.set('depth_cue', 0)\n")
        f.write(f"cmd.set('orthoscopic', 1)\n")
        
        # 添加原子
        for atom in atoms:
            atom_id = atom['atom_id']
            x, y, z = atom['x'], atom['y'], atom['z']
            atom_name = atom['atom_name']
            color = get_pymol_color(atom_name)
            
            f.write(f"cmd.pseudoatom('{obj_name}', pos=[{x}, {y}, {z}], name='{atom_name}', "
                   f"label='{atom_name}', color='{color}')\n")
        
        # 添加键
        for bond in bonds:
            atom1_id = bond['atom1_id']
            atom2_id = bond['atom2_id']
            bond_type = bond['bond_type']
            
            # 找到对应的原子
            atom1 = next((a for a in atoms if a['atom_id'] == atom1_id), None)
            atom2 = next((a for a in atoms if a['atom_id'] == atom2_id), None)
            
            if atom1 and atom2:
                f.write(f"cmd.distance('{obj_name}_bond_{atom1_id}_{atom2_id}', "
                       f"'{obj_name} and name {atom1['atom_name']} and index {atom1_id-1}', "
                       f"'{obj_name} and name {atom2['atom_name']} and index {atom2_id-1}')\n")
        
        # 设置显示样式
        f.write(f"cmd.hide('labels')\n")
        f.write(f"cmd.hide('nonbonded')\n")
        f.write(f"cmd.show('spheres')\n")
        f.write(f"cmd.show('sticks')\n")
        f.write(f"cmd.set('sphere_scale', 0.3)\n")
        f.write(f"cmd.set('stick_radius', 0.15)\n")
        
        # 设置视角
        f.write(f"cmd.zoom('{obj_name}')\n")
        f.write(f"cmd.orient('{obj_name}')\n")
        
        # 设置标题
        f.write(f"cmd.set('ray_trace_mode', 1)\n")
        f.write(f"cmd.set('ray_shadows', 0)\n")
        
        # 保存图像
        f.write(f"cmd.png('{output_path}', width=1200, height=1000, dpi=300, ray=1)\n")
        f.write(f"cmd.quit()\n")
    
    return script_path

def run_pymol_script(script_path):
    """运行PyMOL脚本"""
    try:
        # 尝试使用PyMOL命令行运行脚本
        subprocess.run(['pymol', '-c', script_path], check=True)
        return True
    except (subprocess.SubprocessError, FileNotFoundError) as e:
        logger.error(f"运行PyMOL脚本时出错: {str(e)}")
        return False

def visualize_mol2_file(file_path, output_dir=None, show_plots=True, use_pymol=False):
    """可视化mol2文件中的分子结构"""
    # 解析mol2文件
    atoms, bonds, molecule_info = parse_mol2(file_path)
    
    if not atoms:
        logger.error(f"文件 {file_path} 中没有找到原子数据")
        return
    
    # 创建输出目录
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # 获取分子名称
    molecule_name = molecule_info.get('name', os.path.basename(file_path).split('.')[0])
    logger.info(f"处理分子: {molecule_name}")
    
    # 按子结构分组
    substruct_groups, substruct_bonds = group_by_substruct(atoms, bonds)
    logger.info(f"找到 {len(substruct_groups)} 个子结构")
    
    # 为每个子结构创建图并可视化
    for substruct_name, substruct_atoms in substruct_groups.items():
        logger.info(f"处理子结构: {substruct_name}")
        
        # 获取该子结构的键
        substruct_bond_list = substruct_bonds.get(substruct_name, [])
        
        # 创建NetworkX图
        G = create_networkx_graph(substruct_atoms, substruct_bond_list)
        
        # 设置标题
        title = f"{molecule_name} - {substruct_name}"
        
        # 设置输出路径
        if output_dir:
            output_3d = os.path.join(output_dir, f"{molecule_name}_{substruct_name}_3d.png")
        else:
            output_3d = None
        
        # 使用PyMOL或Matplotlib绘制三维结构
        if use_pymol and output_dir:
            script_path = generate_pymol_script(
                substruct_atoms, substruct_bond_list, output_3d, title, 
                is_substruct=True, substruct_name=substruct_name
            )
            run_pymol_script(script_path)
        else:
            plot_3d_structure_matplotlib(G, title + " (3D)", output_path=output_3d, show=show_plots)
    
    # 创建整个分子的图并可视化
    logger.info("处理整个分子结构")
    G_whole = create_networkx_graph(atoms, bonds)
    
    # 设置输出路径
    if output_dir:
        output_whole_3d = os.path.join(output_dir, f"{molecule_name}_whole_3d.png")
    else:
        output_whole_3d = None
    
    # 使用PyMOL或Matplotlib绘制整个分子的三维结构
    if use_pymol and output_dir:
        script_path = generate_pymol_script(
            atoms, bonds, output_whole_3d, f"{molecule_name} - 完整结构"
        )
        run_pymol_script(script_path)
    else:
        plot_3d_structure_matplotlib(G_whole, f"{molecule_name} - 完整结构 (3D)", 
                                   output_path=output_whole_3d, show=show_plots)
    
    return {
        'molecule_name': molecule_name,
        'num_atoms': len(atoms),
        'num_bonds': len(bonds),
        'num_substructs': len(substruct_groups),
        'substruct_names': list(substruct_groups.keys())
    }

    
    # 创建整个分子的图并可视化
    logger.info("处理整个分子结构")
    G_whole = create_networkx_graph(atoms, bonds)
    
    # 设置输出路径
    if output_dir:
        output_whole_3d = os.path.join(output_dir, f"{molecule_name}_whole_3d.png")
    else:
        output_whole_3d = None
    
    # 使用PyMOL或Matplotlib绘制整个分子的三维结构
    if use_pymol and output_dir:
        script_path = generate_pymol_script(
            atoms, bonds, output_whole_3d, f"{molecule_name} - 完整结构"
        )
        run_pymol_script(script_path)
    else:
        plot_3d_structure_matplotlib(G_whole, f"{molecule_name} - 完整结构 (3D)", 
                                    output_path=output_whole_3d, show=show_plots)
    
    # 绘制整个分子的二维结构
    plot_2d_structure_matplotlib(G_whole, f"{molecule_name} - 完整结构 (2D)", 
                               output_path=output_whole_2d, show=show_plots)
    
    return {
        'molecule_name': molecule_name,
        'num_atoms': len(atoms),
        'num_bonds': len(bonds),
        'num_substructs': len(substruct_groups),
        'substruct_names': list(substruct_groups.keys())
    }

def create_mol2_subset(atoms, bonds, output_path):
    """创建包含指定原子和键的mol2文件子集"""
    with open(output_path, 'w') as f:
        # 写入MOLECULE部分
        f.write("@<TRIPOS>MOLECULE\n")
        f.write("Subset\n")
        f.write(f"{len(atoms)} {len(bonds)} 0 0 0\n")
        f.write("SMALL\n")
        f.write("USER_CHARGES\n\n")
        
        # 写入ATOM部分
        f.write("@<TRIPOS>ATOM\n")
        for atom in atoms:
            f.write(f"{atom['atom_id']} {atom['atom_name']} {atom['x']:.4f} {atom['y']:.4f} {atom['z']:.4f} "
                   f"{atom['atom_type']} {atom['substruct_num']} {atom['substruct_name']} {atom['charge']:.4f}\n")
        
        # 写入BOND部分
        f.write("@<TRIPOS>BOND\n")
        for i, bond in enumerate(bonds, 1):
            f.write(f"{i} {bond['atom1_id']} {bond['atom2_id']} {bond['bond_type']}\n")

def batch_process_mol2_files(input_dir, output_dir, use_pymol=False):
    """批量处理目录中的所有mol2文件"""
    if not os.path.exists(input_dir):
        logger.error(f"输入目录 {input_dir} 不存在")
        return
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 收集所有mol2文件
    mol2_files = []
    for fname in os.listdir(input_dir):
        if fname.endswith(".mol2"):
            file_path = os.path.join(input_dir, fname)
            mol2_files.append(file_path)
    
    if not mol2_files:
        logger.error(f"在 {input_dir} 中没有找到mol2文件")
        return
    
    logger.info(f"找到 {len(mol2_files)} 个mol2文件")
    
    # 处理每个文件
    results = []
    for file_path in mol2_files:
        try:
            result = visualize_mol2_file(
                file_path, 
                output_dir=os.path.join(output_dir, os.path.basename(file_path).split('.')[0]),
                show_plots=False,
                use_pymol=use_pymol
            )
            if result:
                results.append(result)
        except Exception as e:
            logger.error(f"处理文件 {file_path} 时出错: {str(e)}")
    
    # 生成汇总报告
    report_path = os.path.join(output_dir, "visualization_report.txt")
    with open(report_path, 'w') as f:
        f.write("分子结构可视化报告\n")
        f.write("=" * 50 + "\n\n")
        
        for result in results:
            f.write(f"分子: {result['molecule_name']}\n")
            f.write(f"  原子数量: {result['num_atoms']}\n")
            f.write(f"  键数量: {result['num_bonds']}\n")
            f.write(f"  子结构数量: {result['num_substructs']}\n")
            f.write(f"  子结构: {', '.join(result['substruct_names'])}\n")
            f.write("\n")
    
    logger.info(f"批处理完成，结果保存在 {output_dir}")
    logger.info(f"处理了 {len(results)} 个文件")
    
    return results

def create_gui(file_path):
    """创建交互式图形界面"""
    # 解析mol2文件
    atoms, bonds, molecule_info = parse_mol2(file_path)
    if not atoms:
        messagebox.showerror("错误", f"文件 {file_path} 中没有找到原子数据")
        return

    # 获取分子名称和子结构
    molecule_name = molecule_info.get('name', os.path.basename(file_path).split('.')[0])
    substruct_groups, substruct_bonds = group_by_substruct(atoms, bonds)

    # 创建主窗口
    root = tk.Tk()
    root.title(f"分子结构查看器 - {molecule_name}")
    root.geometry("300x400")

    # 创建框架
    frame = ttk.Frame(root, padding="10")
    frame.grid(row=0, column=0, sticky="nsew")

    # 创建标签
    ttk.Label(frame, text="选择要查看的结构：").grid(row=0, column=0, pady=5)

    # 创建列表框
    listbox = tk.Listbox(frame, height=15)
    listbox.grid(row=1, column=0, pady=5, sticky="we")

    # 添加子结构到列表
    listbox.insert(tk.END, "完整分子")
    for substruct_name in sorted(substruct_groups.keys()):
        listbox.insert(tk.END, substruct_name)

    def show_structure():
        """显示选中的结构"""
        selection = listbox.curselection()
        if not selection:
            messagebox.showwarning("警告", "请先选择一个结构")
            return

        selected_item = listbox.get(selection[0])
        
        if selected_item == "完整分子":
            # 显示完整分子
            G = create_networkx_graph(atoms, bonds)
            plot_3d_structure_matplotlib(G, f"{molecule_name} - 完整结构")
        else:
            # 显示选中的子结构
            substruct_atoms = substruct_groups[selected_item]
            substruct_bond_list = substruct_bonds.get(selected_item, [])
            G = create_networkx_graph(substruct_atoms, substruct_bond_list)
            plot_3d_structure_matplotlib(G, f"{molecule_name} - {selected_item}")

    # 创建按钮
    ttk.Button(frame, text="查看结构", command=show_structure).grid(row=2, column=0, pady=10)
    ttk.Button(frame, text="退出", command=root.quit).grid(row=3, column=0, pady=5)

    # 运行主循环
    root.mainloop()

if __name__ == "__main__":
    # 设置输入文件
    mol2_file = r"E:/APTAMER-GEN/mol2/1Q8N-10.mol2"
    
    # 启动图形界面
    create_gui(mol2_file)