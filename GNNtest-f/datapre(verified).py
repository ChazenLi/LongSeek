import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import numpy as np
import torch
from torch_geometric.data import Data
from collections import defaultdict
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def parse_pdb(pdb_path):
    """解析PDB文件，返回每个模型的原子信息和连接信息"""
    models = []
    current_atoms = []
    conect = defaultdict(set)
    
    with open(pdb_path) as f:
        for line in f:
            if line.startswith("MODEL"):
                if current_atoms:
                    models.append((current_atoms, conect))
                    current_atoms = []
                    conect = defaultdict(set)
            elif line.startswith("ATOM") or line.startswith("HETATM"):
                atom_id = int(line[6:11].strip())
                atom_name = line[12:16].strip()
                res_name = line[17:20].strip()
                chain = line[21]
                res_id = int(line[22:26].strip())
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
                element = line[76:78].strip()
                
                current_atoms.append({
                    'id': atom_id,
                    'name': atom_name,
                    'res_name': res_name,
                    'chain': chain,
                    'res_id': res_id,
                    'x': x,
                    'y': y,
                    'z': z,
                    'element': element
                })
                
            elif line.startswith("CONECT"):
                parts = line.split()
                a1 = int(parts[1])
                for a2 in parts[2:]:
                    if a2 != '0':  # 跳过无效连接
                        conect[a1].add(int(a2))
                        conect[int(a2)].add(a1)  # 确保双向连接
    
    if current_atoms:
        models.append((current_atoms, conect))
    
    return models

def element_to_features(element):
    """将元素符号转换为特征向量"""
    periodic_table = {
        'H': [1, 0], 'C': [6, 4], 'N': [7, 3], 'O': [8, 2], 
        'P': [15, 5], 'S': [16, 2]
    }
    return periodic_table.get(element, [0, 0])

def pdb_to_gnn(pdb_path):
    """将PDB转换为GNN数据对象"""
    models = parse_pdb(pdb_path)
    data_list = []
    
    for atoms, conect in models:
        # 创建原子ID到索引的映射
        id_map = {atom['id']: idx for idx, atom in enumerate(atoms)}
        
        # 原子特征矩阵
        node_features = []
        positions = []
        atom_ids = []  # 用于验证连接关系
        
        for atom in atoms:
            atom_ids.append(atom['id'])
            # 特征包括元素序数、价电子数、是否为骨架原子
            element_feat = element_to_features(atom['element'])
            is_backbone = 1 if atom['name'] in ["P", "O5'", "C5'", "C4'", "C3'", "O3'"] else 0
            features = element_feat + [is_backbone]
            node_features.append(features)
            positions.append([atom['x'], atom['y'], atom['z']])
        
        # 转换为有序的边列表
        edges = []
        for a1 in sorted(conect.keys()):
            for a2 in sorted(conect[a1]):
                if a2 > a1:  # 避免重复
                    edges.append((id_map[a1], id_map[a2]))  # 使用id_map进行索引转换
        
        # 边索引
        edge_index = torch.tensor(list(zip(*edges))).long() if edges else torch.empty((2, 0), dtype=torch.long)
        
        # 创建Data对象
        data = Data(
            x=torch.tensor(node_features, dtype=torch.float),
            pos=torch.tensor(positions, dtype=torch.float),
            edge_index=edge_index,
            # 添加元信息
            atom_ids=torch.tensor(atom_ids),
            residues=[(atom['res_name'], atom['res_id']) for atom in atoms]
        )
        
        data_list.append(data)
    
    return data_list

def visualize_gnn_data(data, model_index):
    """可视化GNN数据对象"""
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    pos = data.pos.numpy()
    ax.scatter(pos[:, 0], pos[:, 1], pos[:, 2], c='b', marker='o')
    for edge in data.edge_index.T:
        start, end = pos[edge[0]], pos[edge[1]]
        ax.plot([start[0], end[0]], [start[1], end[1]], [start[2], end[2]], 'r-')
    ax.set_title(f'Model {model_index + 1}')
    plt.show()

def process_pdb_input(input_path, output_dir):
    """处理PDB输入，可以是文件或文件夹，并保存为.pt文件"""
    if os.path.isdir(input_path):
        pdb_files = [os.path.join(input_path, f) for f in os.listdir(input_path) if f.endswith('.pdb')]
    else:
        pdb_files = [input_path]
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    for pdb_file in pdb_files:
        pdb_name = os.path.splitext(os.path.basename(pdb_file))[0]
        data_list = pdb_to_gnn(pdb_file)
        for i, data in enumerate(data_list):
            model_name = f"{pdb_name}-{i+1}.pt"
            output_path = os.path.join(output_dir, model_name)
            torch.save(data, output_path)
            print(f"Saved {output_path}")
            print(f"Processing {pdb_file} - Model {i+1}")
            print("Number of nodes:", data.num_nodes)
            print("Number of edges:", data.num_edges)
            print("Node features:", data.x.shape)
            print("Edge indices:", data.edge_index.shape)
            print("Positions:", data.pos.shape)
            print("Sample atom features:", data.x[0].tolist())
            visualize_gnn_data(data, i)

# 使用示例
process_pdb_input("E:/APTAMER-GEN/GACTCCAGTCGACTGCGGGGCAAAA.pdb", "E:/APTAMER-GEN/pt")