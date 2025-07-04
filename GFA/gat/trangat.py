import os
import torch
import numpy as np
from collections import defaultdict

def parse_mol2(mol2_path):
    """解析mol2文件（与之前相同）"""
    atoms, bonds = [], []
    section = None
    with open(mol2_path, 'r') as f:
        for line in f:
            if line.startswith('@<TRIPOS>ATOM'):
                section = 'ATOM'
                continue
            elif line.startswith('@<TRIPOS>BOND'):
                section = 'BOND'
                continue
            elif line.startswith('@'):
                section = None
            
            if section == 'ATOM' and line.strip():
                atoms.append(line.strip().split())
            elif section == 'BOND' and line.strip():
                bonds.append(line.strip().split())
    return atoms, bonds

def compute_substructure_features(atoms):
    """子结构特征计算（与之前相同）"""
    # ...（保持与之前完全相同的实现）...
    subst_data = defaultdict(lambda: {
        'charges': [], 
        'coords': [],
        'atom_types': []
                                    })
    
    # 收集子结构信息
    for atom in atoms:
        subst_id = int(atom[6])
        subst_data[subst_id]['charges'].append(float(atom[8]) if len(atom)>8 else 0.0)
        subst_data[subst_id]['coords'].append([float(atom[2]), float(atom[3]), float(atom[4])])
        subst_data[subst_id]['atom_types'].append(atom[5])
    
    # 计算统计量
    subst_features = {}
    for subst_id, data in subst_data.items():
        avg_coords = np.mean(data['coords'], axis=0) if data['coords'] else [0.0, 0.0, 0.0]
        subst_features[subst_id] = {
            'size': len(data['coords']),
            'avg_charge': np.mean(data['charges']) if data['charges'] else 0.0,
            'coord_centroid': avg_coords,
            'type_diversity': np.var([sum(ord(c) for c in t) for t in data['atom_types']]) if data['atom_types'] else 0.0
        }
    return subst_features

def convert_for_gat(atoms, bonds):
    """GAT专用特征转换"""
    # 计算子结构特征
    subst_features = compute_substructure_features(atoms)
    
    # 原子特征处理（增加归一化）
    atom_features = []
    for atom in atoms:
        # 基础特征
        name_ascii = sum(ord(c) for c in atom[1]) / 100.0
        x, y, z = map(lambda v: float(v)/10.0, atom[2:5])
        type_ascii = sum(ord(c) for c in atom[5]) / 100.0
        subst_id = int(atom[6]) / 100.0
        subst_name = sum(ord(c) for c in atom[7])/100.0 if len(atom)>7 else 0.0
        charge = float(atom[8]) if len(atom)>8 else 0.0
        
        # 子结构增强特征
        sf = subst_features.get(int(atom[6]), {})
        atom_features.append([
            name_ascii, x, y, z, type_ascii,
            subst_id, subst_name, charge,
            sf.get('size', 0)/10.0,
            sf.get('avg_charge', 0.0),
            sf.get('type_diversity', 0.0)/100.0,
            *(np.array(sf.get('coord_centroid', [0,0,0]))/10.0)
        ])
    
    # 边处理（适配GAT）
    edge_index = []
    for bond in bonds:
        src = int(bond[1])-1  # 0-based
        dst = int(bond[2])-1
        edge_index.append([src, dst])
        edge_index.append([dst, src])  # 无向图双向边
    
    return atom_features, edge_index

def save_gat_data(pt_dir, mol2_path):
    """保存GAT兼容数据"""
    atoms, bonds = parse_mol2(mol2_path)
    atom_feats, edges = convert_for_gat(atoms, bonds)
    
    # 转换为张量
    data = {
        'x': torch.tensor(atom_feats, dtype=torch.float32),
        'edge_index': torch.tensor(edges, dtype=torch.long).t().contiguous(),
        'num_nodes': len(atom_feats)  # 用于处理变长图
    }
    
    # 保存文件
    base_name = os.path.basename(mol2_path).replace('.mol2', '.pt')
    torch.save(data, os.path.join(pt_dir, base_name))

def batch_process_mol2(mol2_dir, pt_dir):
    """批量处理"""
    os.makedirs(pt_dir, exist_ok=True)
    for fname in os.listdir(mol2_dir):
        if fname.endswith('.mol2'):
            save_gat_data(pt_dir, os.path.join(mol2_dir, fname))

if __name__ == "__main__":
    batch_process_mol2(
        mol2_dir="E:/APTAMER-GEN/mol2",
        pt_dir="E:/APTAMER-GEN/gatpt"
    )