import os
import torch
import numpy as np

def parse_mol2(mol2_path):
    """直接解析mol2文件原始数据"""
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

def convert_features(atoms, bonds):
    """特征转换核心逻辑"""
    # 原子特征处理
    atom_features = []
    for atom in atoms:
        # 原子名ASCII编码（示例：C2 -> 67+32=99）
        name_ascii = sum(ord(c) for c in atom[1]) / 1.0
        
        # 坐标处理
        x = float(atom[2]) / 1.0
        y = float(atom[3]) / 1.0
        z = float(atom[4]) / 1.0
        
        # 原子类型ASCII编码（示例：C.3 -> 67+46+51=164）
        type_ascii = sum(ord(c) for c in atom[5]) / 1.0
        
        # 子结构信息
        subst_id = int(atom[6])
        subst_name = sum(ord(c) for c in atom[7]) / 1.0 if len(atom)>7 else 0.0
        
        # 电荷
        charge = float(atom[8]) if len(atom)>8 else 0.0
        
        atom_features.append([
            name_ascii, x, y, z, type_ascii,
            subst_id, subst_name, charge
        ])
    
    # 键特征处理
    bond_map = {'1':1, '2':2, '3':3, 'am':4, 'ar':5, 'du':6, 'un':7, 'nc':8}
    edge_index = []
    bond_features = []
    for bond in bonds:
        src = int(bond[1])-1  # 转换为0-based索引
        dst = int(bond[2])-1
        edge_index.append([src, dst])
        
        # 键类型映射
        bond_type = bond_map.get(bond[3].lower(), 0)
        bond_features.append([bond_type / 1.0])
    
    return atom_features, edge_index, bond_features

def save_validation_file(data, output_path):
    """保存验证文件"""
    with open(output_path, 'w') as f:
        f.write("原子特征验证：\n")
        for i, feat in enumerate(data['atoms']):
            f.write(f"Atom {i+1}: {feat}\n")
        
        f.write("\n键特征验证：\n")
        for i, (edge, feat) in enumerate(zip(data['edges'], data['bonds'])):
            f.write(f"Bond {i+1}: {edge} -> {feat}\n")

# 主流程
mol2_path = "E:/APTAMER-GEN/mol2/1AM0-1.mol2"
output_dir = "E:/APTAMER-GEN/pt"

# 1. 解析原始数据
atoms, bonds = parse_mol2(mol2_path)

# 2. 特征转换
atom_features, edge_index, bond_features = convert_features(atoms, bonds)

# 3. 保存验证文件
validation_data = {
    'atoms': atom_features,
    'edges': edge_index,
    'bonds': bond_features
}
save_validation_file(validation_data, os.path.join(output_dir, "conversion_validation.txt"))

# 4. 转换为张量
atom_tensor = torch.tensor(atom_features, dtype=torch.float32)
edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
bond_tensor = torch.tensor(bond_features, dtype=torch.float32)

# 5. 保存pt文件
torch.save({
    'x': atom_tensor,
    'edge_index': edge_index,
    'edge_attr': bond_tensor
}, os.path.join(output_dir, "molecule_graph.pt"))
