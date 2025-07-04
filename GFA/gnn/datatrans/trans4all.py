import os
import torch
import numpy as np

def parse_mol2(mol2_path):
    """解析mol2文件并提取原子与键信息"""
    atoms, bonds = [], []
    section = None
    with open(mol2_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith('@<TRIPOS>ATOM'):
                section = 'ATOM'
                continue
            elif line.startswith('@<TRIPOS>BOND'):
                section = 'BOND'
                continue
            elif line.startswith('@'):
                section = None
                continue
            
            if section == 'ATOM':
                atoms.append(line.split())
            elif section == 'BOND':
                bonds.append(line.split())
    return atoms, bonds

def convert_features(atoms, bonds):
    """将原子和键数据转换为特征"""
    # 原子特征处理
    atom_features = []
    for atom in atoms:
        # 确保原子行有足够字段
        if len(atom) < 7:
            raise ValueError(f"Invalid ATOM line: {atom}")
        
        # 原子名编码
        name_ascii = sum(ord(c) for c in atom[1]) / 100.0  # 缩放
        
        # 坐标
        x, y, z = map(float, atom[2:5])
        
        # 原子类型编码
        type_ascii = sum(ord(c) for c in atom[5]) / 100.0
        
        # 子结构信息
        subst_id = int(atom[6])
        subst_name = sum(ord(c) for c in atom[7]) / 100.0 if len(atom) >= 8 else 0.0
        
        # 电荷（第9字段）
        charge = float(atom[8]) if len(atom) >= 9 else 0.0
        
        atom_features.append([name_ascii, x, y, z, type_ascii, subst_id, subst_name, charge])
    
    # 键处理
    bond_map = {'1':1, '2':2, '3':3, 'am':4, 'ar':5, 'du':6, 'un':7, 'nc':8, '4':9}
    edge_index = []
    bond_features = []
    for bond in bonds:
        if len(bond) < 4:
            continue  # 跳过无效行
        src = int(bond[1]) - 1
        dst = int(bond[2]) - 1
        edge_index.append([src, dst])
        edge_index.append([dst, src])  # 添加反向边
        
        bond_type = bond_map.get(bond[3].lower(), 0)
        bond_features.extend([[bond_type], [bond_type]])  # 双向特征
    
    return atom_features, edge_index, bond_features

def save_validation_file(data, output_path):
    """保存可读验证文件"""
    with open(output_path, 'w') as f:
        f.write("原子特征（已缩放）：\n")
        for i, feat in enumerate(data['atoms']):
            f.write(f"Atom {i+1}: {feat}\n")
        
        f.write("\n键连接与特征：\n")
        for i, (edge, feat) in enumerate(zip(data['edges'], data['bonds'])):
            f.write(f"Bond {i+1}: {edge[0]+1}-{edge[1]+1} → Type {feat[0]}\n")

# 主流程保持不变，确保异常处理
mol2_dir = "E:/APTAMER-GEN/mol2"
output_dir = "E:/APTAMER-GEN/pt"
os.makedirs(output_dir, exist_ok=True)

for mol2_filename in os.listdir(mol2_dir):
    if mol2_filename.endswith(".mol2"):
        mol2_path = os.path.join(mol2_dir, mol2_filename)
        try:
            atoms, bonds = parse_mol2(mol2_path)
            atom_features, edge_index, bond_features = convert_features(atoms, bonds)
            
            # 保存验证文件
            validation_data = {'atoms': atom_features, 'edges': edge_index, 'bonds': bond_features}
            validation_filename = os.path.splitext(mol2_filename)[0] + "_validation.txt"
            save_validation_file(validation_data, os.path.join(output_dir, validation_filename))
            
            # 转换为张量
            atom_tensor = torch.tensor(atom_features, dtype=torch.float32)
            edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
            bond_tensor = torch.tensor(bond_features, dtype=torch.float32)
            
            # 保存pt文件
            pt_filename = os.path.splitext(mol2_filename)[0] + ".pt"
            torch.save({
                'x': atom_tensor,
                'edge_index': edge_index,
                'edge_attr': bond_tensor
            }, os.path.join(output_dir, pt_filename))
        
        except Exception as e:
            print(f"处理文件 {mol2_filename} 时出错: {str(e)}")


