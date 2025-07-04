import os
import torch
import numpy as np

def decode_atom_features(features):
    """
    解码原子特征向量重建原子信息
    features: [name_ascii, x, y, z, type_ascii, subst_id, subst_name, charge]
    """
    # 解码原子名称（通过ASCII值查找最接近的组合）
    name_ascii = features[0] * 100  # 还原缩放
    # 坐标信息
    x, y, z = features[1:4]
    # 解码原子类型
    type_ascii = features[4] * 100  # 还原缩放
    # 子结构信息
    subst_id = int(features[5])
    subst_name_ascii = features[6] * 100 if features[6] != 0 else 0  # 还原缩放
    # 电荷
    charge = features[7]
    
    # 通用原子类型映射（基于常见原子）
    common_atoms = {
        'C': 67,  # ASCII for 'C'
        'N': 78,  # ASCII for 'N'
        'O': 79,  # ASCII for 'O'
        'P': 80,  # ASCII for 'P'
        'S': 83,  # ASCII for 'S'
        'H': 72   # ASCII for 'H'
    }
    
    # 找到最接近的原子名称
    atom_name = min(common_atoms.items(), key=lambda x: abs(name_ascii - x[1]))[0]
    
    # 找到最接近的原子类型
    atom_type = min(common_atoms.items(), key=lambda x: abs(type_ascii - x[1]))[0]
    
    return {
        'name': atom_name,
        'x': float(x),
        'y': float(y),
        'z': float(z),
        'type': atom_type,
        'subst_id': subst_id,
        'subst_name': f"SUB{subst_id}",  # 简化的子结构名称
        'charge': float(charge)
    }

def decode_bond_features(bond_type):
    """解码键类型"""
    bond_map_inverse = {
        1: '1',
        2: '2',
        3: '3',
        4: 'am',
        5: 'ar',
        6: 'du',
        7: 'un',
        8: 'nc',
        9: '4',
        0: '1'  # 默认值
    }
    return bond_map_inverse.get(int(bond_type[0]), '1')

def generate_mol2_content(atoms_data, edge_index, bond_features, molecule_name="MOLECULE"):
    """生成MOL2文件内容"""
    mol2_lines = []
    
    # 添加MOL2文件头
    mol2_lines.extend([
        "@<TRIPOS>MOLECULE",
        molecule_name,
        f"{len(atoms_data)} {len(edge_index) // 2} 0 0 0",
        "SMALL",
        "USER_CHARGES",
        "",
        "@<TRIPOS>ATOM"
    ])
    
    # 添加原子部分
    for i, atom in enumerate(atoms_data, 1):
        mol2_lines.append(
            f"{i:>4} {atom['name']:<4} {atom['x']:>10.4f} {atom['y']:>10.4f} {atom['z']:>10.4f} "
            f"{atom['type']:<4} {atom['subst_id']:>4} {atom['subst_name']:<8} {atom['charge']:>8.4f}"
        )
    
    # 添加键部分
    mol2_lines.append("@<TRIPOS>BOND")
    
    processed_bonds = set()
    for i, (src, dst) in enumerate(edge_index.t().tolist()):
        # 避免重复添加双向边
        bond_pair = tuple(sorted([src, dst]))
        if bond_pair not in processed_bonds:
            bond_type = decode_bond_features(bond_features[i])
            mol2_lines.append(f"{len(processed_bonds)+1:>4} {src+1:>4} {dst+1:>4} {bond_type}")
            processed_bonds.add(bond_pair)
    
    return "\n".join(mol2_lines)

def convert_pt_to_mol2(pt_path, output_dir):
    """将PT文件转换回MOL2格式"""
    try:
        # 加载PT文件
        data = torch.load(pt_path)
        
        # 解析原子特征
        atoms_data = []
        for features in data['x'].numpy():
            atom_info = decode_atom_features(features)
            atoms_data.append(atom_info)
        
        # 获取边信息
        edge_index = data['edge_index']
        bond_features = data['edge_attr'].numpy()
        
        # 生成mol2内容
        molecule_name = os.path.splitext(os.path.basename(pt_path))[0]
        mol2_content = generate_mol2_content(atoms_data, edge_index, bond_features, molecule_name)
        
        # 保存MOL2文件
        output_path = os.path.join(output_dir, f"{molecule_name}.mol2")
        os.makedirs(output_dir, exist_ok=True)
        
        with open(output_path, 'w') as f:
            f.write(mol2_content)
        
        print(f"Successfully converted {pt_path} to {output_path}")
        return output_path
        
    except Exception as e:
        print(f"Error converting {pt_path}: {str(e)}")
        return None

def batch_convert_pt_to_mol2(pt_dir, output_dir):
    """批量转换PT文件到MOL2格式"""
    os.makedirs(output_dir, exist_ok=True)
    converted_files = []
    failed_files = []
    
    for pt_file in os.listdir(pt_dir):
        if pt_file.endswith('.pt'):
            pt_path = os.path.join(pt_dir, pt_file)
            result = convert_pt_to_mol2(pt_path, output_dir)
            
            if result:
                converted_files.append(result)
            else:
                failed_files.append(pt_file)
    
    # 打印转换结果
    print(f"\nConversion Summary:")
    print(f"Successfully converted: {len(converted_files)} files")
    print(f"Failed conversions: {len(failed_files)} files")
    
    if failed_files:
        print("\nFailed files:")
        for file in failed_files:
            print(f"- {file}")

if __name__ == "__main__":
    # 示例使用
    pt_dir = "E:/APTAMER-GEN/valid/pre"  # PT文件目录
    output_dir = "E:/APTAMER-GEN/valid/pre"  # 输出目录
    
    batch_convert_pt_to_mol2(pt_dir, output_dir)