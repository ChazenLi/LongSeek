import os
import torch
import numpy as np
from collections import defaultdict

# ------------------------ 全局编码配置 ------------------------
# 原子名编码参数
ATOM_NAME_LEN = 4    # 最大长度（如"C12"）
ATOM_CHUNK_SIZE = 2  # 分块字符数
ATOM_NUM_CHUNKS = ATOM_NAME_LEN // ATOM_CHUNK_SIZE
ATOM_SCALE = 10 ** (3 * ATOM_CHUNK_SIZE)  # 1e6

# 原子类型编码参数
TYPE_NAME_LEN = 5    # 最大长度（如"C.ar"）
TYPE_CHUNK_SIZE = 2
TYPE_NUM_CHUNKS = TYPE_NAME_LEN // TYPE_CHUNK_SIZE
TYPE_SCALE = 10 ** (3 * TYPE_CHUNK_SIZE)  # 1e6

# 子结构名称编码参数
SUB_NAME_LEN = 9
SUB_CHUNK_SIZE = 3
SUB_NUM_CHUNKS = SUB_NAME_LEN // SUB_CHUNK_SIZE
SUB_SCALE = 10 ** (3 * SUB_CHUNK_SIZE)  # 1e9

# ------------------------ 编码函数 ------------------------
def chunk_encode(name, max_len, chunk_size, scale):
    """通用分块编码函数"""
    encoded = []
    name_padded = name[:max_len].ljust(max_len, '\x00')
    for i in range(0, max_len, chunk_size):
        chunk = name_padded[i:i+chunk_size]
        code = 0
        for j, c in enumerate(chunk):
            ascii_val = ord(c) if c != '\x00' else 0
            code += ascii_val * (1000 ** (chunk_size - j -1))
        encoded.append(code / scale)
    return encoded

def encode_atom_name(name):
    return chunk_encode(name, ATOM_NAME_LEN, ATOM_CHUNK_SIZE, ATOM_SCALE)

def encode_atom_type(atype):
    return chunk_encode(atype, TYPE_NAME_LEN, TYPE_CHUNK_SIZE, TYPE_SCALE)

def encode_sub_name(subname):
    return chunk_encode(subname, SUB_NAME_LEN, SUB_CHUNK_SIZE, SUB_SCALE)

# ------------------------ 特征转换 ------------------------
def parse_mol2(mol2_path):
    """解析MOL2文件"""
    atoms, bonds = [], []
    section = None
    with open(mol2_path, 'r', encoding='utf-8') as f:
        for line in f:
            if '@<TRIPOS>ATOM' in line:
                section = 'ATOM'
            elif '@<TRIPOS>BOND' in line:
                section = 'BOND'
            elif line.startswith('@'):
                section = None
            elif section == 'ATOM' and line.strip():
                atoms.append(line.strip().split())
            elif section == 'BOND' and line.strip():
                bonds.append(line.strip().split())
    return atoms, bonds

def compute_sub_features(atoms):
    """计算子结构级特征"""
    subst_data = defaultdict(lambda: {
        'coords': [], 'charges': [], 'atom_types': []
    })
    
    for atom in atoms:
        subst_id = int(atom[6])
        record = subst_data[subst_id]
        record['coords'].append(list(map(float, atom[2:5])))
        record['charges'].append(float(atom[8]) if len(atom)>8 else 0.0)
        record['atom_types'].append(atom[5])
    
    features = {}
    for subst_id, data in subst_data.items():
        features[subst_id] = {
            'size': len(data['coords']),
            'avg_charge': np.mean(data['charges']) if data['charges'] else 0.0,
            'centroid': np.mean(data['coords'], axis=0) if data['coords'] else [0.0]*3,
            'type_var': np.var([sum(ord(c) for c in t) for t in data['atom_types']])
        }
    return features

def convert_features(atoms, bonds):
    """特征转换主函数"""
    subst_features = compute_sub_features(atoms)
    atom_features, edge_index, edge_attr = [], [], []
    
    # 处理原子
    for atom in atoms:
        subst_id = int(atom[6])
        # 原子级编码
        atom_feat = [
            *encode_atom_name(atom[1]),
            *encode_atom_type(atom[5]),
            *encode_sub_name(atom[7] if len(atom)>7 else ""),
            float(atom[2])/10.0,  # x
            float(atom[3])/10.0,  # y
            float(atom[4])/10.0,  # z
            int(atom[6]),          # subst_id (原始整数)
            float(atom[8]) if len(atom)>8 else 0.0  # charge
        ]

        atom_features.append(atom_feat)
    
    # 处理键
    bond_map = {'1':1, '2':2, '3':3, 'am':4, 'ar':5, 'du':6, 'un':7, 'nc':8}
    for bond in bonds:
        src, dst = int(bond[1])-1, int(bond[2])-1
        edge_index.extend([[src, dst], [dst, src]])  # 双向边
        btype = bond_map.get(bond[3].lower(), 0)
        edge_attr.extend([[btype/10.0], [btype/10.0]])
    
    return atom_features, edge_index, edge_attr

def process_folder(mol2_dir, pt_dir):
    """批量处理目录"""
    os.makedirs(pt_dir, exist_ok=True)
    for fname in os.listdir(mol2_dir):
        if fname.endswith('.mol2'):
            atoms, bonds = parse_mol2(os.path.join(mol2_dir, fname))
            x, edge_index, edge_attr = convert_features(atoms, bonds)
            data = {
                'x': torch.tensor(x, dtype=torch.float32),
                'edge_index': torch.tensor(edge_index, dtype=torch.long).t().contiguous(),
                'edge_attr': torch.tensor(edge_attr, dtype=torch.float32)
            }
            torch.save(data, os.path.join(pt_dir, fname.replace('.mol2', '.pt')))

if __name__ == '__main__':
    process_folder(
        mol2_dir='E:/APTAMER-GEN/mol2',
        pt_dir='E:/APTAMER-GEN/gcnpt'
    )