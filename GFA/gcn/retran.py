import os
import torch
import numpy as np

# ------------------------ 必须与预处理代码完全一致的配置 ------------------------
# 原子名编码参数
ATOM_NAME_LEN = 4
ATOM_CHUNK_SIZE = 2
ATOM_NUM_CHUNKS = ATOM_NAME_LEN // ATOM_CHUNK_SIZE
ATOM_SCALE = 10 ** (3 * ATOM_CHUNK_SIZE)  # 1e6

# 原子类型编码参数
TYPE_NAME_LEN = 5
TYPE_CHUNK_SIZE = 2
TYPE_NUM_CHUNKS = TYPE_NAME_LEN // TYPE_CHUNK_SIZE
TYPE_SCALE = 10 ** (3 * TYPE_CHUNK_SIZE)  # 1e6

# 子结构名称编码参数
SUB_NAME_LEN = 9
SUB_CHUNK_SIZE = 3
SUB_NUM_CHUNKS = SUB_NAME_LEN // SUB_CHUNK_SIZE
SUB_SCALE = 10 ** (3 * SUB_CHUNK_SIZE)  # 1e9

# ------------------------ 解码函数 ------------------------
def chunk_decode(encoded_chunks, chunk_size, scale, max_len):
    """分块解码函数（修复特殊字符处理）"""
    chars = []
    for chunk_val in encoded_chunks:
        chunk_code = int(round(chunk_val * scale))
        str_code = f"{chunk_code:0{3*chunk_size}d}"  # 三位补零
        for i in range(0, 3*chunk_size, 3):
            part = str_code[i:i+3]
            ascii_val = int(part)
            if 32 <= ascii_val <= 126:  # 仅保留可打印字符
                chars.append(chr(ascii_val))
    # 清理填充字符并截断到最大长度
    return ''.join(chars).replace('\x00', '').strip()[:max_len]

def decode_atom_name(encoded):
    return chunk_decode(encoded, ATOM_CHUNK_SIZE, ATOM_SCALE, ATOM_NAME_LEN)

def decode_atom_type(encoded):
    decoded = chunk_decode(encoded, TYPE_CHUNK_SIZE, TYPE_SCALE, TYPE_NAME_LEN)
    return decoded.replace('#', '.')  # 恢复预处理时替换的符号

def decode_subname(encoded):
    return chunk_decode(encoded, SUB_CHUNK_SIZE, SUB_SCALE, SUB_NAME_LEN)

# ------------------------ 逆转换主逻辑 ------------------------
def pt_to_mol2(pt_path, output_dir):
    """修复版逆转换函数"""
    # 加载数据
    data = torch.load(pt_path)
    atoms = []
    
    # 遍历所有原子特征
    for idx, feat in enumerate(data['x'].numpy()):
        # ------------------------ 解析特征索引 ------------------------
        # 原子名编码（前2维）
        atom_name_enc = feat[0:ATOM_NUM_CHUNKS]
        # 原子类型编码（接下来3维）
        atom_type_enc = feat[ATOM_NUM_CHUNKS : ATOM_NUM_CHUNKS+TYPE_NUM_CHUNKS]
        # 子结构名称编码（接下来3维）
        subname_enc = feat[ATOM_NUM_CHUNKS+TYPE_NUM_CHUNKS : 
                          ATOM_NUM_CHUNKS+TYPE_NUM_CHUNKS+SUB_NUM_CHUNKS]
        # 坐标（接下来3维）
        x = feat[ATOM_NUM_CHUNKS+TYPE_NUM_CHUNKS+SUB_NUM_CHUNKS] * 10.0
        y = feat[ATOM_NUM_CHUNKS+TYPE_NUM_CHUNKS+SUB_NUM_CHUNKS+1] * 10.0
        z = feat[ATOM_NUM_CHUNKS+TYPE_NUM_CHUNKS+SUB_NUM_CHUNKS+2] * 10.0
        # 子结构ID（接下来1维，直接取整）
        subst_id = int(feat[ATOM_NUM_CHUNKS+TYPE_NUM_CHUNKS+SUB_NUM_CHUNKS+3])
        # 电荷（最后1维，直接读取）
        charge = feat[ATOM_NUM_CHUNKS+TYPE_NUM_CHUNKS+SUB_NUM_CHUNKS+4]
        
        # ------------------------ 解码字符串 ------------------------
        atom_name = decode_atom_name(atom_name_enc)
        atom_type = decode_atom_type(atom_type_enc)
        subname = decode_subname(subname_enc)
        
        # 保存原子数据
        atoms.append({
            'id': idx + 1,
            'name': atom_name,
            'x': x,
            'y': y,
            'z': z,
            'type': atom_type,
            'subst_id': subst_id,
            'subname': subname,
            'charge': charge
        })
    
    # ------------------------ 处理键 ------------------------
    seen_bonds = set()
    bonds = []
    edge_index = data['edge_index'].t().tolist()
    edge_attr = data['edge_attr'].numpy()
    
    bond_type_map = {
        0.1: '1', 0.2: '2', 0.3: '3',
        0.4: 'am', 0.5: 'ar', 0.6: 'du',
        0.7: 'un', 0.8: 'nc'
    }
    
    for i, (src, dst) in enumerate(edge_index):
        if (src, dst) in seen_bonds or (dst, src) in seen_bonds:
            continue
        seen_bonds.add((src, dst))
        
        # 获取键类型
        encoded_type = edge_attr[i][0]
        btype = bond_type_map.get(round(encoded_type, 1), 'un')
        
        bonds.append({
            'src': src + 1,  # 恢复1-based索引
            'dst': dst + 1,
            'type': btype
        })
    
    # ------------------------ 生成MOL2内容 ------------------------
    mol2_content = [
        "@<TRIPOS>MOLECULE",
        "RECONSTRUCTED",
        f"{len(atoms)} {len(bonds)} 0 0 0",
        "SMALL",
        "USER_CHARGES",
        "@<TRIPOS>ATOM"
    ]
    
    # 原子部分
    for atom in atoms:
        line = (
            f"{atom['id']:>5} {atom['name']:<4} "
            f"{atom['x']:>9.4f} {atom['y']:>9.4f} {atom['z']:>9.4f} "
            f"{atom['type']:<5} {atom['subst_id']} "
            f"{atom['subname']} {atom['charge']:>7.4f}"
        )
        mol2_content.append(line)
    
    # 键部分
    mol2_content.append("@<TRIPOS>BOND")
    for i, bond in enumerate(bonds, 1):
        mol2_content.append(f"{i:>5} {bond['src']} {bond['dst']} {bond['type']}")
    
    # ------------------------ 保存文件 ------------------------
    os.makedirs(output_dir, exist_ok=True)
    base_name = os.path.basename(pt_path).replace('.pt', '.mol2')
    output_path = os.path.join(output_dir, base_name)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(mol2_content))
    
    print(f"成功转换: {output_path}")

def batch_convert(pt_dir, output_dir):
    """批量转换"""
    for fname in os.listdir(pt_dir):
        if fname.endswith('.pt'):
            pt_path = os.path.join(pt_dir, fname)
            pt_to_mol2(pt_path, output_dir)

if __name__ == '__main__':
    batch_convert(
        pt_dir='E:/APTAMER-GEN/gcnpt',
        output_dir='E:/APTAMER-GEN/gcnvalid/mol2_reconstructed'
    )
    # 批量转换
    # batch_pt_to_mol2(
    #     pt_dir="E:/APTAMER-GEN/gcnvalid/pt",
    #     output_dir="E:/APTAMER-GEN/gcnvalid/mol2_reconstructed"
    # )