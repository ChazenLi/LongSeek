import os
import torch
from torch_geometric.data import Data
from torch_geometric.data.data import DataEdgeAttr

# 添加 DataEdgeAttr 到安全全局列表中
torch.serialization.add_safe_globals([DataEdgeAttr])

def load_pt_file(file_path):
    """加载.pt文件"""
    data = torch.load(file_path, weights_only=False)
    print(f"Loaded data type: {type(data)}")
    if isinstance(data, Data):
        print(f"Data attributes: {data.__dict__}")
    else:
        raise ValueError("Loaded data is not a Data object")
    return data

def parse_pdb(pdb_path):
    """解析PDB文件"""
    atoms = []
    with open(pdb_path, 'r') as f:
        for line in f:
            if line.startswith("ATOM") or line.startswith("HETATM"):
                atom_id = int(line[6:11].strip())
                atom_name = line[12:16].strip()
                res_name = line[17:20].strip()
                chain = line[21].strip()
                res_id = int(line[22:26].strip())
                x = float(line[30:38].strip())
                y = float(line[38:46].strip())
                z = float(line[46:54].strip())
                element = line[76:78].strip()
                atoms.append({
                    'atom_id': atom_id,
                    'atom_name': atom_name,
                    'res_name': res_name,
                    'chain': chain,
                    'res_id': res_id,
                    'x': x,
                    'y': y,
                    'z': z,
                    'element': element
                })
    return atoms

def write_pdb(atoms, output_path):
    """将原子信息写入PDB文件"""
    with open(output_path, 'w') as f:
        f.write("MODEL        1\n")
        for atom in atoms:
            atom_line = f"ATOM  {atom['atom_id']:5d}  {atom['atom_name']:<4} {atom['res_name']:<3} {atom['chain']:<1}{atom['res_id']:4d}    {atom['x']:8.3f}{atom['y']:8.3f}{atom['z']:8.3f}                      {atom['element']:>2}\n"
            f.write(atom_line)
        f.write("ENDMDL\n")

def main():
    pdb_file = "e:/APTAMER-GEN/GACTCCAGTCGACTGCGGGGCAAAA.pdb"
    pt_file = "E:/APTAMER-GEN/pt/GACTCCAGTCGACTGCGGGGCAAAA-1_predicted.pt"
    output_pdb_file = "E:/APTAMER-GEN/decode/GACTCCAGTCGACTGCGGGGCAAAA_predicted.pdb"
    
    # 解析初始PDB文件
    atoms = parse_pdb(pdb_file)
    
    # 加载预测的PT文件
    data = load_pt_file(pt_file)
    
    # 更新原子坐标
    for i, pos in enumerate(data.pos):
        atoms[i]['x'], atoms[i]['y'], atoms[i]['z'] = pos.tolist()
    
    # 写入新的PDB文件
    write_pdb(atoms, output_pdb_file)
    print(f"Predicted PDB saved to {output_pdb_file}")

if __name__ == "__main__":
    main()