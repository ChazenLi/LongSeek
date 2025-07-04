import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from torch_geometric.data import Data
from collections import defaultdict, deque

# 价电子数
VALENCE_ELECTRONS = {
    "H": 1, "C": 4, "N": 5, "O": 6, "P": 5, "S": 6
}

# 键长参考值（含误差范围）
BOND_LENGTHS = {
    ("C", "C"): (1.54, 1.34),
    ("C", "H"): (1.09, None),
    ("C", "O"): (1.43, 1.21),
    ("C", "N"): (1.47, 1.28),
    ("O", "O"): (1.48, 1.21),
    ("N", "N"): (1.45, 1.25),
    ("S", "S"): (2.05, 1.88),
}
TOLERANCE = 0.1  # 允许误差范围

def parse_pdb(pdb_path):
    """ 解析 PDB 文件，支持多个 MODEL """
    models = []
    current_model = None

    with open(pdb_path, 'r') as f:
        for line in f:
            if line.startswith("MODEL"):
                if current_model:
                    models.append(current_model)
                current_model = {"atoms": [], "coords": [], "conect": defaultdict(list)}

            elif line.startswith(("ATOM", "HETATM")):
                if current_model is None:
                    current_model = {"atoms": [], "coords": [], "conect": defaultdict(list)}
                atom_id = int(line[6:11].strip())
                atom_name = line[12:16].strip()
                res_name = line[17:20].strip()
                chain = line[21].strip()
                res_id = int(line[22:26].strip())
                x, y, z = map(float, [line[30:38], line[38:46], line[46:54]])
                element = line[76:78].strip()
                current_model["atoms"].append({
                    "id": atom_id,
                    "name": atom_name,
                    "res_name": res_name,
                    "chain": chain,
                    "res_id": res_id,
                    "element": element
                })
                current_model["coords"].append((x, y, z))

            elif line.startswith("CONECT"):
                if current_model is None:
                    current_model = {"atoms": [], "coords": [], "conect": defaultdict(list)}
                items = list(map(int, line.split()[1:]))
                current_model["conect"][items[0]].extend(items[1:])

            elif line.startswith("ENDMDL") and current_model:
                models.append(current_model)
                current_model = None

    if current_model:
        models.append(current_model)

    return models

def classify_bond(distance, single_bond, double_bond):
    """ 判断键类型 """
    if single_bond and (single_bond - TOLERANCE) <= distance <= (single_bond + TOLERANCE):
        return "single", 1.0
    elif double_bond and (double_bond - TOLERANCE) <= distance <= (double_bond + TOLERANCE):
        return "double", 1.5
    else:
        return "unknown", 0.2  # 低置信度

def bfs_edges(atoms_info, atoms_coords, conect):
    """
    基于广度优先搜索补全边的连接、单双键信息
    """
    edges = []
    edge_attrs = []
    visited = set()
    queue = deque([0])  # 从第一个原子开始

    while queue:
        current = queue.popleft()
        if current in visited:
            continue
        visited.add(current)

        for neighbor in conect[atoms_info[current]['id']]:
            if neighbor not in visited:
                idx1, idx2 = current, [i for i, atom in enumerate(atoms_info) if atom['id'] == neighbor][0]
                coord1, coord2 = np.array(atoms_coords[idx1]), np.array(atoms_coords[idx2])
                distance = np.linalg.norm(coord1 - coord2)

                atom1, atom2 = atoms_info[idx1]["element"], atoms_info[idx2]["element"]
                single_bond, double_bond = BOND_LENGTHS.get((atom1, atom2), (None, None))

                bond_type, bond_confidence = classify_bond(distance, single_bond, double_bond)
                edges.append((idx1, idx2, atom1, atom2, distance))
                edge_attrs.append([distance, bond_confidence])

                queue.append(idx2)

    return edges, edge_attrs

def validate_and_complete_connections(atoms_info, edges):
    """
    只补充相邻的化学键，不连接随机原子
    """
    completed_edges = set((i, j) for i, j, _, _, _ in edges)
    atom_bond_count = {i: 0 for i in range(len(atoms_info))}

    for i, j, _, _, _ in edges:
        atom_bond_count[i] += 1
        atom_bond_count[j] += 1

    for i, atom in enumerate(atoms_info):
        element = atom["element"]
        if element not in VALENCE_ELECTRONS:
            continue  

        max_bonds = VALENCE_ELECTRONS[element] // 2  
        missing_bonds = max_bonds - atom_bond_count[i]

        if missing_bonds > 0:
            for j in range(i + 1, len(atoms_info)):  # 只连接后面的原子
                if (i, j) not in completed_edges and (j, i) not in completed_edges:
                    completed_edges.add((i, j))
                    completed_edges.add((j, i))
                    atom_bond_count[i] += 1
                    atom_bond_count[j] += 1

                    if atom_bond_count[i] >= max_bonds or atom_bond_count[j] >= VALENCE_ELECTRONS[atoms_info[j]["element"]] // 2:
                        break  

    return list(completed_edges)

def pdb_to_gnn_with_bond_completion(pdb_path, output_dir):
    """
    解析 PDB，创建 GNN 结构，并对每个 Model 分别保存
    """
    models = parse_pdb(pdb_path)
    base_name = os.path.splitext(os.path.basename(pdb_path))[0]
    
    os.makedirs(output_dir, exist_ok=True)  # 确保路径存在
    
    data_list = []

    for model_idx, model in enumerate(models, start=1):
        atoms_info, atoms_coords, conect = model["atoms"], model["coords"], model["conect"]
        id_map = {atom['id']: idx for idx, atom in enumerate(atoms_info)}

        node_features = [[0.0] * 38 for _ in atoms_info]
        positions = atoms_coords
        edges, edge_attrs = bfs_edges(atoms_info, atoms_coords, conect)

        completed_edges = validate_and_complete_connections(atoms_info, edges)

        edge_index = torch.tensor(completed_edges, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_attrs, dtype=torch.float) if edge_attrs else torch.empty((0, 2), dtype=torch.float)

        data = Data(
            x=torch.tensor(node_features, dtype=torch.float),
            pos=torch.tensor(positions, dtype=torch.float),
            edge_index=edge_index,
            edge_attr=edge_attr
        )

        file_name = f"{base_name}-{model_idx}.pt"
        output_path = os.path.join(output_dir, file_name)
        torch.save(data, output_path)
        print(f"✅ Saved {output_path} | Nodes={data.num_nodes}, Edges={data.num_edges}")

        if data.num_nodes > 0:
            visualize_structure(positions, completed_edges, f"{base_name}-{model_idx}")

        data_list.append(data)

    return data_list

def visualize_structure(positions, edges, title):
    """
    3D 可视化分子结构
    """
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')

    positions = np.array(positions)
    if positions.size == 0:
        return
    if positions.ndim == 1:
        positions = positions.reshape(-1, 3)
    ax.scatter(positions[:, 0], positions[:, 1], positions[:, 2], c='r', marker='o')

    for i, j in edges:
        pos_i, pos_j = positions[i], positions[j]
        ax.plot([pos_i[0], pos_j[0]], [pos_i[1], pos_j[1]], [pos_i[2], pos_j[2]], 'k-')

    ax.set_title(f"Structure: {title}")
    plt.show()

if __name__ == "__main__":
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"  # 解决OpenMP库初始化问题
    pdb_file = "E:/APTAMER-GEN/pdbdata/1AM0.pdb"
    output_dir = "E:/APTAMER-GEN/pt"
    gnn_data = pdb_to_gnn_with_bond_completion(pdb_file, output_dir)

