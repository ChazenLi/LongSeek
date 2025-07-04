# GNN for aptamer structure prediction
## 1.数据预处理
### 获取aptamer的pdb核酸空间结构数据
    先从rscb获取ID数据，顺便检索aptamer的已知结构数据；----down
    基于获取的ID数据，下载pdb结构文件； ----down(检索到365条，可用pdb数量：363条)
    顺便检索其他的数据库有无aptamer数据的重复、冗余； ----down（）
### 将aptamer的pdb结构数据保存为geometric数据格式
    转换中总计360条可转换数据，获得总共942个可用aptamer空间结构训练数据集 ----down
## 2.GNN模型建立
## 3.GNN模型训练
## 4.结构预测验证与评估
# 设计
## 先纯GNN
    ### **实现 GNN 模型对理论 Aptamer 的 PDB 结构预测**
---
你已经训练了一个 **GNN 模型**，它是基于 **实际 aptamer** 的 **PDB 原子空间结构（折叠后）** 进行训练的。现在，你有一个 **理论 aptamer（未折叠）**，希望用你的 **GNN 模型预测其折叠后的空间结构**，最终输出为 **PDB 格式**。

## **🔹 主要步骤**
1. **加载训练好的 GNN 模型**
2. **准备理论 aptamer 数据（初始 PDB）**
3. **将 PDB 数据转换为 GNN 输入格式**
4. **用 GNN 模型进行预测**
5. **将预测的结构数据转换回 PDB 格式**
6. **保存 PDB 文件**

---

## **📌 1. 加载训练好的 GNN 模型**
首先，我们需要加载训练完成的 `pth` 文件，并把模型设置为 `eval` 模式。
```python
import torch
from torch_geometric.data import Data

# 加载训练好的 GNN 模型
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = GNNModel(input_dim=3, hidden_dim=128, output_dim=3, dropout_rate=0.5)
model.load_state_dict(torch.load("best_model.pth", map_location=device))
model.to(device)
model.eval()
```
> ✅ **确保 `GNNModel` 的结构与你训练时完全一致，否则权重无法正确加载！**

---

## **📌 2. 解析理论 Aptamer 的 PDB 文件**
你的 GNN 模型是基于 **PDB 原子坐标数据** 进行训练的，因此，你需要把 **理论 aptamer 的 PDB 文件** 解析出来，并转换为 **GNN 适用的图结构**。

### **🔸 解析 PDB 获取原子坐标**
可以使用 `Biopython` 解析 PDB 文件：
```python
from Bio import PDB
import numpy as np

def parse_pdb(pdb_file):
    parser = PDB.PDBParser(QUIET=True)
    structure = parser.get_structure("aptamer", pdb_file)
    
    atom_coords = []
    for model in structure:
        for chain in model:
            for residue in chain:
                for atom in residue:
                    atom_coords.append(atom.coord)  # 获取原子坐标 (x, y, z)
                    
    return np.array(atom_coords)  # 返回 N×3 矩阵
```
> ✅ **`parse_pdb("theoretical.pdb")` 会返回一个 (N,3) 形状的 NumPy 数组，表示理论 Aptamer 的所有原子坐标。**

---

## **📌 3. 构造 GNN 输入数据**
GNN 需要 **图数据格式（Graph Data）**，你需要用 `torch_geometric.data.Data` 进行封装：
```python
import torch
from torch_geometric.data import Data

def construct_graph(atom_coords):
    """
    构造 GNN 输入数据，包含原子坐标 (x) 和边连接 (edge_index)
    """
    x = torch.tensor(atom_coords, dtype=torch.float)  # 节点特征 (N,3)

    # 这里可以用最近邻构造边
    from scipy.spatial import distance_matrix
    dist_mat = distance_matrix(atom_coords, atom_coords)
    
    threshold = 4.0  # 设置原子之间的最大边连接距离 (单位: Å)
    edge_index = torch.nonzero(torch.tensor(dist_mat) < threshold, as_tuple=False).T
    
    # 构造 PyG Data 对象
    graph_data = Data(x=x, edge_index=edge_index)

    return graph_data
```
> ✅ **这里 `threshold = 4.0` 代表原子之间的最大连接距离，具体数值可以根据训练时的标准进行调整。**

---

## **📌 4. 使用 GNN 预测折叠后的结构**
有了 `graph_data` 之后，就可以 **输入到 GNN 模型**，得到新的原子坐标：
```python
# 解析理论 Aptamer 的 PDB
atom_coords = parse_pdb("theoretical.pdb")

# 转换为 GNN 输入格式
graph_data = construct_graph(atom_coords).to(device)

# 进行预测
with torch.no_grad():
    predicted_coords = model(graph_data)  # 输出形状: (N,3)
predicted_coords = predicted_coords.cpu().numpy()  # 转换为 NumPy
```
> ✅ `predicted_coords` 是 **(N,3) 形状的数组**，表示 GNN 预测出的原子坐标（折叠后）。

---

## **📌 5. 保存为 PDB 格式**
最后，我们需要将预测得到的 **(N,3) 坐标** 转换回 **PDB 文件**，以便可视化分析：
```python
from Bio.PDB import PDBIO, Structure, Model, Chain, Residue, Atom

def save_pdb(predicted_coords, output_pdb):
    structure = Structure.Structure("predicted_aptamer")
    model = Model.Model(0)
    chain = Chain.Chain("A")
    
    for i, coord in enumerate(predicted_coords):
        residue = Residue.Residue((" ", i, " "), "UNK", i)
        atom = Atom.Atom("C", coord, 1.0, 1.0, " ", "C", i, "C")
        residue.add(atom)
        chain.add(residue)
    
    model.add(chain)
    structure.add(model)

    # 保存 PDB 文件
    io = PDBIO()
    io.set_structure(structure)
    io.save(output_pdb)

# 保存预测结构
save_pdb(predicted_coords, "predicted_aptamer.pdb")
```
> ✅ 这将创建一个新的 `predicted_aptamer.pdb`，其中的原子坐标是 GNN 预测的折叠结构。

---

## **📌 6. 进行可视化**
现在，你可以使用 **PyMOL** 或 **Chimera** 观察 `predicted_aptamer.pdb`：
```shell
pymol predicted_aptamer.pdb
```
或者用 `Chimera`：
```shell
chimera predicted_aptamer.pdb
```
> ✅ **对比理论结构 (`theoretical.pdb`) 和预测结构 (`predicted_aptamer.pdb`)，检查折叠后的空间构象变化。**

---

## **🔹 总结**
| **步骤** | **操作** |
|---------|---------|
| **1. 加载模型** | `torch.load("best_model.pth")` |
| **2. 解析 PDB 文件** | `parse_pdb("theoretical.pdb")` |
| **3. 构造 GNN 输入** | `construct_graph(atom_coords)` |
| **4. 预测折叠结构** | `predicted_coords = model(graph_data)` |
| **5. 保存为 PDB** | `save_pdb(predicted_coords, "predicted_aptamer.pdb")` |
| **6. 可视化** | `pymol predicted_aptamer.pdb` |

这样，你就可以使用 **GNN 模型** 预测 **理论 aptamer 的折叠结构** 并可视化分析！🚀
## 再添加物理约束（最坏情况可能要全部重新预处理和数据转换.....）
### 能量约束（势能动能）
### 作用力约束
### 最小作用量约束
### RSMD约束
### 经验公式约束

