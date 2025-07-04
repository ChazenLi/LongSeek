import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.data import Data, DataLoader
from torch_geometric.utils import to_dense_batch
from sklearn.model_selection import train_test_split
import glob
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

# 设备配置
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

#------------------------ 模型定义 ------------------------#
class SubstructureAttention(nn.Module):
    """子结构注意力计算模块"""
    def __init__(self, feat_dim):
        super().__init__()
        # 注意力网络：学习子结构重要性
        self.attn_net = nn.Sequential(
            nn.Linear(feat_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 1, bias=False)  # 无偏置以增强可解释性
        )
    
    def forward(self, x, subst_ids):
        """
        输入：
        - x: 原子特征矩阵 [num_atoms, feat_dim]
        - subst_ids: 子结构ID向量 [num_atoms]
        
        输出：
        - atom_attn: 原子级注意力权重 [num_atoms, 1]
        """
        unique_ids = torch.unique(subst_ids)
        subst_features = []
        
        # 步骤1：聚合子结构特征（平均池化）
        for sid in unique_ids:
            mask = (subst_ids == sid)
            subst_feat = x[mask].mean(dim=0)  # [feat_dim]
            subst_features.append(subst_feat)
        
        # 步骤2：计算子结构间注意力
        subst_matrix = torch.stack(subst_features)  # [num_substs, feat_dim]
        scores = self.attn_net(subst_matrix)       # [num_substs, 1]
        subst_attn = F.softmax(scores, dim=0)      # 归一化权重
        
        # 步骤3：将权重映射回原子级别
        atom_attn = torch.zeros_like(subst_ids, dtype=torch.float32)
        for i, sid in enumerate(unique_ids):
            atom_attn[subst_ids == sid] = subst_attn[i]
        
        return atom_attn.unsqueeze(1)  # [num_atoms, 1]

class EnhancedGNN(nn.Module):
    """增强的GNN模型（集成子结构注意力）"""
    def __init__(self, input_dim, hidden_dim=128, output_dim=1):
        super().__init__()
        
        # 子结构注意力模块
        self.subst_attn = SubstructureAttention(feat_dim=input_dim)
        
        # 图卷积层序列
        self.conv_layers = nn.ModuleList([
            GCNConv(input_dim + 1, hidden_dim),  # 输入特征 + 注意力权重
            GCNConv(hidden_dim, hidden_dim),
            GCNConv(hidden_dim, hidden_dim),
            GCNConv(hidden_dim, hidden_dim)
        ])
        
        # 标准化层
        self.bns = nn.ModuleList([
            nn.BatchNorm1d(hidden_dim) for _ in range(4)
        ])
        
        # 预测头
        self.pred_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.ELU(),
            nn.Dropout(0.4),
            nn.Linear(hidden_dim//2, output_dim)
        )

    def forward(self, data):
        # 解包数据
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        # 步骤1：计算子结构注意力
        subst_ids = data.x[:, 5].long()  # 假设子结构ID在第5列
        attn_weights = self.subst_attn(x, subst_ids)
        
        # 步骤2：特征拼接（原子特征 + 注意力权重）
        x = torch.cat([x, attn_weights], dim=1)
        
        # 步骤3：多层级图卷积
        for i, (conv, bn) in enumerate(zip(self.conv_layers, self.bns)):
            x = conv(x, edge_index)
            x = bn(x)
            x = F.elu(x) if i < 3 else F.relu(x)  # 最后一层使用ReLU
            x = F.dropout(x, p=0.3, training=self.training)
        
        # 步骤4：图级池化
        x = global_mean_pool(x, batch)
        
        # 步骤5：最终预测
        return self.pred_head(x)

#------------------------ 数据加载 ------------------------#
def load_dataset(pt_dir):
    """加载预处理后的图数据"""
    dataset = []
    for pt_file in glob.glob(os.path.join(pt_dir, "*.pt")):
        data_dict = torch.load(pt_file)
        # 转换为PyG Data对象（需添加真实标签）
        graph_data = Data(
            x=data_dict['x'],
            edge_index=data_dict['edge_index'],
            edge_attr=data_dict['edge_attr'],
            y=torch.tensor([0.0])  # 此处需替换为真实标签
        )
        dataset.append(graph_data)
    return dataset

#------------------------ 训练流水线 ------------------------#
def train_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0
    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()
        
        # 前向传播
        pred = model(data).squeeze()
        
        # 计算损失（假设回归任务）
        target = data.y.to(device)
        loss = F.mse_loss(pred, target)
        
        # 反向传播
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # 梯度裁剪
        optimizer.step()
        
        total_loss += loss.item()
    return total_loss / len(loader)

def validate(model, loader, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            pred = model(data).squeeze()
            loss = F.mse_loss(pred, data.y.to(device))
            total_loss += loss.item()
    return total_loss / len(loader)

def main():
    # 加载数据
    dataset = load_dataset("E:/APTAMER-GEN/gcnpt")
    train_data, test_data = train_test_split(dataset, test_size=0.2, random_state=42)
    
    # 创建数据加载器
    train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=64)
    
    # 初始化模型
    input_dim = dataset[0].x.shape[1]  # 自动获取特征维度
    model = EnhancedGNN(input_dim=input_dim).to(device)
    
    # 优化器配置
    optimizer = Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=8)
    
    # 训练循环
    best_loss = float('inf')
    patience = 15
    bad_epochs = 0
    
    for epoch in range(20):
        train_loss = train_epoch(model, train_loader, optimizer, device)
        val_loss = validate(model, test_loader, device)
        scheduler.step(val_loss)
        
        # 打印进度
        print(f"Epoch {epoch+1:03d} | "
              f"Train Loss: {train_loss:.4f} | "
              f"Val Loss: {val_loss:.4f} | "
              f"LR: {optimizer.param_groups[0]['lr']:.2e}")
        
        # 保存最佳模型
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), 'best_gcn_model.pt')
            bad_epochs = 0
        else:
            bad_epochs += 1
        
        # 早停机制
        if bad_epochs >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break

    # 训练20个epoch后停止
    print("Training completed after 20 epochs.")

if __name__ == "__main__":
    main()