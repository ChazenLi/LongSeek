import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_add_pool
from torch_geometric.data import Data, DataLoader
from torch_geometric.utils import add_self_loops
import glob
from sklearn.model_selection import train_test_split

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class GATSubstAttention(nn.Module):
    """集成子结构注意力的GAT模型"""
    def __init__(self, feat_dim, hidden_dim=64, heads=4, dropout=0.2):
        super().__init__()
        # 子结构注意力层
        self.subst_attn = nn.Sequential(
            nn.Linear(feat_dim, 32),
            nn.LeakyReLU(0.2),
            nn.Linear(32, 1)
        )
        
        # GAT层序列
        self.gat1 = GATConv(
            in_channels=feat_dim + 1,  # 原子特征 + 注意力权重
            out_channels=hidden_dim,
            heads=heads,
            dropout=dropout,
            add_self_loops=False  # 避免与显式边冲突
        )
        self.gat2 = GATConv(
            hidden_dim * heads,
            hidden_dim,
            heads=1,
            dropout=dropout
        )
        
        # 预测层
        self.pred = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim//2, 1)
        )

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        # 步骤1：计算子结构注意力
        subst_ids = x[:, 5].long()
        unique_ids = torch.unique(subst_ids)
        
        # 聚合子结构特征
        subst_feats = []
        for sid in unique_ids:
            mask = (subst_ids == sid)
            subst_feat = x[mask].mean(dim=0)
            subst_feats.append(subst_feat)
        
        # 计算子结构注意力
        subst_matrix = torch.stack(subst_feats)
        subst_weights = F.softmax(self.subst_attn(subst_matrix), dim=0)
        
        # 映射回原子级
        atom_weights = torch.zeros(x.size(0), 1, device=x.device)
        for i, sid in enumerate(unique_ids):
            atom_weights[subst_ids == sid] = subst_weights[i]
        
        # 特征拼接
        x = torch.cat([x, atom_weights], dim=1)
        
        # 步骤2：GAT处理
        edge_index, _ = add_self_loops(edge_index)  # 添加自环
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.gat1(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.gat2(x, edge_index)
        
        # 图级池化
        x = global_add_pool(x, batch)
        return self.pred(x)

def collate_fn(batch):
    """自定义批处理应对变长图"""
    return batch.from_data_list(batch)

def train():
    # 数据加载
    pt_dir = "E:/APTAMER-GEN/gat_pt"
    dataset = []
    for pt_file in glob.glob(os.path.join(pt_dir, "*.pt")):
        data = torch.load(pt_file)
        dataset.append(Data(
            x=data['x'],
            edge_index=data['edge_index'],
            y=torch.randn(1)  # 替换为真实标签
        ))
    
    # 数据拆分
    train_data, test_data = train_test_split(dataset, test_size=0.2, random_state=42)
    
    # 数据加载器（使用自定义collate）
    train_loader = DataLoader(train_data, batch_size=32, shuffle=True, collate_fn=collate_fn)
    test_loader = DataLoader(test_data, batch_size=64, collate_fn=collate_fn)
    
    # 模型初始化
    feat_dim = dataset[0].x.size(1)
    model = GATSubstAttention(feat_dim=feat_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    
    # 训练循环
    for epoch in range(200):
        model.train()
        total_loss = 0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            pred = model(batch).squeeze()
            loss = F.mse_loss(pred, batch.y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
        
        # 验证
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in test_loader:
                batch = batch.to(device)
                pred = model(batch).squeeze()
                val_loss += F.mse_loss(pred, batch.y).item()
        
        print(f"Epoch {epoch+1} | Train Loss: {total_loss/len(train_loader):.4f} | "
              f"Val Loss: {val_loss/len(test_loader):.4f}")

if __name__ == "__main__":
    train()