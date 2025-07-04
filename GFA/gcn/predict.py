import os
import torch
import glob
from torch_geometric.data import Data
import shutil
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

#------------------------ 模型定义（必须与训练时完全一致）------------------------#
class SubstructureAttention(nn.Module):
    """子结构注意力模块（保持与训练代码一致）"""
    def __init__(self, feat_dim):
        super().__init__()
        # 注意：层名称必须与训练代码完全一致！
        self.attn_net = nn.Sequential(
            nn.Linear(feat_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 1, bias=False)
        )
    
    def forward(self, x, subst_ids):
        unique_ids = torch.unique(subst_ids)
        subst_features = []
        for sid in unique_ids:
            mask = (subst_ids == sid)
            subst_feat = x[mask].mean(dim=0)
            subst_features.append(subst_feat)
        
        subst_matrix = torch.stack(subst_features)
        scores = self.attn_net(subst_matrix)
        subst_attn = F.softmax(scores, dim=0)
        
        atom_attn = torch.zeros_like(subst_ids, dtype=torch.float32)
        for i, sid in enumerate(unique_ids):
            atom_attn[subst_ids == sid] = subst_attn[i]
        
        return atom_attn.unsqueeze(1)

class TrainedGCNModel(nn.Module):
    """与训练完全一致的模型结构"""
    def __init__(self, input_dim):
        super().__init__()
        self.subst_attn = SubstructureAttention(input_dim)
        
        # 使用ModuleList并命名为conv_layers（与训练代码匹配）
        self.conv_layers = nn.ModuleList([
            GCNConv(input_dim + 1, 128),
            GCNConv(128, 128),
            GCNConv(128, 128),
            GCNConv(128, 128)
        ])
        
        self.bns = nn.ModuleList([nn.BatchNorm1d(128) for _ in range(4)])
        self.pred_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.ELU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1))
    
    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        subst_ids = x[:, 5].long()
        attn = self.subst_attn(x, subst_ids)
        x = torch.cat([x, attn], dim=1)
        
        # 使用conv_layers索引（与训练代码一致）
        for i, conv in enumerate(self.conv_layers):
            x = conv(x, edge_index)
            x = self.bns[i](x)
            x = F.elu(x) if i < 3 else F.relu(x)
            x = F.dropout(x, p=0.3, training=self.training)
        
        x = global_mean_pool(x, batch)
        return self.pred_head(x)

#------------------------ 预测流程 ------------------------#
def load_model(model_path, input_dim):
    """加载训练好的模型"""
    model = TrainedGCNModel(input_dim=input_dim)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()  # 切换为评估模式
    return model

def predict_single_molecule(model, pt_path):
    """预测单个分子"""
    try:
        # 加载数据
        data_dict = torch.load(pt_path, map_location=device)
        data = Data(
            x=data_dict['x'],
            edge_index=data_dict['edge_index'],
            edge_attr=data_dict.get('edge_attr', None),
            num_nodes=data_dict.get('num_nodes', None)
        )
        
        # 添加虚拟batch维度
        data.batch = torch.zeros(data.num_nodes, dtype=torch.long, device=device)
        
        # 执行预测
        with torch.no_grad():
            prediction = model(data)
        
        return prediction.cpu().numpy().item()
    
    except Exception as e:
        print(f"预测 {os.path.basename(pt_path)} 失败: {str(e)}")
        return None

def save_prediction_result(src_pt_path, pred_value, output_dir):
    """保存预测结果和原始数据"""
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存预测值
    base_name = os.path.basename(src_pt_path)
    pred_txt = os.path.join(output_dir, f"{os.path.splitext(base_name)[0]}_pred.txt")
    with open(pred_txt, 'w') as f:
        f.write(f"{pred_value:.6f}")
    
    # 复制原始pt文件到输出目录（可选）
    dest_pt = os.path.join(output_dir, base_name)
    shutil.copyfile(src_pt_path, dest_pt)
    
    return dest_pt, pred_txt

def batch_predict(model, input_dir, output_dir):
    """批量预测整个目录"""
    pt_files = glob.glob(os.path.join(input_dir, "*.pt"))
    if not pt_files:
        raise FileNotFoundError(f"在 {input_dir} 中未找到任何pt文件")
    
    print(f"开始预测，共发现 {len(pt_files)} 个分子...")
    
    for pt_path in pt_files:
        pred_value = predict_single_molecule(model, pt_path)
        if pred_value is not None:
            _, pred_file = save_prediction_result(
                pt_path, pred_value, output_dir)
            print(f"成功预测 {os.path.basename(pt_path)} -> 结果保存至 {pred_file}")

#------------------------ 主程序 ------------------------#
if __name__ == "__main__":
    # 配置路径
    MODEL_PATH = r"E:/Python/DL/best_gcn_model.pt"
    INPUT_PT = r"E:/APTAMER-GEN/gcnvalid/pt/1.pt"  # 可以改为目录路径
    OUTPUT_DIR = r"E:/APTAMER-GEN/gcnvalid/pre"
    
    # 获取输入维度（通过示例文件）
    sample_data = torch.load(INPUT_PT, map_location='cpu')
    INPUT_DIM = sample_data['x'].shape[1]
    
    # 初始化模型
    try:
        model = load_model(MODEL_PATH, input_dim=INPUT_DIM)
        print("模型加载成功！")
    except Exception as e:
        raise RuntimeError(f"模型加载失败: {str(e)}")
    
    # 执行预测
    if os.path.isdir(INPUT_PT):
        # 输入为目录时批量处理
        batch_predict(model, INPUT_PT, OUTPUT_DIR)
    else:
        # 单文件预测
        pred_value = predict_single_molecule(model, INPUT_PT)
        if pred_value is not None:
            _, pred_file = save_prediction_result(INPUT_PT, pred_value, OUTPUT_DIR)
            print(f"预测结果: {pred_value:.4f}")
            print(f"结果已保存至: {pred_file}")