import torch
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv
import torch.nn.functional as F
import os

# 定义GNN模型
class GNNModel(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout_rate=0.5):
        super(GNNModel, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.bn1 = torch.nn.BatchNorm1d(hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.bn2 = torch.nn.BatchNorm1d(hidden_dim)
        self.conv3 = GCNConv(hidden_dim, hidden_dim)
        self.bn3 = torch.nn.BatchNorm1d(hidden_dim)
        self.conv4 = GCNConv(hidden_dim, output_dim)
        self.dropout = torch.nn.Dropout(p=dropout_rate)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.conv2(x, edge_index)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.conv3(x, edge_index)
        x = self.bn3(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.conv4(x, edge_index)
        return x

# 加载数据
def load_data(file_path):
    data = torch.load(file_path, weights_only=False)
    return data

# 预测函数
def predict(model, data, device):
    model.eval()
    data = data.to(device)
    with torch.no_grad():
        output = model(data)
    return output

def main():
    data_file = "e:/APTAMER-GEN/pt/GACTCCAGTCGACTGCGGGGCAAAA-1.pt"
    model_path = "E:/Python/DL/best_model.pth"
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = GNNModel(input_dim=3, hidden_dim=128, output_dim=3, dropout_rate=0.5).to(device)
    model.load_state_dict(torch.load(model_path))
    
    data = load_data(data_file)
    predicted_positions = predict(model, data, device)
    
    print("Predicted atomic positions:")
    print(predicted_positions)
    
    # 保存预测结果
    data.pos = predicted_positions  # 更新Data对象中的pos属性
    output_file = "e:/APTAMER-GEN/pt/GACTCCAGTCGACTGCGGGGCAAAA-1_predicted.pt"
    torch.save(data, output_file)
    print(f"Predicted positions saved to {output_file}")

if __name__ == "__main__":
    main()