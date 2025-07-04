import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data

class GNNModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, output_dim=1):
        super(GNNModel, self).__init__()
        
        # Graph Convolution Layers
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.conv3 = GCNConv(hidden_dim, hidden_dim)
        self.conv4 = GCNConv(hidden_dim, hidden_dim)
        
        # Batch Normalization layers
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.bn3 = nn.BatchNorm1d(hidden_dim)
        self.bn4 = nn.BatchNorm1d(hidden_dim)
        
        # Final prediction layers
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        
        # Dropout layer
        self.dropout = nn.Dropout(0.3)

    def forward(self, x, edge_index, edge_attr, batch=None):
        # First GCN layer
        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.dropout(x)
        
        # Second GCN layer
        x = self.conv2(x, edge_index)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.dropout(x)
        
        # Third GCN layer
        x = self.conv3(x, edge_index)
        x = self.bn3(x)
        x = F.relu(x)
        x = self.dropout(x)
        
        # Fourth GCN layer
        x = self.conv4(x, edge_index)
        x = self.bn4(x)
        x = F.relu(x)
        
        # Global mean pooling
        if batch is not None:
            x = torch.mean(x, dim=0)
        
        # Final prediction
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x

def predict_structure(model_path, data_path):
    """
    使用训练好的模型预测新的分子结构
    
    Args:
        model_path (str): 训练好的模型路径
        data_path (str): 需要预测的分子数据路径
    """
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    try:
        # 加载分子数据
        print("Loading molecular data...")
        data = torch.load(data_path)
        graph_data = Data(
            x=data['x'],
            edge_index=data['edge_index'],
            edge_attr=data['edge_attr']
        ).to(device)
        
        # 初始化模型
        print("Initializing model...")
        input_dim = data['x'].size(1)
        model = GNNModel(input_dim=input_dim).to(device)
        
        # 加载训练好的模型权重
        print("Loading model weights...")
        model.load_state_dict(torch.load(model_path, map_location=device))
        
        # 设置为评估模式
        model.eval()
        
        # 进行预测
        print("Making prediction...")
        with torch.no_grad():
            prediction = model(graph_data.x, graph_data.edge_index, graph_data.edge_attr)
        
        # 处理预测结果
        prediction = prediction.cpu().numpy()
        
        print("\nPrediction Results:")
        print("-" * 50)
        print(f"Raw prediction values: {prediction.squeeze()}")
        print(f"Mean prediction value: {prediction.mean():.4f}")
        
        # 保存预测结果
        results = {
            'raw_predictions': prediction,
            'mean_prediction': prediction.mean(),
        }
        
        output_path = data_path.replace('.pt', '_pre.pt')
        torch.save(results, output_path)
        print(f"\nResults saved to: {output_path}")
        
        return results
        
    except Exception as e:
        print(f"Error during prediction: {str(e)}")
        return None

if __name__ == "__main__":
    # 设置路径
    MODEL_PATH = r"E:\Python\DL\best_model.pt"
    DATA_PATH = r"E:\APTAMER-GEN\valid\pt\1.pt"
    
    # 运行预测
    results = predict_structure(MODEL_PATH, DATA_PATH)