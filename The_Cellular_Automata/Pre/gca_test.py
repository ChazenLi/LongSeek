# 导入必要的库
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time
import networkx as nx
from scipy.spatial import Delaunay
from torch.optim.lr_scheduler import ReduceLROnPlateau  # 添加学习率调度器

# 添加中文字体支持
import matplotlib
matplotlib.rcParams['font.family'] = ['SimHei', 'sans-serif']  # 设置中文字体，首选SimHei（黑体），后备sans-serif
matplotlib.rcParams['axes.unicode_minus'] = False  # 正确显示负号

# 设置随机种子以确保结果可重现
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)

# 检查是否有可用的GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

# 1. 定义要优化的函数（这里使用Rosenbrock函数作为示例）
def rosenbrock(x, y, a=1, b=100):
    """Rosenbrock函数，一个经典的非凸优化测试函数
    f(x,y) = (a-x)² + b(y-x²)²
    全局最小值在(a,a²)处
    """
    return (a - x)**2 + b * (y - x**2)**2   # 返回Rosenbrock函数值，这是一个常用的优化测试函数

# 2. 创建参数空间图结构
class ParameterGraph:
    def __init__(self, grid_size=32, param_range=(-2, 2), initial_active_points=5, focus_region=None):
        """初始化参数图
        
        参数:
            grid_size: 网格大小 (grid_size x grid_size)
            param_range: 参数范围 (min, max)
            initial_active_points: 初始活跃点数量
            focus_region: 聚焦区域 (x_min, x_max, y_min, y_max)，用于在特定区域增加采样密度
            
        """
        self.grid_size = grid_size           # 存储网格大小
        self.param_range = param_range       # 存储参数范围
        
        # 创建参数网格 - 每个网格点有4个通道：x坐标, y坐标, 函数值, 活跃状态
        self.grid = np.zeros((grid_size, grid_size, 4))  # 初始化网格数组
        
        # 初始化x和y坐标，如果有聚焦区域，则在该区域增加采样密度
        if focus_region is not None:
            x_min, x_max, y_min, y_max = focus_region
            # 在聚焦区域内使用更密集的采样
            focus_size = int(grid_size * 0.6)  # 60%的网格点用于聚焦区域
            regular_size = grid_size - focus_size
            
            # 创建两组坐标值
            x_focus = np.linspace(x_min, x_max, focus_size)
            y_focus = np.linspace(y_min, y_max, focus_size)
            x_regular = np.concatenate([
                np.linspace(param_range[0], x_min, regular_size//2),
                np.linspace(x_max, param_range[1], regular_size//2)
            ])
            y_regular = np.concatenate([
                np.linspace(param_range[0], y_min, regular_size//2),
                np.linspace(y_max, param_range[1], regular_size//2)
            ])
            
            # 合并并排序
            x_values = np.sort(np.concatenate([x_focus, x_regular]))
            y_values = np.sort(np.concatenate([y_focus, y_regular]))
        else:
            # 常规均匀采样
            x_values = np.linspace(param_range[0], param_range[1], grid_size)
            y_values = np.linspace(param_range[0], param_range[1], grid_size)
        
        # 填充网格
        for i, x in enumerate(x_values):     # 遍历所有x坐标
            for j, y in enumerate(y_values): # 遍历所有y坐标
                self.grid[i, j, 0] = x       # 设置x坐标
                self.grid[i, j, 1] = y       # 设置y坐标
                self.grid[i, j, 2] = rosenbrock(x, y)  # 计算并存储函数值
        
        # 创建图结构
        self.create_graph()
        
        # 随机初始化一些活跃点（初始搜索点）
        # 优化：增加在低函数值区域的初始点概率
        function_values = self.grid[:, :, 2].flatten()
        probabilities = 1.0 / (function_values + 1.0)  # 函数值越小，概率越大
        probabilities /= np.sum(probabilities)  # 归一化概率
        
        active_points = np.random.choice(
            grid_size*grid_size, 
            size=initial_active_points, 
            replace=False,
            p=probabilities  # 使用基于函数值的概率
        )
        
        for point in active_points:          # 遍历选中的点
            i, j = point // grid_size, point % grid_size  # 计算点的二维索引
            self.grid[i, j, 3] = 1.0         # 设置为活跃状态
        
        # 添加一些接近全局最小值的点
        self.add_points_near_minimum()
    
    def add_points_near_minimum(self, num_points=3, radius=0.3):
        """在全局最小值附近添加一些活跃点"""
        min_x, min_y = 1.0, 1.0  # Rosenbrock函数的全局最小值
        
        # 找到最接近全局最小值的网格点
        distances = np.sqrt(
            (self.grid[:, :, 0] - min_x)**2 + 
            (self.grid[:, :, 1] - min_y)**2
        )
        
        # 获取距离最小的几个点
        flat_indices = np.argsort(distances.flatten())[:num_points]
        for idx in flat_indices:
            i, j = idx // self.grid_size, idx % self.grid_size
            self.grid[i, j, 3] = 1.0  # 设置为活跃状态
    
    def create_graph(self):
        """创建图结构，使用Delaunay三角剖分连接相邻点"""
        self.graph = nx.Graph()  # 创建无向图
        
        # 添加节点
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                node_id = i * self.grid_size + j
                self.graph.add_node(node_id, pos=(i, j))
        
        # 使用Delaunay三角剖分创建边
        points = np.array([(i, j) for i in range(self.grid_size) for j in range(self.grid_size)])
        tri = Delaunay(points)
        
        # 从三角形中提取边
        edges = set()
        for simplex in tri.simplices:
            for i in range(3):
                for j in range(i+1, 3):
                    edge = (simplex[i], simplex[j])
                    edges.add(tuple(sorted(edge)))
        
        # 添加边到图
        for edge in edges:
            self.graph.add_edge(edge[0], edge[1])
        
        # 存储每个节点的邻居
        self.neighbors = {}
        for node in self.graph.nodes():
            self.neighbors[node] = list(self.graph.neighbors(node))
    
    def get_grid(self):
        """返回当前网格状态"""
        return self.grid.copy()              # 返回网格的副本，避免外部修改原始网格
    
    def update_grid(self, new_grid):
        """更新网格状态"""
        self.grid = new_grid.copy()          # 使用新网格的副本更新当前网格
    
    def get_node_features(self, node_id):
        """获取节点特征"""
        i, j = node_id // self.grid_size, node_id % self.grid_size
        return self.grid[i, j].copy()
    
    def get_neighbor_features(self, node_id):
        """获取节点邻居的特征"""
        neighbor_features = []
        for neighbor in self.neighbors.get(node_id, []):
            i, j = neighbor // self.grid_size, neighbor % self.grid_size
            neighbor_features.append(self.grid[i, j])
        return np.array(neighbor_features)

# 3. 定义图元胞自动机模型 (PyTorch版本) - 增强版
class GraphCA(nn.Module):
    def __init__(self, feature_dim=4, hidden_dim=128):  # 增加隐藏层维度
        super(GraphCA, self).__init__()     # 调用父类的初始化方法
        
        # 节点特征处理 - 增加层数和使用批归一化
        self.node_encoder = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),  # 添加批归一化
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),  # 添加批归一化
            nn.ReLU()
        )
        
        # 邻居聚合 - 增加层数和使用批归一化
        self.neighbor_encoder = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),  # 添加批归一化
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),  # 添加批归一化
            nn.ReLU()
        )
        
        # 更新网络 - 增加层数和使用Dropout
        self.update_network = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.BatchNorm1d(hidden_dim),  # 添加批归一化
            nn.ReLU(),
            nn.Dropout(0.1),  # 添加Dropout防止过拟合
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),  # 添加批归一化
            nn.ReLU(),
            nn.Dropout(0.1),  # 添加Dropout防止过拟合
            nn.Linear(hidden_dim, feature_dim),
            nn.Tanh()  # 限制输出范围在[-1, 1]
        )
        
        # 初始化为"无操作"行为 - 使用更好的初始化方法
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)  # 使用He初始化
                nn.init.zeros_(m.bias)
    
    def forward(self, node_features, neighbor_features):
        """前向传播
        
        参数:
            node_features: 节点特征 [batch_size, feature_dim]
            neighbor_features: 邻居特征 [batch_size, max_neighbors, feature_dim]
        
        返回:
            更新值 [batch_size, feature_dim]
        """
        # 编码节点特征
        node_encoding = self.node_encoder(node_features)  # [batch_size, hidden_dim]
        
        # 编码并聚合邻居特征
        batch_size, max_neighbors, feature_dim = neighbor_features.shape
        neighbor_flat = neighbor_features.reshape(-1, feature_dim)  # [batch_size*max_neighbors, feature_dim]
        neighbor_encoding = self.neighbor_encoder(neighbor_flat)  # [batch_size*max_neighbors, hidden_dim]
        neighbor_encoding = neighbor_encoding.reshape(batch_size, max_neighbors, -1)  # [batch_size, max_neighbors, hidden_dim]
        
        # 平均聚合邻居特征
        mask = (neighbor_features.sum(dim=-1) != 0).float().unsqueeze(-1)  # 创建掩码，排除填充的邻居
        neighbor_encoding = (neighbor_encoding * mask).sum(dim=1) / (mask.sum(dim=1) + 1e-8)  # [batch_size, hidden_dim]
        
        # 连接节点和邻居特征
        combined = torch.cat([node_encoding, neighbor_encoding], dim=1)  # [batch_size, hidden_dim*2]
        
        # 计算更新
        update = self.update_network(combined)  # [batch_size, feature_dim]
        
        return update

# 4. 实现随机更新机制 (PyTorch版本)
def stochastic_update(param_graph, model, update_rate=0.5, max_neighbors=10, temperature=1.0):
    """随机更新图中的节点
    
    参数:
        param_graph: 参数图对象
        model: 图CA模型
        update_rate: 每步更新的节点比例
        max_neighbors: 最大邻居数量
        temperature: 温度参数，控制随机性（越低越确定性）
    
    返回:
        更新后的网格
    """
    grid = param_graph.get_grid()  # 获取当前网格
    grid_size = grid.shape[0]      # 获取网格大小
    
    # 基于函数值计算选择概率
    function_values = grid[:, :, 2].flatten()
    selection_probs = np.exp(-function_values / temperature)  # 使用Boltzmann分布
    selection_probs = selection_probs / np.sum(selection_probs)  # 归一化
    
    # 随机选择要更新的节点，偏向于低函数值区域
    all_nodes = grid_size * grid_size
    update_count = int(all_nodes * update_rate)
    update_nodes = np.random.choice(
        all_nodes, 
        size=update_count, 
        replace=False,
        p=selection_probs  # 使用基于函数值的概率
    )
    
    # 准备批处理数据
    node_features_list = []
    neighbor_features_list = []
    
    for node_id in update_nodes:
        # 获取节点特征
        node_features = param_graph.get_node_features(node_id)
        node_features_list.append(node_features)
        
        # 获取邻居特征
        neighbor_features = param_graph.get_neighbor_features(node_id)
        
        # 填充或截断邻居特征到固定大小
        padded_neighbors = np.zeros((max_neighbors, 4))
        if len(neighbor_features) > 0:
            padded_neighbors[:min(len(neighbor_features), max_neighbors)] = neighbor_features[:max_neighbors]
        
        neighbor_features_list.append(padded_neighbors)
    
    # 转换为PyTorch张量
    if node_features_list:  # 确保列表不为空
        node_features_tensor = torch.tensor(np.stack(node_features_list), dtype=torch.float32, device=device)
        neighbor_features_tensor = torch.tensor(np.stack(neighbor_features_list), dtype=torch.float32, device=device)
        
        # 计算更新
        with torch.no_grad():
            updates = model(node_features_tensor, neighbor_features_tensor)
        
        # 应用更新
        updates_np = updates.cpu().numpy()
        new_grid = grid.copy()
        
        for idx, node_id in enumerate(update_nodes):
            i, j = node_id // grid_size, node_id % grid_size
            
            # 应用更新，使用自适应步长
            step_size = 0.1 * (1.0 / (1.0 + new_grid[i, j, 2]))  # 函数值越小，步长越小
            new_grid[i, j] += updates_np[idx] * step_size
            
            # 确保x和y坐标保持不变
            new_grid[i, j, 0] = grid[i, j, 0]
            new_grid[i, j, 1] = grid[i, j, 1]
            
            # 重新计算函数值
            x, y = new_grid[i, j, 0], new_grid[i, j, 1]
            new_grid[i, j, 2] = rosenbrock(x, y)
        
        # 限制活跃状态在[0, 1]范围内
        new_grid[:, :, 3] = np.clip(new_grid[:, :, 3], 0, 1)
        
        return new_grid
    else:
        return grid.copy()

# 5. 训练图CA模型 (PyTorch版本) - 优化版
def train_graph_ca(param_graph, model, optimizer, scheduler=None, epochs=200, steps_per_epoch=20, max_neighbors=10, 
                  initial_temp=1.0, final_temp=0.01):
    """训练图CA模型以找到函数的最小值 - 增强版
    
    参数:
        param_graph: 参数图对象
        model: 图CA模型
        optimizer: 优化器
        scheduler: 学习率调度器
        epochs: 训练轮数 (增加)
        steps_per_epoch: 每轮的步数 (增加)
        max_neighbors: 最大邻居数量
        initial_temp: 初始温度 (用于模拟退火)
        final_temp: 最终温度 (用于模拟退火)
    
    返回:
        训练历史记录
    """
    history = {                              # 初始化历史记录字典
        'loss': [],                          # 存储损失值
        'min_value': [],                     # 存储最小函数值
        'min_position': [],                  # 存储最小值位置
        'learning_rates': []                 # 存储学习率变化
    }
    
    # 添加早停机制
    best_min_value = float('inf')
    patience = 20
    patience_counter = 0
    
    # 多起点搜索
    restart_count = 0
    max_restarts = 3
    
    # 记录全局最佳结果
    global_best_value = float('inf')
    global_best_position = None
    global_best_grid = None
    
    for epoch in range(epochs):              # 遍历所有训练轮次
        epoch_loss = 0                       # 初始化当前轮次的累计损失
        min_value = float('inf')             # 修复：每轮初始化min_value
        
        # 计算当前温度 (用于模拟退火)
        temp = initial_temp * (final_temp / initial_temp) ** (epoch / epochs)
        
        for step in range(steps_per_epoch):  # 遍历当前轮次的所有步骤
            # 获取当前网格
            grid = param_graph.get_grid()     # 获取当前网格状态
            grid_size = grid.shape[0]
            
            # 准备批处理数据 - 选择活跃节点和一些随机节点
            active_mask = grid[:, :, 3] > 0.5
            active_indices = np.where(active_mask)
            active_nodes = [i * grid_size + j for i, j in zip(active_indices[0], active_indices[1])]
            
            # 如果活跃节点太少，添加一些随机节点
            if len(active_nodes) < 10:
                all_nodes = set(range(grid_size * grid_size))
                active_set = set(active_nodes)
                remaining = list(all_nodes - active_set)
                random_nodes = np.random.choice(remaining, size=min(10-len(active_nodes), len(remaining)), replace=False)
                batch_nodes = active_nodes + list(random_nodes)
            else:
                # 优先选择函数值较低的点 (增强局部搜索)
                if np.random.random() < 0.7 and len(active_nodes) > 0:
                    func_values = np.array([grid[node_id // grid_size, node_id % grid_size, 2] for node_id in active_nodes])
                    probs = np.exp(-func_values / (temp + 1e-8))
                    probs = probs / np.sum(probs)
                    sample_size = min(10, np.count_nonzero(probs))
                    if sample_size > 0:
                        batch_nodes = np.random.choice(active_nodes, size=sample_size, replace=False, p=probs)
                        batch_nodes = list(batch_nodes)
                    else:
                        batch_nodes = active_nodes
            
            # 获取节点和邻居特征
            node_features_list = []
            neighbor_features_list = []
            
            for node_id in batch_nodes:
                # 获取节点特征
                i, j = node_id // grid_size, node_id % grid_size
                node_features = grid[i, j].copy()
                node_features_list.append(node_features)
                
                # 获取邻居特征
                neighbor_features = param_graph.get_neighbor_features(node_id)
                
                # 填充或截断邻居特征到固定大小
                padded_neighbors = np.zeros((max_neighbors, 4))
                if len(neighbor_features) > 0:
                    padded_neighbors[:min(len(neighbor_features), max_neighbors)] = neighbor_features[:max_neighbors]
                
                neighbor_features_list.append(padded_neighbors)
            
            # 转换为PyTorch张量
            if node_features_list:  # 确保列表不为空
                node_features_tensor = torch.tensor(np.stack(node_features_list), dtype=torch.float32, device=device)
                neighbor_features_tensor = torch.tensor(np.stack(neighbor_features_list), dtype=torch.float32, device=device)
                
                # 前向传播
                optimizer.zero_grad()
                updates = model(node_features_tensor, neighbor_features_tensor)
                
                # 计算损失 - 直接使用模型输出
                # 1. 函数值损失 - 使用模型输出的更新来估计函数值改进
                value_loss = torch.mean(updates[:, 2])  # 假设第3通道是函数值
                
                # 2. 活跃状态损失 - 鼓励合适数量的活跃点
                activity_updates = updates[:, 3]  # 假设第4通道是活跃状态
                count_loss = torch.abs(torch.sum(activity_updates > 0) - 15.0)  # 增加目标活跃点数
                
                # 3. 探索损失 - 使用L2正则化鼓励更新的多样性
                exploration_loss = -0.01 * torch.norm(updates, p=2)
                
                # 4. 添加接近全局最小值的损失
                # 计算当前点到全局最小值(1,1)的距离
                x_coords = node_features_tensor[:, 0]
                y_coords = node_features_tensor[:, 1]
                dist_to_global = torch.sqrt((x_coords - 1.0)**2 + (y_coords - 1.0)**2)
                global_min_loss = torch.mean(dist_to_global) * 0.1  # 添加权重
                
                # 总损失 - 动态调整权重
                # 随着训练进行，增加函数值损失的权重，减少探索损失的权重
                value_weight = 2.0 + 3.0 * (epoch / epochs)  # 从2.0增加到5.0
                exploration_weight = 0.01 * (1.0 - 0.9 * (epoch / epochs))  # 从0.01减少到0.001
                
                loss = value_loss * value_weight + 0.05 * count_loss + exploration_weight * exploration_loss + global_min_loss
                
                # 反向传播
                loss.backward()
                optimizer.step()
                
                # 应用更新并获取新网格
                updates_np = updates.detach().cpu().numpy()
                new_grid = grid.copy()
                
                for idx, node_id in enumerate(batch_nodes):
                    i, j = node_id // grid_size, node_id % grid_size
                    
                    # 应用更新，使用自适应步长
                    step_size = 0.1 * (1.0 - 0.5 * (epoch / epochs))  # 随着训练进行减小步长
                    new_grid[i, j] += updates_np[idx] * step_size
                    
                    # 确保x和y坐标保持不变
                    new_grid[i, j, 0] = grid[i, j, 0]
                    new_grid[i, j, 1] = grid[i, j, 1]
                    
                    # 重新计算函数值
                    x, y = new_grid[i, j, 0], new_grid[i, j, 1]
                    new_grid[i, j, 2] = rosenbrock(x, y)
                
                # 限制活跃状态
                new_grid[:, :, 3] = np.clip(new_grid[:, :, 3], 0, 1)
                
                # 更新网格
                param_graph.update_grid(new_grid)
                
                epoch_loss += loss.item()
        
        # 更新学习率
        if scheduler is not None:
            scheduler.step(min_value)  # 传入监控指标
            history['learning_rates'].append(optimizer.param_groups[0]['lr'])
        
        # 计算当前最小值
        grid = param_graph.get_grid()
        min_idx = np.argmin(grid[:, :, 2])
        min_i, min_j = min_idx // grid.shape[1], min_idx % grid.shape[1]
        min_value = grid[min_i, min_j, 2]
        min_position = (grid[min_i, min_j, 0], grid[min_i, min_j, 1])
        
        # 更新全局最佳结果
        if min_value < global_best_value:
            global_best_value = min_value
            global_best_position = min_position
            global_best_grid = grid.copy()
        
        # 记录历史
        history['loss'].append(epoch_loss / steps_per_epoch)
        history['min_value'].append(min_value)
        history['min_position'].append(min_position)
        
        # 打印进度
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss/steps_per_epoch:.4f}, "
                  f"Min Value: {min_value:.4f} at {min_position}, "
                  f"Temperature: {temp:.6f}")
        
        # 早停检查
        if min_value < best_min_value:
            best_min_value = min_value
            patience_counter = 0
        else:
            patience_counter += 1
        
        # 如果连续patience轮没有改进，考虑重启或早停
        if patience_counter >= patience:
            if restart_count < max_restarts:
                print(f"\n重启优化过程 ({restart_count+1}/{max_restarts})...")
                # 重新初始化一些活跃点，但保留最佳点
                grid = param_graph.get_grid()
                grid[:, :, 3] = 0.0  # 清除所有活跃点
                
                # 保留最佳点为活跃点
                best_i, best_j = min_idx // grid.shape[1], min_idx % grid.shape[1]
                grid[best_i, best_j, 3] = 1.0
                
                # 在最佳点周围添加一些活跃点
                for _ in range(14):  # 总共15个活跃点
                    i = np.random.randint(0, grid_size)
                    j = np.random.randint(0, grid_size)
                    grid[i, j, 3] = 1.0
                
                param_graph.update_grid(grid)
                
                # 重置早停计数器
                patience_counter = 0
                restart_count += 1
            else:
                print("\n达到最大重启次数，提前停止训练")
                break
    
    # 恢复全局最佳结果
    if global_best_grid is not None:
        param_graph.update_grid(global_best_grid)
    
    return history

# 6. 可视化函数
def visualize_optimization(param_graph, history, save_animation=False):
    """可视化优化过程
    
    参数:
        param_graph: 参数图对象
        history: 训练历史记录
        save_animation: 是否保存动画
    """
    grid_size = param_graph.grid_size
    param_range = param_graph.param_range
    
    # 创建网格用于等高线图
    x = np.linspace(param_range[0], param_range[1], 100)
    y = np.linspace(param_range[0], param_range[1], 100)
    X, Y = np.meshgrid(x, y)
    Z = rosenbrock(X, Y)
    
    # 创建图形
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # 绘制等高线图
    contour = ax1.contourf(X, Y, Z, levels=50, cmap='viridis')
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_title('Rosenbrock函数优化过程 (GCA)')
    fig.colorbar(contour, ax=ax1)
    
    # 绘制图结构
    pos = {node: (j, i) for node, (i, j) in nx.get_node_attributes(param_graph.graph, 'pos').items()}
    
    # 初始化散点图 - 显示活跃点
    grid = param_graph.get_grid()
    active_mask = grid[:, :, 3] > 0.5
    active_x = grid[:, :, 0][active_mask]
    active_y = grid[:, :, 1][active_mask]
    scatter = ax1.scatter(active_x, active_y, c='red', s=50, label='活跃点')
    
    # 绘制最优点轨迹
    min_positions = np.array(history['min_position'])
    line, = ax1.plot(min_positions[:1, 0], min_positions[:1, 1], 'r-', linewidth=2, label='最优点轨迹')
    min_point = ax1.scatter(min_positions[0, 0], min_positions[0, 1], c='white', s=100, edgecolors='black', label='当前最优点')
    
    # 绘制损失曲线
    ax2.plot(history['loss'], label='总损失')
    ax2.plot(history['min_value'], label='最小函数值')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Value')
    ax2.set_title('训练损失和最小函数值 (GCA)')
    ax2.legend()
    ax2.grid(True)
    
    # 添加图例
    ax1.legend()
    
    # 标记全局最小值
    ax1.scatter(1, 1, c='yellow', s=200, alpha=0.5, edgecolors='black', label='全局最小值')
    
    # 更新函数
    def update(frame):
        # 更新最优点轨迹
        line.set_data(min_positions[:frame+1, 0], min_positions[:frame+1, 1])
        
        # 更新当前最优点
        min_point.set_offsets([min_positions[frame, 0], min_positions[frame, 1]])
        
        return scatter, line, min_point
    
    # 创建动画
    ani = FuncAnimation(fig, update, frames=len(history['min_value']), 
                        interval=100, blit=True)
    
    # 保存动画
    if save_animation:
        ani.save('gca_optimization_animation.gif', writer='pillow', fps=10)
    
    plt.tight_layout()
    plt.show()

# 7. 主函数
def main():
    # 创建参数图 - 使用更大的网格和更多的初始活跃点
    param_graph = ParameterGraph(grid_size=64, param_range=(-2, 2), initial_active_points=15)
    
    # 创建图CA模型 - 使用增强版模型
    model = GraphCA().to(device)
    
    # 创建优化器 - 使用更低的学习率和权重衰减
    optimizer = optim.Adam(model.parameters(), lr=0.0005, weight_decay=1e-5)
    
    # 添加学习率调度器
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)  # 不要加 verbose 参数
    
    # 训练模型 - 使用增强版训练函数
    print("开始训练图元胞自动机(GCA)...")
    start_time = time.time()
    history = train_graph_ca(param_graph, model, optimizer, scheduler, 
                           epochs=200, steps_per_epoch=20,
                           initial_temp=1.0, final_temp=0.01)
    end_time = time.time()
    print(f"训练完成，耗时: {end_time - start_time:.2f}秒")
    
    # 打印最终结果
    final_min_value = history['min_value'][-1]
    final_min_position = history['min_position'][-1]
    print(f"\n最终结果:")
    print(f"最小函数值: {final_min_value:.6f}")
    print(f"最优参数: x={final_min_position[0]:.6f}, y={final_min_position[1]:.6f}")
    print(f"全局最小值: 0.0 at (1.0, 1.0)")
    print(f"与全局最小值的距离: {((final_min_position[0]-1.0)**2 + (final_min_position[1]-1.0)**2)**0.5:.6f}")
    
    # 可视化优化过程
    visualize_optimization(param_graph, history, save_animation=True)

# 程序入口点
if __name__ == "__main__":
    main()