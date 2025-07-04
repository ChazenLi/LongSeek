# 导入必要的库
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time

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

# 2. 创建参数空间网格
class ParameterGrid:
    def __init__(self, grid_size=32, param_range=(-2, 2)):
        """初始化参数网格
        
        参数:
            grid_size: 网格大小 (grid_size x grid_size)
            param_range: 参数范围 (min, max)
        """
        self.grid_size = grid_size           # 存储网格大小
        self.param_range = param_range       # 存储参数范围
        
        # 创建参数网格 - 每个网格点有4个通道：x坐标, y坐标, 函数值, 活跃状态
        self.grid = np.zeros((grid_size, grid_size, 4))  # 初始化网格数组
        
        # 初始化x和y坐标
        x_values = np.linspace(param_range[0], param_range[1], grid_size)  # 在参数范围内均匀生成x坐标值
        y_values = np.linspace(param_range[0], param_range[1], grid_size)  # 在参数范围内均匀生成y坐标值
        
        # 填充网格
        for i, x in enumerate(x_values):     # 遍历所有x坐标
            for j, y in enumerate(y_values): # 遍历所有y坐标
                self.grid[i, j, 0] = x       # 设置x坐标
                self.grid[i, j, 1] = y       # 设置y坐标
                self.grid[i, j, 2] = rosenbrock(x, y)  # 计算并存储函数值
                
        # 随机初始化一些活跃点（初始搜索点）
        active_points = np.random.choice(grid_size*grid_size, size=5, replace=False)  # 随机选择5个不重复的点
        for point in active_points:          # 遍历选中的点
            i, j = point // grid_size, point % grid_size  # 计算点的二维索引
            self.grid[i, j, 3] = 1.0         # 设置为活跃状态
    
    def get_grid(self):
        """返回当前网格状态"""
        return self.grid.copy()              # 返回网格的副本，避免外部修改原始网格
    
    def update_grid(self, new_grid):
        """更新网格状态"""
        self.grid = new_grid.copy()          # 使用新网格的副本更新当前网格

# 3. 定义神经元胞自动机模型 (PyTorch版本)
class NeuralCA(nn.Module):
    def __init__(self):
        super(NeuralCA, self).__init__()     # 调用父类的初始化方法
        
        # 卷积层用于感知邻居状态
        self.perception = nn.Conv2d(
            in_channels=4,                    # 输入通道数为4（x坐标、y坐标、函数值、活跃状态）
            out_channels=128,                 # 输出128个特征图
            kernel_size=3,                   # 使用3x3的卷积核
            padding=1                        # 使用padding=1保持空间维度不变
        )
        
        # 处理感知信息的隐藏层
        self.hidden1 = nn.Linear(128, 64)    # 第一个全连接层，将128维特征降至64维
        self.hidden2 = nn.Linear(64, 32)     # 第二个全连接层，将64维特征降至32维
        
        # 输出层 - 预测网格点的状态更新
        # 使用tanh激活函数，输出范围为[-1, 1]
        self.output_layer = nn.Linear(32, 4)  # 输出层，产生4个通道的更新值
        
        # 初始化为"无操作"行为 - 最后一层权重初始化为接近零
        nn.init.normal_(self.output_layer.weight, std=0.01)  # 使用小标准差的正态分布初始化权重
        nn.init.zeros_(self.output_layer.bias)  # 将偏置初始化为零
    
    def forward(self, x):
        """前向传播"""
        # 卷积层处理
        x = F.relu(self.perception(x))       # 应用卷积层并使用ReLU激活函数
        
        # 重塑张量以便于全连接层处理
        batch_size, channels, height, width = x.shape  # 获取张量的形状
        x = x.permute(0, 2, 3, 1)            # [B, C, H, W] -> [B, H, W, C]，调整维度顺序
        
        # 应用全连接层
        x = F.relu(self.hidden1(x))          # 应用第一个全连接层并使用ReLU激活
        x = F.relu(self.hidden2(x))          # 应用第二个全连接层并使用ReLU激活
        x = torch.tanh(self.output_layer(x))  # 应用输出层并使用tanh激活，限制输出在[-1,1]范围
        
        # 恢复原始形状
        x = x.permute(0, 3, 1, 2)            # [B, H, W, C] -> [B, C, H, W]，恢复维度顺序
        
        return x                             # 返回更新值

# 4. 实现随机更新机制 (PyTorch版本)
def stochastic_update(grid, model, update_rate=0.5):
    """随机更新网格中的单元
    
    参数:
        grid: 参数网格
        model: 神经CA模型
        update_rate: 每步更新的单元比例
    
    返回:
        更新后的网格
    """
    grid_size = grid.shape[0]                # 获取网格大小
    
    # 转换为PyTorch张量
    grid_tensor = torch.tensor(grid, dtype=torch.float32, device=device)  # 将NumPy数组转换为PyTorch张量
    grid_tensor = grid_tensor.permute(2, 0, 1).unsqueeze(0)  # [H, W, C] -> [1, C, H, W]，调整维度顺序
    
    # 计算所有单元的更新
    with torch.no_grad():                    # 不计算梯度，节省内存
        update = model(grid_tensor)          # 使用模型计算更新值
    
    # 随机选择要更新的单元
    update_mask = np.random.random((grid_size, grid_size)) < update_rate  # 生成随机掩码，决定哪些单元需要更新
    update_mask = update_mask.astype(np.float32)  # 将布尔掩码转换为浮点数
    update_mask = np.stack([update_mask] * 4, axis=-1)  # 扩展掩码到4个通道
    
    # 应用更新
    new_grid = grid.copy()                  # 复制原始网格
    update_np = update.squeeze(0).permute(1, 2, 0).cpu().numpy()  # [1, C, H, W] -> [H, W, C]，转换为NumPy数组
    new_grid += update_np * update_mask * 0.1  # 应用更新，缩小更新步长
    
    # 确保x和y坐标保持不变（我们只更新函数值和活跃状态）
    new_grid[:, :, 0] = grid[:, :, 0]        # 保持x坐标不变
    new_grid[:, :, 1] = grid[:, :, 1]        # 保持y坐标不变
    
    # 重新计算函数值
    for i in range(grid_size):               # 遍历所有行
        for j in range(grid_size):           # 遍历所有列
            x, y = new_grid[i, j, 0], new_grid[i, j, 1]  # 获取当前点的x和y坐标
            new_grid[i, j, 2] = rosenbrock(x, y)  # 重新计算函数值
    
    # 限制活跃状态在[0, 1]范围内
    new_grid[:, :, 3] = np.clip(new_grid[:, :, 3], 0, 1)  # 裁剪活跃状态值
    
    return new_grid                         # 返回更新后的网格

# 5. 训练神经CA模型 (PyTorch版本)
def train_neural_ca(param_grid, model, optimizer, epochs=100, steps_per_epoch=10):
    """训练神经CA模型以找到函数的最小值
    
    参数:
        param_grid: 参数网格对象
        model: 神经CA模型
        optimizer: 优化器
        epochs: 训练轮数
        steps_per_epoch: 每轮的步数
    
    返回:
        训练历史记录
    """
    history = {                              # 初始化历史记录字典
        'loss': [],                          # 存储损失值
        'min_value': [],                     # 存储最小函数值
        'min_position': []                   # 存储最小值位置
    }
    
    for epoch in range(epochs):              # 遍历所有训练轮次
        epoch_loss = 0                       # 初始化当前轮次的累计损失
        
        for step in range(steps_per_epoch):  # 遍历当前轮次的所有步骤
            # 获取当前网格
            grid = param_grid.get_grid()     # 获取当前网格状态
            
            # 转换为PyTorch张量
            grid_tensor = torch.tensor(grid, dtype=torch.float32, device=device)  # 将网格转换为PyTorch张量
            grid_tensor = grid_tensor.permute(2, 0, 1).unsqueeze(0)  # [H, W, C] -> [1, C, H, W]，调整维度顺序
            
            # 前向传播
            optimizer.zero_grad()             # 清除之前的梯度
            update = model(grid_tensor)       # 使用模型计算更新值
            
            # 应用更新并获取新网格
            update_np = update.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()  # 将更新值转换为NumPy数组
            update_mask = np.random.random(grid.shape[:2]) < 0.5  # 生成随机掩码
            update_mask = update_mask.astype(np.float32)  # 将布尔掩码转换为浮点数
            update_mask = np.stack([update_mask] * 4, axis=-1)  # 扩展掩码到4个通道
            
            new_grid = grid.copy()           # 复制原始网格
            new_grid += update_np * update_mask * 0.1  # 应用更新，缩小更新步长
            
            # 确保x和y坐标保持不变
            new_grid[:, :, 0] = grid[:, :, 0]  # 保持x坐标不变
            new_grid[:, :, 1] = grid[:, :, 1]  # 保持y坐标不变
            
            # 重新计算函数值
            for i in range(grid.shape[0]):    # 遍历所有行
                for j in range(grid.shape[1]):  # 遍历所有列
                    x, y = new_grid[i, j, 0], new_grid[i, j, 1]  # 获取当前点的x和y坐标
                    new_grid[i, j, 2] = rosenbrock(x, y)  # 重新计算函数值
            
            # 限制活跃状态
            new_grid[:, :, 3] = np.clip(new_grid[:, :, 3], 0, 1)  # 裁剪活跃状态值
            
            # 将新网格转换为PyTorch张量，并确保它连接到计算图
            new_grid_tensor = torch.tensor(new_grid, dtype=torch.float32, device=device)  # 将新网格转换为PyTorch张量
            
            # 计算损失
            # 1. 函数值损失 - 活跃点的平均函数值
            active_mask = new_grid[:, :, 3] > 0.5  # 找出活跃点
            if np.sum(active_mask) > 0:      # 如果有活跃点
                function_values = new_grid[:, :, 2][active_mask]  # 获取活跃点的函数值
                # 创建一个需要梯度的张量
                value_loss_tensor = torch.tensor(function_values, dtype=torch.float32, device=device, requires_grad=True)  # 转换为需要梯度的张量
                value_loss = torch.mean(value_loss_tensor)  # 计算平均函数值作为损失
            else:
                value_loss = torch.tensor(100.0, device=device, requires_grad=True)  # 惩罚没有活跃点的情况
            
            # 2. 活跃点数量损失 - 我们希望保持一定数量的活跃点
            active_count = torch.sum(new_grid_tensor[:, :, 3] > 0.5)  # 计算活跃点数量
            count_loss = torch.abs(active_count.float() - 10.0)  # 计算与目标数量(10)的差距
            
            # 3. 探索损失 - 鼓励活跃点分散
            if np.sum(active_mask) > 1:      # 如果有多个活跃点
                active_positions = np.argwhere(active_mask)  # 获取活跃点的位置
                dists = np.zeros((len(active_positions), len(active_positions)))  # 初始化距离矩阵
                for i in range(len(active_positions)):  # 遍历所有活跃点
                    for j in range(i+1, len(active_positions)):  # 遍历其他活跃点
                        pos1, pos2 = active_positions[i], active_positions[j]  # 获取两个点的位置
                        dist = np.sqrt(((pos1 - pos2) ** 2).sum())  # 计算欧氏距离
                        dists[i, j] = dists[j, i] = dist  # 填充距离矩阵（对称）
                # 创建一个需要梯度的张量
                exploration_loss_tensor = torch.tensor(dists, dtype=torch.float32, device=device, requires_grad=True)  # 转换为需要梯度的张量
                exploration_loss = -torch.mean(exploration_loss_tensor)  # 计算负平均距离作为损失（最大化距离）
            else:
                exploration_loss = torch.tensor(0.0, device=device, requires_grad=True)  # 如果活跃点不足，不计算探索损失
            
            # 总损失 - 确保所有损失都连接到计算图
            loss = value_loss + 0.1 * count_loss + 0.01 * exploration_loss  # 组合三种损失，使用不同的权重
            
            # 反向传播
            loss.backward()                   # 计算梯度
            optimizer.step()                  # 更新模型参数
            
            # 更新网格
            param_grid.update_grid(new_grid)  # 更新参数网格
            
            epoch_loss += loss.item()        # 累加损失值
        
        # 计算当前最小值
        grid = param_grid.get_grid()         # 获取当前网格
        min_idx = np.argmin(grid[:, :, 2])   # 找出函数值最小的点的索引
        min_i, min_j = min_idx // grid.shape[1], min_idx % grid.shape[1]  # 计算二维索引
        min_value = grid[min_i, min_j, 2]    # 获取最小函数值
        min_position = (grid[min_i, min_j, 0], grid[min_i, min_j, 1])  # 获取最小值位置
        
        # 记录历史
        history['loss'].append(epoch_loss / steps_per_epoch)  # 记录平均损失
        history['min_value'].append(min_value)  # 记录最小函数值
        history['min_position'].append(min_position)  # 记录最小值位置
        
        # 打印进度
        if (epoch + 1) % 10 == 0:            # 每10轮打印一次进度
            print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss/steps_per_epoch:.4f}, "
                  f"Min Value: {min_value:.4f} at {min_position}")  # 打印当前轮次、损失和最小值信息
    
    return history                          # 返回训练历史记录

# 6. 可视化函数
def visualize_optimization(param_grid, history, save_animation=False):
    """可视化优化过程
    
    参数:
        param_grid: 参数网格对象
        history: 训练历史记录
        save_animation: 是否保存动画
    """
    grid_size = param_grid.grid_size         # 获取网格大小
    param_range = param_grid.param_range     # 获取参数范围
    
    # 创建网格用于等高线图
    x = np.linspace(param_range[0], param_range[1], 100)  # 生成x坐标点
    y = np.linspace(param_range[0], param_range[1], 100)  # 生成y坐标点
    X, Y = np.meshgrid(x, y)                # 创建坐标网格
    Z = rosenbrock(X, Y)                    # 计算每个网格点的函数值
    
    # 创建图形
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))  # 创建1行2列的子图
    
    # 绘制等高线图
    contour = ax1.contourf(X, Y, Z, levels=50, cmap='viridis')  # 绘制填充等高线图
    ax1.set_xlabel('x')                     # 设置x轴标签
    ax1.set_ylabel('y')                     # 设置y轴标签
    ax1.set_title('Rosenbrock函数优化过程')   # 设置标题
    fig.colorbar(contour, ax=ax1)           # 添加颜色条
    
    # 初始化散点图 - 显示活跃点
    grid = param_grid.get_grid()            # 获取当前网格
    active_mask = grid[:, :, 3] > 0.5       # 找出活跃点
    active_x = grid[:, :, 0][active_mask]   # 获取活跃点的x坐标
    active_y = grid[:, :, 1][active_mask]   # 获取活跃点的y坐标
    scatter = ax1.scatter(active_x, active_y, c='red', s=50, label='活跃点')  # 绘制活跃点散点图
    
    # 绘制最优点轨迹
    min_positions = np.array(history['min_position'])  # 获取最优点轨迹
    line, = ax1.plot(min_positions[:1, 0], min_positions[:1, 1], 'r-', linewidth=2, label='最优点轨迹')  # 绘制轨迹线
    min_point = ax1.scatter(min_positions[0, 0], min_positions[0, 1], c='white', s=100, edgecolors='black', label='当前最优点')  # 绘制当前最优点
    
    # 绘制损失曲线
    ax2.plot(history['loss'], label='总损失')  # 绘制总损失曲线
    ax2.plot(history['min_value'], label='最小函数值')  # 绘制最小函数值曲线
    ax2.set_xlabel('Epoch')                 # 设置x轴标签
    ax2.set_ylabel('Value')                 # 设置y轴标签
    ax2.set_title('训练损失和最小函数值')      # 设置标题
    ax2.legend()                            # 添加图例
    ax2.grid(True)                          # 显示网格线
    
    # 添加图例
    ax1.legend()                            # 为第一个子图添加图例
    
    # 标记全局最小值
    ax1.scatter(1, 1, c='yellow', s=200, alpha=0.5, edgecolors='black', label='全局最小值')  # 标记全局最小值点(1,1)
    
    # 更新函数
    def update(frame):
        # 更新最优点轨迹
        line.set_data(min_positions[:frame+1, 0], min_positions[:frame+1, 1])  # 更新轨迹线数据
        
        # 更新当前最优点
        min_point.set_offsets([min_positions[frame, 0], min_positions[frame, 1]])  # 更新最优点位置
        
        return scatter, line, min_point      # 返回需要更新的图形元素
    
    # 创建动画
    ani = FuncAnimation(fig, update, frames=len(history['min_value']), 
                        interval=100, blit=True)  # 创建动画，每100毫秒更新一帧
    
    # 保存动画
    if save_animation:
        ani.save('optimization_animation.gif', writer='pillow', fps=10)  # 保存为GIF文件，每秒10帧
    
    plt.tight_layout()                      # 自动调整子图参数，使之填充整个图像区域
    plt.show()                              # 显示图形

# 7. 主函数
def main():
    # 创建参数网格
    param_grid = ParameterGrid(grid_size=32, param_range=(-2, 2))  # 创建32x32的参数网格，范围为[-2,2]
    
    # 创建神经CA模型
    model = NeuralCA().to(device)           # 创建模型并移至指定设备(CPU/GPU)
    
    # 创建优化器
    optimizer = optim.Adam(model.parameters(), lr=0.01)  # 使用Adam优化器，学习率为0.01
    
    # 训练模型
    print("开始训练神经元胞自动机...")        # 打印训练开始信息
    start_time = time.time()                # 记录开始时间
    history = train_neural_ca(param_grid, model, optimizer, epochs=100, steps_per_epoch=5)  # 训练模型
    end_time = time.time()                  # 记录结束时间
    print(f"训练完成，耗时: {end_time - start_time:.2f}秒")  # 打印训练耗时
    
    # 打印最终结果
    final_min_value = history['min_value'][-1]  # 获取最终的最小函数值
    final_min_position = history['min_position'][-1]  # 获取最终的最小值位置
    print(f"\n最终结果:")                     # 打印结果标题
    print(f"最小函数值: {final_min_value:.6f}")  # 打印最小函数值
    print(f"最优参数: x={final_min_position[0]:.6f}, y={final_min_position[1]:.6f}")  # 打印最优参数
    print(f"全局最小值: 0.0 at (1.0, 1.0)")   # 打印理论上的全局最小值
    
    # 可视化优化过程
    visualize_optimization(param_grid, history, save_animation=True)  # 可视化并保存动画

# 程序入口点
if __name__ == "__main__":
    main()