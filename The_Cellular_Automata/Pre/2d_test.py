# 导入必要的库
import numpy as np  # 用于数值计算和网格处理
import matplotlib.pyplot as plt  # 数据可视化
from matplotlib.animation import FuncAnimation  # 创建动画

# 初始化网格函数
def initialize_grid(size):
    """创建随机初始化的二维网格
    参数:
        size: 网格尺寸（size x size）
    返回:
        随机初始化的二维网格，p=[0.7,0.3]表示70%概率生成0(死细胞)，30%概率生成1(活细胞)
    """
    return np.random.choice([0, 1], size=(size, size), p=[0.7, 0.3]).astype(np.int8)

# 邻居计数器函数
def count_neighbors(grid, x, y, size):
    """计算 (x,y) 位置的活邻居数量
    参数:
        grid: 当前网格状态
        x, y: 当前细胞坐标
        size: 网格尺寸
    返回:
        邻居中活细胞的数量
    注意:
        使用循环边界条件处理网格边缘，使网格呈现无限延伸效果
    """
    neighbors = 0
    for dx in [-1, 0, 1]:  # 水平方向偏移
        for dy in [-1, 0, 1]:  # 垂直方向偏移
            if dx == 0 and dy == 0:  # 排除自身
                continue
            nx, ny = (x + dx) % size, (y + dy) % size  # 循环边界处理
            neighbors += grid[nx, ny]  # 累计邻居状态
    return neighbors

# 网格状态更新函数
def update_grid(grid, add_disturbance=False):
    """康威生命游戏规则更新
    参数:
        grid: 当前网格状态
        add_disturbance: 是否添加随机扰动
    返回:
        更新后的网格状态
    规则:
        1. 活细胞邻居 <2 或 >3 → 死亡（模拟孤独或拥挤导致死亡）
        2. 死细胞邻居 ==3 → 复活（模拟繁殖）
        3. 如果系统稳定，添加随机扰动（优化规则）
    """
    size = grid.shape[0]
    new_grid = grid.copy()  # 创建新网格副本
    
    # 应用标准康威生命游戏规则
    for x in range(size):
        for y in range(size):
            neighbors = count_neighbors(grid, x, y, size)
            if grid[x, y] == 1:  # 活细胞判断
                if neighbors < 2 or neighbors > 3:
                    new_grid[x, y] = 0  # 死亡规则：孤独或拥挤
            else:  # 死细胞判断
                if neighbors == 3:
                    new_grid[x, y] = 1  # 复活规则：繁殖
    
    # 如果需要添加扰动（系统稳定时）
    if add_disturbance:
        # 随机选择网格中的少量点进行状态翻转
        disturbance_points = int(size * size * 0.01)  # 扰动1%的细胞
        for _ in range(disturbance_points):
            x, y = np.random.randint(0, size), np.random.randint(0, size)
            new_grid[x, y] = 1 - new_grid[x, y]  # 翻转细胞状态
            
    return new_grid

# 检查网格是否稳定（没有变化或循环）
def is_stable(grid_history, current_grid, stability_threshold=5):
    """检查网格是否达到稳定状态
    参数:
        grid_history: 历史网格状态列表
        current_grid: 当前网格状态
        stability_threshold: 稳定性判断阈值
    返回:
        布尔值，表示系统是否稳定
    """
    # 如果历史记录不足，认为不稳定
    if len(grid_history) < stability_threshold:
        return False
    
    # 检查是否与最近的网格状态相同（静态稳定）
    for i in range(1, min(stability_threshold + 1, len(grid_history))):
        if np.array_equal(grid_history[-i], current_grid):
            return True
    
    # 检查是否存在循环模式（周期稳定）
    for period in range(2, min(stability_threshold, len(grid_history) // 2 + 1)):
        is_cycle = True
        for offset in range(period):
            if not np.array_equal(grid_history[-(offset+1)], grid_history[-(offset+1+period)]):
                is_cycle = False
                break
        if is_cycle:
            return True
            
    return False

# 动画更新函数
def animate(frame):
    """动画帧更新函数
    参数:
        frame: 当前帧号
    返回:
        更新后的图像对象
    """
    global current_grid, grid_history, stability_counter, disturbance_counter
    
    # 检查系统是否稳定
    stable = is_stable(grid_history, current_grid)
    
    # 更新网格状态
    if stable:
        stability_counter += 1
        # 每隔10帧添加一次扰动
        if stability_counter % 10 == 0:
            current_grid = update_grid(current_grid, add_disturbance=True)
            disturbance_counter += 1
        else:
            current_grid = update_grid(current_grid)
    else:
        stability_counter = 0
        current_grid = update_grid(current_grid)
    
    # 保存历史状态（最多保留50个状态以节省内存）
    grid_history.append(current_grid.copy())
    if len(grid_history) > 50:
        grid_history.pop(0)
    
    # 计算活细胞比例
    alive_ratio = np.sum(current_grid) / (current_grid.shape[0] * current_grid.shape[1]) * 100
    
    # 更新图像和标题
    img.set_array(current_grid)
    ax.set_title(f"步骤: {frame} | 活细胞: {alive_ratio:.1f}% | {'稳定' if stable else '变化中'} | 扰动次数: {disturbance_counter}")
    
    return img,  # 返回更新后的图像对象

# 主程序入口
if __name__ == "__main__":
    # 参数配置
    grid_size = 100       # 100x100网格
    update_interval = 50  # 50ms更新间隔
    
    # 初始化全局变量
    current_grid = initialize_grid(grid_size)  # 当前网格状态
    grid_history = [current_grid.copy()]       # 网格历史状态
    stability_counter = 0                      # 稳定状态计数器
    disturbance_counter = 0                    # 扰动计数器
    
    # 创建绘图窗口
    fig, ax = plt.subplots(figsize=(10, 10))  # 10x10英寸画布
    ax.set_axis_off()  # 隐藏坐标轴
    img = ax.imshow(current_grid, cmap='viridis', interpolation='nearest')  # 创建初始图像，使用viridis配色
    plt.colorbar(img, ax=ax, label='细胞状态')  # 添加颜色条
    
    # 创建动画
    ani = FuncAnimation(
        fig, 
        animate,
        frames=500,       # 生成500帧动画
        interval=update_interval,  # 每帧间隔
        blit=True        # 优化渲染性能
    )
    
    # 显示动画
    plt.tight_layout()  # 优化布局
    plt.show()  # 启动matplotlib事件循环