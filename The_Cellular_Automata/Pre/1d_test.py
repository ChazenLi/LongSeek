import numpy as np  # 导入NumPy库，用于数值计算
import matplotlib.pyplot as plt  # 导入matplotlib的pyplot模块，用于绘图
from matplotlib.animation import FuncAnimation  # 导入动画功能

# 将规则编号转换为状态转换表
# 例如规则30会被转换为一个集合，集合中包含哪些三元组会变为1
# 规则编号0~255，对应8位二进制

def apply_rule(rule_number):
    """将规则编号转换为状态转换表"""
    rule_bin = np.binary_repr(rule_number, width=8)  # 转为8位二进制字符串
    rule_function = set()  # 用于存储会变为1的三元组

    for i in range(8):  # 遍历所有可能的三元组（000~111）
        if rule_bin[7 - i] == '1':  # 若该位为1
            config = tuple(map(int, format(i, '03b')))  # 转为三元组
            rule_function.add(config)  # 加入集合

    return rule_function  # 返回规则集合

# 根据规则更新元胞状态

def update_cells(current_state, rule_function):
    """根据规则更新元胞状态"""
    next_state = current_state.copy()  # 复制当前状态
    n = len(current_state)  # 元胞总数

    for i in range(n):  # 遍历每个元胞
        left = current_state[(i - 1) % n]  # 左邻居（循环边界）
        center = current_state[i]  # 当前元胞
        right = current_state[(i + 1) % n]  # 右邻居

        if (left, center, right) in rule_function:  # 若三元组在规则集合中
            next_state[i] = 1  # 变为1
        else:
            next_state[i] = 0  # 否则变为0

    return next_state  # 返回下一步状态

# 初始化并运行元胞自动机，返回初始状态和步数

def run_ca(rule_number, steps=50, size=100):
    """初始化并运行元胞自动机"""
    state = np.zeros(size, dtype=int)  # 所有元胞初始化为0
    state[size // 2] = 1  # 中间元胞置为1
    return state, steps  # 返回初始状态和步数

# 设置图形窗口
fig, ax = plt.subplots(figsize=(10, 6))  # 创建绘图窗口，指定大小
ax.set_axis_off()  # 隐藏坐标轴

im = None  # 用于保存图像对象

# 动画每一帧的更新函数

def animate(frame):
    """动画每一帧的更新函数"""
    global current_state, rule_function, im  # 使用全局变量

    # 更新状态
    current_state = update_cells(current_state, rule_function)
    # 转换为二维数组（单行），便于imshow显示
    img_array = np.expand_dims(current_state, axis=0)

    # 如果是第一帧，创建图像对象
    if im is None:
        im = ax.imshow(img_array, cmap='binary', interpolation='nearest', animated=True)
        ax.set_title(f"Rule {rule_number} - Step {frame}")  # 设置标题
    else:
        im.set_data(img_array)  # 更新图像数据
        ax.set_title(f"Rule {rule_number} - Step {frame}")  # 更新标题

    return im,  # 返回图像对象

if __name__ == "__main__":  # 主程序入口
    rule_number = 30  # 可修改为其他规则（0~255）
    size = 100  # 元胞数量
    steps = 50  # 演化步数

    # 初始化元胞自动机
    current_state, total_steps = run_ca(rule_number, steps=steps, size=size)
    rule_function = apply_rule(rule_number)  # 获取规则集合

    # 创建动画
    ani = FuncAnimation(fig, animate, frames=total_steps, interval=200, blit=True)

    # 显示动画
    plt.show()