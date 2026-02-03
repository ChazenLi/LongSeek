import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib import rcParams
import numpy as np
from collections import defaultdict
import re


rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS', 'DejaVu Sans']
rcParams['axes.unicode_minus'] = False
rcParams['figure.figsize'] = [14, 8]


class GameVisualizer:
    def __init__(self):
        self.scores = defaultdict(list)
        self.reputations = defaultdict(list)
        self.rounds = []
        self.alliances = []
        self.betrayals = []
        self.agent_colors = {
            0: '#FF6B6B',
            1: '#4ECDC4',
            2: '#45B7D1',
            3: '#96CEB4'
        }
        self.agent_names = {}
    
    def parse_game_log(self, log_file: str = None, log_text: str = None):
        """解析游戏日志"""
        if log_text:
            lines = log_text.split('\n')
        else:
            with open(log_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()
        
        current_round = 0
        
        for line in lines:
            # 检测回合
            if '第' in line and '回合' in line:
                match = re.search(r'第\s*(\d+)\s*回合', line)
                if match:
                    current_round = int(match.group(1))
                    if current_round not in self.rounds:
                        self.rounds.append(current_round)
            
            # 检测agent信息（声明与决策阶段）
            if 'Agent' in line and '位置:' in line:
                match = re.search(r'Agent\s+(\d+)\s+\((.+?)\)', line)
                if match:
                    agent_id = int(match.group(1))
                    strategy_name = match.group(2).strip()
                    if agent_id not in self.agent_names:
                        self.agent_names[agent_id] = strategy_name
            
            # 检测分数和信誉
            if '分数:' in line and '信誉:' in line:
                match = re.search(r'Agent\s+(\d+).*?分数:\s*(-?\d+),\s*信誉:\s*([\d.]+)', line)
                if match:
                    agent_id = int(match.group(1))
                    score = int(match.group(2))
                    reputation = float(match.group(3))
                    
                    self.scores[agent_id].append(score)
                    self.reputations[agent_id].append(reputation)
            
            # 检测联盟
            if '结成联盟' in line:
                match = re.search(r'Agent\s+(\d+)\s+和\s+Agent\s+(\d+)\s+结成联盟', line)
                if match:
                    agent1 = int(match.group(1))
                    agent2 = int(match.group(2))
                    self.alliances.append((current_round, agent1, agent2))
            
            # 检测背叛
            if '背叛了' in line:
                match = re.search(r'Agent\s+(\d+)\s+背叛了\s+Agent\s+(\d+)', line)
                if match:
                    betrayer = int(match.group(1))
                    victim = int(match.group(2))
                    self.betrayals.append((current_round, betrayer, victim))
            
            # 检测当前状态
            if 'Agent' in line and '分数=' in line and ', 信誉=' in line:
                match = re.search(r'Agent\s+(\d+):\s*分数=(-?\d+),\s*信誉=([\d.]+)', line)
                if match:
                    agent_id = int(match.group(1))
                    score = int(match.group(2))
                    reputation = float(match.group(3))
                    
                    # 只在当前回合结束时更新
                    if len(self.scores[agent_id]) < len(self.rounds):
                        self.scores[agent_id].append(score)
                        self.reputations[agent_id].append(reputation)
        
        # 确保数据长度一致
        max_len = len(self.rounds)
        for agent_id in self.scores:
            while len(self.scores[agent_id]) < max_len:
                if self.scores[agent_id]:
                    self.scores[agent_id].append(self.scores[agent_id][-1])
                else:
                    self.scores[agent_id].append(0)
        
        for agent_id in self.reputations:
            while len(self.reputations[agent_id]) < max_len:
                if self.reputations[agent_id]:
                    self.reputations[agent_id].append(self.reputations[agent_id][-1])
                else:
                    self.reputations[agent_id].append(1.0)
    
    def plot_score_evolution(self, save_path: str = 'score_evolution.png'):
        """绘制分数变化图"""
        fig = plt.figure(figsize=(14, 8))
        
        # 绘制每个agent的分数曲线
        for agent_id, scores in sorted(self.scores.items()):
            if scores:
                name = self.agent_names.get(agent_id, f'Agent {agent_id}')
                plt.plot(self.rounds, scores, 
                        marker='o', 
                        linewidth=3, 
                        markersize=8,
                        color=self.agent_colors.get(agent_id, '#333333'),
                        label=name,
                        alpha=0.8)
        
        # 标记联盟事件
        for round_num, agent1, agent2 in self.alliances:
            if round_num <= len(self.rounds):
                y1 = self.scores[agent1][round_num - 1] if round_num - 1 < len(self.scores[agent1]) else 0
                y2 = self.scores[agent2][round_num - 1] if round_num - 1 < len(self.scores[agent2]) else 0
                avg_y = (y1 + y2) / 2
                plt.scatter(round_num, avg_y, marker='*', s=300, c='gold', 
                           edgecolors='orange', linewidths=2, zorder=10)
                plt.text(round_num, avg_y + 1, '联盟', ha='center', va='bottom', 
                        fontsize=9, fontweight='bold', color='orange')
        
        # 标记背叛事件
        for round_num, betrayer, victim in self.betrayals:
            if round_num <= len(self.rounds):
                y1 = self.scores[betrayer][round_num - 1] if round_num - 1 < len(self.scores[betrayer]) else 0
                y2 = self.scores[victim][round_num - 1] if round_num - 1 < len(self.scores[victim]) else 0
                avg_y = (y1 + y2) / 2
                plt.scatter(round_num, avg_y, marker='X', s=300, c='red', 
                           edgecolors='darkred', linewidths=2, zorder=10)
                plt.text(round_num, avg_y - 1.5, '背叛', ha='center', va='top', 
                        fontsize=9, fontweight='bold', color='red')
        
        plt.xlabel('回合数', fontsize=14, fontweight='bold')
        plt.ylabel('分数', fontsize=14, fontweight='bold')
        plt.title('各Agent分数变化趋势', fontsize=18, fontweight='bold', pad=20)
        plt.grid(True, alpha=0.3, linestyle='--')
        plt.legend(loc='best', fontsize=11, framealpha=0.9)
        
        # 添加胜利线
        plt.axhline(y=10, color='green', linestyle='--', linewidth=2, alpha=0.6, label='胜利线(10分)')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"[OK] 分数变化图已保存到: {save_path}")
        plt.close(fig)
    
    def plot_reputation_evolution(self, save_path: str = 'reputation_evolution.png'):
        """绘制信誉变化图"""
        fig = plt.figure(figsize=(14, 8))
        
        # 绘制每个agent的信誉曲线
        for agent_id, reputations in sorted(self.reputations.items()):
            if reputations:
                name = self.agent_names.get(agent_id, f'Agent {agent_id}')
                plt.plot(self.rounds, reputations, 
                        marker='s', 
                        linewidth=3, 
                        markersize=8,
                        color=self.agent_colors.get(agent_id, '#333333'),
                        label=name,
                        alpha=0.8)
        
        plt.xlabel('回合数', fontsize=14, fontweight='bold')
        plt.ylabel('信誉值', fontsize=14, fontweight='bold')
        plt.title('各Agent信誉变化趋势', fontsize=18, fontweight='bold', pad=20)
        plt.grid(True, alpha=0.3, linestyle='--')
        plt.legend(loc='best', fontsize=11, framealpha=0.9)
        plt.ylim(0, 1.2)
        
        # 标记背叛导致的信誉变化
        for round_num, betrayer, victim in self.betrayals:
            if round_num <= len(self.rounds):
                plt.axvline(x=round_num, color='red', linestyle=':', linewidth=1.5, alpha=0.5)
                plt.text(round_num, 1.1, f'背叛!\nAgent {betrayer}→{victim}', 
                        ha='center', va='top', fontsize=9, color='red', 
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"[OK] 信誉变化图已保存到: {save_path}")
        plt.close(fig)
    
    def plot_combined_chart(self, save_path: str = 'combined_chart.png'):
        """绘制综合图表（分数+信誉）"""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 12))
        
        # 上图：分数变化
        for agent_id, scores in sorted(self.scores.items()):
            if scores:
                name = self.agent_names.get(agent_id, f'Agent {agent_id}')
                ax1.plot(self.rounds, scores, 
                        marker='o', 
                        linewidth=2.5, 
                        markersize=7,
                        color=self.agent_colors.get(agent_id, '#333333'),
                        label=name,
                        alpha=0.8)
        
        # 标记联盟和背叛
        for round_num, agent1, agent2 in self.alliances:
            if round_num <= len(self.rounds):
                y1 = self.scores[agent1][round_num - 1] if round_num - 1 < len(self.scores[agent1]) else 0
                y2 = self.scores[agent2][round_num - 1] if round_num - 1 < len(self.scores[agent2]) else 0
                avg_y = (y1 + y2) / 2
                ax1.scatter(round_num, avg_y, marker='*', s=200, c='gold', 
                           edgecolors='orange', linewidths=2, zorder=10)
        
        for round_num, betrayer, victim in self.betrayals:
            if round_num <= len(self.rounds):
                y1 = self.scores[betrayer][round_num - 1] if round_num - 1 < len(self.scores[betrayer]) else 0
                y2 = self.scores[victim][round_num - 1] if round_num - 1 < len(self.scores[victim]) else 0
                avg_y = (y1 + y2) / 2
                ax1.scatter(round_num, avg_y, marker='X', s=200, c='red', 
                           edgecolors='darkred', linewidths=2, zorder=10)
        
        ax1.axhline(y=10, color='green', linestyle='--', linewidth=2, alpha=0.6, label='胜利线')
        ax1.set_xlabel('回合数', fontsize=12, fontweight='bold')
        ax1.set_ylabel('分数', fontsize=12, fontweight='bold')
        ax1.set_title('各Agent分数变化趋势', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3, linestyle='--')
        ax1.legend(loc='best', fontsize=10)
        
        # 创建图例
        alliance_patch = mpatches.Patch(color='gold', label='★ 联盟')
        betrayal_patch = mpatches.Patch(color='red', label='X 背叛')
        ax1.legend(handles=[alliance_patch, betrayal_patch], loc='upper right')
        
        # 下图：信誉变化
        for agent_id, reputations in sorted(self.reputations.items()):
            if reputations:
                name = self.agent_names.get(agent_id, f'Agent {agent_id}')
                ax2.plot(self.rounds, reputations, 
                        marker='s', 
                        linewidth=2.5, 
                        markersize=7,
                        color=self.agent_colors.get(agent_id, '#333333'),
                        label=name,
                        alpha=0.8)
        
        ax2.set_xlabel('回合数', fontsize=12, fontweight='bold')
        ax2.set_ylabel('信誉值', fontsize=12, fontweight='bold')
        ax2.set_title('各Agent信誉变化趋势', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3, linestyle='--')
        ax2.set_ylim(0, 1.2)
        
        # 标记背叛事件
        for round_num, betrayer, victim in self.betrayals:
            if round_num <= len(self.rounds):
                ax2.axvline(x=round_num, color='red', linestyle=':', linewidth=1.5, alpha=0.5)
                ax2.text(round_num, 1.1, f'背叛', 
                        ha='center', va='top', fontsize=8, color='red',
                        bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"[OK] 综合图表已保存到: {save_path}")
        plt.close(fig)
    
    def plot_radar_chart(self, save_path: str = 'radar_chart.png'):
        """绘制雷达图（最终状态）"""
        if not self.rounds:
            print("[警告] 没有数据，无法绘制雷达图")
            return
        
        categories = ['分数', '信誉', '胜率', '稳定性', '社交']
        
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection='polar')
        
        angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
        angles += angles[:1]
        
        # 计算每个agent的指标
        agent_stats = {}
        for agent_id in self.scores:
            scores = self.scores[agent_id]
            reputations = self.reputations[agent_id]
            
            if not scores:
                continue
            
            # 分数（归一化到0-1）
            max_score = max(max(s) for s in self.scores.values()) if self.scores else 1
            score_norm = scores[-1] / max_score if max_score > 0 else 0
            
            # 信誉
            reputation = reputations[-1] if reputations else 0.5
            
            # 胜率（假设每次对战获胜率为0.5，这里简化处理）
            win_rate = 0.5 + (score_norm - 0.5) * 0.3
            
            # 稳定性（分数变化的方差倒数）
            if len(scores) > 1:
                stability = 1 / (1 + np.var(scores))
            else:
                stability = 0.5
            
            # 社交（联盟数 - 背叛数）
            social = 0.5
            alliance_count = sum(1 for _, a1, a2 in self.alliances if a1 == agent_id or a2 == agent_id)
            betrayal_count = sum(1 for _, b, _ in self.betrayals if b == agent_id)
            social = 0.5 + (alliance_count - betrayal_count) * 0.2
            social = max(0, min(1, social))
            
            agent_stats[agent_id] = [score_norm, reputation, win_rate, stability, social]
        
        # 绘制雷达图
        for agent_id, stats in sorted(agent_stats.items()):
            name = self.agent_names.get(agent_id, f'Agent {agent_id}')
            values = stats + stats[:1]
            
            ax.plot(angles, values, 'o-', linewidth=2, 
                   label=name, color=self.agent_colors.get(agent_id, '#333333'))
            ax.fill(angles, values, alpha=0.15, color=self.agent_colors.get(agent_id, '#333333'))
        
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories)
        ax.set_ylim(0, 1)
        ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'])
        ax.grid(True)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
        plt.title('各Agent综合能力雷达图', fontsize=16, fontweight='bold', pad=20)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"[OK] 雷达图已保存到: {save_path}")
        plt.close(fig)
    
    def plot_all(self):
        """生成所有图表"""
        if not self.rounds:
            print("没有数据，请先运行游戏")
            return
        
        print("\n[可视化] 开始生成游戏成长图...")
        
        self.plot_score_evolution()
        self.plot_reputation_evolution()
        self.plot_combined_chart()
        self.plot_radar_chart()
        
        print("\n[完成] 所有图表生成完成！")


def main():
    import game
    
    print("[游戏] 开始运行游戏...")
    print("=" * 70)
    
    # 运行游戏并保存输出到文件
    import sys
    from io import StringIO
    
    old_stdout = sys.stdout
    captured_output = StringIO()
    sys.stdout = captured_output
    
    try:
        game_instance = game.Game()
        game_instance.agents = [
            game.DiplomaticAgent(0, (0, 0)),
            game.VengefulAgent(1, (0, 1)),
            game.OpportunistAgent(2, (1, 0)),
            game.ManipulatorAgent(3, (1, 1))
        ]
        game_instance.run()
    except Exception as e:
        print(f"[错误] 游戏运行错误: {e}")
    finally:
        # 获取游戏输出
        game_output = captured_output.getvalue()
        sys.stdout = old_stdout
        
        # 保存到文件
        log_file = 'game_log.txt'
        with open(log_file, 'w', encoding='utf-8') as f:
            f.write(game_output)
        print(f"[完成] 游戏日志已保存到: {log_file}")
        print("=" * 70)
        
        # 创建可视化
        print("\n[可视化] 开始生成图表...")
        visualizer = GameVisualizer()
        visualizer.parse_game_log(log_file=log_file)
        visualizer.plot_all()
        print("\n[完成] 所有图表生成完毕！")


if __name__ == "__main__":
    main()
