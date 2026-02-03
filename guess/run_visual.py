import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from collections import defaultdict
import re
import sys
from io import StringIO

rcParams = plt.rcParams
rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans', 'Arial']
rcParams['axes.unicode_minus'] = False
rcParams['figure.figsize'] = [14, 8]


def generate_visualizations(log_text: str):
    """从游戏日志生成图表"""
    
    # 解析数据
    scores = defaultdict(list)
    reputations = defaultdict(list)
    rounds = []
    alliances = []
    betrayals = []
    agent_names = {}
    agent_colors = {
        0: '#FF6B6B',
        1: '#4ECDC4',
        2: '#45B7D1',
        3: '#96CEB4'
    }
    
    current_round = 0
    lines = log_text.split('\n')
    
    for line in lines:
        # 检测回合
        if '第' in line and '回合' in line:
            match = re.search(r'第\s*(\d+)\s*回合', line)
            if match:
                current_round = int(match.group(1))
                if current_round not in rounds:
                    rounds.append(current_round)
        
        # 检测agent名称
        if 'Agent' in line and '位置:' in line:
            match = re.search(r'Agent\s+(\d+)\s+\((.+?)\)', line)
            if match:
                agent_id = int(match.group(1))
                strategy_name = match.group(2).strip()
                if agent_id not in agent_names:
                    agent_names[agent_id] = strategy_name
        
        # 检测分数和信誉
        if '分数:' in line and '信誉:' in line:
            match = re.search(r'Agent\s+(\d+).*?分数:\s*(-?\d+),\s*信誉:\s*([\d.]+)', line)
            if match:
                agent_id = int(match.group(1))
                score = int(match.group(2))
                reputation = float(match.group(3))
                
                scores[agent_id].append(score)
                reputations[agent_id].append(reputation)
        
        # 检测联盟
        if '结成联盟' in line:
            match = re.search(r'Agent\s+(\d+)\s+和\s+Agent\s+(\d+)\s+结成联盟', line)
            if match:
                agent1 = int(match.group(1))
                agent2 = int(match.group(2))
                alliances.append((current_round, agent1, agent2))
        
        # 检测背叛
        if '背叛了' in line:
            match = re.search(r'Agent\s+(\d+)\s+背叛了\s+Agent\s+(\d+)', line)
            if match:
                betrayer = int(match.group(1))
                victim = int(match.group(2))
                betrayals.append((current_round, betrayer, victim))
    
    # 确保数据长度一致
    max_len = len(rounds)
    for agent_id in scores:
        while len(scores[agent_id]) < max_len:
            if scores[agent_id]:
                scores[agent_id].append(scores[agent_id][-1])
            else:
                scores[agent_id].append(0)
    
    for agent_id in reputations:
        while len(reputations[agent_id]) < max_len:
            if reputations[agent_id]:
                reputations[agent_id].append(reputations[agent_id][-1])
            else:
                reputations[agent_id].append(1.0)
    
    # 生成图表1：分数变化
    print("[图表1] 生成分数变化图...")
    fig1 = plt.figure(figsize=(14, 8))
    
    for agent_id, agent_scores in sorted(scores.items()):
        if agent_scores:
            name = agent_names.get(agent_id, f'Agent {agent_id}')
            plt.plot(rounds, agent_scores, 
                    marker='o', linewidth=3, markersize=8,
                    color=agent_colors.get(agent_id, '#333333'),
                    label=name, alpha=0.8)
    
    # 标记联盟
    for round_num, agent1, agent2 in alliances:
        if round_num <= len(rounds):
            y1 = scores[agent1][round_num - 1] if round_num - 1 < len(scores[agent1]) else 0
            y2 = scores[agent2][round_num - 1] if round_num - 1 < len(scores[agent2]) else 0
            avg_y = (y1 + y2) / 2
            plt.scatter(round_num, avg_y, marker='*', s=300, c='gold', 
                       edgecolors='orange', linewidths=2, zorder=10)
            plt.text(round_num, avg_y + 1, '联盟', ha='center', va='bottom', 
                    fontsize=9, fontweight='bold', color='orange')
    
    # 标记背叛
    for round_num, betrayer, victim in betrayals:
        if round_num <= len(rounds):
            y1 = scores[betrayer][round_num - 1] if round_num - 1 < len(scores[betrayer]) else 0
            y2 = scores[victim][round_num - 1] if round_num - 1 < len(scores[victim]) else 0
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
    plt.axhline(y=10, color='green', linestyle='--', linewidth=2, alpha=0.6, label='胜利线(10分)')
    plt.tight_layout()
    plt.savefig('score_evolution.png', dpi=200, bbox_inches='tight')
    plt.close(fig1)
    print("[完成] score_evolution.png 已保存")
    
    # 生成图表2：信誉变化
    print("[图表2] 生成信誉变化图...")
    fig2 = plt.figure(figsize=(14, 8))
    
    for agent_id, agent_reputations in sorted(reputations.items()):
        if agent_reputations:
            name = agent_names.get(agent_id, f'Agent {agent_id}')
            plt.plot(rounds, agent_reputations, 
                    marker='s', linewidth=3, markersize=8,
                    color=agent_colors.get(agent_id, '#333333'),
                    label=name, alpha=0.8)
    
    plt.xlabel('回合数', fontsize=14, fontweight='bold')
    plt.ylabel('信誉值', fontsize=14, fontweight='bold')
    plt.title('各Agent信誉变化趋势', fontsize=18, fontweight='bold', pad=20)
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.legend(loc='best', fontsize=11, framealpha=0.9)
    plt.ylim(0, 1.2)
    plt.tight_layout()
    plt.savefig('reputation_evolution.png', dpi=200, bbox_inches='tight')
    plt.close(fig2)
    print("[完成] reputation_evolution.png 已保存")
    
    # 生成图表3：综合图表
    print("[图表3] 生成综合图表...")
    fig3, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 12))
    
    # 上图：分数变化
    for agent_id, agent_scores in sorted(scores.items()):
        if agent_scores:
            name = agent_names.get(agent_id, f'Agent {agent_id}')
            ax1.plot(rounds, agent_scores, 
                    marker='o', linewidth=2.5, markersize=7,
                    color=agent_colors.get(agent_id, '#333333'),
                    label=name, alpha=0.8)
    
    for round_num, agent1, agent2 in alliances:
        if round_num <= len(rounds):
            y1 = scores[agent1][round_num - 1] if round_num - 1 < len(scores[agent1]) else 0
            y2 = scores[agent2][round_num - 1] if round_num - 1 < len(scores[agent2]) else 0
            avg_y = (y1 + y2) / 2
            ax1.scatter(round_num, avg_y, marker='*', s=200, c='gold', 
                       edgecolors='orange', linewidths=2, zorder=10)
    
    for round_num, betrayer, victim in betrayals:
        if round_num <= len(rounds):
            y1 = scores[betrayer][round_num - 1] if round_num - 1 < len(scores[betrayer]) else 0
            y2 = scores[victim][round_num - 1] if round_num - 1 < len(scores[victim]) else 0
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
    for agent_id, agent_reputations in sorted(reputations.items()):
        if agent_reputations:
            name = agent_names.get(agent_id, f'Agent {agent_id}')
            ax2.plot(rounds, agent_reputations, 
                    marker='s', linewidth=2.5, markersize=7,
                    color=agent_colors.get(agent_id, '#333333'),
                    label=name, alpha=0.8)
    
    ax2.set_xlabel('回合数', fontsize=12, fontweight='bold')
    ax2.set_ylabel('信誉值', fontsize=12, fontweight='bold')
    ax2.set_title('各Agent信誉变化趋势', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.set_ylim(0, 1.2)
    
    plt.tight_layout()
    plt.savefig('combined_chart.png', dpi=200, bbox_inches='tight')
    plt.close(fig3)
    print("[完成] combined_chart.png 已保存")
    
    print("\n[总结] 所有图表生成完成！")
    print("  - score_evolution.png: 分数变化趋势")
    print("  - reputation_evolution.png: 信誉变化趋势")
    print("  - combined_chart.png: 综合图表")


def main():
    import game
    
    print("=" * 70)
    print("[1/2] 运行游戏并捕获日志...")
    print("=" * 70)
    
    # 捕获游戏输出
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
        print(f"[错误] 游戏运行错误: {e}", file=old_stdout)
    finally:
        game_output = captured_output.getvalue()
        sys.stdout = old_stdout
        
        # 输出游戏结果
        print(game_output)
        
        print("\n" + "=" * 70)
        print("[2/2] 生成可视化图表...")
        print("=" * 70)
        
        # 生成图表
        generate_visualizations(game_output)
        
        print("\n" + "=" * 70)
        print("[完成] 游戏和可视化全部完成！")
        print("=" * 70)


if __name__ == "__main__":
    main()
