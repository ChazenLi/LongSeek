import random
from typing import List, Dict, Optional, Tuple, Set
from enum import Enum


class Move(Enum):
    ROCK = "石头"
    SCISSORS = "剪刀"
    PAPER = "布"


class DeclarationType(Enum):
    TRUTH = "真实"
    BLUFF = "虚张声势"
    NONE = "不声明"
    PARTIAL = "模糊声明"


class AllianceStatus(Enum):
    NONE = "无联盟"
    ALLIANCE = "联盟中"
    BETRAYED = "被背叛"
    BETRAYER = "背叛者"


class Agent:
    def __init__(self, id: int, position: tuple):
        self.id = id
        self.position = position
        self.score = 0
        self.alive = True
        self.declared_move: Optional[Move] = None
        self.declaration_type: DeclarationType = DeclarationType.NONE
        self.actual_move: Optional[Move] = None
        
        # 信誉系统
        self.reputation = 1.0  # 信誉值，0-1之间
        self.trustworthiness: Dict[int, float] = {}  # 对对手的信任度
        
        # 联盟系统
        self.alliances: Dict[int, AllianceStatus] = {}
        self.betrayal_cooldown = 0  # 背叛冷却回合
        
        # 历史记录
        self.opponents_history: Dict[int, List[Tuple[Move, DeclarationType]]] = {}
        self.consecutive_losses = 0
        self.consecutive_wins = 0
        
        # 特殊能力
        self.can_peek = False  # 能否偷看
        self.is_protected = False  # 是否受保护
        
        self.strategy_name = ""
        self.personality = ""  # 性格特征

    def get_declaration(self, game_state: Dict) -> tuple:
        """决定声明策略"""
        raise NotImplementedError

    def choose_move(self, game_state: Dict) -> Move:
        """选择出什么"""
        raise NotImplementedError

    def think(self, game_state: Dict) -> str:
        """思考过程"""
        raise NotImplementedError

    def propose_alliance(self, target_id: int, game_state: Dict) -> bool:
        """提议结盟"""
        raise NotImplementedError

    def decide_betrayal(self, game_state: Dict) -> int:
        """决定是否背叛联盟，返回要背叛的目标ID，-1表示不背叛"""
        raise NotImplementedError

    def update_trust(self, opponent_id: int, declared: Optional[Move], actual: Move):
        """更新对对手的信任度"""
        if opponent_id not in self.trustworthiness:
            self.trustworthiness[opponent_id] = 0.5

        if declared is None:
            pass
        elif declared == actual:
            self.trustworthiness[opponent_id] = min(1.0, self.trustworthiness[opponent_id] + 0.15)
        else:
            self.trustworthiness[opponent_id] = max(0.0, self.trustworthiness[opponent_id] - 0.25)

    def record_opponent_move(self, opponent_id: int, move: Move, decl_type: DeclarationType):
        """记录对手的出牌"""
        if opponent_id not in self.opponents_history:
            self.opponents_history[opponent_id] = []
        self.opponents_history[opponent_id].append((move, decl_type))


class DiplomaticAgent(Agent):
    """外交型：擅长结盟，但也有可能背叛"""

    def __init__(self, id: int, position: tuple):
        super().__init__(id, position)
        self.strategy_name = "外交型"
        self.personality = "圆滑善变"

    def get_declaration(self, game_state: Dict) -> tuple:
        move = self.choose_move(game_state)
        
        # 对盟友说真话，对其他人虚张声势
        has_alliance = any(status == AllianceStatus.ALLIANCE for status in self.alliances.values())
        
        if has_alliance:
            self.declaration_type = DeclarationType.TRUTH
            self.declared_move = move
            return DeclarationType.TRUTH, move
        else:
            self.declaration_type = DeclarationType.BLUFF
            fake_moves = [m for m in Move if m != move]
            fake_move = random.choice(fake_moves)
            self.declared_move = fake_move
            return DeclarationType.BLUFF, fake_move

    def choose_move(self, game_state: Dict) -> Move:
        move_counts = {Move.ROCK: 0.0, Move.SCISSORS: 0.0, Move.PAPER: 0.0}
        
        # 对盟友的对手分析减少权重
        for opp_id in self.trustworthiness:
            if opp_id in self.alliances and self.alliances[opp_id] == AllianceStatus.ALLIANCE:
                continue  # 不针对盟友
            if opp_id in self.opponents_history and len(self.opponents_history[opp_id]) >= 2:
                last_moves = [m[0] for m in self.opponents_history[opp_id][-3:]]
                for m in last_moves:
                    move_counts[m] += 0.4
        
        beat_counts = {
            Move.ROCK: move_counts[Move.SCISSORS],
            Move.SCISSORS: move_counts[Move.PAPER],
            Move.PAPER: move_counts[Move.ROCK]
        }
        
        max_count = max(beat_counts.values())
        if max_count == 0:
            return random.choice(list(Move))
        candidates = [k for k, v in beat_counts.items() if v == max_count]
        return random.choice(candidates)

    def propose_alliance(self, target_id: int, game_state: Dict) -> bool:
        # 只与分数高的agent结盟
        alive_agents = game_state.get('alive_agents', [])
        target_agent = next((a for a in alive_agents if a.id == target_id), None)
        
        if target_agent and target_agent.score >= 3:
            return True
        return False

    def decide_betrayal(self, game_state: Dict) -> int:
        # 如果盟友分数比自己高很多，可能背叛
        for opp_id, status in self.alliances.items():
            if status == AllianceStatus.ALLIANCE:
                alive_agents = game_state.get('alive_agents', [])
                ally = next((a for a in alive_agents if a.id == opp_id), None)
                if ally and ally.score > self.score + 5:
                    return opp_id
        return -1

    def think(self, game_state: Dict) -> str:
        thought = f"我是Agent {self.id}，{self.personality}的{self.strategy_name}。\n"
        thought += f"我的信誉：{self.reputation:.2f}\n"
        thought += f"我的联盟：{self.alliances}\n"
        
        if self.declaration_type == DeclarationType.TRUTH:
            thought += f"声明：{self.declared_move}（真实，对盟友）\n"
        elif self.declaration_type == DeclarationType.BLUFF:
            thought += f"声明：{self.declared_move}（虚张声势，对敌人）\n"
        else:
            thought += "声明：不声明\n"
        
        thought += f"实际出：{self.actual_move}\n"
        
        # 计算背叛倾向
        allies = [k for k, v in self.alliances.items() if v == AllianceStatus.ALLIANCE]
        if allies:
            thought += f"我当前有{len(allies)}个盟友。"
            if random.random() < 0.3:
                thought += "在考虑是否背叛..."
        else:
            thought += "我正在寻找结盟对象..."
        
        return thought


class VengefulAgent(Agent):
    """复仇型：记仇，一旦被背叛就会报复"""

    def __init__(self, id: int, position: tuple):
        super().__init__(id, position)
        self.strategy_name = "复仇型"
        self.personality = "睚眦必报"
        self.enemies: Set[int] = set()

    def get_declaration(self, game_state: Dict) -> tuple:
        move = self.choose_move(game_state)
        
        # 对敌人虚张声势
        if self.enemies:
            self.declaration_type = DeclarationType.BLUFF
            fake_moves = [m for m in Move if m != move]
            fake_move = random.choice(fake_moves)
            self.declared_move = fake_move
            return DeclarationType.BLUFF, fake_move
        else:
            self.declaration_type = DeclarationType.TRUTH
            self.declared_move = move
            return DeclarationType.TRUTH, move

    def choose_move(self, game_state: Dict) -> Move:
        move_counts = {Move.ROCK: 0.0, Move.SCISSORS: 0.0, Move.PAPER: 0.0}
        
        # 专门针对敌人
        for enemy_id in self.enemies:
            if enemy_id in self.opponents_history:
                last_moves = [m[0] for m in self.opponents_history[enemy_id][-3:]]
                for m in last_moves:
                    move_counts[m] += 0.8  # 给敌人更高权重
        
        # 也考虑其他对手
        for opp_id in self.trustworthiness:
            if opp_id not in self.enemies and opp_id in self.opponents_history:
                last_moves = [m[0] for m in self.opponents_history[opp_id][-2:]]
                for m in last_moves:
                    move_counts[m] += 0.3
        
        beat_counts = {
            Move.ROCK: move_counts[Move.SCISSORS],
            Move.SCISSORS: move_counts[Move.PAPER],
            Move.PAPER: move_counts[Move.ROCK]
        }
        
        max_count = max(beat_counts.values())
        if max_count == 0:
            return random.choice(list(Move))
        candidates = [k for k, v in beat_counts.items() if v == max_count]
        return random.choice(candidates)

    def propose_alliance(self, target_id: int, game_state: Dict) -> bool:
        # 拒绝与敌人结盟
        if target_id in self.enemies:
            return False
        return random.random() < 0.3  # 30%概率接受

    def decide_betrayal(self, game_state: Dict) -> int:
        # 复仇型不会主动背叛
        return -1

    def think(self, game_state: Dict) -> str:
        thought = f"我是Agent {self.id}，{self.personality}的{self.strategy_name}。\n"
        thought += f"我的信誉：{self.reputation:.2f}\n"
        thought += f"我的敌人：{list(self.enemies)}\n"
        
        if self.enemies:
            thought += "我正在策划复仇！\n"
            thought += f"声明：{self.declared_move}（虚张声势迷惑敌人）\n"
        else:
            thought += "目前没有敌人，保持诚实。\n"
            thought += f"声明：{self.declared_move}（真实）\n"
        
        thought += f"实际出：{self.actual_move}"
        return thought


class OpportunistAgent(Agent):
    """机会主义型：没有固定原则，见风使舵"""

    def __init__(self, id: int, position: tuple):
        super().__init__(id, position)
        self.strategy_name = "机会主义型"
        self.personality = "见风使舵"

    def get_declaration(self, game_state: Dict) -> tuple:
        move = self.choose_move(game_state)
        
        # 根据当前局势调整策略
        alive_agents = game_state.get('alive_agents', [])
        avg_score = sum(a.score for a in alive_agents) / len(alive_agents) if alive_agents else 0
        
        if self.score >= avg_score:
            # 领先时说真话建立信誉
            self.declaration_type = DeclarationType.TRUTH
            self.declared_move = move
            return DeclarationType.TRUTH, move
        elif self.score < avg_score - 2:
            # 落后严重时虚张声势
            self.declaration_type = DeclarationType.BLUFF
            fake_moves = [m for m in Move if m != move]
            fake_move = random.choice(fake_moves)
            self.declared_move = fake_move
            return DeclarationType.BLUFF, fake_move
        else:
            # 保持神秘
            self.declaration_type = DeclarationType.NONE
            self.declared_move = None
            return DeclarationType.NONE, None

    def choose_move(self, game_state: Dict) -> Move:
        move_counts = {Move.ROCK: 0.0, Move.SCISSORS: 0.0, Move.PAPER: 0.0}
        
        for opp_id in self.trustworthiness:
            if opp_id in self.opponents_history and len(self.opponents_history[opp_id]) >= 2:
                last_moves = [m[0] for m in self.opponents_history[opp_id][-2:]]
                for m in last_moves:
                    move_counts[m] += 0.4
        
        beat_counts = {
            Move.ROCK: move_counts[Move.SCISSORS],
            Move.SCISSORS: move_counts[Move.PAPER],
            Move.PAPER: move_counts[Move.ROCK]
        }
        
        max_count = max(beat_counts.values())
        if max_count == 0:
            return random.choice(list(Move))
        candidates = [k for k, v in beat_counts.items() if v == max_count]
        return random.choice(candidates)

    def propose_alliance(self, target_id: int, game_state: Dict) -> bool:
        # 只与当前分数最高者结盟
        alive_agents = game_state.get('alive_agents', [])
        if alive_agents:
            max_score = max(a.score for a in alive_agents)
            target_agent = next((a for a in alive_agents if a.id == target_id), None)
            if target_agent and target_agent.score >= max_score:
                return True
        return False

    def decide_betrayal(self, game_state: Dict) -> int:
        # 随机背叛
        if random.random() < 0.2:  # 20%概率背叛
            allies = [k for k, v in self.alliances.items() if v == AllianceStatus.ALLIANCE]
            if allies:
                return random.choice(allies)
        return -1

    def think(self, game_state: Dict) -> str:
        thought = f"我是Agent {self.id}，{self.personality}的{self.strategy_name}。\n"
        thought += f"我的信誉：{self.reputation:.2f}\n"
        
        alive_agents = game_state.get('alive_agents', [])
        avg_score = sum(a.score for a in alive_agents) / len(alive_agents) if alive_agents else 0
        
        if self.declaration_type == DeclarationType.TRUTH:
            thought += f"声明：{self.declared_move}（真实，领先时建立信誉）\n"
        elif self.declaration_type == DeclarationType.BLUFF:
            thought += f"声明：{self.declared_move}（虚张声势，落后时搏一搏）\n"
        else:
            thought += "声明：不声明（保持神秘）\n"
        
        thought += f"实际出：{self.actual_move}\n"
        thought += f"平均分数：{avg_score:.1f}，我的分数：{self.score}"
        return thought


class ManipulatorAgent(Agent):
    """操纵型：喜欢操纵其他agent的决定"""

    def __init__(self, id: int, position: tuple):
        super().__init__(id, position)
        self.strategy_name = "操纵型"
        self.personality = "心机深沉"
        self.manipulation_targets: Dict[int, int] = {}  # 操纵目标及次数

    def get_declaration(self, game_state: Dict) -> tuple:
        move = self.choose_move(game_state)
        
        # 分析对手信任度，针对性声明
        avg_trust = sum(self.trustworthiness.values()) / len(self.trustworthiness) if self.trustworthiness else 0.5
        
        if avg_trust > 0.7:
            # 对手信任度高时，说假话降低其信任
            self.declaration_type = DeclarationType.BLUFF
            fake_moves = [m for m in Move if m != move]
            fake_move = random.choice(fake_moves)
            self.declared_move = fake_move
            return DeclarationType.BLUFF, fake_move
        elif avg_trust < 0.3:
            # 对手不信任时，突然说真话重建信任
            self.declaration_type = DeclarationType.TRUTH
            self.declared_move = move
            return DeclarationType.TRUTH, move
        else:
            # 模糊声明，让对手猜不透
            self.declaration_type = DeclarationType.PARTIAL
            fake_moves = [m for m in Move if m != move]
            fake_move = random.choice(fake_moves)
            self.declared_move = fake_move
            return DeclarationType.PARTIAL, fake_move

    def choose_move(self, game_state: Dict) -> Move:
        move_counts = {Move.ROCK: 0.0, Move.SCISSORS: 0.0, Move.PAPER: 0.0}
        
        # 详细分析对手
        for opp_id in self.trustworthiness:
            trust = self.trustworthiness[opp_id]
            
            # 对手信任我，可能会相信我的声明
            if opp_id in self.opponents_history:
                last_moves = [m[0] for m in self.opponents_history[opp_id][-3:]]
                for m in last_moves:
                    move_counts[m] += trust * 0.5
        
        beat_counts = {
            Move.ROCK: move_counts[Move.SCISSORS],
            Move.SCISSORS: move_counts[Move.PAPER],
            Move.PAPER: move_counts[Move.ROCK]
        }
        
        max_count = max(beat_counts.values())
        if max_count == 0:
            return random.choice(list(Move))
        candidates = [k for k, v in beat_counts.items() if v == max_count]
        return random.choice(candidates)

    def propose_alliance(self, target_id: int, game_state: Dict) -> bool:
        # 与容易被操纵的agent结盟
        if target_id in self.trustworthiness:
            # 信任度在0.3-0.7之间的agent容易被操纵
            if 0.3 <= self.trustworthiness[target_id] <= 0.7:
                return True
        return random.random() < 0.2

    def decide_betrayal(self, game_state: Dict) -> int:
        # 精心计算背叛时机
        for opp_id, status in self.alliances.items():
            if status == AllianceStatus.ALLIANCE:
                alive_agents = game_state.get('alive_agents', [])
                ally = next((a for a in alive_agents if a.id == opp_id), None)
                if ally and ally.score >= 5:
                    # 盟友接近胜利时背叛
                    return opp_id
        return -1

    def think(self, game_state: Dict) -> str:
        thought = f"我是Agent {self.id}，{self.personality}的{self.strategy_name}。\n"
        thought += f"我的信誉：{self.reputation:.2f}\n"
        
        if self.trustworthiness:
            avg_trust = sum(self.trustworthiness.values()) / len(self.trustworthiness)
            thought += f"对手平均信任度：{avg_trust:.2f}\n"
            thought += f"我在操纵{len(self.trustworthiness)}个对手...\n"
        
        if self.declaration_type == DeclarationType.TRUTH:
            thought += f"声明：{self.declared_move}（突然说真话重建信任）\n"
        elif self.declaration_type == DeclarationType.BLUFF:
            thought += f"声明：{self.declared_move}（说假话降低对手信任）\n"
        else:
            thought += f"声明：{self.declared_move}（模糊声明，让对手猜不透）\n"
        
        thought += f"实际出：{self.actual_move}"
        return thought


class Game:
    def __init__(self):
        self.agents: List[Agent] = []
        self.round = 0
        self.alliance_log = []
        self.betrayal_log = []

    def get_neighbors(self, agent: Agent) -> List[Agent]:
        neighbors = []
        x, y = agent.position
        directions = [(0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)]

        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            for other in self.agents:
                if other.alive and other.position == (nx, ny) and other.id != agent.id:
                    neighbors.append(other)

        return neighbors

    def rps_result(self, move1: Move, move2: Move) -> int:
        if move1 == move2:
            return 0
        if (move1 == Move.ROCK and move2 == Move.SCISSORS) or \
           (move1 == Move.SCISSORS and move2 == Move.PAPER) or \
           (move1 == Move.PAPER and move2 == Move.ROCK):
            return 1
        return -1

    def alliance_phase(self, alive_agents: List[Agent]):
        """联盟阶段"""
        print("\n【联盟阶段】")
        game_state = {'alive_agents': alive_agents}
        
        # 随机选择一对agent进行结盟提议
        if len(alive_agents) >= 2:
            proposer, target = random.sample(alive_agents, 2)
            
            if target.id not in proposer.alliances and proposer.id not in target.alliances:
                if proposer.propose_alliance(target.id, game_state):
                    # 检查目标是否接受
                    if target.propose_alliance(proposer.id, game_state):
                        proposer.alliances[target.id] = AllianceStatus.ALLIANCE
                        target.alliances[proposer.id] = AllianceStatus.ALLIANCE
                        print(f"  Agent {proposer.id} 和 Agent {target.id} 结成联盟！")
                        self.alliance_log.append((self.round, proposer.id, target.id))
                    else:
                        print(f"  Agent {proposer.id} 提议与 Agent {target.id} 结盟，被拒绝")

    def betrayal_phase(self, alive_agents: List[Agent]):
        """背叛阶段"""
        print("\n【背叛阶段】")
        game_state = {'alive_agents': alive_agents}
        
        for agent in alive_agents:
            if agent.betrayal_cooldown > 0:
                agent.betrayal_cooldown -= 1
                continue
            
            betray_target = agent.decide_betrayal(game_state)
            
            if betray_target != -1 and betray_target in agent.alliances:
                target_agent = next((a for a in alive_agents if a.id == betray_target), None)
                if target_agent:
                    # 执行背叛
                    agent.alliances[betray_target] = AllianceStatus.BETRAYER
                    target_agent.alliances[agent.id] = AllianceStatus.BETRAYED
                    
                    # 背叛惩罚和奖励
                    agent.score += 1
                    agent.reputation -= 0.3
                    agent.betrayal_cooldown = 3
                    
                    target_agent.score -= 2
                    target_agent.reputation += 0.1
                    
                    # 复仇型agent会将背叛者加入敌人列表
                    if isinstance(target_agent, VengefulAgent):
                        target_agent.enemies.add(agent.id)
                    
                    print(f"  Agent {agent.id} 背叛了 Agent {betray_target}！")
                    print(f"    Agent {agent.id} 获得+1分，信誉-0.3")
                    print(f"    Agent {betray_target} 失去-2分，信誉+0.1")
                    self.betrayal_log.append((self.round, agent.id, betray_target))

    def play_round(self):
        self.round += 1
        print(f"\n{'='*70}")
        print(f"第 {self.round} 回合")
        print(f"{'='*70}")

        alive_agents = [a for a in self.agents if a.alive]
        if len(alive_agents) <= 1:
            return False

        if self.check_winner():
            return False

        # 联盟阶段
        self.alliance_phase(alive_agents)
        
        # 背叛阶段
        self.betrayal_phase(alive_agents)

        # 声明与决策阶段
        print("\n【声明与决策阶段】")
        game_state = {'alive_agents': alive_agents}
        declarations = {}
        thoughts = {}
        actual_moves = {}

        for agent in alive_agents:
            decl_type, decl_move = agent.get_declaration(game_state)
            actual_move = agent.choose_move(game_state)
            agent.actual_move = actual_move
            
            declarations[agent.id] = (decl_type, decl_move)
            thoughts[agent.id] = agent.think(game_state)
            actual_moves[agent.id] = actual_move

        for agent_id, thought in thoughts.items():
            agent = self.agents[agent_id]
            print(f"\nAgent {agent.id} ({agent.strategy_name} - {agent.personality}):")
            print(f"  位置: {agent.position}")
            print(f"  分数: {agent.score}, 信誉: {agent.reputation:.2f}")
            print(f"  {thought}")

        # 对决阶段
        print("\n【对决阶段】")
        battle_results = {}

        for agent in alive_agents:
            neighbors = self.get_neighbors(agent)
            for neighbor in neighbors:
                battle_key = tuple(sorted((agent.id, neighbor.id)))
                if battle_key not in battle_results:
                    move1 = actual_moves[agent.id]
                    move2 = actual_moves[neighbor.id]

                    # 检查是否是盟友
                    is_alliance = (neighbor.id in agent.alliances and 
                                 agent.alliances[neighbor.id] == AllianceStatus.ALLIANCE)
                    
                    result = self.rps_result(move1, move2)

                    if is_alliance:
                        # 盟友对战，平局
                        battle_results[battle_key] = (None, agent.id, move1, move2)
                    elif result == 1:
                        agent.score += 2
                        neighbor.score -= 1
                        agent.consecutive_losses = 0
                        neighbor.consecutive_losses += 1
                        battle_results[battle_key] = (agent.id, neighbor.id, move1, move2)
                    elif result == -1:
                        agent.score -= 1
                        neighbor.score += 2
                        agent.consecutive_losses += 1
                        neighbor.consecutive_losses = 0
                        battle_results[battle_key] = (neighbor.id, agent.id, move2, move1)
                    else:
                        battle_results[battle_key] = (None, agent.id, move1, move2)

        print("\n【对战结果】")
        for battle_key, result in battle_results.items():
            winner_id, loser_id, move_winner, move_loser = result
            agent1_id, agent2_id = battle_key
            
            # 检查是否是盟友
            agent1 = self.agents[agent1_id]
            agent2 = self.agents[agent2_id]
            is_alliance = (agent2_id in agent1.alliances and 
                          agent1.alliances[agent2_id] == AllianceStatus.ALLIANCE)
            
            if is_alliance:
                print(f"  Agent {agent1_id}({move_winner}) vs Agent {agent2_id}({move_loser}): 盟友对决，和平共处")
            elif winner_id is not None:
                print(f"  Agent {winner_id}({move_winner}) vs Agent {loser_id}({move_loser}): Agent {winner_id} 获胜!")
            else:
                print(f"  Agent {agent1_id}({move_winner}) vs Agent {agent2_id}({move_loser}): 平局")

        # 更新信任和信誉
        print("\n【信任与信誉更新】")
        for battle_key, result in battle_results.items():
            winner_id, loser_id, move_winner, move_loser = result
            agent1_id, agent2_id = battle_key
            
            if winner_id is not None:
                winner = self.agents[winner_id]
                loser = self.agents[loser_id]
                
                # 赢家获得信誉提升
                winner.reputation = min(1.0, winner.reputation + 0.05)
                
                # 更新信任度
                winner.update_trust(loser_id, loser.declared_move, move_loser)
                loser.update_trust(winner_id, winner.declared_move, move_winner)
                winner.record_opponent_move(loser_id, move_loser, loser.declaration_type)
                loser.record_opponent_move(winner_id, move_winner, winner.declaration_type)

        print("\n【当前状态】")
        for agent in alive_agents:
            alliances = [k for k, v in agent.alliances.items() if v == AllianceStatus.ALLIANCE]
            status = f"分数={agent.score}, 信誉={agent.reputation:.2f}"
            if alliances:
                status += f", 盟友={alliances}"
            print(f"  Agent {agent.id}: {status}")

        self.check_elimination()

        return True

    def check_winner(self) -> bool:
        alive_agents = [a for a in self.agents if a.alive]
        for agent in alive_agents:
            if agent.score >= 10:
                return True
        return False

    def check_elimination(self):
        alive_agents = [a for a in self.agents if a.alive]
        if len(alive_agents) <= 1:
            return

        eliminated = []

        for agent in alive_agents:
            if agent.consecutive_losses >= 4 or agent.score <= -12:
                agent.alive = False
                eliminated.append(agent.id)

        if eliminated:
            print(f"\n【淘汰】Agent {eliminated} 被淘汰！")

    def print_final_result(self):
        print(f"\n{'='*70}")
        print("游戏结束！")
        print(f"{'='*70}")

        alive_agents = [a for a in self.agents if a.alive]
        winners = [a for a in alive_agents if a.score >= 10]

        print("\n【联盟历史】")
        for round_num, agent1, agent2 in self.alliance_log:
            print(f"  第{round_num}回合: Agent {agent1} <-> Agent {agent2}")

        print("\n【背叛记录】")
        for round_num, betrayer, victim in self.betrayal_log:
            print(f"  第{round_num}回合: Agent {betrayer} 背叛了 Agent {victim}")

        if winners:
            winner = winners[0]
            print(f"\n[冠军] 最终胜利者: Agent {winner.id}")
            print(f"   策略: {winner.strategy_name}")
            print(f"   性格: {winner.personality}")
            print(f"   最终分数: {winner.score}")
            print(f"   最终信誉: {winner.reputation:.2f}")
            print(f"   位置: {winner.position}")
            print(f"   获胜回合数: {self.round}")
        elif alive_agents:
            winner = alive_agents[0]
            print(f"\n[冠军] 最终胜利者: Agent {winner.id}")
            print(f"   策略: {winner.strategy_name}")
            print(f"   性格: {winner.personality}")
            print(f"   最终分数: {winner.score}")
            print(f"   最终信誉: {winner.reputation:.2f}")

        print("\n【最终排名】")
        ranked = sorted([a for a in self.agents], key=lambda x: (-x.score, x.id))
        for i, agent in enumerate(ranked, 1):
            status = "存活" if agent.alive else "淘汰"
            print(f"  第{i}名: Agent {agent.id} ({agent.strategy_name}) - 分数: {agent.score}, 信誉: {agent.reputation:.2f} ({status})")

    def run(self):
        print("游戏开始！2x2格子，4个agent进行勾心斗角的石头剪刀布")
        print("新增特性：")
        print("  - 联盟系统：可以结成临时联盟，盟友对战时和平共处")
        print("  - 背叛机制：可以背叛盟友获得分数，但会降低信誉")
        print("  - 信誉系统：信誉高的agent更容易获得信任")
        print("  - 获胜+2分，失败-1分，平局不扣分")
        print("  - 最先达到10分的agent获胜")

        while self.play_round():
            pass

        self.print_final_result()


def main():
    game = Game()

    game.agents = [
        DiplomaticAgent(0, (0, 0)),
        VengefulAgent(1, (0, 1)),
        OpportunistAgent(2, (1, 0)),
        ManipulatorAgent(3, (1, 1))
    ]

    game.run()


if __name__ == "__main__":
    main()
