from collections import defaultdict
import string
import yaml
from typing import Generator
import os
from utils.path import get_config_path  # 新增导入
import torch
from collections import defaultdict
from typing import Generator
import os
import string
import yaml

class MolTokenizer:
    def __init__(self):
        self.config = self._load_config()
        self.special_tokens = ['<pad>', '<sos>', '<eos>', '<unk>']
        self.vocab = self.special_tokens.copy()  # 初始化基础词汇
        self.char2idx = {}
        self.idx2char = {}

    def _load_config(self):
        config_path = os.path.join(
            os.path.dirname(__file__), 
            '..', '..', 'configs', 'params.yaml'
        )
        with open(config_path) as f:
            return yaml.safe_load(f)

    def build_vocab(self, block_stream: Generator[str, None, None]) -> None:
        """带进度显示的增强版词汇表构建"""
        char_counts = defaultdict(int)
        total_blocks = 0
        
        print("\n[1/3] 扫描字符分布...")
        for block in block_stream:
            total_blocks += 1
            for char in block:
                char_counts[char] += 1
            if total_blocks % 1000 == 0:
                print(f"已处理 {total_blocks} 个分子块", end='\r')

        print(f"\n[2/3] 完成扫描，共发现 {len(char_counts)} 种字符")
        min_freq = self.config['data']['min_char_freq']
        valid_chars = [
            c for c, cnt in char_counts.items()
            if cnt >= min_freq or c in string.whitespace
        ]
        
        self.vocab += sorted(valid_chars)
        self.char2idx = {c:i for i,c in enumerate(self.vocab)}
        self.idx2char = {i:c for i,c in enumerate(self.vocab)}
        print(f"[3/3] 最终词汇表大小: {len(self.vocab)}")
        
    def encode(self, text: str) -> torch.Tensor:  # 修改返回类型
        encoded = [self.char2idx['<sos>']]
        for c in text.replace('\n', '↨'):
            encoded.append(self.char2idx.get(c, self.char2idx['<unk>']))
        encoded.append(self.char2idx['<eos>'])
    
        max_len = self.config['training']['max_seq_len']
        if len(encoded) > max_len:
            encoded = encoded[:max_len-1] + [self.char2idx['<eos>']]
        else:
            encoded += [self.char2idx['<pad>']] * (max_len - len(encoded))
    
        return torch.LongTensor(encoded)  # 直接返回张量

    def decode(self, tokens: list[int]) -> str:
        return ''.join(
            self.idx2char.get(idx, '<unk>') for idx in tokens
        ).replace('↨', '\n').replace('<sos>', '').replace('<eos>', '')