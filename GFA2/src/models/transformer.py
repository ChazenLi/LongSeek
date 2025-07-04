import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import yaml
import os
from utils.path import get_config_path  # 新增导入


class MolTransformer(nn.Module):
    def __init__(self, vocab_size: int):
        super().__init__()
        self.config = self._load_config()
        
        self.embedding = nn.Embedding(vocab_size, self.config['model']['d_model'])
        encoder_layers = TransformerEncoderLayer(
            d_model=self.config['model']['d_model'],
            nhead=self.config['model']['nhead'],
            dim_feedforward=self.config['model']['dim_feedforward'],
            batch_first=True  # 修复警告
        )
        self.encoder = TransformerEncoder(encoder_layers, self.config['model']['num_layers'])
        self.decoder = nn.Linear(self.config['model']['d_model'], vocab_size)
        
    def _load_config(self):
        config_path = get_config_path()  # 统一路径获取
        with open(config_path) as f:
            return yaml.safe_load(f)
    
    def forward(self, src: torch.Tensor) -> torch.Tensor:
        src_emb = self.embedding(src)
        memory = self.encoder(src_emb)
        return self.decoder(memory)
    
    def encode(self, src: torch.Tensor) -> torch.Tensor:
        return self.encoder(self.embedding(src))