import torch
import os
import sys
import numpy as np
from torch.utils.data import DataLoader, IterableDataset
from data_utils.parser import Mol2StreamParser
from data_utils.tokenizer import MolTokenizer
from models.transformer import MolTransformer
from utils.logger import TrainingLogger
from utils.memory import HDF5FeatureWriter
from torch.cuda.amp import autocast
import h5py
import torch.nn as nn

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class StreamDataset(IterableDataset):
    def __init__(self, tokenizer, parser):
        self.tokenizer = tokenizer
        self.parser = parser
        
    def __iter__(self):
        for block in self.parser.stream_blocks():
            yield self.tokenizer.encode(block)  # 返回张量

def collate_fn(batch):
    """将张量列表堆叠为批次"""
    return torch.stack(batch)

def main():
    # ================== 初始化阶段 ==================
    print("="*50)
    print("初始化分子编码器...")
    
    # 初始化组件
    parser = Mol2StreamParser("E:/APTAMER-GEN/mol2")
    tokenizer = MolTokenizer()
    logger = TrainingLogger()
    
    # ================== 构建词汇表 ==================
    print("\n" + "="*50)
    print("开始构建词汇表")
    tokenizer.build_vocab(parser.stream_blocks())
    if len(tokenizer.vocab) < 10:
        raise ValueError("词汇表构建失败，请检查输入数据")
    
    # ================== 模型初始化 ==================
    print("\n" + "="*50)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用计算设备: {device}")
    
    model = MolTransformer(len(tokenizer.vocab)).to(device)
    if torch.cuda.device_count() > 1:
        print(f"检测到 {torch.cuda.device_count()} 块GPU，启用并行计算")
        model = torch.nn.DataParallel(model)
    
    # ================== 优化器配置 ==================
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(tokenizer.config['training']['learning_rate'])
    )
    criterion = nn.CrossEntropyLoss(
        ignore_index=tokenizer.char2idx['<pad>']
    ).to(device)
    
    # ================== 数据管道 ==================
    print("\n" + "="*50)
    print("初始化数据管道")
    
    # 修复点1：创建数据集实例
    dataset = StreamDataset(tokenizer, parser)
    dataloader = DataLoader(
        dataset,
        batch_size=int(tokenizer.config['training']['batch_size']),
        collate_fn=collate_fn,
        num_workers=8,
        pin_memory=True,
        prefetch_factor=2
    )
    
    # ================== 训练准备 ==================
    # 修复点2：正确的混合精度初始化
    scaler = torch.amp.GradScaler()
    
    # ================== 训练循环 ==================
    print("\n" + "="*50)
    print("开始训练")
    try:
        with HDF5FeatureWriter("features.h5") as feat_writer:
            for epoch in range(int(tokenizer.config['training']['epochs'])):
                model.train()
                total_loss = 0
                
                for batch_idx, batch in enumerate(dataloader):
                    # 数据转移至GPU
                    batch = batch.to(device, non_blocking=True)
                    
                    # 混合精度训练
                    with autocast():
                        outputs = model(batch)
                        loss = criterion(
                            outputs.view(-1, len(tokenizer.vocab)),
                            batch.view(-1)
                        )
                    
                    # 反向传播优化
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
                    
                    # 特征存储
                    with torch.no_grad():
                        # 修复点3：正确获取编码器输出
                        features = model.module.encode(batch) if isinstance(model, nn.DataParallel) else model.encode(batch)
                        features = features.cpu().numpy()
                        if features.size > 0:
                            feat_writer.write_batch(features)
                    
                    total_loss += loss.item()
                    if batch_idx % 100 == 0:
                        print(f"批次 {batch_idx} | 当前损失: {loss.item():.4f}", end='\r')
                
                avg_loss = total_loss / len(dataloader)
                logger.log_metrics(epoch+1, avg_loss)
                print(f"\nEpoch {epoch+1} | 平均损失: {avg_loss:.4f}")
                
    except Exception as e:
        logger.log_error(e)
        raise
    finally:
        # ================== 收尾工作 ==================
        print("\n训练完成，保存最终模型...")
        torch.save(model.state_dict(), "mol_transformer.pth")
        
        # 特征矩阵验证
        if os.path.exists("features.h5"):
            with h5py.File("features.h5", 'r') as f:
                if 'features' in f:
                    print("\n特征矩阵信息:")
                    print(f"样本总数: {f['features'].shape[0]}")
                    print(f"特征维度: {f['features'].shape[1]}")
                else:
                    print("\n警告: 未生成特征矩阵")
        else:
            print("\n错误: 特征文件未创建")

if __name__ == "__main__":
    main()