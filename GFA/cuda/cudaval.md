import torch

# 检查 CUDA 是否可用
cuda_available = torch.export.cuda.is_available()
print("CUDA available:", cuda_available)  # 应输出 True

if cuda_available:
    # 查看当前 CUDA 设备信息
    print("Current CUDA device:", torch.cuda.current_device())  # 默认 GPU 索引（通常为 0）
    print("CUDA device name:", torch.cuda.get_device_name(0))  # GPU 型号（如 "NVIDIA GeForce RTX 3090"）

    # 测试 CUDA 加速
    x = torch.randn(3, 3).cuda()
    print("Tensor device:", x.device)  # 应输出 "cuda:0"

# 查看 PyTorch 版本
print("PyTorch version:", torch.__version__)  # 查看版本是否包含 "cuXXX"（如 "2.1.0+cu121" 表示 CUDA 12.1")