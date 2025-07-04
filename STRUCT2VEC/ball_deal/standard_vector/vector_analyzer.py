import os
import sys
import argparse

# 添加父目录到系统路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 导入向量分析工具
from standard_vector.vector_utils import MoleculeVectorAnalyzer, analyze_vectors


def main():
    """主函数，处理命令行参数并启动向量分析"""
    parser = argparse.ArgumentParser(description='分子结构向量分析工具')
    parser.add_argument('--mol2', type=str, help='mol2文件路径')
    parser.add_argument('--interactive', action='store_true', help='是否使用交互模式')
    parser.add_argument('--substruct', type=int, help='要分析的子结构索引（从0开始）')
    parser.add_argument('--vector-type', type=str, default='all', 
                        choices=['all', 'backbone', 'non_backbone', 'mixed'],
                        help='要分析的向量类型')
    parser.add_argument('--operation', type=str, 
                        choices=['magnitude', 'similarity', 'visualize'],
                        help='要执行的操作')
    args = parser.parse_args()
    
    # 如果提供了mol2文件路径，则使用该路径
    if args.mol2 and os.path.exists(args.mol2):
        mol2_file = args.mol2
    else:
        # 否则提示用户输入
        mol2_file = input("请输入mol2文件路径: ")
        if not os.path.exists(mol2_file):
            print(f"文件 {mol2_file} 不存在")
            return
    
    # 启动向量分析
    analyze_vectors(
        mol2_file, 
        interactive=args.interactive, 
        substruct_idx=args.substruct,
        vector_type=args.vector_type,
        operation=args.operation
    )


if __name__ == "__main__":
    main()