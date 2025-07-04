import os
from collections import defaultdict

def process_folder(input_folder, output_dir):
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 遍历文件夹获取所有mol2文件 [4][6]
    for filename in os.listdir(input_folder):
        if not filename.lower().endswith('.mol2'):
            continue
            
        input_path = os.path.join(input_folder, filename)
        output_path = os.path.join(output_dir, f"{os.path.splitext(filename)[0]}_stats.txt")
        
        # 处理单个文件
        process_single_file(input_path, output_path)

def process_single_file(input_path, output_path):
    # 初始化数据结构
    atom_counts = {
        'atom_name': defaultdict(int),
        'atom_type': defaultdict(int),
        'subst_name': defaultdict(int)
    }
    bond_counts = defaultdict(int)
    
    current_section = None
    
    with open(input_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('@<TRIPOS>ATOM'):
                current_section = 'ATOM'
                continue
            elif line.startswith('@<TRIPOS>BOND'):
                current_section = 'BOND'
                continue
            elif line.startswith('@<TRIPOS>'):
                current_section = None
                continue

            if current_section == 'ATOM':
                parts = line.split()
                if len(parts) >= 9:
                    atom_counts['atom_name'][parts[1]] += 1
                    atom_counts['atom_type'][parts[5]] += 1
                    atom_counts['subst_name'][parts[7]] += 1
            elif current_section == 'BOND':
                parts = line.split()
                if len(parts) >= 4:
                    bond_counts[parts[3]] += 1

    # 写入统计结果 [4]
    with open(output_path, 'w') as f:
        f.write("Atom Name Counts:\n")
        for name, count in atom_counts['atom_name'].items():
            f.write(f"{name}: {count}\n")
        
        f.write("\nAtom Type Counts:\n")
        for type_, count in atom_counts['atom_type'].items():
            f.write(f"{type_}: {count}\n")
        
        f.write("\nSubstructure Counts:\n")
        for subst, count in atom_counts['subst_name'].items():
            f.write(f"{subst}: {count}\n")
        
        f.write("\nBond Type Counts:\n")
        for bond, count in bond_counts.items():
            f.write(f"{bond}: {count}\n")

# 使用示例
process_folder(
    input_folder="E:/APTAMER-GEN/mol2data",
    output_dir="E:/APTAMER-GEN/character"
)

