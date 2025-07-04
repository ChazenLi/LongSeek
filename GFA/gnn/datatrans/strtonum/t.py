import os
import re
from collections import defaultdict

def process_mol2_folder(input_folder, output_file):
    # 初始化全局统计字典
    atom_stats = {
        'atom_name': defaultdict(int),
        'atom_type': defaultdict(int),
        'subst_name': defaultdict(int)
    }
    bond_stats = defaultdict(int)

    # 遍历文件夹处理所有mol2文件
    for filename in os.listdir(input_folder):
        if not filename.lower().endswith('.mol2'):
            continue
            
        filepath = os.path.join(input_folder, filename)
        
        with open(filepath, 'r') as f:
            current_section = None
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
                    if len(parts) >= 8:
                        # 原子名称统计
                        atom_stats['atom_name'][parts[1]] += 1
                        # 原子类型统计
                        atom_stats['atom_type'][parts[5]] += 1
                        # 子结构名称统计（去除数字）
                        subst = re.sub(r'\d+', '', parts[7])  # 关键修改点[5]
                        atom_stats['subst_name'][subst] += 1

                elif current_section == 'BOND':
                    parts = line.split()
                    if len(parts) >= 4:
                        bond_type = parts[3].lower()  # 统一转为小写
                        bond_stats[bond_type] += 1

    # 写入汇总结果
    with open(output_file, 'w') as f:
        f.write("=== Atom Name Statistics ===\n")
        for name, count in sorted(atom_stats['atom_name'].items()):
            f.write(f"{name}: {count}\n")

        f.write("\n=== Atom Type Statistics ===\n")
        for type_, count in sorted(atom_stats['atom_type'].items()):
            f.write(f"{type_}: {count}\n")

        f.write("\n=== Substructure Name Statistics (letters only) ===\n")
        for subst, count in sorted(atom_stats['subst_name'].items()):
            f.write(f"{subst}: {count}\n")

        f.write("\n=== Bond Type Statistics ===\n")
        for bond, count in sorted(bond_stats.items()):
            f.write(f"{bond}: {count}\n")


# 使用示例
process_mol2_folder(
    input_folder='E:/APTAMER-GEN/mol2data',
    output_file='E:/APTAMER-GEN/character/combined_aggregate_stats.txt'
)

