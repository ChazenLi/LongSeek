import os

def split_mol2(input_file, output_dir):
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 生成基础文件名（不带扩展名）
    base_name = os.path.basename(input_file).replace('.mol2', '')
    
    # 读取原始文件内容
    with open(input_file, 'r') as f:
        lines = f.readlines()
    
    # 分割分子块
    molecules = []
    current_mol = []
    in_molecule = False
    
    for line in lines:
        if line.startswith("@<TRIPOS>MOLECULE"):
            if in_molecule:  # 结束前一个分子
                molecules.append(current_mol)
                current_mol = []
            in_molecule = True
        current_mol.append(line)
    
    if current_mol:  # 添加最后一个分子
        molecules.append(current_mol)
    
    # 写入分割后的文件
    for i, mol in enumerate(molecules, 1):
        output_path = os.path.join(
            output_dir,
            f"{base_name}-{i}.mol2"
        )
        with open(output_path, 'w') as f:
            f.writelines(mol)
    
    return len(molecules)

def process_directory(input_dir, output_dir):
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 遍历输入目录中的所有文件
    for file_name in os.listdir(input_dir):
        if file_name.endswith('.mol2'):
            input_file = os.path.join(input_dir, file_name)
            split_mol2(input_file, output_dir)

# 使用示例
if __name__ == "__main__":
    input_dir = "E:\\APTAMER-GEN\\mol2data"  # 原始文件目录
    output_dir = "E:\\APTAMER-GEN\\mol2"     # 输出目录
    
    process_directory(input_dir, output_dir)
    print(f"成功处理目录 {input_dir} 中的所有 .mol2 文件，并将结果存储到 {output_dir}")