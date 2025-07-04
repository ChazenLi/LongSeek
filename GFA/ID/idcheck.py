import os

def list_files_without_extension(directory):
    """获取指定目录下的所有文件名（不包括扩展名）"""
    try:
        files = set()
        for filename in os.listdir(directory):
            full_path = os.path.join(directory, filename)
            if os.path.isfile(full_path):
                # 分离文件名和扩展名，并将文件名添加到集合中
                name_without_extension = os.path.splitext(filename)[0]
                files.add(name_without_extension)
        return files
    except FileNotFoundError:
        print(f"目录 {directory} 不存在")
        return set()
    except PermissionError:
        print(f"没有权限访问目录 {directory}")
        return set()

def compare_directories(dir1, dir2):
    """比较两个目录中的文件（不包括扩展名），找出差异"""
    files_dir1 = list_files_without_extension(dir1)
    files_dir2 = list_files_without_extension(dir2)

    # 找出在dir1但不在dir2中的文件
    only_in_dir1 = files_dir1 - files_dir2
    # 找出在dir2但不在dir1中的文件
    only_in_dir2 = files_dir2 - files_dir1

    return only_in_dir1, only_in_dir2

# 定义目录路径
dir_pdbdata = r"E:/APTAMER-GEN/pdbdata"
dir_mol2data = r"E:/APTAMER-GEN/mol2data"

# 比较目录中的文件
only_in_pdbdata, only_in_mol2data = compare_directories(dir_pdbdata, dir_mol2data)

# 输出结果
print("仅在 {} 中的文件:".format(dir_pdbdata))
for file in sorted(only_in_pdbdata):
    print(file)

print("\n仅在 {} 中的文件:".format(dir_mol2data))
for file in sorted(only_in_mol2data):
    print(file)