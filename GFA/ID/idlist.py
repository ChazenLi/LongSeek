import os

def list_files_in_directory(directory):
    # 获取目录下的所有文件和文件夹
    files = [filename for filename in os.listdir(directory) if os.path.isfile(os.path.join(directory, filename))]
    
    # 遍历文件列表并输出文件名（不包括扩展名）
    for filename in files:
        # 分离文件名和扩展名
        name_without_extension = os.path.splitext(filename)[0]
        
        # 打印文件名
        print(name_without_extension)

# 调用函数并传入目标目录
directory_path = r"E:/APTAMER-GEN/pdbdata"
list_files_in_directory(directory_path)