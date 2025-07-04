import os
import requests

def read_pdb_ids(file_path):
    """从文件中读取 PDB ID 列表"""
    with open(file_path, 'r') as file:
        pdb_ids = [line.strip() for line in file.readlines()]
    return pdb_ids

def download_pdb(pdb_id, save_dir):
    """下载指定 PDB ID 的结构文件并保存"""
    pdb_url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
    response = requests.get(pdb_url)
    
    if response.status_code == 200:
        save_path = os.path.join(save_dir, f"{pdb_id}.pdb")
        with open(save_path, "w") as file:
            file.write(response.text)
        print(f"[INFO] Downloaded {pdb_id}.pdb")
    else:
        print(f"[WARNING] Failed to download {pdb_id}.pdb")

def main():
    # 读取 PDB ID 列表
    pdb_ids_file = r"E:\Python\DL\GNNtest\apt_ids_rscb.txt"
    pdb_ids = read_pdb_ids(pdb_ids_file)
    
    # 下载并保存 PDB 文件
    save_dir = r"E:\APTAMER-GEN\pdbdata"
    os.makedirs(save_dir, exist_ok=True)
    
    for pdb_id in pdb_ids:
        download_pdb(pdb_id, save_dir)

if __name__ == "__main__":
    main()
