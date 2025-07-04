import os
import requests

def fetch_pdb_ids(query="aptamer"):
    """查询 RCSB PDB 数据库，获取包含指定查询词的 PDB ID 列表"""
    url = "https://search.rcsb.org/rcsbsearch/v2/query"
    query_object = {
        "query": {
            "type": "terminal",
            "service": "full_text",
            "parameters": {
                "value": query
            }
        },
        "request_options": {
            "return_all_hits": True
        },
        "return_type": "entry"
    }
    
    response = requests.post(url, json=query_object)
    if response.status_code == 200:
        data = response.json()
        pdb_ids = [entry["identifier"] for entry in data.get("result_set", [])]
        return pdb_ids
    else:
        print(f"[ERROR] Failed to fetch PDB IDs. Status code: {response.status_code}")
        return []

def save_pdb_ids_to_file(pdb_ids, file_path):
    """将 PDB ID 列表保存到文件"""
    with open(file_path, 'w') as file:
        for pdb_id in pdb_ids:
            file.write(f"{pdb_id}\n")
    print(f"Saved PDB IDs to {file_path}")

# 获取包含 "aptamer" 的 PDB ID 列表
pdb_ids = fetch_pdb_ids("aptamer")

# 输出 PDB ID 列表
print("Found PDB IDs:", pdb_ids)

# 保存 PDB ID 列表到文件
file_path = os.path.join(os.path.dirname(__file__), "apt_ids_rscb.txt")
save_pdb_ids_to_file(pdb_ids, file_path)