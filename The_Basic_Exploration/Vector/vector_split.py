import numpy as np
import matplotlib.pyplot as plt
import torch
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial Unicode MS', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False

def vector_dim_elimination_analysis(vec):
    from torch.nn.functional import cosine_similarity
    vec = vec.clone().float()
    n = vec.shape[0]
    mask = torch.ones_like(vec)
    current_vec = vec.clone()
    elimination_order = []
    similarity_list = [1.0]
    while mask.sum() > 0:
        sims = []
        candidates = []
        for i in range(n):
            if mask[i] == 0:
                sims.append(-float('inf'))
                candidates.append(None)
                continue
            temp_vec = current_vec.clone()
            temp_vec[i] = 0
            sim = cosine_similarity(temp_vec.unsqueeze(0), vec.unsqueeze(0)).item()
            sims.append(sim)
            candidates.append(temp_vec.clone())
        max_idx = int(torch.tensor(sims).argmax())
        elimination_order.append(max_idx)
        current_vec[max_idx] = 0
        mask[max_idx] = 0
        similarity_list.append(sims[max_idx])
    return elimination_order, similarity_list

def single_dim_zero_similarity(vec):
    from torch.nn.functional import cosine_similarity
    n = vec.shape[0]
    sims = []
    for i in range(n):
        temp_vec = vec.clone()
        temp_vec[i] = 0
        sim = cosine_similarity(temp_vec.unsqueeze(0), vec.unsqueeze(0)).item()
        sims.append(sim)
    return sims

def plot_elimination_similarity(sims):
    angles = np.arccos(np.clip(sims, -1, 1))
    fig, axs = plt.subplots(1, 2, figsize=(14, 5))
    axs[0].plot(range(len(sims)), sims, marker='o')
    axs[0].set_title('高维向量逐步置零与原向量余弦相似度')
    axs[0].set_xlabel('消去步数（已置零维度数）')
    axs[0].set_ylabel('与原向量余弦相似度')
    axs[0].grid(True)
    axs[1].plot(range(len(angles)), angles, marker='o', color='orange')
    axs[1].set_title('高维向量逐步置零与原向量夹角')
    axs[1].set_xlabel('消去步数（已置零维度数）')
    axs[1].set_ylabel('与原向量夹角（弧度）')
    axs[1].grid(True)
    plt.tight_layout()
    plt.show()

def plot_single_dim_similarity(sims_single, dim):
    cos2_sum = np.sum(np.array(sims_single)**2)
    plt.figure(figsize=(8,5))
    plt.bar(range(len(sims_single)), sims_single)
    plt.xlabel('被置零的维度编号')
    plt.ylabel('与原向量余弦相似度')
    plt.title(f'单一维度置零后与原向量余弦相似度分布\ncos$^2$和={cos2_sum:.3f}，理论值={dim-1}')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

vec = torch.randn(10)
order, sims = vector_dim_elimination_analysis(vec)
plot_elimination_similarity(sims)
sims_single = single_dim_zero_similarity(vec)
plot_single_dim_similarity(sims_single, vec.shape[0])

