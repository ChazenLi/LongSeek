# 分子结构向量分析工具

这个模块提供了分析分子结构中原子向量的工具，支持交互式和非交互式两种使用模式。

## 功能特点

- 解析mol2文件中的原子和键信息
- 识别backbone和非backbone原子
- 构建原子之间的向量
- 计算向量模长和相似度
- 可视化分子结构和向量
- 支持交互式和非交互式两种使用模式
- 支持将向量数据（包含起始atom_id、结束atom_id、向量坐标和模长）保存为CSV或JSON格式

## 文件结构

- `__init__.py`: 包初始化文件
- `vector.py`: 定义backbone原子列表
- `vector_utils.py`: 核心向量分析功能
- `vector_analyzer.py`: 命令行工具入口
- `example_usage.py`: 使用示例

## 使用方法

### 命令行使用

```bash
# 交互式模式
python vector_analyzer.py --mol2 "E:\APTAMER-GEN\mol2\1AW4-1.mol2"

# 非交互式模式，查看第一个子结构的backbone向量模长
python vector_analyzer.py --mol2 "E:\APTAMER-GEN\mol2\1AW4-1.mol2" --interactive --substruct 0 --vector-type backbone --operation magnitude

# 保存第一个子结构的所有向量数据为CSV格式
python vector_analyzer.py --mol2 "E:\APTAMER-GEN\mol2\1AW4-1.mol2" --interactive --substruct 0 --vector-type all --operation save --save-to "./output/vectors.csv" --format csv

# 保存第一个子结构的backbone向量数据为JSON格式
python vector_analyzer.py --mol2 "E:\APTAMER-GEN\mol2\1AW4-1.mol2" --interactive --substruct 0 --vector-type backbone --operation save --save-to "./output/backbone_vectors.json" --format json
```

### 在代码中使用

```python
from standard_vector.vector_utils import analyze_vectors

# 直接指定mol2文件路径
mol2_file = r"E:\APTAMER-GEN\mol2\1AW4-1.mol2"

# 非交互模式，查看第一个子结构的backbone向量模长
result = analyze_vectors(
    mol2_file,
    interactive=False,
    substruct_idx=0,  # 第一个子结构
    vector_type='backbone',
    operation='magnitude'
)

# 保存第一个子结构的所有向量数据为CSV格式
csv_output = "./output/vectors.csv"
result = analyze_vectors(
    mol2_file,
    interactive=False,
    substruct_idx=0,  # 第一个子结构
    vector_type='all',
    operation='save',
    save_to_file=csv_output,
    file_format='csv'
)

# 保存第一个子结构的backbone向量数据为JSON格式
json_output = "./output/backbone_vectors.json"
result = analyze_vectors(
    mol2_file,
    interactive=False,
    substruct_idx=0,  # 第一个子结构
    vector_type='backbone',
    save_to_file=json_output,
    file_format='json'
)
```

更多使用示例请参考 `example_usage.py`。

## 参数说明

- `mol2_file`: mol2文件路径
- `interactive`: 是否使用交互模式，默认为True
- `substruct_idx`: 非交互模式下要分析的子结构索引（从0开始）
- `vector_type`: 向量类型，可选值为'all'、'backbone'、'non_backbone'、'mixed'
- `operation`: 要执行的操作，可选值为'magnitude'、'similarity'、'visualize'、'save'
- `save_to_file`: 保存向量数据的输出文件路径
- `file_format`: 输出文件格式，可选值为'csv'、'json'，默认为'csv'