# data_utils/parser.py
from pathlib import Path
from typing import Generator
import os
import yaml

from pathlib import Path
from typing import Generator, List
import os
import yaml

class Mol2StreamParser:
    def __init__(self, folder_path: str):
        self.folder_path = Path(folder_path)
        self.config = self._load_config()
        self._validate_path()

    def _load_config(self):
        config_path = os.path.join(
            os.path.dirname(__file__),
            '..', '..', 'configs', 'params.yaml'
        )
        with open(config_path) as f:
            return yaml.safe_load(f)

    def _validate_path(self):
        if not self.folder_path.exists():
            raise FileNotFoundError(f"分子库路径不存在: {self.folder_path}")
        if not list(self.folder_path.glob("**/*.mol2")):
            raise ValueError("指定路径下未找到任何.mol2文件")

    def _parse_file(self, file_path: Path) -> List[str]:
        """返回单个文件的所有有效块"""
        blocks = []
        current_block = []
        in_target = False
        
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                
                if line.startswith(("@<TRIPOS>ATOM", "@<TRIPOS>BOND")):
                    if current_block:  # 保存上一个未闭合的块
                        blocks.append('\n'.join(current_block))
                    current_block = [line]
                    in_target = True
                elif in_target:
                    if line.startswith("@<TRIPOS>"):
                        blocks.append('\n'.join(current_block))
                        current_block = []
                        in_target = line.startswith(("@<TRIPOS>ATOM", "@<TRIPOS>BOND"))
                        if in_target:
                            current_block.append(line)
                    elif line:
                        current_block.append(line)
            
            if current_block:
                blocks.append('\n'.join(current_block))
        
        return blocks

    def stream_blocks(self) -> Generator[str, None, None]:
        """带错误处理的增强版块生成器"""
        for mol2_file in self.folder_path.glob("**/*.mol2"):
            try:
                yield from self._parse_file(mol2_file)
            except Exception as e:
                print(f"解析文件失败: {mol2_file}，错误: {str(e)}")
                continue