# utils/path.py
import os

def get_project_root():
    """获取项目根目录"""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.normpath(os.path.join(current_dir, '..', '..'))

def get_config_path():
    """获取配置文件绝对路径"""
    root = get_project_root()
    return os.path.join(root, 'configs', 'params.yaml')