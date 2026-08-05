"""
文件与IO工具
提供 pickle 序列化、结果文件夹创建、调试打印等通用IO工具函数。
"""
import os
import pickle
import shutil
import sys
from datetime import datetime


def print_debug_info(message, print_label=True):
    """打印调试信息"""
    if print_label:
        timestamp = datetime.now().strftime('%H:%M:%S')
        print(f"[{timestamp}] {message}")


def save_pickle(data, filepath):
    """保存pickle文件"""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'wb') as f:
        pickle.dump(data, f)


def load_pickle(filepath):
    """加载pickle文件"""
    if os.path.exists(filepath):
        with open(filepath, 'rb') as f:
            return pickle.load(f)
    return None


def make_result_folder(experiment_label):
    """创建带时间戳的实验结果文件夹，并备份当前脚本

    结果目录锚定在仓库根目录的 result/ 下，而不是相对当前工作目录。
    以前用 '../../result/...'，从 examples/ 里运行脚本时会解析成
    examples/result/，把输出和脚本备份都写错位置。
    """
    current_time = datetime.now()
    # folder_name = current_time.strftime('%Y%m%d_%H%M%S')
    folder_name = 'test'
    # file_tools.py 位于 <repo>/digitaltwin/utils/，向上三级即仓库根目录。
    repo_root = os.path.dirname(os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))))
    result_folder = os.path.join(
        repo_root, 'result', experiment_label, folder_name)
    os.makedirs(result_folder, exist_ok=True)

    code_dir = os.path.join(result_folder, 'code')
    os.makedirs(code_dir, exist_ok=True)
    current_file = sys.argv[0]
    file_name = os.path.basename(current_file)
    target_path = os.path.join(code_dir, file_name)
    shutil.copy2(current_file, target_path)
    return result_folder