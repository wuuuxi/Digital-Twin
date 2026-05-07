"""
变负载规划数据加载工具

加载 generate_variable_load() 输出的 vload csv (高度-负载-激活规划值)。
这类文件位于 subject.vload_load_folder 下，由 vload 实验中的 vload_file
字段指定。

用法示例:
    from digitaltwin.analysis.vload_planning import load_planned_vload
    planned_df = load_planned_vload(subject, 'M2_GL_0.4.csv')
"""
import os
import pandas as pd


def load_planned_vload(subject, vload_file, verbose=True):
    """
    从 subject.vload_load_folder 加载规划好的 vload csv。

    期望列：Height, Load, Activation（后二者可选）。
    自动剔除 'Unnamed:*' 列。

    Returns
    -------
    pd.DataFrame or None
        路径不存在或 vload_file 为空时返回 None。
    """
    if not vload_file:
        return None
    path = os.path.join(subject.vload_load_folder, vload_file)
    if not os.path.exists(path):
        if verbose:
            print(f'  规划文件不存在: {path}')
        return None
    df = pd.read_csv(path)
    df = df.loc[:, ~df.columns.astype(str).str.startswith('Unnamed')]
    return df