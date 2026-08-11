"""
通用数据 IO 工具：load key 规范化 + 缓存 CSV 读取。

从 analysis/result_analysis.py 拆出，供分析层与流水线层共用。
"""
import numpy as np
import pandas as pd


def canonical_load_key(value):
    """
    统一 load key 的字符串格式。

    CSV 读写后，原来的 "20" 可能变成 20.0 / "20.0"。
    这里统一成 "20"，避免缓存切片数据时 load key 对不上。
    """
    try:
        f = float(value)
        if np.isfinite(f) and abs(f - round(f)) < 1e-9:
            return str(int(round(f)))
        return f'{f:g}'
    except Exception:
        return str(value)


def read_data_csv(path):
    '''读取 aligned_data / cutted_data 缓存 CSV。

    组名现在混有 '20' 与 'IM-1' 两类，pandas 默认分块（low_memory=True）
    推断 dtype，同一列在不同块里得到不同类型，于是报
    DtypeWarning: Columns (19) have mixed types。这不只是警告：它意味着
    load_weight 可能部分行被读成 20.0、部分行是 '20'，后面按组名
    匹配窗口时就会静静对不上。

    因此统一：low_memory=False 全列一次性推断，并把 load_weight 转成
    字符串（NaN 保留），再交由 canonical_load_key 做格式归一。
    '''
    df = pd.read_csv(path, low_memory=False)
    if 'load_weight' in df.columns:
        df['load_weight'] = df['load_weight'].apply(
            lambda v: v if pd.isna(v) else str(v))
    return df