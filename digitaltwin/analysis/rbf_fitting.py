"""
RBF（径向基函数）拟合模块
提供位置-负载-肌肉激活关系的 RBF 拟合与预测功能。

原始实现位于 src/heatmap.py，现已重构并集成到 Digital Twin 框架中。

用法示例:
    from digitaltwin.analysis.rbf_fitting import rbf_fit, rbf_predict, fit_activation_map
"""
import numpy as np
import pickle
import os
from scipy.spatial.distance import cdist
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans

# ============================================================
#  默认参数
# ============================================================
DEFAULT_DATA_LEN = 50
DEFAULT_NUM_CENTERS = 20
DEFAULT_SIGMA = 1.0
DEFAULT_RANDOM_STATE = 10


# ============================================================
#  核心 RBF 函数
# ============================================================

def rbf_function(x, c, s):
    """
    计算 RBF 核矩阵。

    Parameters
    ----------
    x : np.ndarray, shape (n_samples, n_features)
        输入数据点
    c : np.ndarray, shape (n_centers, n_features)
        RBF 中心
    s : float
        RBF 宽度参数 (sigma)

    Returns
    -------
    np.ndarray, shape (n_samples, n_centers)
    """
    return np.exp(-cdist(x, c) ** 2 / (2 * s ** 2))


def rbf_fit(XY, Z, num_centers=DEFAULT_NUM_CENTERS, sigma=DEFAULT_SIGMA,
            random_state=DEFAULT_RANDOM_STATE):
    """
    使用 KMeans 聚类 + RBF 拟合多维输入到标量输出的映射。

    Parameters
    ----------
    XY : tuple of array-like
        输入变量元组，如 (position, load) 或 (pos, load, velocity)
    Z : array-like
        输出变量（如肌肉激活值）
    num_centers : int
        KMeans 聚类中心数
    sigma : float
        RBF 宽度参数
    random_state : int
        随机种子

    Returns
    -------
    tuple
        (centers, weights, scaler, sigma)
    """
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(np.column_stack(XY))

    kmeans = KMeans(n_clusters=num_centers, random_state=random_state)
    kmeans.fit(X_scaled)
    centers = kmeans.cluster_centers_

    R = rbf_function(X_scaled, centers, sigma)
    weights = np.linalg.pinv(R).dot(Z)

    return centers, weights, scaler, sigma


def rbf_predict(XY, centers, weights, scaler, sigma):
    """
    使用已训练的 RBF 模型进行预测。

    Parameters
    ----------
    XY : tuple of array-like
        输入变量元组
    centers, weights, scaler, sigma
        rbf_fit 返回的参数

    Returns
    -------
    np.ndarray
        预测值
    """
    X_scaled = scaler.transform(np.column_stack(XY))
    R = rbf_function(X_scaled, centers, sigma)
    return R.dot(weights)


# ============================================================
#  高级拟合接口
# ============================================================

def fit_activation_map(data, pos_col='pos_l', load_col='load', emg_col='emg0',
                       num_centers=DEFAULT_NUM_CENTERS, sigma=DEFAULT_SIGMA,
                       data_len=DEFAULT_DATA_LEN, random_state=DEFAULT_RANDOM_STATE):
    """
    拟合位置-负载-肌肉激活的 RBF 映射并生成网格预测。

    Parameters
    ----------
    data : pd.DataFrame
        包含 pos, load, emg 列的 DataFrame
    pos_col, load_col, emg_col : str
        对应列名
    num_centers, sigma, data_len, random_state
        RBF 参数

    Returns
    -------
    dict
        {'xi', 'yi', 'zi', 'centers', 'weights', 'scaler', 'sigma'}
    """
    x = data[pos_col].values
    y = data[load_col].values
    z = data[emg_col].values

    xi = np.linspace(x.min(), x.max(), data_len)
    yi = np.linspace(y.min(), y.max(), data_len)
    xi, yi = np.meshgrid(xi, yi)

    centers, weights, scaler, sigma_out = rbf_fit(
        (x, y), z, num_centers=num_centers, sigma=sigma,
        random_state=random_state)

    zi = rbf_predict(
        (xi.flatten(), yi.flatten()), centers, weights, scaler, sigma_out
    ).reshape(xi.shape)

    return {
        'xi': xi, 'yi': yi, 'zi': zi,
        'centers': centers, 'weights': weights,
        'scaler': scaler, 'sigma': sigma_out,
    }


def fit_activation_map_3d(data, a, b, c, d, num_centers=DEFAULT_NUM_CENTERS,
                          sigma=DEFAULT_SIGMA, data_len=DEFAULT_DATA_LEN):
    """
    拟合三维输入 (pos, load, velocity) → activation 的 RBF 映射。

    Parameters
    ----------
    data : pd.DataFrame
    a, b, c, d : str
        分别为 pos_col, load_col, emg_col, vel_col

    Returns
    -------
    dict
        {'xi', 'yi', 'mi', 'zi', 'centers', 'weights', 'scaler', 'sigma'}
    """
    x, y, m, z = data[a], data[b], data[d], data[c]
    xi = np.linspace(min(x), max(x), data_len)
    yi = np.linspace(min(y), max(y), data_len)
    mi = np.linspace(min(m), max(m), data_len)
    xi, yi, mi = np.meshgrid(xi, yi, mi)

    centers, weights, scaler, sigma_out = rbf_fit(
        (x, y, m), z, num_centers=num_centers, sigma=sigma)
    zi = rbf_predict(
        (xi.flatten(), yi.flatten(), mi.flatten()),
        centers, weights, scaler, sigma_out
    ).reshape(xi.shape)

    return {
        'xi': xi, 'yi': yi, 'mi': mi, 'zi': zi,
        'centers': centers, 'weights': weights,
        'scaler': scaler, 'sigma': sigma_out,
    }


# ============================================================
#  参数持久化
# ============================================================

def save_rbf_params(centers, weights, scaler, sigma, filepath):
    """保存 RBF 参数到 pickle 文件"""
    os.makedirs(os.path.dirname(filepath) or '.', exist_ok=True)
    params = {'centers': centers, 'weights': weights,
              'scaler': scaler, 'sigma': sigma}
    with open(filepath, 'wb') as f:
        pickle.dump(params, f)


def load_rbf_params(filepath):
    """从 pickle 文件加载 RBF 参数，返回 (centers, weights, scaler, sigma)"""
    with open(filepath, 'rb') as f:
        p = pickle.load(f)
    return p['centers'], p['weights'], p['scaler'], p['sigma']


# ============================================================
#  评估指标
# ============================================================

def compute_rmse_percentage(data, pos_col, load_col, emg_col,
                            centers, weights, scaler, sigma):
    """
    计算 RBF 预测的 RMSE 百分比误差。

    Returns
    -------
    float
        RMSE / mean(prediction)
    """
    pred = rbf_predict((data[pos_col], data[load_col]),
                       centers, weights, scaler, sigma)
    rmse = np.sqrt(np.mean((pred - data[emg_col]) ** 2))
    mean_pred = np.mean(pred)
    return rmse / mean_pred if mean_pred != 0 else float('inf')


def compute_rmse_by_load(data, pos_col, load_col, emg_col,
                         centers, weights, scaler, sigma):
    """
    按负载分组计算 RMSE 百分比。

    Returns
    -------
    dict
        {load_value: rmse_percentage}
    """
    results = {}
    for lv in data[load_col].unique():
        subset = data[data[load_col] == lv]
        results[lv] = compute_rmse_percentage(
            subset, pos_col, load_col, emg_col,
            centers, weights, scaler, sigma)
    return results