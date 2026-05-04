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

os.environ['OMP_NUM_THREADS'] = '2'
from scipy.spatial.distance import cdist
from scipy.interpolate import RegularGridInterpolator
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans

from digitaltwin.analysis.smooth_monotonic import smooth_monotonic_projection

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


def apply_smooth_monotonic_projection(
    zi,
    yi,
    lambda_data=1.0,
    lambda_height=0.01,
    lambda_load=5.0,
    lambda_cross=0.1,
    min_slope=0.0,
    solver='auto',
):
    """
    对 RBF 拟合得到的 zi 网格沿负载轴做平滑单调投影。

    该方法不同于 isotonic regression：
    - isotonic regression 容易产生分段平台；
    - 平滑单调投影通过二阶平滑项得到连续自然的曲面。

    该方法也不同于在 RBF 权重上加入单调性惩罚：
    - 约束作用在最终规则网格上；
    - 高度方向和负载方向的平滑强度可以独立控制；
    - 不直接修改 RBF 权重，因此更不容易压平高度方向细节。
    """
    load_values = np.asarray(yi)[:, 0]
    return smooth_monotonic_projection(
        zi,
        load_values,
        lambda_data=lambda_data,
        lambda_height=lambda_height,
        lambda_load=lambda_load,
        lambda_cross=lambda_cross,
        min_slope=min_slope,
        solver=solver,
    )


def predict_at(params, heights, loads):
    """
    在任意 (height, load) 点上评估 fit_activation_map 返回的曲面。

    - 当 params 是普通 RBF 拟合（未做 isotonic 修正）时，直接用 rbf_predict；
    - 当 params 是 isotonic 修正后的曲面时，centers/weights 与修正后的 zi
      不再一致，改用网格 (xi, yi, zi) 做双线性插值。

    Parameters
    ----------
    params : dict
        fit_activation_map 返回的字典。
    heights, loads : array-like
        相同形状的查询点。

    Returns
    -------
    np.ndarray
        与输入同形的预测值。
    """
    heights = np.asarray(heights, dtype=float)
    loads = np.asarray(loads, dtype=float)
    shape = heights.shape

    if params.get('monotonic_load'):
        yi_1d = np.asarray(params['yi'])[:, 0]
        xi_1d = np.asarray(params['xi'])[0, :]
        zi = np.asarray(params['zi'])
        if yi_1d[0] > yi_1d[-1]:
            yi_1d = yi_1d[::-1]
            zi = zi[::-1, :]
        if xi_1d[0] > xi_1d[-1]:
            xi_1d = xi_1d[::-1]
            zi = zi[:, ::-1]
        interp = RegularGridInterpolator(
            (yi_1d, xi_1d), zi,
            bounds_error=False, fill_value=None)
        pts = np.column_stack([loads.ravel(), heights.ravel()])
        return interp(pts).reshape(shape)

    return rbf_predict(
        (heights.ravel(), loads.ravel()),
        params['centers'], params['weights'],
        params['scaler'], params['sigma']
    ).reshape(shape)


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
                       data_len=DEFAULT_DATA_LEN, random_state=DEFAULT_RANDOM_STATE,
                       height_range=None,
                       monotonic_load=False,
                       monotonic_method='smooth_projection',
                       projection_lambda_data=1.0,
                       projection_lambda_height=0.01,
                       projection_lambda_load=5.0,
                       projection_lambda_cross=0.1,
                       projection_min_slope=0.0,
                       projection_solver='auto'):
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
    height_range : list[float] or None
        高度范围 [h_min, h_max]。若指定，则用该范围生成网格
        并过滤超出范围的数据点；若为 None，则使用数据本身的 min/max。
    monotonic_load : bool
        是否对 RBF 拟合结果沿负载轴做平滑单调投影。
        该后处理只作用在最终 zi 网格上，不改变 RBF 权重。
    monotonic_method : str
        当前支持 'smooth_projection'。
    projection_lambda_data : float
        保持接近原始 RBF 曲面的权重。越大越贴近原始 RBF。
    projection_lambda_height : float
        高度方向二阶平滑强度。为保留高度细节，建议从较小值开始。
    projection_lambda_load : float
        负载方向二阶平滑强度。越大，敏感度图越连续。
    projection_lambda_cross : float
        高度-负载交叉平滑强度。越大，曲面扭曲越少。
    projection_min_slope : float
        负载方向最小斜率。0 表示非递减；小正数可减少大片 0 敏感度。
    projection_solver : {'auto', 'cvxpy', 'penalty'}
        'auto' 会优先使用 cvxpy 严格约束；若未安装 cvxpy，则回退到 scipy 罚函数。

    Returns
    -------
    dict
        {'xi', 'yi', 'zi', 'centers', 'weights', 'scaler', 'sigma'}
    """
    x = data[pos_col].values
    y = data[load_col].values
    z = data[emg_col].values

    # 如果指定了 height_range，过滤数据并使用指定范围
    if height_range is not None:
        h_min, h_max = height_range
        mask = (x >= h_min) & (x <= h_max)
        x, y, z = x[mask], y[mask], z[mask]
        xi = np.linspace(h_min, h_max, data_len)
    else:
        xi = np.linspace(x.min(), x.max(), data_len)
    yi = np.linspace(y.min(), y.max(), data_len)
    xi, yi = np.meshgrid(xi, yi)

    centers, weights, scaler, sigma_out = rbf_fit(
        (x, y), z, num_centers=num_centers, sigma=sigma,
        random_state=random_state)

    zi = rbf_predict(
        (xi.flatten(), yi.flatten()), centers, weights, scaler, sigma_out
    ).reshape(xi.shape)

    if monotonic_load:
        if monotonic_method != 'smooth_projection':
            raise ValueError("monotonic_method currently supports only "
                             "'smooth_projection'.")
        zi = apply_smooth_monotonic_projection(
            zi, yi,
            lambda_data=projection_lambda_data,
            lambda_height=projection_lambda_height,
            lambda_load=projection_lambda_load,
            lambda_cross=projection_lambda_cross,
            min_slope=projection_min_slope,
            solver=projection_solver)

    return {
        'xi': xi, 'yi': yi, 'zi': zi,
        'centers': centers, 'weights': weights,
        'scaler': scaler, 'sigma': sigma_out,
        'monotonic_load': monotonic_load,
        'monotonic_method': monotonic_method if monotonic_load else None,
        'projection_lambda_data': (
            projection_lambda_data if monotonic_load else None),
        'projection_lambda_height': (
            projection_lambda_height if monotonic_load else None),
        'projection_lambda_load': (
            projection_lambda_load if monotonic_load else None),
        'projection_lambda_cross': (
            projection_lambda_cross if monotonic_load else None),
        'projection_min_slope': (
            projection_min_slope if monotonic_load else None),
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