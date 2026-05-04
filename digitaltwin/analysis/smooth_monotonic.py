"""
平滑单调投影模块。

用于对已经拟合好的二维激活曲面 Z(height, load) 做网格层面的
平滑单调修正：

    原始 RBF 曲面 Z0
        -> smooth_monotonic_projection
        -> 沿负载方向单调、同时保持平滑的新曲面 Z

该方法不同于 isotonic regression：
- isotonic regression 的解通常是分段常数，容易出现平台；
- 平滑单调投影通过二阶平滑项控制曲面曲率，敏感度图会更连续自然。

该方法也不同于在 RBF 权重上加单调惩罚：
- 约束作用在拟合后的规则网格上；
- 高度方向和负载方向的平滑强度可以分别调节；
- 不直接改变 RBF basis 权重，因此更不容易压平高度方向细节。
"""
import numpy as np


def _second_diff_height(Z):
    """高度方向二阶差分。"""
    return Z[:, 2:] - 2.0 * Z[:, 1:-1] + Z[:, :-2]


def _second_diff_load(Z):
    """负载方向二阶差分。"""
    return Z[2:, :] - 2.0 * Z[1:-1, :] + Z[:-2, :]


def _cross_diff(Z):
    """高度-负载交叉差分，用于控制曲面扭曲。"""
    return Z[1:, 1:] - Z[1:, :-1] - Z[:-1, 1:] + Z[:-1, :-1]


def _smooth_monotonic_projection_cvxpy(
    Z0,
    load_values,
    lambda_data=1.0,
    lambda_height=0.01,
    lambda_load=5.0,
    lambda_cross=0.1,
    min_slope=0.0,
    solver='OSQP',
):
    """
    使用 cvxpy 求解严格约束的平滑单调投影。
    """
    import cvxpy as cp

    Z0 = np.asarray(Z0, dtype=float)
    load_values = np.asarray(load_values, dtype=float)
    n_load, n_height = Z0.shape

    Z = cp.Variable((n_load, n_height))

    d_load = Z[1:, :] - Z[:-1, :]
    d2_height = Z[:, 2:] - 2.0 * Z[:, 1:-1] + Z[:, :-2]
    d2_load = Z[2:, :] - 2.0 * Z[1:-1, :] + Z[:-2, :]
    d_cross = (
        Z[1:, 1:] - Z[1:, :-1]
        - Z[:-1, 1:] + Z[:-1, :-1]
    )

    load_step = np.diff(load_values).reshape(-1, 1)
    min_diff = min_slope * load_step

    objective = lambda_data * cp.sum_squares(Z - Z0)
    if n_height >= 3 and lambda_height > 0:
        objective += lambda_height * cp.sum_squares(d2_height)
    if n_load >= 3 and lambda_load > 0:
        objective += lambda_load * cp.sum_squares(d2_load)
    if n_load >= 2 and n_height >= 2 and lambda_cross > 0:
        objective += lambda_cross * cp.sum_squares(d_cross)

    constraints = [d_load >= min_diff]

    problem = cp.Problem(cp.Minimize(objective), constraints)

    try:
        problem.solve(solver=solver)
    except Exception:
        problem.solve()

    if Z.value is None:
        raise RuntimeError('smooth_monotonic_projection: cvxpy failed.')

    return np.asarray(Z.value, dtype=float)


def _smooth_monotonic_projection_penalty(
    Z0,
    load_values,
    lambda_data=1.0,
    lambda_height=0.01,
    lambda_load=5.0,
    lambda_cross=0.1,
    min_slope=0.0,
    monotonic_penalty=1e4,
    max_iter=1000,
):
    """
    无 cvxpy 时的 scipy 罚函数回退版本。

    注意：这是 soft constraint，不像 cvxpy 版本那样严格保证单调；
    但由于约束作用在网格上，而不是 RBF 权重上，通常仍比 RBF 权重惩罚
    更少压制高度方向细节。
    """
    from scipy.optimize import minimize

    Z0 = np.asarray(Z0, dtype=float)
    load_values = np.asarray(load_values, dtype=float)
    n_load, n_height = Z0.shape

    load_step = np.diff(load_values).reshape(-1, 1)
    min_diff = min_slope * load_step

    def objective(z_flat):
        Z = z_flat.reshape(Z0.shape)
        loss = lambda_data * np.mean((Z - Z0) ** 2)

        if n_height >= 3 and lambda_height > 0:
            loss += lambda_height * np.mean(_second_diff_height(Z) ** 2)
        if n_load >= 3 and lambda_load > 0:
            loss += lambda_load * np.mean(_second_diff_load(Z) ** 2)
        if n_load >= 2 and n_height >= 2 and lambda_cross > 0:
            loss += lambda_cross * np.mean(_cross_diff(Z) ** 2)

        d_load = Z[1:, :] - Z[:-1, :]
        violation = np.minimum(d_load - min_diff, 0.0)
        loss += monotonic_penalty * np.mean(violation ** 2)
        return loss

    result = minimize(
        objective,
        Z0.ravel(),
        method='L-BFGS-B',
        options={'maxiter': max_iter},
    )

    return result.x.reshape(Z0.shape)


def smooth_monotonic_projection(
    Z0,
    load_values,
    lambda_data=1.0,
    lambda_height=0.01,
    lambda_load=5.0,
    lambda_cross=0.1,
    min_slope=0.0,
    solver='auto',
    monotonic_penalty=1e4,
):
    """
    对二维曲面做平滑单调投影。

    Parameters
    ----------
    Z0 : np.ndarray, shape (n_load, n_height)
        原始拟合曲面。axis=0 为负载方向，axis=1 为高度方向。
    load_values : array-like, shape (n_load,)
        负载坐标，与 Z0 的行对应。
    lambda_data : float
        保持接近原始曲面的权重。越大越贴近原始 RBF。
    lambda_height : float
        高度方向二阶平滑强度。越大，高度方向越平滑。
        为了保留高度细节，建议从较小值开始，如 0.001~0.05。
    lambda_load : float
        负载方向二阶平滑强度。越大，负载方向敏感度越连续。
    lambda_cross : float
        高度-负载交叉平滑强度。越大，曲面扭曲越少。
    min_slope : float
        负载方向最小斜率。0 表示非递减；
        一个很小的正数可减少敏感度图中大片接近 0 的区域。
    solver : {'auto', 'cvxpy', 'penalty'}
        - 'auto': 优先使用 cvxpy，若未安装则回退到 scipy 罚函数。
        - 'cvxpy': 使用严格约束版本。
        - 'penalty': 使用 scipy 罚函数版本。
    monotonic_penalty : float
        仅在 penalty 版本中使用，控制违反单调性的惩罚强度。

    Returns
    -------
    np.ndarray
        平滑单调修正后的曲面。
    """
    Z0 = np.asarray(Z0, dtype=float)
    load_values = np.asarray(load_values, dtype=float)

    # 若负载是降序，先翻转，投影后再翻回。
    reversed_load = False
    if load_values[0] > load_values[-1]:
        reversed_load = True
        load_values = load_values[::-1]
        Z0 = Z0[::-1, :]

    if solver not in ('auto', 'cvxpy', 'penalty'):
        raise ValueError("solver must be 'auto', 'cvxpy', or 'penalty'.")

    if solver in ('auto', 'cvxpy'):
        try:
            Z = _smooth_monotonic_projection_cvxpy(
                Z0,
                load_values,
                lambda_data=lambda_data,
                lambda_height=lambda_height,
                lambda_load=lambda_load,
                lambda_cross=lambda_cross,
                min_slope=min_slope,
            )
        except Exception:
            if solver == 'cvxpy':
                raise
            Z = _smooth_monotonic_projection_penalty(
                Z0,
                load_values,
                lambda_data=lambda_data,
                lambda_height=lambda_height,
                lambda_load=lambda_load,
                lambda_cross=lambda_cross,
                min_slope=min_slope,
                monotonic_penalty=monotonic_penalty,
            )
    else:
        Z = _smooth_monotonic_projection_penalty(
            Z0,
            load_values,
            lambda_data=lambda_data,
            lambda_height=lambda_height,
            lambda_load=lambda_load,
            lambda_cross=lambda_cross,
            min_slope=min_slope,
            monotonic_penalty=monotonic_penalty,
        )

    if reversed_load:
        Z = Z[::-1, :]

    return Z