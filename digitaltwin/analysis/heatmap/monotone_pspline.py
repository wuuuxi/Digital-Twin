"""
2D 单调 P-spline 拟合模块。

模型（在高度、负载两个轴上用张量积 B-spline）：

    A(h, l) = sum_{i, j} theta_{ij} * B^h_i(h) * B^l_j(l)

对负载方向的单调性（递增）等价于系数上的一阶差分约束：

    theta_{i, j+1} - theta_{i, j} >= 0   for all i, j

平滑性使用 P-spline 二阶差分惩罚（Eilers & Marx）：

    L = ||Phi vec(theta) - z||^2 / N
      + lambda_h * ||D2_h theta||_F^2
      + lambda_l * ||theta D2_l^T||_F^2

优点：
  - 从模型结构上保证负载方向单调，不是后处理修正；
  - lambda_h 与 lambda_l 独立控制高度 / 负载方向的平滑强度；
  - 敏感度 ∂A/∂l 连续、不会出现 isotonic 那样的平台。
"""
import numpy as np
from scipy.interpolate import BSpline


# ===========================================================
#  B-spline 设计矩阵
# ===========================================================

def _bspline_basis(x, knots, degree):
    """在点 x 上评估所有 B-spline basis，返回设计矩阵 (N, n_basis)。"""
    x = np.asarray(x, dtype=float)
    n_basis = len(knots) - degree - 1
    span = float(knots[-1] - knots[0])
    eps = max(span * 1e-9, 1e-12)
    # 钕制到 [knots[0], knots[-1] - eps]，避免 BSpline 在右端点返回 0/NaN
    x_eval = np.clip(x, knots[0], knots[-1] - eps)
    B = np.zeros((len(x_eval), n_basis))
    for i in range(n_basis):
        coef = np.zeros(n_basis)
        coef[i] = 1.0
        spl = BSpline(knots, coef, degree, extrapolate=True)
        B[:, i] = spl(x_eval)
    return B


def _make_uniform_knots(x_min, x_max, n_basis, degree):
    """构造均匀钉住节点向量，使 basis 总数为 n_basis。"""
    n_internal = n_basis - degree - 1
    if n_internal < 0:
        raise ValueError(
            f'n_basis ({n_basis}) too small for degree {degree}.')
    if n_internal > 0:
        internal = np.linspace(x_min, x_max, n_internal + 2)[1:-1]
    else:
        internal = np.array([])
    knots = np.concatenate([
        np.repeat(x_min, degree + 1),
        internal,
        np.repeat(x_max, degree + 1),
    ])
    return knots


def _build_d2(n):
    """二阶差分矩阵 D2: (n-2, n)。"""
    if n < 3:
        return np.zeros((0, n))
    D = np.zeros((n - 2, n))
    for i in range(n - 2):
        D[i, i] = 1.0
        D[i, i + 1] = -2.0
        D[i, i + 2] = 1.0
    return D


# ===========================================================
#  拟合
# ===========================================================

def fit_monotone_pspline_2d(
    h, l, z,
    n_basis_h=12,
    n_basis_l=10,
    degree=3,
    lambda_h=1.0,
    lambda_l=1.0,
    increasing=True,
    h_range=None,
    l_range=None,
    solver='auto',
    max_iter=2000,
):
    """
    2D 张量积 B-spline、负载方向单调拟合。

    Parameters
    ----------
    h, l, z : array-like
        数据点。
    n_basis_h, n_basis_l : int
        高度 / 负载方向 B-spline basis 个数。越大越灵活。
    degree : int
        B-spline 阶数（3 = 三次）。
    lambda_h, lambda_l : float
        二阶差分平滑惩罚强度。越大越平滑。
    increasing : bool
        True 表示负载方向递增单调。
    h_range, l_range : (float, float) or None
        basis 的定义域。为 None 时取数据的 min/max。
    solver : {'auto', 'cvxpy', 'lbfgs'}
        - 'auto'  : 优先 cvxpy 严格 QP，缺失时回退 L-BFGS-B。
        - 'cvxpy' : 必须 cvxpy，严格约束。
        - 'lbfgs' : 重参数化（增量 = d^2 · cumsum）使单调性隐含成立。
    max_iter : int
        L-BFGS-B 最大迭代次数。

    Returns
    -------
    dict
        含 'theta', 'knots_h', 'knots_l', 'degree', 'h_range', 'l_range',
        'n_basis_h', 'n_basis_l', 'increasing', 'model'='pspline'。
    """
    h = np.asarray(h, dtype=float)
    l = np.asarray(l, dtype=float)
    z = np.asarray(z, dtype=float)

    if h_range is None:
        h_range = (float(h.min()), float(h.max()))
    if l_range is None:
        l_range = (float(l.min()), float(l.max()))

    knots_h = _make_uniform_knots(h_range[0], h_range[1],
                                  n_basis_h, degree)
    knots_l = _make_uniform_knots(l_range[0], l_range[1],
                                  n_basis_l, degree)

    Bh = _bspline_basis(h, knots_h, degree)  # (N, n_h)
    Bl = _bspline_basis(l, knots_l, degree)  # (N, n_l)

    N = len(z)
    # Phi[k, i*n_l + j] = Bh[k, i] * Bl[k, j]
    Phi = (Bh[:, :, None] * Bl[:, None, :]).reshape(N, -1)

    n_h, n_l = n_basis_h, n_basis_l
    D2_h = _build_d2(n_h)
    D2_l = _build_d2(n_l)

    if solver in ('auto', 'cvxpy'):
        try:
            theta = _fit_cvxpy(Phi, z, n_h, n_l, D2_h, D2_l,
                               lambda_h, lambda_l, increasing)
            return _pack(theta, knots_h, knots_l, degree,
                         h_range, l_range, n_h, n_l, increasing)
        except Exception as e:
            if solver == 'cvxpy':
                raise
            print(f'  cvxpy 求解失败，回退到 L-BFGS-B: {e}')

    theta = _fit_lbfgs(Phi, z, n_h, n_l, D2_h, D2_l,
                       lambda_h, lambda_l, increasing, max_iter)
    return _pack(theta, knots_h, knots_l, degree,
                 h_range, l_range, n_h, n_l, increasing)


def _pack(theta, knots_h, knots_l, degree,
          h_range, l_range, n_h, n_l, increasing):
    return {
        'model': 'pspline',
        'theta': theta,
        'knots_h': knots_h,
        'knots_l': knots_l,
        'degree': degree,
        'h_range': h_range,
        'l_range': l_range,
        'n_basis_h': n_h,
        'n_basis_l': n_l,
        'increasing': increasing,
    }


def _fit_cvxpy(Phi, z, n_h, n_l, D2_h, D2_l,
               lambda_h, lambda_l, increasing):
    import cvxpy as cp
    theta = cp.Variable((n_h, n_l))
    # row-major flatten via transpose trick：vec(theta^T) = row-major(theta)
    theta_flat = cp.reshape(cp.transpose(theta), (n_h * n_l,))
    pred = Phi @ theta_flat
    objective = cp.sum_squares(pred - z) / max(1, len(z))
    if D2_h.shape[0] > 0 and lambda_h > 0:
        objective += lambda_h * cp.sum_squares(D2_h @ theta)
    if D2_l.shape[0] > 0 and lambda_l > 0:
        objective += lambda_l * cp.sum_squares(theta @ D2_l.T)
    if increasing:
        constraints = [theta[:, 1:] - theta[:, :-1] >= 0]
    else:
        constraints = [theta[:, 1:] - theta[:, :-1] <= 0]
    prob = cp.Problem(cp.Minimize(objective), constraints)
    prob.solve()
    if theta.value is None:
        raise RuntimeError('cvxpy failed to solve.')
    return np.asarray(theta.value, dtype=float)


def _fit_lbfgs(Phi, z, n_h, n_l, D2_h, D2_l,
               lambda_h, lambda_l, increasing, max_iter):
    """重参数化使单调性隐含成立：

        theta[:, 0]   = a
        theta[:, j>0] = a + cumsum(d**2)   (increasing)
                      = a - cumsum(d**2)   (decreasing)

    这样 d 是无约束变量，d**2 自然非负，theta 沿 j 单调。
    """
    from scipy.optimize import minimize
    n_diff = n_l - 1
    n_params = n_h + n_h * n_diff

    def unpack(p):
        a = p[:n_h]
        d = p[n_h:].reshape(n_h, n_diff)
        inc = d ** 2
        if not increasing:
            inc = -inc
        cum = np.zeros((n_h, n_l))
        cum[:, 0] = a
        cum[:, 1:] = a[:, None] + np.cumsum(inc, axis=1)
        return cum

    def loss(p):
        theta = unpack(p)
        pred = Phi @ theta.flatten()
        residuals = pred - z
        loss_val = np.mean(residuals ** 2)
        if D2_h.shape[0] > 0 and lambda_h > 0:
            loss_val += lambda_h * np.sum((D2_h @ theta) ** 2)
        if D2_l.shape[0] > 0 and lambda_l > 0:
            loss_val += lambda_l * np.sum((theta @ D2_l.T) ** 2)
        return loss_val

    p0 = np.zeros(n_params)
    p0[:n_h] = float(np.mean(z))
    result = minimize(loss, p0, method='L-BFGS-B',
                      options={'maxiter': max_iter})
    return unpack(result.x)


# ===========================================================
#  预测
# ===========================================================

def predict_monotone_pspline(params, h, l):
    """在任意 (h, l) 点上评估单调 P-spline 曲面。"""
    h = np.asarray(h, dtype=float)
    l = np.asarray(l, dtype=float)
    shape = h.shape
    Bh = _bspline_basis(h.ravel(), params['knots_h'], params['degree'])
    Bl = _bspline_basis(l.ravel(), params['knots_l'], params['degree'])
    theta = params['theta']
    pred = np.einsum('ki,ij,kj->k', Bh, theta, Bl)
    return pred.reshape(shape)