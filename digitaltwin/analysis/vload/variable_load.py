"""
变负载优化模块
基于 Pyomo + Ipopt 求解最优负载曲线，使肌肉激活达到目标值。

原始实现位于 src/variable_load.py 和 src/load.py，
现已重构并集成到 Digital Twin 框架中。

用法示例:
    from digitaltwin.analysis.variable_load import (
        variable_load_optimization,
        generate_variable_load,
    )
"""
import numpy as np
import pandas as pd
import pickle
import os

from pyomo.environ import (
    ConcreteModel, RangeSet, Var, Objective, ConstraintList,
    NonNegativeReals, minimize, value, exp, sqrt,
)
from pyomo.opt import SolverFactory


# ============================================================
#  Pyomo 内部 RBF 预测（符号表达式）
# ============================================================

def _rbf_predict_pyomo(x, y, centers, weights, scaler, sigma):
    """Pyomo 符号兼容的 RBF 预测"""
    dmin = scaler.data_min_
    dmax = scaler.data_max_
    xs = (x - dmin[0]) / (dmax[0] - dmin[0])
    ys = (y - dmin[1]) / (dmax[1] - dmin[1])
    return sum(
        weights[k] * exp(-((xs - centers[k, 0]) ** 2
                           + (ys - centers[k, 1]) ** 2) / (2 * sigma ** 2))
        for k in range(len(weights))
    )


# ============================================================
#  Pyomo 内部 P-spline 预测（符号表达式）
# ============================================================
#
# 思路：张量积 2D B-spline 在固定高度 h 处退化为关于负载 l 的
# 1D 三次 B-spline，即 f(l) = Σ_m c_m * B^l_m(l)。任何 C^2 三次样条都
# 可以等价写成“截断幂次基”形式：
#     f(l) = A + B*l + C*l^2 + D*l^3
#          + Σ_k jump_k * max(0, l - break_k)^3
# 而 max(0, u)^3 本身就是 C^2 光滑的，用 sqrt(u^2 + eps) 做一个极小
# 扰动即可实现 C^∞ 光滑，与 ipopt 内点法需要的二阶可微要求完全兼容。

def _pspline_to_trunc_power(params, h):
    """
    给定固定高度 h，把 2D pspline 曲面 a(h, l) 转换为关于 l 的截断幂次基：

        a(h, l) = A + B*l + C*l^2 + D*l^3
                + Σ_k jump_k * max(0, l - break_k)^3

    Returns
    -------
    base : tuple of 4 floats (A, B, C, D)
    breaks : list of float
    jumps : list of float
    """
    from scipy.interpolate import BSpline, PPoly
    from digitaltwin.analysis.heatmap.monotone_pspline import (
        _bspline_basis as _bspl,
    )

    knots_h = params['knots_h']
    knots_l = params['knots_l']
    degree = params['degree']
    theta = params['theta']

    if degree != 3:
        raise NotImplementedError(
            f'_pspline_to_trunc_power requires degree=3, got {degree}.')

    # 在 h 处把 2D theta 投影成 l 方向的 1D B-spline 系数
    Bh = _bspl(np.array([h], dtype=float), knots_h, degree)[0]
    c_l = theta.T @ Bh

    # 转 PPoly：(degree+1, n_pieces) 的局部多项式系数
    spl = BSpline(knots_l, c_l, degree, extrapolate=False)
    pp = PPoly.from_spline((spl.t, spl.c, spl.k))
    breaks_all = pp.x
    coefs_all = pp.c

    # 把每段 (l - x_i) 形式的局部多项式展开到全局 l 多项式系数
    pieces = []
    for i in range(coefs_all.shape[1]):
        x_i = float(breaks_all[i])
        x_next = float(breaks_all[i + 1])
        if x_next - x_i < 1e-12:
            continue  # 跳过 clamped 端点产生的零宽度段
        c0 = float(coefs_all[0, i])
        c1 = float(coefs_all[1, i])
        c2 = float(coefs_all[2, i])
        c3 = float(coefs_all[3, i])
        D = c0
        C = -3 * c0 * x_i + c1
        B = 3 * c0 * x_i ** 2 - 2 * c1 * x_i + c2
        A = -c0 * x_i ** 3 + c1 * x_i ** 2 - c2 * x_i + c3
        pieces.append((x_i, A, B, C, D))

    if not pieces:
        return (0.0, 0.0, 0.0, 0.0), [], []

    # 第一段作为 base，后续每段相对前一段的 D 跳跃 = 截断幂次系数
    _, A0, B0, C0, D0 = pieces[0]
    base = (A0, B0, C0, D0)
    breaks, jumps = [], []
    prev_D = D0
    for x_i, _, _, _, D_i in pieces[1:]:
        jmp = D_i - prev_D
        if abs(jmp) > 1e-12:
            breaks.append(x_i)
            jumps.append(jmp)
        prev_D = D_i

    return base, breaks, jumps


def _pspline_predict_pyomo(l_var, base, breaks, jumps, eps=1e-10):
    """
    Pyomo 符号兼容的截断幂次 P-spline 求值：
        f(l) = A + B*l + C*l^2 + D*l^3
             + Σ_k jump_k * max(0, l - break_k)^3
    其中 max(0, u)^3 用 ((u + sqrt(u^2 + eps))/2)^3 做 C^∞ 光滑实现。
    """
    A, B, C, D = base
    expr = A + B * l_var + C * l_var ** 2 + D * l_var ** 3
    for x_k, c_k in zip(breaks, jumps):
        u = l_var - x_k
        relu_cubed = ((u + sqrt(u ** 2 + eps)) / 2) ** 3
        expr = expr + c_k * relu_cubed
    return expr


def _warm_start_from_pspline(pspline_list, w, g, xi, l_min, l_max,
                              n_grid=500, match_tol_ratio=0.05):
    """
    从 P-spline 曲面上查找贴近目标激活 g 的负载曲线，作为 ipopt 的 warm start。

    策略：
      1. 在 \[l_min, l_max\] 上取 n_grid 点。
      2. 对每个高度 xi\[i\]，计算所有肌肉在 grid 上的加权偏差
         dev\[i, k\] = Σ_j w_j \* |a_j(xi\[i\], l_grid\[k\]) - g_j|
         取 argmin 作为 l_init\[i\]。
      3. 把 dev_min\[i\] < tol 的点标为“匹配成功”，取最长一段连续
         匹配区间 \[i_start, i_end\]；其以前用 l_init\[i_start\] 填充，
         其以后用 l_init\[i_end\] 填充（避免初始值跳变冲撞 epsilon 约束）。
      4. 同时返回 a_init\[i, j\] = a_j(xi\[i\], l_init\[i\])，以保证等式约束
         a == surface(l) 在初始点几乎成立。
    """
    from digitaltwin.analysis.heatmap.monotone_pspline import (
        predict_monotone_pspline,
    )

    timestep_num = len(xi)
    muscle_num = len(pspline_list)
    l_grid = np.linspace(l_min, l_max, n_grid)

    H = np.broadcast_to(np.asarray(xi)[:, None],
                        (timestep_num, n_grid))
    L = np.broadcast_to(l_grid[None, :], (timestep_num, n_grid))

    dev = np.zeros((timestep_num, n_grid), dtype=float)
    a_grid_per_muscle = []
    for j in range(muscle_num):
        a_j = predict_monotone_pspline(pspline_list[j], H, L)
        a_grid_per_muscle.append(a_j)
        dev += float(w[j]) * np.abs(a_j - float(g[j]))

    k_best = np.argmin(dev, axis=1)
    l_init = l_grid[k_best]
    dev_min = dev[np.arange(timestep_num), k_best]

    # “匹配成功”阈值：相对于加权目标幅度
    target_scale = sum(float(w[j]) * max(float(g[j]), 1e-3)
                       for j in range(muscle_num))
    tol = match_tol_ratio * max(target_scale, 1e-6)
    is_match = dev_min < tol

    if is_match.any():
        # 取最长连续 True 区间
        best_seg = None
        i = 0
        while i < timestep_num:
            if is_match[i]:
                j = i
                while j + 1 < timestep_num and is_match[j + 1]:
                    j += 1
                if best_seg is None or (j - i) > (best_seg[1] - best_seg[0]):
                    best_seg = (i, j)
                i = j + 1
            else:
                i += 1
        i_start, i_end = best_seg
        l_init = l_init.copy()
        l_init[:i_start] = l_init[i_start]
        l_init[i_end + 1:] = l_init[i_end]
        k_best_filled = np.searchsorted(l_grid, l_init)
        k_best_filled = np.clip(k_best_filled, 0, n_grid - 1)
    else:
        k_best_filled = k_best

    # 初始 a：用调整后的 l_init 重新查表
    a_init = np.zeros((timestep_num, muscle_num), dtype=float)
    rows = np.arange(timestep_num)
    for j in range(muscle_num):
        a_init[:, j] = a_grid_per_muscle[j][rows, k_best_filled]

    return l_init, a_init


# ============================================================
#  核心优化函数
# ============================================================

def variable_load_optimization(rbf_params, w, g=0.0, s=1, c=0,
                                epsilon=0.5, max_iter=20000,
                                h_min=1.35, h_max=2.05,
                                timestep_num=100, l_min=20, l_max=70):
    """
    变负载优化：求解使肌肉激活接近目标值的最优负载曲线。

    Parameters
    ----------
    rbf_params : list
        [centers_list, weights_list, scaler_list, sigma_list]
    w : list
        各肌肉权重
    g : float or list
        目标激活值（g=0 时自动切换为最小化模式 c=1）
    s : float or list
        激活上限安全阈值
    c : int
        0 = 最小化偏差, 1 = 最大化偏差
    epsilon : float
        相邻时间步负载变化的平滑约束
    max_iter : int
        Ipopt 最大迭代次数
    h_min, h_max : float
        位置（高度）范围
    timestep_num : int
        离散时间步数
    l_min, l_max : float
        负载范围 (kg)

    Returns
    -------
    tuple
        (heights, optimal_loads, activations)
    """
    centers, weights, scaler, sigma = rbf_params
    muscle_num = len(centers)
    assert len(w) == muscle_num

    if g == 0:
        c = 1
    if isinstance(s, (float, int)) or (hasattr(s, '__len__') and len(s) == 1):
        s = [s if isinstance(s, (float, int)) else s[0]] * muscle_num
    if isinstance(g, (float, int)) or (hasattr(g, '__len__') and len(g) == 1):
        g = [g if isinstance(g, (float, int)) else g[0]] * muscle_num

    xi = np.linspace(h_min, h_max, timestep_num)

    model = ConcreteModel()
    model.I = RangeSet(0, timestep_num - 1)
    model.J = RangeSet(0, muscle_num - 1)
    model.l = Var(model.I, within=NonNegativeReals,
                  initialize=(l_min + l_max) / 2)
    model.a = Var(model.I, model.J, within=NonNegativeReals,
                  initialize=g[0])

    model.constr = ConstraintList()
    for i in model.I:
        for j in model.J:
            model.constr.add(
                model.a[i, j] == _rbf_predict_pyomo(
                    xi[i], model.l[i],
                    centers[j], weights[j], scaler[j], sigma[j]))
            model.constr.add(model.a[i, j] <= s[j])
            model.constr.add(model.l[i] <= l_max)
            model.constr.add(model.l[i] >= l_min)
            if i > 0:
                model.constr.add(model.l[i] <= model.l[i - 1] + epsilon)
                model.constr.add(model.l[i] >= model.l[i - 1] - epsilon)

    obj = sum(
        sum(abs(model.a[i, j] - g[j]) * w[j] for j in model.J)
        for i in model.I
    ) * (-1) ** c
    model.obj = Objective(expr=obj, sense=minimize)

    solver = SolverFactory('ipopt')
    solver.options['max_iter'] = max_iter
    solver.solve(model)

    load = np.array([value(model.l[i]) for i in model.I])
    activation = np.array([
        [value(model.a[i, j]) for j in model.J]
        for i in model.I
    ])
    return xi, load, activation


def variable_load_optimization_max(rbf_params, w, g=0.0, s=1, c=0,
                                    epsilon=0.5, max_iter=20000,
                                    h_min=1.35, h_max=2.05,
                                    timestep_num=100, l_min=20, l_max=70):
    """
    变负载优化（最大效率模式）：目标函数使用 activation / load 比值。
    参数与 variable_load_optimization 相同。
    """
    centers, weights, scaler, sigma = rbf_params
    muscle_num = len(centers)
    assert len(w) == muscle_num

    if g == 0:
        c = 1
    if isinstance(s, (float, int)) or (hasattr(s, '__len__') and len(s) == 1):
        s = [s if isinstance(s, (float, int)) else s[0]] * muscle_num
    if isinstance(g, (float, int)) or (hasattr(g, '__len__') and len(g) == 1):
        g = [g if isinstance(g, (float, int)) else g[0]] * muscle_num

    xi = np.linspace(h_min, h_max, timestep_num)

    model = ConcreteModel()
    model.I = RangeSet(0, timestep_num - 1)
    model.J = RangeSet(0, muscle_num - 1)
    model.l = Var(model.I, within=NonNegativeReals,
                  initialize=(l_min + l_max) / 2)
    model.a = Var(model.I, model.J, within=NonNegativeReals,
                  initialize=g[0])

    model.constr = ConstraintList()
    for i in model.I:
        for j in model.J:
            model.constr.add(
                model.a[i, j] == _rbf_predict_pyomo(
                    xi[i], model.l[i],
                    centers[j], weights[j], scaler[j], sigma[j]))
            model.constr.add(model.a[i, j] <= s[j])
            model.constr.add(model.l[i] <= l_max)
            model.constr.add(model.l[i] >= l_min)
            if i > 0:
                model.constr.add(model.l[i] <= model.l[i - 1] + epsilon)
                model.constr.add(model.l[i] >= model.l[i - 1] - epsilon)

    obj = sum(
        sum(abs(model.a[i, j] / model.l[i] - g[j]) * w[j] for j in model.J)
        for i in model.I
    ) * (-1) ** c
    model.obj = Objective(expr=obj, sense=minimize)

    solver = SolverFactory('ipopt')
    solver.options['max_iter'] = max_iter
    solver.solve(model)

    load = np.array([value(model.l[i]) for i in model.I])
    activation = np.array([
        [value(model.a[i, j]) for j in model.J]
        for i in model.I
    ])
    return xi, load, activation


def variable_load_optimization_pspline(pspline_params_list, w, g=0.0, s=1, c=0,
                                        epsilon=0.5, max_iter=20000,
                                        h_min=1.35, h_max=2.05,
                                        timestep_num=100,
                                        l_min=20, l_max=70, tee=False):
    """
    变负载优化（P-spline 版）：用 monotone P-spline 拟合曲面作为 Pyomo 符号约束。

    实现路径：对每个时间步 i，高度 xi[i] 是常量，曲面 a(xi[i], ·) 退化为
    关于 l 的 1D 三次 B-spline。先数值地转为“截断幂次基”表示：
        a(xi[i], l) = A + B*l + C*l^2 + D*l^3
                    + Σ_k jump_k * max(0, l - break_k)^3
    在 Pyomo 中用 ((u + sqrt(u^2 + eps))/2)^3 实现 max(0, u)^3 的 C^∞ 光滑版本，
    满足 ipopt 内点法的二阶可微要求，同时保留 P-spline 曲面的精确性。

    Parameters
    ----------
    pspline_params_list : list of dict
        每块肌肉的 P-spline 参数（来自 monotone_pspline.fit_monotone_pspline_2d
        的返回字典，含 'theta', 'knots_h', 'knots_l', 'degree' 等键）。
    其它参数同 variable_load_optimization。
    """
    muscle_num = len(pspline_params_list)
    assert len(w) == muscle_num

    if g == 0:
        c = 1
    if isinstance(s, (float, int)) or (hasattr(s, '__len__') and len(s) == 1):
        s = [s if isinstance(s, (float, int)) else s[0]] * muscle_num
    if isinstance(g, (float, int)) or (hasattr(g, '__len__') and len(g) == 1):
        g = [g if isinstance(g, (float, int)) else g[0]] * muscle_num

    xi = np.linspace(h_min, h_max, timestep_num)

    # 预算每个 (时间步 i, 肌肉 j) 处曲面在 l 上的截断幂次系数
    coefs_table = [[None] * muscle_num for _ in range(timestep_num)]
    for i in range(timestep_num):
        for j in range(muscle_num):
            coefs_table[i][j] = _pspline_to_trunc_power(
                pspline_params_list[j], float(xi[i]))

    # 从曲面上查 a ≈ g 的负载作为 warm start；并 clip 到 [0, s_j] 内
    # 避免 a_init 负值或超上限导致 ipopt 起步就不可行。
    l_init_arr, a_init_arr = _warm_start_from_pspline(
        pspline_params_list, w, g, xi, l_min, l_max)
    a_init_arr = np.clip(a_init_arr, 0.0, None)
    for jj in range(muscle_num):
        a_init_arr[:, jj] = np.minimum(a_init_arr[:, jj], float(s[jj]))

    model = ConcreteModel()
    model.I = RangeSet(0, timestep_num - 1)
    model.J = RangeSet(0, muscle_num - 1)
    model.l = Var(model.I, bounds=(l_min, l_max),
                  initialize=lambda m, i: float(l_init_arr[i]))
    model.a = Var(model.I, model.J, within=NonNegativeReals,
                  initialize=lambda m, i, j: float(a_init_arr[i, j]))

    model.constr = ConstraintList()
    for i in model.I:
        for j in model.J:
            base, breaks, jumps = coefs_table[i][j]
            model.constr.add(
                model.a[i, j] == _pspline_predict_pyomo(
                    model.l[i], base, breaks, jumps))
            model.constr.add(model.a[i, j] <= s[j])
            if i > 0:
                model.constr.add(model.l[i] <= model.l[i - 1] + epsilon)
                model.constr.add(model.l[i] >= model.l[i - 1] - epsilon)

    # 目标函数：用松弛变量 dev[i, j] 把 |a[i, j] - g[j]| 拆掉，
    # 让 ipopt 看到的全是 C^∞ 光滑表达式、避免 abs 带来的 kink。
    if c == 0:
        model.dev = Var(model.I, model.J, within=NonNegativeReals,
                        initialize=0.0)
        for i in model.I:
            for j in model.J:
                model.constr.add(model.dev[i, j] >= model.a[i, j] - g[j])
                model.constr.add(model.dev[i, j] >= g[j] - model.a[i, j])
        obj = sum(model.dev[i, j] * w[j]
                  for i in model.I for j in model.J)
    else:
        # c==1（g 全为 0，a≥0 所以 |a-g|=a）→ 等价于最大化 sum w*a
        obj = -sum(model.a[i, j] * w[j]
                   for i in model.I for j in model.J)
    model.obj = Objective(expr=obj, sense=minimize)

    solver = SolverFactory('ipopt')
    solver.options['max_iter'] = max_iter
    solver.solve(model, tee=tee)

    load = np.array([value(model.l[i]) for i in model.I])
    activation = np.array([
        [value(model.a[i, j]) for j in model.J]
        for i in model.I
    ])
    return xi, load, activation


# ============================================================
#  编排函数
# ============================================================

def one_muscle_variable_load(subject, title, muscle_files, goal, epsilons,
                              max_iter, vload_dir, variable_mode=1,
                              plot_fn=None, rbf_predict_fn=None,
                              use_pspline=False, tee=False):
    """
    单肌肉（或肌肉组合）的变负载优化 + 绘图 + CSV 导出。

    Parameters
    ----------
    subject : Subject
        实验配置
    title : str
        肌肉标题
    muscle_files : list of str
        RBF 参数 pkl 文件名列表（{musc}_rbf_params.pkl）。use_pspline=True
        时会自动转换为对应的 {musc}_pspline_params.pkl。
    goal : float
        目标激活值
    epsilons : list of float
        平滑约束列表
    max_iter : int
        最大迭代次数
    vload_dir : str
        输出目录
    variable_mode : int
        优化模式 (1=目标, 2=最小化, 3=效率，仅 RBF)
    plot_fn : callable, optional
        绘图函数 (plot_variable_load_result)
    rbf_predict_fn : callable, optional
        RBF 预测函数（use_pspline=False 时用于绘图）
    use_pspline : bool, default False
        True 时使用 P-spline 曲面（从 {musc}_pspline_params.pkl 加载并在
        Pyomo 中以截断幂次基作 C² 光滑的符号求值）；False 时使用 RBF。
    """
    from digitaltwin.analysis.heatmap.rbf_fitting import rbf_predict as _rbf_predict
    rbf_predict_fn = rbf_predict_fn or _rbf_predict

    height_min, height_max = subject.height_range
    base = (subject.muscle_folder if subject.load_previous_data
            else os.path.join(subject.result_folder, 'heatmap/params'))

    # ---- 加载曲面参数 ----
    if use_pspline:
        psp_files = [
            f.replace('_rbf_params.pkl', '_pspline_params.pkl')
            for f in muscle_files
        ]
        pspline_list = []
        for file in psp_files:
            with open(os.path.join(base, file), 'rb') as f:
                pspline_list.append(pickle.load(f))
    else:
        centers, weights, scaler, sigma = [], [], [], []
        for file in muscle_files:
            with open(os.path.join(base, file), 'rb') as f:
                p = pickle.load(f)
                centers.append(p['centers'])
                weights.append(p['weights'])
                scaler.append(p['scaler'])
                sigma.append(p['sigma'])
        rbf_params = [centers, weights, scaler, sigma]

    # ---- 执行优化 ----
    heights, opti_loads, activations = [], [], []
    for eps in epsilons:
        kw = dict(w=[1] * len(muscle_files), epsilon=eps,
                  max_iter=max_iter, h_min=height_min, h_max=height_max)
        if variable_mode != 2:
            kw['g'] = goal

        if use_pspline:
            if variable_mode == 3:
                raise NotImplementedError(
                    'variable_mode=3 (效率最大化) 当前仅在 RBF 路径上实现。')
            h, l, a = variable_load_optimization_pspline(
                pspline_list, tee=tee, **kw)
        else:
            opt_fn = (variable_load_optimization_max if variable_mode == 3
                      else variable_load_optimization)
            h, l, a = opt_fn(rbf_params, **kw)

        heights.append(h)
        opti_loads.append(l)
        activations.append(a)

    # ---- 导出 CSV（机器人指令）----
    h_out = (-0.7 + heights[0]) if subject.turn_position else (0.7 - heights[0])
    df = pd.DataFrame({
        'Load_l': opti_loads[0] / 2, 'Height_l': h_out,
        'Load_r': opti_loads[0] / 2, 'Height_r': h_out,
    }).T
    csv_name = f'M{subject.target_motion}_{title}_{goal}.csv'
    df.to_csv(os.path.join(vload_dir, csv_name), index=False, header=False)

    # ---- 绘图 ----
    if plot_fn is not None:
        load_min, load_max = _get_load_range(subject)
        xi = np.linspace(height_min, height_max, 100)
        yi = np.linspace(load_min, load_max, 100)
        xi, yi = np.meshgrid(xi, yi)
        for j in range(len(muscle_files)):
            if use_pspline:
                from digitaltwin.analysis.heatmap.monotone_pspline import (
                    predict_monotone_pspline,
                )
                zi = predict_monotone_pspline(pspline_list[j], xi, yi)
            else:
                zi = rbf_predict_fn(
                    (xi.flatten(), yi.flatten()),
                    centers[j], weights[j], scaler[j], sigma[j]
                ).reshape(xi.shape)
            # 诊断用：检查 goal 是否落在曲面取值范围内，
            # 否则 ax1.contour(levels=[goal]) 会静默不画白色虚线。
            zi_min = float(np.nanmin(zi))
            zi_max = float(np.nanmax(zi))
            in_range = zi_min <= goal <= zi_max
            print(f'[vload-plot] {title}[m{j}] '
                  f'zi.min={zi_min:.3f}, zi.max={zi_max:.3f}, '
                  f'goal={goal} -> goal_in_range={in_range}')
            plot_fn(subject, xi, yi, zi, heights, opti_loads,
                    activations, epsilons, j, goal, title, vload_dir)

    return heights, opti_loads, activations


def generate_variable_load(subject, variable_mode=1, plot_fn=None,
                            use_pspline=False, tee=False):
    """
    批量生成变负载优化结果（对 subject.titles 中的所有肌肉）。

    Parameters
    ----------
    subject : Subject
    variable_mode : int
    plot_fn : callable, optional
        plot_variable_load_result 函数引用
    use_pspline : bool, default False
        True 时使用 P-spline 曲面作为优化中的激活曲面（需 generate_heatmaps
        已产出 {musc}_pspline_params.pkl）；False 时使用 RBF。
    """
    titles = subject.titles
    goals = subject.goal
    max_iter = subject.max_iter or [10000] * len(titles)
    epsilons = subject.epsilons or [0.5]

    vload_dir = os.path.join(subject.result_folder, 'vload')
    os.makedirs(vload_dir, exist_ok=True)

    for i in range(len(titles)):
        if isinstance(titles[i], list):
            title = '+'.join(titles[i])
            muscle_files = [f'{x}_rbf_params.pkl' for x in titles[i]]
        else:
            title = titles[i]
            muscle_files = [f'{titles[i]}_rbf_params.pkl']

        one_muscle_variable_load(
            subject, title, muscle_files, goals[i], epsilons,
            max_iter[i], vload_dir, variable_mode, plot_fn=plot_fn,
            use_pspline=use_pspline, tee=tee)


# ============================================================
#  工具函数
# ============================================================

def _get_load_range(subject):
    """从 subject 获取负载范围"""
    if subject.load_range is not None:
        return subject.load_range
    keys = [int(k) for k in subject.robot_files.keys()]
    return [min(keys), max(keys)]