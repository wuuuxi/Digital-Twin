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
    NonNegativeReals, minimize, value, exp,
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


# ============================================================
#  编排函数
# ============================================================

def one_muscle_variable_load(subject, title, muscle_files, goal, epsilons,
                              max_iter, vload_dir, variable_mode=1,
                              plot_fn=None, rbf_predict_fn=None):
    """
    单肌肉（或肌肉组合）的变负载优化 + 绘图 + CSV 导出。

    Parameters
    ----------
    subject : Subject
        实验配置
    title : str
        肌肉标题
    muscle_files : list of str
        RBF 参数 pkl 文件名列表
    goal : float
        目标激活值
    epsilons : list of float
        平滑约束列表
    max_iter : int
        最大迭代次数
    vload_dir : str
        输出目录
    variable_mode : int
        优化模式 (1=目标, 2=最小化, 3=效率)
    plot_fn : callable, optional
        绘图函数 (plot_variable_load_result)
    rbf_predict_fn : callable, optional
        RBF 预测函数
    """
    from digitaltwin.analysis.rbf_fitting import rbf_predict as _rbf_predict
    rbf_predict_fn = rbf_predict_fn or _rbf_predict

    height_min, height_max = subject.height_range

    # 加载 RBF 参数
    centers, weights, scaler, sigma = [], [], [], []
    for file in muscle_files:
        # base = (subject.muscle_folder if subject.load_previous_data
        #         else subject.result_folder)
        base = (subject.muscle_folder if subject.load_previous_data
                else os.path.join(subject.result_folder, 'heatmap/params'))
        with open(os.path.join(base, file), 'rb') as f:
            p = pickle.load(f)
            centers.append(p['centers'])
            weights.append(p['weights'])
            scaler.append(p['scaler'])
            sigma.append(p['sigma'])

    rbf_params = [centers, weights, scaler, sigma]

    # 执行优化
    heights, opti_loads, activations = [], [], []
    for eps in epsilons:
        opt_fn = (variable_load_optimization_max if variable_mode == 3
                  else variable_load_optimization)
        kw = dict(w=[1] * len(muscle_files), epsilon=eps,
                  max_iter=max_iter, h_min=height_min, h_max=height_max)
        if variable_mode != 2:
            kw['g'] = goal
        h, l, a = opt_fn(rbf_params, **kw)
        heights.append(h)
        opti_loads.append(l)
        activations.append(a)

    # 导出 CSV（机器人指令）
    h_out = (-0.7 + heights[0]) if subject.turn_position else (0.7 - heights[0])
    df = pd.DataFrame({
        'Load_l': opti_loads[0] / 2, 'Height_l': h_out,
        'Load_r': opti_loads[0] / 2, 'Height_r': h_out,
    }).T
    csv_name = f'M{subject.target_motion}_{title}_{goal}.csv'
    df.to_csv(os.path.join(vload_dir, csv_name), index=False, header=False)

    # 绘图
    if plot_fn is not None:
        load_min, load_max = _get_load_range(subject)
        xi = np.linspace(height_min, height_max, 100)
        yi = np.linspace(load_min, load_max, 100)
        xi, yi = np.meshgrid(xi, yi)
        for j in range(len(muscle_files)):
            zi = rbf_predict_fn(
                (xi.flatten(), yi.flatten()),
                centers[j], weights[j], scaler[j], sigma[j]
            ).reshape(xi.shape)
            plot_fn(subject, xi, yi, zi, heights, opti_loads,
                    activations, epsilons, j, goal, title, vload_dir)

    return heights, opti_loads, activations


def generate_variable_load(subject, variable_mode=1, plot_fn=None):
    """
    批量生成变负载优化结果（对 subject.titles 中的所有肌肉）。

    Parameters
    ----------
    subject : Subject
    variable_mode : int
    plot_fn : callable, optional
        plot_variable_load_result 函数引用
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
            max_iter[i], vload_dir, variable_mode, plot_fn=plot_fn)


# ============================================================
#  工具函数
# ============================================================

def _get_load_range(subject):
    """从 subject 获取负载范围"""
    if subject.load_range is not None:
        return subject.load_range
    keys = [int(k) for k in subject.robot_files.keys()]
    return [min(keys), max(keys)]