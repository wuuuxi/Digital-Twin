"""
digitaltwin/data/insole/calibration.py

批量标定 insole_time_offset 并写回 json。

这里是整个标定流程的主体，example 只负责改参数和调用：

    calibrate_insole_offsets(subject, ...)   # 标定 + 写回 + 画图
    check_insole_offsets(subject)            # 跑分析前的预检查

写回 json 的内容
----------------
    "insole_time_offset": 1.234

只写这一个字段，不再往 json 里写 corr / method / calibrated_at 等元数据。
标定质量（相关系数、拟合时长、段数、是否写回）仍然会在标定时由
print_report 打印到终端，供当场判断；不可信的组（corr < min_corr）
本来就不会被写回，所以 json 里出现 insole_time_offset 就意味着它通过了阀值。
"""
import os

import numpy as np

from digitaltwin.utils.logger import beauty_print
from digitaltwin.data.robot_processor import RobotProcessor
from . import io as insole_io
from .sync import estimate_time_offset
from .timebase import OFFSET_KEY, has_offset


def _data_groups(subject):
    return subject.config.get('modeling_file', {}).get('data', {}) or {}


def load_robot_reference(subject, load_key, file_info, verbose=True):
    """读机器人参考力，返回 (time_from_zero, force_l + force_r)。

    用左右之和而不是单侧：与鞋垫 L+R 同量纲、同相，且对单侧噪声不敏感。
    """
    robot_file = (file_info or {}).get('robot_file')
    if not robot_file:
        beauty_print('组 {}：缺 robot_file，跳过'.format(load_key),
                     type='warning')
        return None, None

    try:
        data = RobotProcessor.process(
            robot_file,
            file_info.get('load_kg'),
            robot_folder=getattr(subject, 'modeling_robot_folder', None),
            folder=subject.folder,
            turn_position=getattr(subject, 'turn_position', False),
        )
    except Exception as e:
        beauty_print('组 {}：机器人数据读取失败 ({})'.format(load_key, e),
                     type='warning')
        return None, None

    if data is None or 'time' not in data:
        beauty_print('组 {}：机器人数据为空，跳过'.format(load_key),
                     type='warning')
        return None, None

    t = np.asarray(data['time'], dtype=float)
    t = t - t[0]
    f = (np.asarray(data['force_l'], dtype=float)
         + np.asarray(data['force_r'], dtype=float))
    if verbose:
        print('  [Robot] {}: {} frames, {:.1f}s'.format(
            os.path.basename(str(robot_file)), t.size, t[-1] - t[0]))
    return t, f


def load_insole_raw(subject, file_info, key, verbose=False):
    """读单侧鞋垫的【原始】时间与力。

    标定阶段必须不带任何 offset：否则就是拿已经移过位的信号去重新求移位，
    重跑一次就多加一个 offset。
    """
    rel = (file_info or {}).get(key)
    if not rel:
        return None, None
    path = insole_io.resolve_insole_path(subject, rel)
    if path is None:
        beauty_print('鞋垫文件未找到: {}'.format(rel), type='warning')
        return None, None
    return insole_io.load(path, verbose=verbose, time_offset=None)


def calibrate_group(subject, load_key, file_info, max_lag=30.0, corr_thr=0.5,
                    verbose=True):
    """标定单一采集组，返回 (info, t_rob, f_rob)。失败时 info 为 None。"""
    if verbose:
        print('\n=== 组 {} ==='.format(load_key))

    tl, fl = load_insole_raw(subject, file_info, 'insole_file_l',
                             verbose=verbose)
    tr, fr = load_insole_raw(subject, file_info, 'insole_file_r',
                             verbose=verbose)
    if tl is None or tr is None:
        beauty_print('组 {}：左右鞋垫文件不齐，无法标定'.format(load_key),
                     type='warning')
        return None, None, None

    t_rob, f_rob = load_robot_reference(subject, load_key, file_info,
                                        verbose=verbose)
    if t_rob is None:
        return None, None, None

    info = estimate_time_offset(tl, fl, tr, fr, t_rob, f_rob,
                                max_lag=max_lag, corr_thr=corr_thr,
                                verbose=verbose)
    return info, t_rob, f_rob


def calibrate_insole_offsets(subject, load_keys=None, write_json=True,
                             min_corr=0.5, max_lag=30.0, corr_thr=0.5,
                             plot=True, show=True, verbose=True):
    """批量标定鞋垫时间偏移，写回 json，并画诊断图。

    Parameters
    ----------
    subject : Subject
    load_keys : list[str], optional -- None 表示全部采集组
    write_json : bool -- 是否把结果写回配置文件
    min_corr : float -- 低于此相关系数不写回，只报告
    max_lag : float -- 滞后搜索范围 ±(s)
    corr_thr : float -- 左右一致性门槛，决定哪些段算深蹲
    plot / show : bool

    Returns
    -------
    dict -- {load_key: {'offset','corr','reliable','written',...}}
    """
    groups = _data_groups(subject)
    keys = list(groups.keys()) if load_keys is None else list(load_keys)

    report = {}
    figs = []
    n_written = 0

    for load_key in keys:
        file_info = groups.get(str(load_key))
        if file_info is None:
            beauty_print('配置中没有组 {}，跳过'.format(load_key),
                         type='warning')
            continue

        info, t_rob, f_rob = calibrate_group(
            subject, load_key, file_info, max_lag=max_lag,
            corr_thr=corr_thr, verbose=verbose)
        if info is None:
            report[str(load_key)] = {'offset': None, 'corr': None,
                                     'reliable': False, 'written': False}
            continue

        ok = bool(abs(info['corr']) >= min_corr and info['reliable'])

        if write_json and ok:
            # 只写 offset。标定质量信息只进终端报表，不进配置文件。
            file_info[OFFSET_KEY] = round(float(info['offset']), 4)
            n_written += 1
        elif write_json and not ok:
            beauty_print(
                '组 {}：corr={:.3f}, reliable={}，不写回 json。'
                '写一个不可信的 offset 比不写更危险——后续流程会把它当成'
                '已标定而不再告警。'.format(
                    load_key, info['corr'], info['reliable']),
                type='warning')

        report[str(load_key)] = {
            'offset': float(info['offset']),
            'corr': float(info['corr']),
            'reliable': bool(info['reliable']),
            'at_edge': bool(info['at_edge']),
            'fit_duration_s': float(info['fit_duration_s']),
            'n_segments': int(len(info['segments'])),
            'fallback': bool(info['fallback']),
            'written': bool(write_json and ok),
        }

        if plot:
            # 延迟导入：不画图时不应该强制依赖 matplotlib
            from digitaltwin.visualization.insole_sync_plot import (
                plot_sync_diagnosis)
            figs.append(plot_sync_diagnosis(str(load_key), info,
                                            t_rob, f_rob))

    if write_json and n_written:
        subject.save_config()
        if verbose:
            print('\n已写回 {} 组的 {} 到 {}'.format(
                n_written, OFFSET_KEY, subject.config_path))

    if plot and figs and show:
        import matplotlib.pyplot as plt
        plt.show()

    return report


def check_insole_offsets(subject, load_keys=None, verbose=True):
    """跑分析前的预检查：哪些组还没有 insole_time_offset。

    放在流水线开始前调，一次性把缺失情况告诉使用者，而不是等读到第
    一个文件时才一条一条地警告。

    Returns
    -------
    list[str] -- 缺少标定的组 key
    """
    groups = _data_groups(subject)
    keys = list(groups.keys()) if load_keys is None else list(load_keys)

    missing = [str(k) for k in keys if not has_offset(groups.get(str(k), {}))]

    if missing:
        beauty_print(
            '以下采集组没有 {}：{}\n'
            '这些组的鞋垫与机器人【未对齐】，两路力会整体错位。\n'
            '请先运行 example_insole_sync_offset.py 完成互相关标定。'.format(
                OFFSET_KEY, ', '.join(missing)),
            type='warning')
    elif verbose:
        print('[Insole] 所有采集组均已标定 {}。'.format(OFFSET_KEY))

    return missing


def print_report(report):
    """把标定结果打成一张表。

    json 里只保留 insole_time_offset，所以这张表是判断标定质量的唯一场合：
      - corr 区分“标得准”与“拟合失败但也给了个数”；
      - fit(s) / segs 暂露样本量不足（只用 2 s、1 段），这是单看 corr
        发现不了的；
      - written=False 的组没有写回，需要先查数据再重标。
    建议标定时把这张表连同日期一并记在实验记录里。
    """
    print('\n' + '=' * 78)
    print('{:<8}{:>12}{:>10}{:>12}{:>10}{:>10}{:>10}'.format(
        'group', 'offset(s)', 'corr', 'fit(s)', 'segs', 'reliable', 'written'))
    print('-' * 78)
    for key, r in report.items():
        if r.get('offset') is None:
            print('{:<8}{:>12}{:>10}{:>12}{:>10}{:>10}{:>10}'.format(
                key, '-', '-', '-', '-', 'False', 'False'))
            continue
        print('{:<8}{:>+12.3f}{:>10.3f}{:>12.1f}{:>10d}{:>10}{:>10}'.format(
            key, r['offset'], r['corr'], r['fit_duration_s'],
            r['n_segments'], str(r['reliable']), str(r['written'])))
    print('=' * 78)