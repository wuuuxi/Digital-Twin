"""
digitaltwin/visualization/insole_sync_plot.py

鞋垫 / 机器人时间同步的三联诊断图。

图内文字全部用英文：matplotlib 默认字体 (DejaVu Sans) 没有中文字形，
改 rcParams 又依赖本机装了什么字体，不可移植。
"""
import matplotlib.pyplot as plt
import numpy as np


def zscore(x):
    """标准化。两路信号量纲差一个量级，不归一化就只能看到其中一条。"""
    x = np.asarray(x, float)
    s = np.nanstd(x)
    return (x - np.nanmean(x)) / (s if s > 1e-12 else 1.0)


def plot_sync_diagnosis(load_key, info, t_rob, f_rob, t_ins=None, f_ins=None):
    """画一组的三联诊断图。

    子图 1 Before：鞋垫原始时间 vs 机器人。看错位有多大。
    子图 2 After ：鞋垫时间 + offset vs 机器人。验收图，两条曲线峰谷应重合。
    子图 3       ：深蹲段上的滞后-相关曲线。峰尖锐且高 = 标定可信；
                    峰宽、多峰、或峰在边界 = 不可信。
    灰色阴影带  ：被判定为深蹲、参与拟合的时间段。

    Parameters
    ----------
    load_key : str
    info : dict -- sync.estimate_time_offset 的返回值
    t_rob, f_rob : array-like -- 机器人时间与参考力
    t_ins, f_ins : array-like, optional -- 鞋垫时间与力。默认用 info 里的
        grid / insole_total，与拟合时用的信号一致。
    """
    if t_ins is None or f_ins is None:
        t_ins, f_ins = info['grid'], info['insole_total']

    fig, axes = plt.subplots(3, 1, figsize=(11, 8), constrained_layout=True)
    fig.suptitle(
        '{}: insole vs robot  offset={:+.3f}s, corr={:.3f}, overlap={:.0%}'
        .format(load_key, info['offset'], info['corr'],
                info.get('overlap_fraction', 1.0)))

    for ax, shift, title in (
            (axes[0], 0.0, 'Before: raw insole time'),
            (axes[1], info['offset'], 'After: insole time + offset')):
        ax.plot(np.asarray(t_ins) + shift, zscore(f_ins),
                label='insole L+R', lw=1.2)
        ax.plot(t_rob, zscore(f_rob), label='robot force', lw=1.2, alpha=0.8)
        for (s, e) in info['segments']:
            ax.axvspan(s + shift, e + shift, color='0.85', zorder=0)
        ax.set_title(title, fontsize=10)
        ax.set_ylabel('z-score')
        ax.legend(loc='upper right', fontsize=8)

    axes[2].plot(info['lags'], info['corrs'], lw=1.0)
    axes[2].axvline(info['offset'], color='r', ls='--',
                    label='peak {:+.3f}s'.format(info['offset']))
    axes[2].axhline(0.0, color='0.7', lw=0.8)
    axes[2].set_xlabel('lag (s)   [insole time + lag -> robot time]')
    axes[2].set_ylabel('Pearson r')
    axes[2].set_title('Cross-correlation over motion segments', fontsize=10)
    axes[2].legend(loc='upper right', fontsize=8)
    return fig
