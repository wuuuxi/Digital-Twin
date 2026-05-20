"""
基于交互力与加速度自行估算实际负载，并可视化：
    位置-速度 / 位置-估算负载 / 位置-交互力均值

估算公式：
    estimated_load (kg) = (force_l + force_r) / ((acc_l + acc_r) / 2 + g)
    g = 9.81 m/s²

用法：
    python example_load_estimation.py
"""
import matplotlib.pyplot as plt
from digitaltwin import Subject, MultiLoadPipeline


def main():
    subject = Subject('../config/20260513_squat_FTS09_mvc.json')
    pipeline = MultiLoadPipeline(subject)
    pipeline.debug = True

    pipeline.run(include_xsens=False)
    pipeline.visualize_load_estimation(movement_types=['upward'])
    plt.show()


if __name__ == '__main__':
    main()