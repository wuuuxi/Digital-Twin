import numpy as np


class MuscleStateManager:
    """肌肉激活计算与管理类"""

    # 上肢肌肉回归系数 (6组肌肉 × 6个多项式系数)
    UPPER_LIMB_REGRESSION_COEFFS = [
        [-2.83847443e-02, 3.78972232e-04, 5.73952879e-04,
         -4.93123956e-07, -4.39014837e-06, -1.86593838e-06],
        [-1.59330326e-01, 2.30448754e-03, 2.35025582e-03,
         -5.08526370e-06, -2.24674247e-05, -6.13742225e-07],
        [-1.37639327e-01, 2.00722246e-03, 2.18952440e-03,
         -4.10670254e-06, -2.10775861e-05, -2.03264374e-06],
        [-2.37691947e-03, 7.78713575e-05, 5.35114834e-05,
         5.08632500e-08, -8.37050041e-07, -6.00804626e-08],
        [-1.01442551e-01, 1.47043898e-03, 1.56184263e-03,
         -3.23510388e-06, -1.43861358e-05, -1.25572653e-06],
        [-1.07357358e-03, 1.15538266e-04, 2.14702498e-04,
         -7.47505564e-07, -1.02141327e-07, -2.59922512e-06],
    ]

    def __init__(self, motion_label):
        self.motion_label = motion_label
        self.arm_length = 0.37
        self.shoulder_height = 0.45

    def _height_to_joint_angles_bp(self, height):
        """卧推: 根据手部高度计算关节角度 (7个自由度)"""
        height = np.clip(height, 0.68, 0.68 + self.arm_length)

        poly_coeffs = [
            [-88.62, 431.6, -283.8],   # arm_flex
            [507.7, -786.8, 255.1],     # arm_abd
            [-90.15, 379.6, -255.7],    # arm_rot
            [-442.9, 416.9, 51.31],     # elb_flex
            [-154, 333.8, -63.38],      # pro_sup
            [-48.62, 83.46, -45.51],    # wst_flex
            [-17.15, 23.32, -6.703],    # wst_dev
        ]
        qt = np.array([p[0] * height**2 + p[1] * height + p[2]
                        for p in poly_coeffs])
        return qt

    def _height_to_joint_angles_sq(self, height):
        """深蹲: 根据高度计算下肢关节角度 (5个自由度)"""
        poly_coeffs = [
            [-549.4, 1116, -461.8],     # hip_flexion
            [52.59, -124.4, 57.34],     # hip_adduction
            [127.5, -382.4, 276.1],     # hip_rotation
            [-642.7, 1364, -628.9],     # knee_flexion
            [-267, 616, -328.4],        # ankle_flexion
        ]
        qt = np.array([p[0] * height**2 + p[1] * height + p[2]
                        for p in poly_coeffs])
        return qt

    def _height_to_joint_angles_sq_pelvis(self, height):
        """深蹲: 根据高度计算骨盆关节角度 (6个自由度)"""
        poly_coeffs = [
            [165.6, -351.8, 162.8],     # hip_flexion
            [27.08, -63.36, 35.16],     # hip_adduction
            [-17.25, 48.61, -31.55],    # hip_rotation
            [1.854, -4.177, 2.224],     # knee_flexion
            [0.3581, 0.167, -0.7378],   # ankle_flexion
            [-1.62, 4.064, -2.76],      # ankle_flexion_2
        ]
        qt = np.array([p[0] * height**2 + p[1] * height + p[2]
                        for p in poly_coeffs])
        return qt

    def calculate_joint_angle(self, input_data):
        """根据运动类型计算关节角度"""
        if self.motion_label == 'benchpress':
            qt_r = self._height_to_joint_angles_bp(input_data[2])
            qt_l = self._height_to_joint_angles_bp(input_data[3])
            return [qt_r, qt_l]
        elif self.motion_label == 'squat':
            qt_r = self._height_to_joint_angles_sq(input_data[2])
            qt_l = self._height_to_joint_angles_sq(input_data[3])
            qt_pelv = self._height_to_joint_angles_sq_pelvis(
                (input_data[2] + input_data[3]) / 2)
            return [qt_r, qt_l, qt_pelv]

    def calculate_activation(self, input_data):
        """计算左右共12组肌肉的激活水平"""
        qt_r, qt_l = self.calculate_joint_angle(input_data)

        if len(input_data) == 4:
            shoulder_flex_r, elbow_flex_r = qt_r[3], qt_r[0]
            shoulder_flex_l, elbow_flex_l = qt_l[3], qt_l[0]
        else:
            shoulder_flex_r = elbow_flex_r = 0
            shoulder_flex_l = elbow_flex_l = 0

        activations = [0.0] * 12
        coeffs = self.UPPER_LIMB_REGRESSION_COEFFS

        # 右侧6组肌肉
        for i in range(6):
            c = coeffs[i]
            act = (c[0]
                   + c[1] * shoulder_flex_r
                   + c[2] * elbow_flex_r
                   + c[3] * shoulder_flex_r ** 2
                   + c[4] * shoulder_flex_r * elbow_flex_r
                   + c[5] * elbow_flex_r ** 2)
            activations[i] = np.clip(act, 0.0, 1.0)

        # 左侧6组肌肉
        for i in range(6):
            c = coeffs[i]
            act = (c[0]
                   + c[1] * shoulder_flex_l
                   + c[2] * elbow_flex_l
                   + c[3] * shoulder_flex_l ** 2
                   + c[4] * shoulder_flex_l * elbow_flex_l
                   + c[5] * elbow_flex_l ** 2)
            activations[i + 6] = np.clip(act, 0.0, 1.0)

        return activations