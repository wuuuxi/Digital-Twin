import numpy as np

class MuscleStateManager:
    """肌肉激活计算与管理类"""

    def __init__(self, motion_label):
        self.motion_label = motion_label
        self.armlenth = 0.37
        self.ShouHeight = 0.45
        UpperLimbMusR1 = [-2.83847443e-02, 3.78972232e-04, 5.73952879e-04,
                          -4.93123956e-07, -4.39014837e-06, -1.86593838e-06]
        UpperLimbMusR2 = [-1.59330326e-01, 2.30448754e-03, 2.35025582e-03,
                          -5.08526370e-06, -2.24674247e-05, -6.13742225e-07]
        UpperLimbMusR3 = [-1.37639327e-01, 2.00722246e-03, 2.18952440e-03,
                          -4.10670254e-06, -2.10775861e-05, -2.03264374e-06]
        UpperLimbMusR4 = [-2.37691947e-03, 7.78713575e-05, 5.35114834e-05,
                          5.08632500e-08, -8.37050041e-07, -6.00804626e-08]
        UpperLimbMusR5 = [-1.01442551e-01, 1.47043898e-03, 1.56184263e-03,
                          -3.23510388e-06, -1.43861358e-05, -1.25572653e-06]
        UpperLimbMusR6 = [-1.07357358e-03, 1.15538266e-04, 2.14702498e-04,
                          -7.47505564e-07, -1.02141327e-07, -2.59922512e-06]
        UpperLimbMusR = [UpperLimbMusR1, UpperLimbMusR2, UpperLimbMusR3,
                         UpperLimbMusR4, UpperLimbMusR5, UpperLimbMusR6]
        self.UpperLimbMusR = UpperLimbMusR

    def _height_to_joint_angles_bp(self, height):
        # height_remap = (height-0.68) / self.armlenth *37 + 0.68
        if height < 0.68:
            height = 0.68
        elif height > 0.68 + self.armlenth:
            height = 0.68 + self.armlenth
        P1 = [-88.62, 431.6, -283.8]  # arm_flex
        P2 = [507.7, -786.8, 255.1]  # armabd
        P3 = [-90.15, 379.6, -255.7]  # arm rot
        P4 = [-442.9, 416.9, 51.31]  # elb flex
        P5 = [-154, 333.8, -63.38]  # prosup
        P6 = [-48.62, 83.46, -45.51]  # wstFlex
        P7 = [-17.15, 23.32, -6.703]  # wstDev
        P = [P1, P2, P3, P4, P5, P6, P7]

        qt = np.zeros(7)

        for i in range(7):
            qt[i] = P[i][0] * height ** 2 + P[i][1] * height + P[i][2]

        return qt

    def _height_to_joint_angles_sq(self, height):
        height_remap = (height - 0.68) / self.ShouHeight * 37 + 0.68
        P1 = [-549.4, 1116, -461.8]  # hip flexion
        P2 = [52.59, -124.4, 57.34]  # hip adduction
        P3 = [127.5, -382.4, 276.1]  # hip rotation
        P4 = [-642.7, 1364, -628.9]  # knee flexion
        P5 = [-267, 616, -328.4]  # ankle flexion

        P = [P1, P2, P3, P4, P5]

        qt = np.zeros(5)

        for i in range(0, 5):
            qt[i] = P[i][0] * height ** 2 + P[i][1] * height + P[i][2]

        return qt

    def _height_to_joint_angles_sq_pelvis(self, height):
        height_remap = (height - 0.68) / self.ShouHeight * 37 + 0.68
        P1 = [165.6, -351.8, 162.8]  # hip flexion
        P2 = [27.08, -63.36, 35.16]  # hip adduction
        P3 = [-17.25, 48.61, -31.55]  # hip rotation
        P4 = [1.854, -4.177, 2.224]  # knee flexion
        P5 = [0.3581, 0.167, -0.7378]  # ankle flexion
        P6 = [-1.62, 4.064, -2.76]  # ankle flexion

        P = [P1, P2, P3, P4, P5, P6]

        qt = np.zeros(6)

        for i in range(0, 6):
            qt[i] = P[i][0] * height ** 2 + P[i][1] * height + P[i][2]

        return qt

    def calculate_joint_angle(self, input):
        if self.motion_label == 'benchpress':
            qt_r = self._height_to_joint_angles_bp(input[2])
            qt_l = self._height_to_joint_angles_bp(input[3])
            return [qt_r, qt_l]
        elif self.motion_label == 'squat':
            qt_r = self._height_to_joint_angles_sq(input[2])
            qt_l = self._height_to_joint_angles_sq(input[3])
            qt_pelv = self._height_to_joint_angles_sq_pelvis((input[2] + input[3]) / 2)
            return [qt_r, qt_l, qt_pelv]

    def calculate_activation(self, input):
        qt_r, qt_l = self.calculate_joint_angle(input)

        if len(input) == 4:
            q_R_Shou_Flex = qt_r[3]
            q_R_Elbow_Flex = qt_r[0]
            q_L_Shou_Flex = qt_l[3]
            q_L_Elbow_Flex = qt_l[0]
        else:
            q_R_Shou_Flex = 0
            q_R_Elbow_Flex = 0
            q_L_Shou_Flex = 0
            q_L_Elbow_Flex = 0

        w1 = 1
        w2 = 1
        act = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        for i in range(6):
            tempAct = (self.UpperLimbMusR[i][0] +
                       self.UpperLimbMusR[i][1] * q_R_Shou_Flex * w1 +
                       self.UpperLimbMusR[i][2] * q_R_Elbow_Flex * w2 +
                       self.UpperLimbMusR[i][3] * q_R_Shou_Flex ** 2 +
                       self.UpperLimbMusR[i][4] * q_R_Shou_Flex * q_R_Elbow_Flex +
                       self.UpperLimbMusR[i][5] * q_R_Elbow_Flex ** 2)
            act[i] = max(0.0, min(1.0, tempAct))

        for i in range(0, 6):
            tempAct = (self.UpperLimbMusR[i][0] +
                       self.UpperLimbMusR[i][1] * q_L_Shou_Flex * w1 +
                       self.UpperLimbMusR[i][2] * q_L_Elbow_Flex * w2 +
                       self.UpperLimbMusR[i][3] * q_L_Shou_Flex ** 2 +
                       self.UpperLimbMusR[i][4] * q_L_Shou_Flex * q_L_Elbow_Flex +
                       self.UpperLimbMusR[i][5] * q_L_Elbow_Flex ** 2)
            act[i + 6] = max(0.0, min(1.0, tempAct))

        return act