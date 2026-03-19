import opensim as osim
import numpy as np
import math
from .muscle_state import MuscleStateManager
from digitaltwin.utils.logger import beauty_print


class OpenSimModel:
    """OpenSim模型管理类"""

    # 各运动类型的初始坐标索引和角度值
    INIT_COORDS = {
        'squat': {
            'indices': [51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64],
            'values':  [50, -40, -90, 125, 90, -70, -15, 50, -40, -90, 125, 90, -70, -15],
        },
        'benchpress': {
            'indices': [1, 8, 11, 13, 18, 53, 60, 55, 62],
            'values':  [90, -25, -25, 85, 85, 90, 90, 90, 90],
        },
        'default': {
            'indices': [53, 60, 55, 62],
            'values':  [90, 90, 90, 90],
        },
    }

    # 各运动类型的坐标映射 (用于 update_state_q)
    BP_RIGHT_COORDS = [51, 52, 53, 54, 55, 56, 57]
    BP_LEFT_COORDS = [58, 59, 60, 61, 62, 63, 64]
    SQ_RIGHT_COORDS = [7, 8, 9, 13, 15]
    SQ_LEFT_COORDS = [10, 11, 12, 18, 20]
    SQ_PELVIS_COORDS = [1, 2, 3, 4, 5, 6]

    # 肌肉组索引 (右侧6组 + 左侧6组)
    MUSCLE_GROUPS = [
        # 右侧肌肉组
        [30, 31, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44],
        [16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29],
        [12, 13, 14, 15],
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
        [215, 216, 217, 218, 219, 220],
        [242, 243, 244, 245, 246, 247],
        # 左侧肌肉组
        [145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156],
        [131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144],
        [127, 128, 129, 130],
        [116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126],
        [227, 228, 229, 230, 231, 232],
        [465, 466, 467, 468, 469, 470],
    ]

    DISPLAY_MULTIPLIER = 23

    def __init__(self, opensim_settings, motion_label):
        self.model_path = opensim_settings.get("model_path")
        self.model_name = opensim_settings.get(
            "model_name", 'whole body model_yuetian_realtime.osim')
        self.model_file_path = f"{self.model_path}/{self.model_name}"
        self.geometry_path = opensim_settings.get(
            "geometry_path", f"{self.model_path}/Geometry")

        coord_cfg = self.INIT_COORDS.get(motion_label, self.INIT_COORDS['default'])
        self.init_coord_indices = coord_cfg['indices']
        self.init_coord_values = coord_cfg['values']

        self.motion_label = motion_label
        self.model = None
        self.state = None
        self.visualizer = None
        self.muscle_state = MuscleStateManager(self.motion_label)

        self.load_model()

    def load_model(self):
        """加载OpenSim模型"""
        try:
            self.model = osim.Model(self.model_file_path)
            osim.ModelVisualizer.addDirToGeometrySearchPaths(self.geometry_path)

            self.model.setUseVisualizer(True)
            self.state = self.model.initSystem()
            init_state = osim.State(self.state)

            self.visualizer = self.model.updVisualizer()
            self.visualizer.show(init_state)

            beauty_print("OpenSim model loaded successfully")
            return True
        except Exception as e:
            beauty_print(f"Error loading OpenSim model: {e}", type='warning')
            return False

    def initialize_pose(self):
        """设置初始姿态"""
        q_rad = np.deg2rad(self.init_coord_values)
        for i, coord_idx in enumerate(self.init_coord_indices):
            coord = self.model.updCoordinateSet().get(coord_idx - 1)
            coord.setValue(self.state, q_rad[i])
        # 设置骨盆高度使双脚接触地面
        pelvis_coord = self.model.updCoordinateSet().get(4)  # index 5-1
        pelvis_coord.setValue(self.state, 0.52)

    def _set_coordinates(self, coord_indices, angles_rad):
        """批量设置坐标值 (通用辅助方法)"""
        for i, coord_idx in enumerate(coord_indices):
            coord = self.model.updCoordinateSet().get(coord_idx - 1)
            coord.setValue(self.state, angles_rad[i])

    def update_state_q(self, input_data):
        """更新选定坐标的状态变量 q"""
        if len(input_data) != 4:
            return

        if self.motion_label == 'benchpress':
            qt_r, qt_l = self.muscle_state.calculate_joint_angle(input_data)
            q_rad_r = np.deg2rad(qt_r)
            q_rad_l = np.deg2rad(qt_l)
            self._set_coordinates(self.BP_RIGHT_COORDS, q_rad_r)
            self._set_coordinates(self.BP_LEFT_COORDS, q_rad_l)

        elif self.motion_label == 'squat':
            qt_r, qt_l, qt_pelv = self.muscle_state.calculate_joint_angle(input_data)
            q_rad_r = np.deg2rad(qt_r)
            q_rad_l = np.deg2rad(qt_l)

            # 骨盆: 前3个为角度需转弧度，后3个为平移量保持不变
            q_rad_pelv = np.zeros(6)
            q_rad_pelv[:3] = np.deg2rad(qt_pelv[:3])
            q_rad_pelv[3:] = qt_pelv[3:]
            q_rad_pelv[4] += 0.7  # 骨盆高度补偿

            self._set_coordinates(self.SQ_RIGHT_COORDS, q_rad_r)
            self._set_coordinates(self.SQ_LEFT_COORDS, q_rad_l)
            self._set_coordinates(self.SQ_PELVIS_COORDS, q_rad_pelv)

    def update_state_activation(self, input_data):
        """更新肌肉激活水平"""
        activations = self.muscle_state.calculate_activation(input_data)
        scaled_act = [a * self.DISPLAY_MULTIPLIER for a in activations]

        muscle_set = self.model.updForceSet()
        muscles = muscle_set.getMuscles()

        for group_idx, muscle_group in enumerate(self.MUSCLE_GROUPS):
            if group_idx >= len(scaled_act):
                break
            for muscle_idx in muscle_group:
                muscle = muscles.get(muscle_idx - 1)
                muscle.setActivation(self.state, scaled_act[group_idx])

    def update_visualization(self):
        """更新可视化"""
        if not self.model:
            return
        self.model.equilibrateMuscles(self.state)
        self.model.updForceSet()
        self.model.updControls(self.state)
        self.model.realizeDynamics(self.state)
        self.model.getVisualizer().show(self.state)