import opensim as osim
import numpy as np
import math
from .muscle_state import MuscleStateManager
from digitaltwin.utils.logger import beauty_print


class OpenSimModel:
    """OpenSim模型管理类"""

    def __init__(self, opensim_settings, motion_label):
        self.model_path = opensim_settings.get("model_path")
        self.model_name = opensim_settings.get("model_name", 'whole body model_yuetian_realtime.osim')
        self.modelFilePath = self.model_path + '/' + self.model_name

        self.geometry_path = opensim_settings.get("geometry_path", self.model_path + "/Geometry")

        if motion_label == 'squat':
            self.coordinates = [51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64]
            self.coordinates_value = [50, -40, -90, 125, 90, -70, -15, 50, -40, -90, 125, 90, -70, -15]
        elif motion_label == 'benchpress':
            self.coordinates = [1, 8, 11, 13, 18, 53, 60, 55, 62]
            self.coordinates_value = [90, -25, -25, 85, 85, 90, 90, 90, 90]
        else:
            self.coordinates = [53, 60, 55, 62]
            self.coordinates_value = [90, 90, 90, 90]

        self.motion_label = motion_label
        self.model = None
        self.state = None
        self.visualizer = None
        self.muscle_state = MuscleStateManager(self.motion_label)

        self.load_model()

    def load_model(self):
        """加载OpenSim模型"""
        try:
            self.model = osim.Model(self.modelFilePath)
            osim.ModelVisualizer.addDirToGeometrySearchPaths(self.geometry_path)

            self.model.setUseVisualizer(True)
            self.state = self.model.initSystem()
            init_state = osim.State(self.state)

            self.visualizer = self.model.updVisualizer()
            sviz = self.visualizer.updSimbodyVisualizer()
            self.visualizer.show(init_state)

            # print("OpenSim model loaded successfully")
            beauty_print(f"OpenSim model loaded successfully")
            return True
        except Exception as e:
            beauty_print(f"Error loading OpenSim model: {e}", type='warning')
            return False

    def initQ(self):
        """Initialize pose"""
        # initial position coordinates
        Coord_Seq = self.coordinates  # what does each item in q represent in model coordinate (in seq # way)
        value_Seq = self.coordinates_value
        q_in_rad = np.zeros(len(value_Seq))
        for i in range(len(value_Seq)):
            q_in_rad[i] = value_Seq[i] / 180 * math.pi

        for i in range(len(Coord_Seq)):
            Coord = self.model.updCoordinateSet().get(Coord_Seq[i] - 1)
            Coord.setValue(self.state, q_in_rad[i])
        # set pelvis height sothat feets touch the ground
        Coord = self.model.updCoordinateSet().get(5 - 1)
        Coord.setValue(self.state, 0.52)
        return

    def updStateQ(self, input):
        """Update state variable q for selected coordinates"""
        if self.motion_label == 'benchpress':
            Coord_Seq = [52, 51, 59, 58, 54, 61]

            if len(input) == 4:
                [qt_r, qt_l] = self.muscle_state.calculate_joint_angle(input)

                q_in_rad_r = np.zeros(7)
                q_in_rad_l = np.zeros(7)

                for i in range(7):
                    q_in_rad_r[i] = qt_r[i] / 180 * math.pi
                    q_in_rad_l[i] = qt_l[i] / 180 * math.pi

                right_coords = [51, 52, 53, 54, 55, 56, 57]
                for i, coord_idx in enumerate(right_coords):
                    Coord = self.model.updCoordinateSet().get(coord_idx - 1)
                    Coord.setValue(self.state, q_in_rad_r[i])
                    # coord_name = Coord.getName()
                    # print(f"索引 {coord_idx}: {coord_name:30s} | 角度: {qt_r}° ({q_in_rad_r} rad)")

                left_coords = [58, 59, 60, 61, 62, 63, 64]
                for i, coord_idx in enumerate(left_coords):
                    Coord = self.model.updCoordinateSet().get(coord_idx - 1)
                    Coord.setValue(self.state, q_in_rad_l[i])
        elif self.motion_label == 'squat':
            Coord_Seq = [7, 8, 9, 13, 15, 10, 11, 12, 18,
                         20]  # what does each item in q represent in model coordinate (in seq # way)

            if len(input) == 4:
                [qt_r, qt_l, qt_pelv] = self.muscle_state.calculate_joint_angle(input)
                # qt_r = self.muscle_state._height_to_joint_angles_sq(input[2])
                # qt_l = self.muscle_state._height_to_joint_angles_sq(input[3])
                # qt_pelv = self.muscle_state._height_to_joint_angles_sq_Pelvis((input[2] + input[3]) / 2)

                # qt_r[1] = qt_r[1] -20;
                # qt_l[1] = qt_l[1] -20;

                # qt_r[3] = qt_r[3] +20;
                # qt_l[3] = qt_l[3] +20;

                q_in_rad_r = np.zeros(5)
                q_in_rad_l = np.zeros(5)
                q_in_rad_pelv = np.zeros(6)

                for i in range(5):
                    q_in_rad_r[i] = qt_r[i] / 180 * math.pi
                    q_in_rad_l[i] = qt_l[i] / 180 * math.pi

                for i in range(3):  # pelvis transation doesn't need this conversion
                    q_in_rad_pelv[i] = qt_pelv[i] / 180 * math.pi
                    q_in_rad_pelv[i + 3] = qt_pelv[i + 3]

                right_coords = [7, 8, 9, 13, 15]
                for i, coord_idx in enumerate(right_coords):
                    Coord = self.model.updCoordinateSet().get(coord_idx - 1)
                    Coord.setValue(self.state, q_in_rad_r[i])
                    # coord_name = Coord.getName()
                    # print(f"索引 {coord_idx}: {coord_name:30s} | 角度: {qt_r}° ({q_in_rad_r} rad)")

                left_coords = [10, 11, 12, 18, 20]
                for i, coord_idx in enumerate(left_coords):
                    Coord = self.model.updCoordinateSet().get(coord_idx - 1)
                    Coord.setValue(self.state, q_in_rad_l[i])
                    # coord_name = Coord.getName()
                    # print(f"索引 {coord_idx}: {coord_name:30s} | 角度: {qt_r}° ({q_in_rad_r} rad)")

                # Coord = self.model.updCoordinateSet().get(10 - 1)
                # Coord.setValue(self.state, q_in_rad_l[0])
                # Coord = self.model.updCoordinateSet().get(11 - 1)
                # Coord.setValue(self.state, q_in_rad_l[1])
                # Coord = self.model.updCoordinateSet().get(12 - 1)
                # Coord.setValue(self.state, q_in_rad_l[2])
                # Coord = self.model.updCoordinateSet().get(18 - 1)
                # Coord.setValue(self.state, q_in_rad_l[3])
                # Coord = self.model.updCoordinateSet().get(20 - 1)
                # Coord.setValue(self.state, q_in_rad_l[4])

                pelv_coords = [1, 2, 3, 4, 5, 6]
                q_in_rad_pelv[4] = q_in_rad_pelv[4] + 0.7
                for i, coord_idx in enumerate(pelv_coords):
                    Coord = self.model.updCoordinateSet().get(coord_idx - 1)
                    Coord.setValue(self.state, q_in_rad_pelv[i])
                    # coord_name = Coord.getName()
                    # print(f"索引 {coord_idx}: {coord_name:30s} | 角度: {qt_r}° ({q_in_rad_pelv} rad)")

    def updStateAct(self, input):
        """Update state activation level for all/selected muscles - same as original"""
        DisplayMultiplier = 23

        act = self.muscle_state.calculate_activation(input)
        for i in range(len(act)):
            act[i] = act[i] * DisplayMultiplier

        # left muscle groups
        MuscleGroup1 = [30, 31, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44]
        MuscleGroup2 = [16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29]
        MuscleGroup3 = [12, 13, 14, 15]
        MuscleGroup4 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
        MuscleGroup5 = [215, 216, 217, 218, 219, 220]
        MuscleGroup6 = [242, 243, 244, 245, 246, 247]

        # left muscle groups
        MuscleGroup7 = [145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156]
        MuscleGroup8 = [131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144]
        MuscleGroup9 = [127, 128, 129, 130]
        MuscleGroup10 = [116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126]
        MuscleGroup11 = [227, 228, 229, 230, 231, 232]
        MuscleGroup12 = [465, 466, 467, 468, 469, 470]

        muscle_groups = [MuscleGroup1, MuscleGroup2, MuscleGroup3, MuscleGroup4,
                         MuscleGroup5, MuscleGroup6, MuscleGroup7, MuscleGroup8,
                         MuscleGroup9, MuscleGroup10, MuscleGroup11, MuscleGroup12]

        MuscleSet = self.model.updForceSet()
        Muscles = MuscleSet.getMuscles()

        for group_idx, muscle_group in enumerate(muscle_groups):
            if group_idx < len(act):
                for i in range(len(muscle_group)):
                    Act = Muscles.get(muscle_group[i] - 1)
                    Act.setActivation(self.state, act[group_idx])

        return

    def update_visualization(self):
        """更新可视化"""
        if not self.model:
            return

        silo = self.model.getVisualizer().getInputSilo()
        self.model.equilibrateMuscles(self.state)

        self.model.updForceSet()
        self.model.updControls(self.state)
        self.model.realizeDynamics(self.state)
        self.model.getVisualizer().show(self.state)
