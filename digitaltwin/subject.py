import os
import json
import numpy as np

from digitaltwin.utils.file_tools import make_result_folder


class Subject:
    """实验参数配置类，从 JSON 配置文件加载参数

    使用方式：
        subject = Subject("config/20250708_BenchPress_Chenzui.json")

    JSON 配置格式说明：
        - folder: 实验根目录，所有相对路径的基准
        - modeling_file: 建模数据文件配置
            - emg_folder / robot_folder / xsens_folder: 可选子目录
            - data: {load_key: {robot_file, emg_file, xsens_file?, start_time?}}
              路径拼接规则: folder + emg_folder + emg_file, 如果 emg_folder 为空则不加
        - variable_load_file: 变负载数据文件配置
            - emg_folder / robot_folder / load_folder: 可选子目录
            - read_ori_robot: True 则使用 RobotOriginProcessor, 否则 RobotProcessor
            - data: {label: {robot_file, emg_file, vload_file?, xsens_file?,
                     target_activation, start_time?, load_range?, target_muscle}}
        - heatmap_settings: 热力图与分析参数
            - muscle_folder, height_range, load_range, titles, goal, epsilons, max_iter
    """

    DEFAULTS = {
        "musc_label": [
            "TA", "GL", "SOL", "FibLon", "VL", "RF",
            "Abs", "ES", "PMCla", "PMSte", "LD", "DelAnt",
            "VM", "Addl", "BF", "ST", "GlutMax", "GlutMed",
            "DelMed", "DelPos", "Bic", "TriLong", "TriLat", "BRD"
        ],
        "musc_mvc": [
            0.1276, 0.2980, 0.3023, 0.3392, 0.3792, 0.3061,
            0.7502, 0.2241, 0.2260, 0.3334, 0.0539, 0.3480,
            0.1276, 0.2980, 0.3023, 0.3392, 0.3792, 0.3061,
            0.7502, 0.1041, 0.0660, 0.4334, 0.1539, 0.0480
        ],
        "emg_fs": 2000,
        "motion_flag": "all",
        "target_motion": "squat",
        "turn_position": False,
        "remove_leading_zeros": False,
        "variable_mode": 1,
        "load_previous_data": False,
    }

    def __init__(self, config_path: str):
        self.config_path = config_path
        self.config = {}

        if not self._load_config():
            raise FileNotFoundError(f"无法加载配置文件: {config_path}")

        self._parse_config()

    # ------------------------------------------------------------------
    #  配置加载
    # ------------------------------------------------------------------

    def _load_config(self) -> bool:
        """从 JSON 文件加载配置"""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                self.config = json.load(f)
            print(f"实验配置加载成功: {self.config_path}")
            return True
        except Exception as e:
            print(f"加载配置文件失败: {e}")
            return False

    def _resolve_path(self, path, base=None):
        """解析路径：绝对路径直接返回，空字符串返回 base，相对路径基于 base 拼接"""
        if path is None:
            return None
        base = base or self.folder
        if not path or path == ".":
            return base
        if os.path.isabs(path):
            return path
        return os.path.join(base, path)

    # ------------------------------------------------------------------
    #  配置解析
    # ------------------------------------------------------------------

    def _parse_config(self):
        """将 JSON 配置解析为实例属性"""

        # ---- 实验标识 ----
        self.experiment_label = self.config.get("experiment_label", "")

        # ---- 根目录 ----
        self.folder = self.config.get("folder", "")

        # ---- EMG 设置 ----
        emg = self.config.get("emg_settings", {})
        self.musc_label = emg.get("musc_label", self.DEFAULTS["musc_label"])
        self.musc_mvc = emg.get("musc_mvc", self.DEFAULTS["musc_mvc"])
        self.emg_fs = emg.get("fs", self.DEFAULTS["emg_fs"])

        # ---- 运动设置 ----
        motion = self.config.get("motion_settings", {})
        self.motion_flag = motion.get(
            "motion_flag", self.DEFAULTS["motion_flag"])
        self.target_motion = motion.get(
            "target_motion", self.DEFAULTS["target_motion"])
        self.turn_position = motion.get(
            "turn_position", self.DEFAULTS["turn_position"])
        self.remove_leading_zeros = motion.get(
            "remove_leading_zeros", self.DEFAULTS["remove_leading_zeros"])

        # ---- 建模文件 (modeling_file) ----
        modeling = self.config.get("modeling_file", {})
        self.modeling_emg_folder = self._resolve_path(
            modeling.get("emg_folder", ""))
        self.modeling_robot_folder = self._resolve_path(
            modeling.get("robot_folder", ""))
        self.modeling_xsens_folder = self._resolve_path(
            modeling.get("xsens_folder", ""))
        self.modeling_data = modeling.get("data", {})

        # ---- 变负载文件 (variable_load_file) ----
        vload_cfg = self.config.get("variable_load_file", {})
        self.vload_emg_folder = self._resolve_path(
            vload_cfg.get("emg_folder", ""))
        self.vload_robot_folder = self._resolve_path(
            vload_cfg.get("robot_folder", ""))
        self.vload_load_folder = self._resolve_path(
            vload_cfg.get("load_folder", ""))
        self.read_ori_robot = vload_cfg.get("read_ori_robot", False)
        self.vload_data = vload_cfg.get("data", {})

        # ---- 热力图 / 分析设置 (heatmap_settings) ----
        heatmap = self.config.get("heatmap_settings", {})

        # muscle_folder 特殊处理
        mf = heatmap.get("muscle_folder", None)
        if mf is not None and not os.path.isabs(mf):
            if mf.startswith("result/") or mf.startswith("result\\"):
                self.muscle_folder = mf
            else:
                self.muscle_folder = os.path.join(self.folder, mf)
        else:
            self.muscle_folder = mf

        self.height_range = heatmap.get("height_range", None)
        self.load_range = heatmap.get("load_range", None)
        self.titles = heatmap.get("titles", None)
        self.goal = heatmap.get("goal", None)
        self.epsilons = heatmap.get("epsilons", None)
        self.max_iter = heatmap.get("max_iter", None)
        self.plot_muscle_idx = heatmap.get("plot_muscle_idx", [])

        # ---- 其他 ----
        self.variable_mode = self.config.get(
            "variable_mode", self.DEFAULTS["variable_mode"])
        self.load_previous_data = self.config.get(
            "load_previous_data", self.DEFAULTS["load_previous_data"])
        self.var_data = []
        self.test = False
        self.robot_files_test = None

        # 创建结果文件夹（复用 data/utils 中的函数）
        self.result_folder = make_result_folder(self.experiment_label)
        if self.muscle_folder is None:
            self.muscle_folder = self.result_folder

        # 验证
        if self.titles and self.goal:
            assert len(self.titles) == len(self.goal), \
                f"titles ({len(self.titles)}) 和 goal ({len(self.goal)}) 长度不匹配"

    # ------------------------------------------------------------------
    #  工具方法
    # ------------------------------------------------------------------

    def save_config(self, save_path: str = None) -> bool:
        """将当前配置保存为 JSON 文件"""
        save_path = save_path or self.config_path
        try:
            with open(save_path, 'w', encoding='utf-8') as f:
                json.dump(self.config, f, indent=2, ensure_ascii=False)
            print(f"配置保存成功: {save_path}")
            return True
        except Exception as e:
            print(f"保存配置失败: {e}")
            return False