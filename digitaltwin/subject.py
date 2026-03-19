import os
import json
import numpy as np

from digitaltwin.data.helpers import make_result_folder


class Subject:
    """实验参数配置类，从 JSON 配置文件加载参数

    使用方式：
        subject = Subject("config/20250708_BenchPress_Chenzui.json")

    JSON 路径约定：
        - emg_folder / robot_folder / load_folder: 相对于 folder 的路径
        - muscle_folder: null 则自动生成 result 目录；
          以 "result/" 开头视为相对于工作目录，否则相对于 folder
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
        "emg_fs": 1000,
        "motion_flag": "all",
        "target_motion": "squat",
        "turn_position": False,
        "remove_leading_zeros": False,
        "read_ori_robot_var": False,
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

        # ---- 路径设置 ----
        paths = self.config.get("paths", {})
        self.folder = paths.get("folder", "")
        self.emg_folder = self._resolve_path(paths.get("emg_folder", ""))
        self.robot_folder = self._resolve_path(paths.get("robot_folder", ""))
        self.load_folder = self._resolve_path(paths.get("load_folder", ""))

        # muscle_folder 特殊处理
        mf = paths.get("muscle_folder", None)
        if mf is not None and not os.path.isabs(mf):
            if mf.startswith("result/") or mf.startswith("result\\"):
                self.muscle_folder = mf
            else:
                self.muscle_folder = os.path.join(self.folder, mf)
        else:
            self.muscle_folder = mf

        # ---- EMG 设置 ----
        emg = self.config.get("emg_settings", {})
        self.musc_label = emg.get("musc_label", self.DEFAULTS["musc_label"])
        self.musc_mvc = emg.get("musc_mvc", self.DEFAULTS["musc_mvc"])
        self.emg_fs = emg.get("emg_Fs", self.DEFAULTS["emg_fs"])

        # ---- 运动设置 ----
        motion = self.config.get("motion_settings", {})
        self.motion_flag = motion.get("motion_flag", self.DEFAULTS["motion_flag"])
        self.target_motion = motion.get("target_motion", self.DEFAULTS["target_motion"])
        self.turn_position = motion.get("turn_position", self.DEFAULTS["turn_position"])
        self.remove_leading_zeros = motion.get(
            "remove_leading_zeros", self.DEFAULTS["remove_leading_zeros"])
        self.read_ori_robot_var = motion.get(
            "read_ori_robot_var", self.DEFAULTS["read_ori_robot_var"])

        # ---- 数据文件 ----
        self.robot_files = self.config.get("robot_files", {})
        self.vload_parameters = self.config.get("vload_parameters", {})

        # ---- 分析设置 ----
        analysis = self.config.get("analysis_settings", {})
        self.height_range = analysis.get("height_range", None)
        self.load_range = analysis.get("load_range", None)
        self.titles = analysis.get("titles", None)
        self.goal = analysis.get("goal", None)
        self.epsilons = analysis.get("epsilons", None)
        self.max_iter = analysis.get("max_iter", None)
        self.plot_muscle_idx = analysis.get("plot_muscle_idx", [])

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