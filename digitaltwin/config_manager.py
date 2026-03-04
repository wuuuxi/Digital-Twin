from typing import Dict, Any, Optional
import json
import os

from digitaltwin.utils.logger import beauty_print


class ConfigManager:
    """配置文件管理器"""

    def __init__(self, config_path: str = "config.json"):
        """
        初始化配置管理器

        Args:
            config_path: 配置文件路径
        """
        self.config_path = config_path
        self.config: Dict[str, Any] = {}

    def load_config(self) -> bool:
        """从文件加载配置"""
        try:
            if not os.path.exists(self.config_path):
                beauty_print(f"配置文件 {self.config_path} 不存在，使用默认配置", type='warning')
                self._create_default_config()
                return True

            with open(self.config_path, 'r', encoding='utf-8') as f:
                self.config = json.load(f)
            print(f"配置文件加载成功: {self.config_path}")
            return True
        except Exception as e:
            beauty_print(f"加载配置文件失败: {e}", type='warning')
            self._create_default_config()
            return False

    def save_config(self, config_path: str = None) -> bool:
        """保存配置到文件"""
        save_path = config_path or self.config_path
        try:
            with open(save_path, 'w', encoding='utf-8') as f:
                json.dump(self.config, f, indent=2, ensure_ascii=False)
            print(f"配置文件保存成功: {save_path}")
            return True
        except Exception as e:
            beauty_print(f"保存配置文件失败: {e}", type='warning')
            return False

    def _create_default_config(self):
        """创建默认配置"""
        self.config = {
            "data_settings": {
                "folder": "",
                "data_files": {},
                "musc_mvc": [],
                "fs": 1000
            },
            "opensim_settings": {
                "model_path": "workspace",
                "geometry_path": "workspace/Geometry"
            },
            "audio_settings": {
                "sound_path": "workspace",
                "fixed_beep_count": 5
            },
            "playback_settings": {
                "target_lift_duration": None,
                "target_lower_duration": None,
                "lift_speed_ratio": 1.0,
                "lower_speed_ratio": 1.0
            },
            "visualization_settings": {
                "display_multiplier": 23,
                "sample_num": 100,
                "arm_length": 0.37,
                "shoulder_height": 0.45
            }
        }

    def get_motion(self) -> Dict[str, Any]:
        """获取音频设置"""
        return self.config.get("motion", {})

    def get_data_settings(self) -> Dict[str, Any]:
        """获取数据设置"""
        return self.config.get("data_settings", {})

    def get_opensim_settings(self) -> Dict[str, Any]:
        """获取OpenSim设置"""
        return self.config.get("opensim_settings", {})

    def get_audio_settings(self) -> Dict[str, Any]:
        """获取音频设置"""
        return self.config.get("audio_settings", {})

    def get_playback_settings(self) -> Dict[str, Any]:
        """获取播放设置"""
        return self.config.get("playback_settings", {})

    def get_visualization_settings(self) -> Dict[str, Any]:
        """获取可视化设置"""
        return self.config.get("visualization_settings", {})

    def update_setting(self, section: str, key: str, value: Any) -> bool:
        """更新特定设置"""
        try:
            if section in self.config:
                self.config[section][key] = value
                return True
            return False
        except Exception as e:
            print(f"更新设置失败: {e}")
            return False
