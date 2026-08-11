from typing import Dict, Any, Optional, List
import json
import os
import re

import numpy as np

from digitaltwin.utils.logger import beauty_print

G = 9.81

# 负载模式。三种模式的区别只在于“实际负载从哪里来”：
#
#   isotonic   定负载。配重已知，直接用 load_kg。
#   isokinetic 等速。限速深蹲，发力较大时杆以恒速运动，加速度为 0，
#              此时杆力全部用于平衡重力，可由受力反推等效负载。
#   isometric  等长。杆固定在某高度不动，速度与加速度均为 0，
#              同样可由受力反推等效负载。
#
# 后两种模式的 load_kg 写 null，由 estimate_load_kg() 算出。
LOAD_MODES = ('isotonic', 'isokinetic', 'isometric')
FORCE_DERIVED_MODES = ('isokinetic', 'isometric')
DEFAULT_LOAD_MODE = 'isotonic'


def get_data_groups(config: Dict[str, Any]) -> Dict[str, Any]:
    """取 modeling_file.data。"""
    return config.get('modeling_file', {}).get('data', {}) or {}


def get_group(config: Dict[str, Any], load_key: str) -> Dict[str, Any]:
    """取单个采集组的配置。"""
    return get_data_groups(config).get(str(load_key), {}) or {}


_KEY_NUMBER_RE = re.compile(r'\d+(?:\.\d+)?')

# Windows 上不能出现在文件名里的字符。'/' 尤其危险：组名 'IK-0.3m/s'
# 会被当成两级目录。
_PATH_UNSAFE_CHARS = ('/', '\\', ':', '*', '?', '"', '<', '>', '|')


def parse_key_number(load_key) -> Optional[float]:
    """
    从组名里取出第一个数字，取不到返回 None。

    组名允许带单位：'20kg' -> 20.0，'IK-0.3m/s' -> 0.3，'IM-1m' -> 1.0。
    刻意不接受负号：'IK-0.3m/s' 里的 '-' 是分隔符，不是负号。
    """
    m = _KEY_NUMBER_RE.search(str(load_key))
    if not m:
        return None
    try:
        return float(m.group(0))
    except ValueError:
        return None


def safe_load_key(load_key) -> str:
    """
    把组名变成可以安全用作文件名 / 文件夹名的字符串。

    组名现在允许写成 'IK-0.3m/s' 这种带单位的形式，但它同时被用来拼
    result/.../inverse_dynamics/{load_key}/ 这类路径。凡是要落盘的地方
    都必须先过这个函数，否则 'IK-0.3m/s' 会变成一级子目录 'IK-0.3m'
    加一个名为 's' 的文件。
    """
    text = str(load_key).strip()
    for ch in _PATH_UNSAFE_CHARS:
        text = text.replace(ch, '_')
    text = text.replace(' ', '_')
    return text or 'unnamed'


def numeric_load_value(load_key, file_info=None):
    """把组名解析成数值负载 (kg)；解析不出来返回 nan。

    为什么不能直接 float(load_key)：等长 / 等速组的组名是
    'IM-1' / 'IK-0.3'，float() 会抛 ValueError。而且这一步在 try 外面，
    异常会直接穿出流水线，把整批组一起打挂。

    也不能从组名里抽数字：'IM-1' 的 1 是杆高 1.0 m，
    'IK-0.3' 的 0.3 是最高速度 0.3 m/s，都不是负载。误用会把
    等长组当成 1 kg 的定负载组静静带进热力图拟合。
    这两类组的实际负载必须由受力反推，因此先给 nan。

    Parameters
    ----------
    load_key : str
        组名。
    file_info : dict, optional
        采集组配置，优先取其 'load_kg' 或 'load' 字段再回退组名。
    """
    info = file_info or {}
    for key in ('load_kg', 'load'):
        value = info.get(key)
        if value is None:
            continue
        try:
            f = float(value)
        except (TypeError, ValueError):
            continue
        if np.isfinite(f):
            return f
    try:
        return float(load_key)
    except (TypeError, ValueError):
        return float('nan')


def _infer_legacy_mode(load_key: str) -> Dict[str, Any]:
    """
    老格式兼容：组里没有 mode 字段时，由组名推断。

    旧约定是组名即配重公斤数，但 0.3 / 0.15 这种小数实际上是等速速度。
    新命名（'20kg' / 'IK-0.3m/s' / 'IM-1m'）也一并识别。
    """
    text = str(load_key).lower()
    value = parse_key_number(load_key)

    if 'isometric' in text or text.startswith('im'):
        return {'mode': 'isometric', 'bar_height': value, 'load_kg': None}
    if 'isokinetic' in text or text.startswith('ik') or 'm/s' in text:
        return {'mode': 'isokinetic', 'target_velocity': value,
                'load_kg': None}
    if value is None:
        return {'mode': DEFAULT_LOAD_MODE, 'load_kg': None}
    if value < 1.0:
        return {'mode': 'isokinetic', 'target_velocity': value, 'load_kg': None}
    return {'mode': DEFAULT_LOAD_MODE, 'load_kg': value}


def get_load_mode(config: Dict[str, Any], load_key: str,
                  warn: bool = True) -> str:
    """
    返回采集组的负载模式。

    组里没有 mode 字段时退回旧约定推断，并告警。
    """
    group = get_group(config, load_key)
    mode = group.get('mode')

    if mode is None:
        guess = _infer_legacy_mode(load_key)
        if warn:
            beauty_print(
                '[Config] 组 {} 缺少 mode 字段，按旧约定推断为 {}。'
                '建议在 json 中显式声明。'.format(load_key, guess['mode']),
                type='warning')
        return guess['mode']

    mode = str(mode).strip().lower()
    if mode not in LOAD_MODES:
        if warn:
            beauty_print(
                '[Config] 组 {} 的 mode="{}" 不在已知模式 {} 中'.format(
                    load_key, mode, LOAD_MODES),
                type='warning')
        return DEFAULT_LOAD_MODE
    return mode


def get_nominal_load_kg(config: Dict[str, Any],
                        load_key: str) -> Optional[float]:
    """
    返回名义负载 (kg)。等速 / 等长模式返回 None，应改用 estimate_load_kg。
    """
    group = get_group(config, load_key)

    if 'load_kg' in group:
        value = group.get('load_kg')
        return None if value is None else float(value)

    guess = _infer_legacy_mode(load_key)
    return guess.get('load_kg')


def estimate_load_kg(force_l, force_r, acc=None, bar_mass: float = 0.0,
                     g: float = G) -> float:
    """
    由杆力反推等效负载 (kg)。

    等速段与等长段的加速度为 0，此时

        force_l + force_r = M * g

    所以 M = (force_l + force_r) / g。若传入 acc，则按

        force_l + force_r = M * (g + acc)

    修正，并只使用 |acc| 较小的帧。bar_mass 为杆自身质量，会加到结果上。

    Parameters
    ----------
    force_l, force_r : array-like or float -- 机器人两侧致动器力 (N)
    acc : array-like or float, optional -- 杆的竖直加速度 (m/s^2)
    bar_mass : float -- 杆自身质量 (kg)

    Returns
    -------
    float -- 等效负载 (kg)
    """
    total = np.asarray(force_l, dtype=float) + np.asarray(force_r, dtype=float)

    if acc is None:
        denom = np.full_like(total, g)
        keep = np.ones_like(total, dtype=bool)
    else:
        acc = np.asarray(acc, dtype=float)
        denom = g + acc
        # 只用接近恒速 / 静止的帧，避免加减速段污染估计
        keep = np.abs(acc) < 0.1 * g
        if not np.any(keep):
            keep = np.ones_like(total, dtype=bool)

    with np.errstate(invalid='ignore', divide='ignore'):
        mass = total / denom

    mass = mass[keep & np.isfinite(mass)]
    if mass.size == 0:
        return float('nan')

    return float(np.median(mass)) + float(bar_mass)


def resolve_load_kg(config: Dict[str, Any], load_key: str,
                    force_l=None, force_r=None, acc=None) -> Optional[float]:
    """
    统一入口：拿到该组的实际负载 (kg)。

    定负载组直接返回 load_kg；等速 / 等长组在给出杆力时由受力反推，
    未给出杆力时返回 None 并告警。
    """
    mode = get_load_mode(config, load_key, warn=False)

    if mode not in FORCE_DERIVED_MODES:
        return get_nominal_load_kg(config, load_key)

    if force_l is None or force_r is None:
        beauty_print(
            '[Config] 组 {} 是 {} 模式，需要杆力才能反推负载，'
            '但未传入 force_l / force_r'.format(load_key, mode),
            type='warning')
        return None

    bar_mass = config.get('opensim_settings', {}).get('bar_mass', 0.0) or 0.0
    return estimate_load_kg(force_l, force_r, acc=acc, bar_mass=0.0)


def filter_load_keys(config: Dict[str, Any], load_keys=None,
                     modes=None, exclude=None) -> List[str]:
    """
    按模式筛选采集组。

    代替各 example 里硬写的 EXCLUDE_LOAD_KEYS=['0.15','0.3']：
    只要干定负载组，写 modes=('isotonic',) 即可，
    以后新增等长 / 等速组也不用再改脚本。

    Parameters
    ----------
    load_keys : iterable, optional -- None 表示全部
    modes : str or iterable, optional -- 保留的模式
    exclude : iterable, optional -- 额外排除的组名
    """
    groups = get_data_groups(config)
    keys = list(groups.keys()) if load_keys is None else [
        str(k) for k in load_keys]

    if isinstance(modes, str):
        modes = (modes,)
    exclude = set(str(k) for k in (exclude or ()))

    # 组名改过之后，脚本里硬写的 EXCLUDE_LOAD_KEYS 可能一个都对不上，
    # 于是本该被排除的组会悄悄进入统计，而且不会有任何报错。必须报出来。
    stale = [k for k in exclude if k not in groups]
    if stale:
        beauty_print(
            '[Config] exclude 里的组名在 config 中不存在: {}。'
            '组名可能已改（如 0.3 -> IK-0.3m/s），该排除项当前无效，'
            '请改用 modes=(...) 按模式筛选。'.format(', '.join(sorted(stale))),
            type='warning')

    result = []
    for key in keys:
        if key in exclude or key not in groups:
            continue
        if modes is not None and get_load_mode(config, key, warn=False) \
                not in modes:
            continue
        result.append(key)
    return result


def describe_load_key(config: Dict[str, Any], load_key: str) -> str:
    """生成用于打印 / 图例的一行描述。"""
    mode = get_load_mode(config, load_key, warn=False)
    group = get_group(config, load_key)

    if mode == 'isokinetic':
        v = group.get('target_velocity', load_key)
        return '{} (等速 {} m/s)'.format(load_key, v)
    if mode == 'isometric':
        h = group.get('bar_height')
        return '{} (等长{})'.format(
            load_key, ' {} m'.format(h) if h is not None else '')

    mass = get_nominal_load_kg(config, load_key)
    return '{} ({} kg)'.format(load_key, mass if mass is not None else '?')


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
        """获取运动设置"""
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

    # ------------------------------------------------------------------
    # 负载模式（定负载 / 等速 / 等长）
    # ------------------------------------------------------------------
    def get_data_groups(self) -> Dict[str, Any]:
        """获取全部采集组"""
        return get_data_groups(self.config)

    def get_group(self, load_key: str) -> Dict[str, Any]:
        """获取单个采集组"""
        return get_group(self.config, load_key)

    def get_load_mode(self, load_key: str, warn: bool = True) -> str:
        """获取该组的负载模式"""
        return get_load_mode(self.config, load_key, warn=warn)

    def get_nominal_load_kg(self, load_key: str) -> Optional[float]:
        """获取名义负载，等速 / 等长组返回 None"""
        return get_nominal_load_kg(self.config, load_key)

    def resolve_load_kg(self, load_key: str, force_l=None, force_r=None,
                        acc=None) -> Optional[float]:
        """获取实际负载，必要时由杆力反推"""
        return resolve_load_kg(self.config, load_key,
                               force_l=force_l, force_r=force_r, acc=acc)

    def filter_load_keys(self, load_keys=None, modes=None,
                         exclude=None) -> List[str]:
        """按模式筛选采集组"""
        return filter_load_keys(self.config, load_keys=load_keys,
                                modes=modes, exclude=exclude)

    def describe_load_key(self, load_key: str) -> str:
        """生成用于打印 / 图例的描述"""
        return describe_load_key(self.config, load_key)

    @staticmethod
    def safe_load_key(load_key) -> str:
        """组名转可落盘的文件名 / 文件夹名"""
        return safe_load_key(load_key)

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
