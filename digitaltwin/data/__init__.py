from .data_manager import *

from .emg_processor import EMGProcessor
from .robot_processor import RobotProcessor
from .xsens_processor import XsensProcessor

__all__ = ['EMGProcessor', 'RobotProcessor', 'XsensProcessor']

# from emg_processor import EMGProcessor
# from xsens_processor import XsensProcessor
# from robot_processor import RobotProcessor
# # from visualization import MovementVisualizer
# from utils import (
#     find_nearest_idx,
#     resample_data,
#     extract_continuous_segments,
#     select_longest_segment,
#     print_debug_info,
#     save_pickle,
#     load_pickle,
#     make_result_folder
# )
#
# __all__ = [
#     'EMGProcessor',
#     'XsensProcessor',
#     'RobotProcessor',
#     # 'MovementVisualizer',
#     'find_nearest_idx',
#     'resample_data',
#     'extract_continuous_segments',
#     'select_longest_segment',
#     'print_debug_info',
#     'save_pickle',
#     'load_pickle',
#     'make_result_folder'
# ]