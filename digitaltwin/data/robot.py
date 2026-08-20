"""Robot sensor data API under the data domain."""

from .robot_processor import RobotOriginProcessor, RobotProcessor

__all__ = ["RobotProcessor", "RobotOriginProcessor"]
