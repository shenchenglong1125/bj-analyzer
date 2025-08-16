# Detectors Package
from .base_detector import DetectorBase
from .triple_screen_detector import TripleScreenDetector
from .lower_body_detector import LowerBodyDetector

__all__ = ['DetectorBase', 'TripleScreenDetector', 'LowerBodyDetector']


def create_detector(detector_type: str, config):
    """
    检测器工厂函数
    
    Args:
        detector_type: 检测器类型
        config: 检测器配置
        
    Returns:
        检测器实例
    """
    if detector_type == "triple_screen_detector":
        return TripleScreenDetector(config)
    elif detector_type == "lower_body_detector":
        return LowerBodyDetector(config)
    else:
        raise ValueError(f"不支持的检测器类型: {detector_type}")
