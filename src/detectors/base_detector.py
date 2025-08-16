"""
检测器基类
"""
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
import cv2
import numpy as np
from ..models.data_models import DetectionResult, DetectorConfig
from ..utils.logger import get_logger

logger = get_logger(__name__)


class DetectorBase(ABC):
    """检测器基类"""
    
    def __init__(self, config: DetectorConfig):
        self.config = config
        self.logger = get_logger(self.__class__.__name__)
    
    @abstractmethod
    def detect_frame(self, frame: np.ndarray) -> Dict[str, Any]:
        """
        检测单帧图像
        
        Args:
            frame: 输入帧图像
            
        Returns:
            检测结果字典
        """
        pass
    
    @abstractmethod
    def is_split_screen(self, detection_result: Dict[str, Any]) -> bool:
        """
        判断是否为分屏场景
        
        Args:
            detection_result: 检测结果
            
        Returns:
            是否为分屏
        """
        pass
    
    def process_segment(self, frames: List[np.ndarray], segment_start: float, segment_end: float) -> DetectionResult:
        """
        处理视频片段
        
        Args:
            frames: 帧列表
            segment_start: 片段开始时间
            segment_end: 片段结束时间
            
        Returns:
            检测结果
        """
        import time
        start_time = time.time()
        
        detection_frames = 0
        total_confidence = 0.0
        is_split = False
        
        # 按检测间隔处理帧
        interval_frames = max(1, int(len(frames) * self.config.detection_interval / (segment_end - segment_start)))
        
        for i in range(0, len(frames), interval_frames):
            if i >= len(frames):
                break
                
            frame = frames[i]
            try:
                result = self.detect_frame(frame)
                detection_frames += 1
                
                if result:
                    confidence = result.get('confidence', 0.0)
                    total_confidence += confidence
                    
                    # 判断是否为分屏
                    if self.is_split_screen(result):
                        is_split = True
                        
                        # 如果启用提前终止，检测到分屏后立即停止
                        if self.config.enable_early_stop:
                            self.logger.debug(f"检测到分屏，提前终止片段处理")
                            break
                
            except Exception as e:
                self.logger.error(f"处理帧失败: {e}")
                continue
        
        processing_time = time.time() - start_time
        
        # 计算平均置信度
        avg_confidence = total_confidence / max(detection_frames, 1)
        
        return DetectionResult(
            segment_start=segment_start,
            segment_end=segment_end,
            is_split_screen=is_split,
            confidence=avg_confidence,
            detection_frames=detection_frames,
            processing_time=processing_time
        )
    
    def preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        预处理帧图像
        
        Args:
            frame: 原始帧
            
        Returns:
            预处理后的帧
        """
        # 转换为RGB格式
        if len(frame.shape) == 3 and frame.shape[2] == 3:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        else:
            frame_rgb = frame
        
        return frame_rgb
    
    def get_detector_info(self) -> Dict[str, Any]:
        """获取检测器信息"""
        return {
            'name': self.__class__.__name__,
            'type': self.config.detector_type,
            'config': {
                'segment_duration': self.config.segment_duration,
                'detection_interval': self.config.detection_interval,
                'confidence_threshold': self.config.confidence_threshold,
                'enable_early_stop': self.config.enable_early_stop
            }
        }
