"""
配置管理器
"""
import os
from pathlib import Path
from typing import Dict, Any, Optional
from ..models.data_models import ProcessingConfig, DetectorConfig, AdaptiveConfig
from ..utils.file_utils import FileUtils
from ..utils.logger import get_logger

logger = get_logger(__name__)


class ConfigManager:
    """配置管理器"""
    
    def __init__(self, config_file: str = ""):
        self.config_file = config_file
        self.config = ProcessingConfig()
        self._load_config()
    
    def _load_config(self):
        """加载配置文件 - 现在直接使用硬编码配置"""
        logger.info("使用硬编码配置")
        
        self._set_hardcoded_config()
    
    def _set_hardcoded_config(self):
        """设置硬编码配置"""
        # 硬编码三联屏检测器配置
        self.config.triple_screen_config.detector_type = "triple_screen_detector"
        self.config.triple_screen_config.similarity_threshold = 0.85  # 提高相似度阈值，减少误判
        self.config.triple_screen_config.method = "histogram"
        self.config.triple_screen_config.min_region_size = 100
        self.config.triple_screen_config.confidence_threshold = 0.5
        
        # 硬编码下半身检测器配置
        self.config.detector_config.detector_type = "lower_body_detector"
        self.config.detector_config.confidence_threshold = 0.3  # 降低置信度阈值，提高检测灵敏度
        
        # 硬编码ROI配置
        self.config.detector_config.roi_horizontal_start = 0.25
        self.config.detector_config.roi_horizontal_end = 0.75
        self.config.detector_config.roi_vertical_start = 0.5
        self.config.detector_config.roi_vertical_end = 1.0
        
        # 硬编码直方图配置
        self.config.detector_config.histogram_sample_interval = 60  # 改为1分钟采样一次，只计算基准
        self.config.detector_config.histogram_comparison_method = "correlation"
        self.config.detector_config.histogram_threshold = 0.1  # 进一步降低直方图阈值，减少MediaPipe调用
        self.config.detector_config.histogram_bins = 32
        self.config.detector_config.histogram_channels = [0, 1, 2]
        
        # 硬编码MediaPipe配置
        self.config.detector_config.mediapipe_confidence_threshold = 0.5  # 降低MediaPipe置信度阈值
        self.config.detector_config.mediapipe_min_detection_confidence = 0.3  # 降低检测置信度
        self.config.detector_config.mediapipe_min_tracking_confidence = 0.3  # 降低跟踪置信度
        self.config.detector_config.mediapipe_enable_buttocks_detection = True
        
        # 硬编码时序配置
        self.config.detector_config.temporal_min_consecutive_frames = 3
        self.config.detector_config.temporal_max_gap_frames = 5
        
        # 硬编码自适应配置（简化版）
        self.config.adaptive_config.initial_interval = 1.0  # 初始间隔1秒
        self.config.adaptive_config.max_interval = 32.0    # 最大间隔32秒
        self.config.adaptive_config.interval_multiplier = 2.0  # 间隔倍数2.0
        
        # 硬编码其他配置
        self.config.output_path = "output"
        self.config.log_level = "INFO"
    
    def _parse_config(self, config_data: Dict[str, Any]):
        """解析配置数据"""
        try:
            # 处理基本配置
            if 'input_path' in config_data:
                self.config.input_path = config_data['input_path']
            if 'output_path' in config_data:
                self.config.output_path = config_data['output_path']
            if 'enable_parallel' in config_data:
                self.config.enable_parallel = config_data['enable_parallel']
            if 'max_workers' in config_data:
                self.config.max_workers = config_data['max_workers']
            if 'log_level' in config_data:
                self.config.log_level = config_data['log_level']
            
            # 处理检测器配置（支持新旧两种格式）
            if 'detector_config' in config_data:
                # 新格式：detector_config
                detector_config = config_data['detector_config']
                if 'detector_type' in detector_config:
                    self.config.detector_config.detector_type = detector_config['detector_type']
                if 'segment_duration' in detector_config:
                    self.config.detector_config.segment_duration = detector_config['segment_duration']
                if 'detection_interval' in detector_config:
                    self.config.detector_config.detection_interval = detector_config['detection_interval']
                if 'face_threshold' in detector_config:
                    self.config.detector_config.face_threshold = detector_config['face_threshold']
                if 'confidence_threshold' in detector_config:
                    self.config.detector_config.confidence_threshold = detector_config['confidence_threshold']
                if 'max_faces' in detector_config:
                    self.config.detector_config.max_faces = detector_config['max_faces']
                if 'enable_early_stop' in detector_config:
                    self.config.detector_config.enable_early_stop = detector_config['enable_early_stop']
                # 三联屏检测器特有配置
                if 'similarity_threshold' in detector_config:
                    self.config.detector_config.similarity_threshold = detector_config['similarity_threshold']
                if 'method' in detector_config:
                    self.config.detector_config.method = detector_config['method']
                if 'min_region_size' in detector_config:
                    self.config.detector_config.min_region_size = detector_config['min_region_size']
                
                # 下半身检测器特有配置
                if 'roi_horizontal_start' in detector_config:
                    self.config.detector_config.roi_horizontal_start = detector_config['roi_horizontal_start']
                if 'roi_horizontal_end' in detector_config:
                    self.config.detector_config.roi_horizontal_end = detector_config['roi_horizontal_end']
                if 'roi_vertical_start' in detector_config:
                    self.config.detector_config.roi_vertical_start = detector_config['roi_vertical_start']
                if 'roi_vertical_end' in detector_config:
                    self.config.detector_config.roi_vertical_end = detector_config['roi_vertical_end']
                if 'histogram_sample_interval' in detector_config:
                    self.config.detector_config.histogram_sample_interval = detector_config['histogram_sample_interval']
                if 'histogram_comparison_method' in detector_config:
                    self.config.detector_config.histogram_comparison_method = detector_config['histogram_comparison_method']
                if 'histogram_threshold' in detector_config:
                    self.config.detector_config.histogram_threshold = detector_config['histogram_threshold']
                if 'histogram_bins' in detector_config:
                    self.config.detector_config.histogram_bins = detector_config['histogram_bins']
                if 'histogram_channels' in detector_config:
                    self.config.detector_config.histogram_channels = detector_config['histogram_channels']
                if 'mediapipe_confidence_threshold' in detector_config:
                    self.config.detector_config.mediapipe_confidence_threshold = detector_config['mediapipe_confidence_threshold']
                if 'mediapipe_min_detection_confidence' in detector_config:
                    self.config.detector_config.mediapipe_min_detection_confidence = detector_config['mediapipe_min_detection_confidence']
                if 'mediapipe_min_tracking_confidence' in detector_config:
                    self.config.detector_config.mediapipe_min_tracking_confidence = detector_config['mediapipe_min_tracking_confidence']
                if 'mediapipe_enable_buttocks_detection' in detector_config:
                    self.config.detector_config.mediapipe_enable_buttocks_detection = detector_config['mediapipe_enable_buttocks_detection']
                if 'temporal_min_consecutive_frames' in detector_config:
                    self.config.detector_config.temporal_min_consecutive_frames = detector_config['temporal_min_consecutive_frames']
                if 'temporal_max_gap_frames' in detector_config:
                    self.config.detector_config.temporal_max_gap_frames = detector_config['temporal_max_gap_frames']
            
            # 处理下半身检测器的独立配置部分
            if 'roi_config' in config_data:
                roi_config = config_data['roi_config']
                if 'horizontal_start' in roi_config:
                    self.config.detector_config.roi_horizontal_start = roi_config['horizontal_start']
                if 'horizontal_end' in roi_config:
                    self.config.detector_config.roi_horizontal_end = roi_config['horizontal_end']
                if 'vertical_start' in roi_config:
                    self.config.detector_config.roi_vertical_start = roi_config['vertical_start']
                if 'vertical_end' in roi_config:
                    self.config.detector_config.roi_vertical_end = roi_config['vertical_end']
            
            if 'histogram_config' in config_data:
                histogram_config = config_data['histogram_config']
                if 'sample_interval' in histogram_config:
                    self.config.detector_config.histogram_sample_interval = histogram_config['sample_interval']
                if 'comparison_method' in histogram_config:
                    self.config.detector_config.histogram_comparison_method = histogram_config['comparison_method']
                if 'threshold' in histogram_config:
                    self.config.detector_config.histogram_threshold = histogram_config['threshold']
                if 'bins' in histogram_config:
                    self.config.detector_config.histogram_bins = histogram_config['bins']
                if 'channels' in histogram_config:
                    self.config.detector_config.histogram_channels = histogram_config['channels']
            
            if 'mediapipe_config' in config_data:
                mediapipe_config = config_data['mediapipe_config']
                if 'confidence_threshold' in mediapipe_config:
                    self.config.detector_config.mediapipe_confidence_threshold = mediapipe_config['confidence_threshold']
                if 'min_detection_confidence' in mediapipe_config:
                    self.config.detector_config.mediapipe_min_detection_confidence = mediapipe_config['min_detection_confidence']
                if 'min_tracking_confidence' in mediapipe_config:
                    self.config.detector_config.mediapipe_min_tracking_confidence = mediapipe_config['min_tracking_confidence']
                if 'enable_buttocks_detection' in mediapipe_config:
                    self.config.detector_config.mediapipe_enable_buttocks_detection = mediapipe_config['enable_buttocks_detection']
            
            # 处理三联屏检测器配置
            if 'triple_screen_config' in config_data:
                triple_screen_config = config_data['triple_screen_config']
                if 'detector_type' in triple_screen_config:
                    self.config.triple_screen_config.detector_type = triple_screen_config['detector_type']
                if 'similarity_threshold' in triple_screen_config:
                    self.config.triple_screen_config.similarity_threshold = triple_screen_config['similarity_threshold']
                if 'method' in triple_screen_config:
                    self.config.triple_screen_config.method = triple_screen_config['method']
                if 'min_region_size' in triple_screen_config:
                    self.config.triple_screen_config.min_region_size = triple_screen_config['min_region_size']
                if 'segment_duration' in triple_screen_config:
                    self.config.triple_screen_config.segment_duration = triple_screen_config['segment_duration']
                if 'detection_interval' in triple_screen_config:
                    self.config.triple_screen_config.detection_interval = triple_screen_config['detection_interval']
                if 'confidence_threshold' in triple_screen_config:
                    self.config.triple_screen_config.confidence_threshold = triple_screen_config['confidence_threshold']
                if 'enable_early_stop' in triple_screen_config:
                    self.config.triple_screen_config.enable_early_stop = triple_screen_config['enable_early_stop']
            
            if 'temporal_config' in config_data:
                temporal_config = config_data['temporal_config']
                if 'min_consecutive_frames' in temporal_config:
                    self.config.detector_config.temporal_min_consecutive_frames = temporal_config['min_consecutive_frames']
                if 'max_gap_frames' in temporal_config:
                    self.config.detector_config.temporal_max_gap_frames = temporal_config['max_gap_frames']
            elif 'detector' in config_data:
                # 旧格式：detector
                detector_config = config_data['detector']
                if 'type' in detector_config:
                    self.config.detector_config.detector_type = detector_config['type']
                if 'segment_duration' in detector_config:
                    self.config.detector_config.segment_duration = detector_config['segment_duration']
                if 'detection_interval' in detector_config:
                    self.config.detector_config.detection_interval = detector_config['detection_interval']
                if 'face_threshold' in detector_config:
                    self.config.detector_config.face_threshold = detector_config['face_threshold']
                if 'confidence_threshold' in detector_config:
                    self.config.detector_config.confidence_threshold = detector_config['confidence_threshold']
                if 'max_faces' in detector_config:
                    self.config.detector_config.max_faces = detector_config['max_faces']
                if 'enable_early_stop' in detector_config:
                    self.config.detector_config.enable_early_stop = detector_config['enable_early_stop']
            
            # 处理自适应配置
            if 'adaptive' in config_data:
                adaptive_config = config_data['adaptive']
                if 'initial_interval' in adaptive_config:
                    self.config.adaptive_config.initial_interval = adaptive_config['initial_interval']
                if 'max_interval' in adaptive_config:
                    self.config.adaptive_config.max_interval = adaptive_config['max_interval']
                if 'interval_multiplier' in adaptive_config:
                    self.config.adaptive_config.interval_multiplier = adaptive_config['interval_multiplier']
                if 'min_interval' in adaptive_config:
                    self.config.adaptive_config.min_interval = adaptive_config['min_interval']
                if 'precision_mode_interval' in adaptive_config:
                    self.config.adaptive_config.precision_mode_interval = adaptive_config['precision_mode_interval']
                if 'enable_adaptive' in adaptive_config:
                    self.config.adaptive_config.enable_adaptive = adaptive_config['enable_adaptive']
        
        except Exception as e:
            logger.error(f"解析配置数据失败: {e}")
    
    def get_config(self) -> ProcessingConfig:
        """获取配置"""
        return self.config
    
    def update_config(self, **kwargs):
        """更新配置"""
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
            elif hasattr(self.config.detector_config, key):
                setattr(self.config.detector_config, key, value)
            elif hasattr(self.config.adaptive_config, key):
                setattr(self.config.adaptive_config, key, value)
    
    def save_config(self, file_path: str = "") -> bool:
        """保存配置到文件"""
        if not file_path:
            file_path = self.config_file or "config/default.yaml"
        
        config_data = {
            'input_path': self.config.input_path,
            'output_path': self.config.output_path,
            'enable_parallel': self.config.enable_parallel,
            'max_workers': self.config.max_workers,
            'log_level': self.config.log_level,
            'detector_config': {
                'detector_type': self.config.detector_config.detector_type,
                'segment_duration': self.config.detector_config.segment_duration,
                'detection_interval': self.config.detector_config.detection_interval,
                'confidence_threshold': self.config.detector_config.confidence_threshold,
                'enable_early_stop': self.config.detector_config.enable_early_stop,
                'similarity_threshold': self.config.detector_config.similarity_threshold,
                'method': self.config.detector_config.method,
                'min_region_size': self.config.detector_config.min_region_size
            },
            'adaptive': {
                'initial_interval': self.config.adaptive_config.initial_interval,
                'max_interval': self.config.adaptive_config.max_interval,
                'interval_multiplier': self.config.adaptive_config.interval_multiplier,
                'min_interval': self.config.adaptive_config.min_interval,
                'precision_mode_interval': self.config.adaptive_config.precision_mode_interval,
                'enable_adaptive': self.config.adaptive_config.enable_adaptive
            }
        }
        
        return FileUtils.save_yaml(config_data, file_path)
    
    def validate_config(self) -> bool:
        """验证配置有效性"""
        try:
            # 检查输入路径
            if self.config.input_path and not os.path.exists(self.config.input_path):
                logger.error(f"输入路径不存在: {self.config.input_path}")
                return False
            
            # 检查检测器配置
            if self.config.detector_config.segment_duration <= 0:
                logger.error("片段时长必须大于0")
                return False
            
            if self.config.detector_config.detection_interval <= 0:
                logger.error("检测间隔必须大于0")
                return False
            
            if not (0 <= self.config.detector_config.confidence_threshold <= 1):
                logger.error("置信度阈值必须在0-1之间")
                return False
            
            if not (0 <= self.config.detector_config.similarity_threshold <= 1):
                logger.error("相似度阈值必须在0-1之间")
                return False
            
            # 检查自适应配置
            if self.config.adaptive_config.initial_interval <= 0:
                logger.error("初始抽帧间隔必须大于0")
                return False
            
            if self.config.adaptive_config.max_interval <= self.config.adaptive_config.initial_interval:
                logger.error("最大抽帧间隔必须大于初始间隔")
                return False
            
            if self.config.adaptive_config.interval_multiplier <= 1:
                logger.error("间隔倍数必须大于1")
                return False
            
            return True
        
        except Exception as e:
            logger.error(f"配置验证失败: {e}")
            return False
