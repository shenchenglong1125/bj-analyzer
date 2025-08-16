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
        """加载配置文件"""
        if self.config_file and os.path.exists(self.config_file):
            config_data = FileUtils.load_yaml(self.config_file)
            if config_data:
                self._parse_config(config_data)
                logger.info(f"配置文件加载成功: {self.config_file}")
            else:
                logger.warning(f"配置文件加载失败，使用默认配置: {self.config_file}")
        else:
            logger.info("使用默认配置")
    
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
