"""
数据模型定义
"""
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from datetime import datetime


@dataclass
class DetectionResult:
    """检测结果数据模型"""
    segment_start: float  # 片段开始时间（秒）
    segment_end: float    # 片段结束时间（秒）
    is_split_screen: bool # 是否为分屏
    confidence: float     # 置信度
    detection_frames: int = 0  # 检测的帧数
    processing_time: float = 0.0  # 处理时间（秒）
    similarity_scores: List[float] = field(default_factory=list)  # 相似度分数


@dataclass
class VideoInfo:
    """视频信息数据模型"""
    file_path: str
    file_name: str
    duration: float       # 视频时长（秒）
    fps: float           # 帧率
    width: int           # 视频宽度
    height: int          # 视频高度
    total_frames: int    # 总帧数


@dataclass
class ProcessingResult:
    """处理结果数据模型"""
    video_info: VideoInfo
    segments: List[DetectionResult] = field(default_factory=list)
    total_processing_time: float = 0.0
    total_segments: int = 0
    split_screen_segments: int = 0
    processing_status: str = "completed"
    error_message: Optional[str] = None


@dataclass
class BatchResult:
    """批量处理结果数据模型"""
    processed_files: List[ProcessingResult] = field(default_factory=list)
    total_files: int = 0
    successful_files: int = 0
    failed_files: int = 0
    start_time: datetime = field(default_factory=datetime.now)
    end_time: Optional[datetime] = None
    total_processing_time: float = 0.0


@dataclass
class AdaptiveConfig:
    """自适应抽帧配置"""
    initial_interval: float = 0.5  # 初始抽帧间隔（秒）
    max_interval: float = 32.0     # 最大抽帧间隔（秒）
    interval_multiplier: float = 2.0  # 间隔倍数
    min_interval: float = 0.1      # 最小抽帧间隔（秒）
    precision_mode_interval: float = 0.1  # 精确模式间隔（秒）
    enable_adaptive: bool = True   # 是否启用自适应抽帧


@dataclass
class DetectorConfig:
    """检测器配置数据模型"""
    detector_type: str = "triple_screen_detector"
    segment_duration: float = 15.0  # 片段时长（秒）
    detection_interval: float = 1.0  # 检测间隔（秒）
    confidence_threshold: float = 0.5  # 置信度阈值
    enable_early_stop: bool = True  # 是否启用提前终止
    # 三联屏检测器特有配置
    similarity_threshold: float = 0.85  # 相似度阈值
    method: str = "histogram"  # 检测方法: histogram, orb, template
    min_region_size: int = 100  # 最小区域尺寸


@dataclass
class TripleScreenConfig:
    """三联屏检测器配置数据模型"""
    detector_type: str = "triple_screen_detector"
    similarity_threshold: float = 0.85  # 相似度阈值
    method: str = "histogram"  # 检测方法: histogram, orb, template
    min_region_size: int = 100  # 最小区域尺寸
    segment_duration: float = 15.0  # 片段时长（秒）
    detection_interval: float = 1.0  # 检测间隔（秒）
    confidence_threshold: float = 0.5  # 置信度阈值
    enable_early_stop: bool = True  # 是否启用提前终止


@dataclass
class ProcessingConfig:
    """处理配置数据模型"""
    input_path: str = ""
    output_path: str = ""
    config_file: str = ""
    detector_config: DetectorConfig = field(default_factory=DetectorConfig)
    triple_screen_config: TripleScreenConfig = field(default_factory=TripleScreenConfig)  # 三联屏检测器配置
    adaptive_config: AdaptiveConfig = field(default_factory=AdaptiveConfig)  # 自适应配置
    enable_parallel: bool = False  # 是否启用并行处理
    max_workers: int = 4  # 最大工作线程数
    log_level: str = "INFO"  # 日志级别


@dataclass
class ROIConfig:
    """检测区域配置"""
    horizontal_start: float = 0.25  # 左右1/4
    horizontal_end: float = 0.75    # 左右3/4
    vertical_start: float = 0.5     # 上下1/2
    vertical_end: float = 1.0       # 底部


@dataclass
class HistogramConfig:
    """直方图检测配置"""
    sample_interval: int = 30       # 采样间隔（帧）
    comparison_method: str = "correlation"  # 比较方法: correlation, chi_square, intersection
    threshold: float = 0.3          # 直方图变化阈值
    bins: int = 32                  # 直方图bin数量
    channels: List[int] = field(default_factory=lambda: [0, 1, 2])  # 检测的通道 (BGR)


@dataclass
class MediaPipeConfig:
    """MediaPipe配置"""
    confidence_threshold: float = 0.7  # 置信度阈值
    min_detection_confidence: float = 0.5
    min_tracking_confidence: float = 0.5
    enable_buttocks_detection: bool = True


@dataclass
class TemporalConfig:
    """时序验证配置"""
    min_consecutive_frames: int = 3  # 最少连续检测帧数
    max_gap_frames: int = 5         # 最大间隔帧数


@dataclass
class LowerBodyDetectorConfig:
    """下半身检测器配置"""
    roi_config: ROIConfig = field(default_factory=ROIConfig)
    histogram_config: HistogramConfig = field(default_factory=HistogramConfig)
    mediapipe_config: MediaPipeConfig = field(default_factory=MediaPipeConfig)
    temporal_config: TemporalConfig = field(default_factory=TemporalConfig)
