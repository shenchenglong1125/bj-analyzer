"""
下半身检测器
基于平均直方图和MediaPipe的综合检测算法
"""
import cv2
import numpy as np
import mediapipe as mp
from typing import List, Dict, Any, Optional, Tuple
from tqdm import tqdm
from .base_detector import DetectorBase
from ..models.data_models import (
    DetectionResult, DetectorConfig, VideoInfo,
    LowerBodyDetectorConfig, ROIConfig, HistogramConfig, 
    MediaPipeConfig, TemporalConfig
)
from ..utils.logger import get_logger


class LowerBodyDetector(DetectorBase):
    """下半身检测器"""
    
    def __init__(self, config: DetectorConfig):
        super().__init__(config)
        
        # 加载下半身检测器特定配置
        self.lower_body_config = self._load_lower_body_config()
        
        # 初始化MediaPipe
        self.mp_pose = None
        self._init_mediapipe()
        
        # 全局平均直方图
        self.global_avg_histogram = None
        
        self.logger.info(f"下半身检测器初始化完成")
    
    def _load_lower_body_config(self) -> LowerBodyDetectorConfig:
        """加载下半身检测器配置"""
        try:
            # 从主配置中提取下半身检测器配置
            config_dict = self.config.__dict__.copy()
            
            # 创建配置对象
            roi_config = ROIConfig(
                horizontal_start=config_dict.get('roi_horizontal_start', 0.25),
                horizontal_end=config_dict.get('roi_horizontal_end', 0.75),
                vertical_start=config_dict.get('roi_vertical_start', 0.5),
                vertical_end=config_dict.get('roi_vertical_end', 1.0)
            )
            
            histogram_config = HistogramConfig(
                sample_interval=config_dict.get('histogram_sample_interval', 30),
                comparison_method=config_dict.get('histogram_comparison_method', 'correlation'),
                threshold=config_dict.get('histogram_threshold', 0.3),
                bins=config_dict.get('histogram_bins', 32),
                channels=config_dict.get('histogram_channels', [0, 1, 2])
            )
            
            mediapipe_config = MediaPipeConfig(
                confidence_threshold=config_dict.get('mediapipe_confidence_threshold', 0.7),
                min_detection_confidence=config_dict.get('mediapipe_min_detection_confidence', 0.5),
                min_tracking_confidence=config_dict.get('mediapipe_min_tracking_confidence', 0.5),
                enable_buttocks_detection=config_dict.get('mediapipe_enable_buttocks_detection', True)
            )
            
            temporal_config = TemporalConfig(
                min_consecutive_frames=config_dict.get('temporal_min_consecutive_frames', 3),
                max_gap_frames=config_dict.get('temporal_max_gap_frames', 5)
            )
            
            return LowerBodyDetectorConfig(
                roi_config=roi_config,
                histogram_config=histogram_config,
                mediapipe_config=mediapipe_config,
                temporal_config=temporal_config
            )
            
        except Exception as e:
            self.logger.warning(f"加载下半身检测器配置失败，使用默认配置: {e}")
            return LowerBodyDetectorConfig()
    
    def _init_mediapipe(self):
        """初始化MediaPipe姿态检测"""
        try:
            self.mp_pose = mp.solutions.pose.Pose(
                static_image_mode=False,
                model_complexity=1,  # 使用轻量级模型
                smooth_landmarks=True,
                enable_segmentation=False,
                smooth_segmentation=True,
                min_detection_confidence=self.lower_body_config.mediapipe_config.min_detection_confidence,
                min_tracking_confidence=self.lower_body_config.mediapipe_config.min_tracking_confidence
            )
            self.logger.info("MediaPipe姿态检测初始化成功")
        except Exception as e:
            self.logger.error(f"MediaPipe初始化失败: {e}")
            self.mp_pose = None
    
    def _preprocess_video_histogram(self, video_path: str, video_info: VideoInfo):
        """预处理视频，计算全局平均直方图"""
        self.logger.info("开始计算全局平均直方图...")
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"无法打开视频文件: {video_path}")
        
        try:
            total_histogram = None
            sample_count = 0
            sample_interval = self.lower_body_config.histogram_config.sample_interval
            
            # 计算总帧数和预估采样数
            total_frames = int(video_info.total_frames)
            estimated_samples = total_frames // sample_interval
            
            frame_count = 0
            with tqdm(total=total_frames, desc="计算全局平均直方图", unit="帧", 
                     bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] {postfix}') as pbar:
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    
                    # 每隔指定帧数采样一次
                    if frame_count % sample_interval == 0:
                        roi = self._extract_roi(frame, video_info)
                        if roi is not None:
                            hist = self._calculate_histogram(roi)
                            
                            if total_histogram is None:
                                total_histogram = hist
                            else:
                                total_histogram += hist
                            
                            sample_count += 1
                    
                    frame_count += 1
                    pbar.update(1)
                    pbar.set_postfix({
                        '采样': f"{sample_count}/{estimated_samples}",
                        '间隔': f"{sample_interval}帧"
                    }, refresh=True)
            
            # 计算平均直方图
            if total_histogram is not None and sample_count > 0:
                self.global_avg_histogram = total_histogram / sample_count
                self.logger.info(f"全局平均直方图计算完成，采样帧数: {sample_count}")
            else:
                self.logger.warning("无法计算全局平均直方图")
                
        finally:
            cap.release()
    
    def _preprocess_negative_periods_histogram(self, video_path: str, video_info: VideoInfo, negative_periods: List[Tuple[float, float]]):
        """预处理阴性时间段，计算全局平均直方图（只计算一阶段阴性时间段）"""
        self.logger.info("开始计算阴性时间段全局平均直方图...")
        
        try:
            total_histogram = None
            sample_count = 0
            sample_interval = self.lower_body_config.histogram_config.sample_interval
            
            # 计算总采样时长和预估采样次数
            total_duration = sum(end - start for start, end in negative_periods)
            estimated_samples = int(total_duration / sample_interval) + 1
            
            with tqdm(total=estimated_samples, desc="计算阴性时间段直方图", 
                     unit="采样", bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] {postfix}') as pbar:
                
                for period_start, period_end in negative_periods:
                    period_duration = period_end - period_start
                    self.logger.info(f"采样时间段: {period_start:.1f}s - {period_end:.1f}s (时长: {period_duration:.1f}s)")
                    
                    # 使用固定间隔采样，类似一阶段的方法
                    current_time = period_start
                    while current_time < period_end:
                        # 使用extract_frame_at_time方式提取帧
                        frame = self._extract_frame_at_time(video_path, current_time)
                        if frame is None:
                            break
                        
                        # 添加调试信息，验证采样精度
                        fps = 60.0  # 视频帧率
                        frame_index = int(current_time * fps)
                        actual_time = frame_index / fps
                        self.logger.debug(f"采样时间点: 目标={current_time:.1f}s, 实际={actual_time:.1f}s, 帧索引={frame_index}")
                        
                        import time
                        start_time = time.time()
                        
                        roi = self._extract_roi(frame, video_info)
                        roi_time = time.time()
                        
                        if roi is not None:
                            hist = self._calculate_histogram(roi)
                            hist_time = time.time()
                            
                            if total_histogram is None:
                                total_histogram = hist
                            else:
                                total_histogram += hist
                            
                            sample_count += 1
                            
                            # 更新进度条（按采样次数更新）
                            pbar.update(1)
                            pbar.set_postfix({
                                '当前时间': f"{current_time:.1f}s",
                                '采样数': sample_count,
                                '时间段': f"{period_start:.1f}s-{period_end:.1f}s",
                                'ROI耗时': f"{(roi_time-start_time)*1000:.1f}ms",
                                '直方图耗时': f"{(hist_time-roi_time)*1000:.1f}ms"
                            }, refresh=True)
                        
                        # 跳到下一个采样点
                        current_time += sample_interval
            
            # 计算平均直方图
            if total_histogram is not None and sample_count > 0:
                self.global_avg_histogram = total_histogram / sample_count
                self.logger.info(f"阴性时间段全局平均直方图计算完成，采样帧数: {sample_count}")
            else:
                self.logger.warning("无法计算阴性时间段全局平均直方图")
                
        except Exception as e:
            self.logger.error(f"计算全局平均直方图失败: {e}")
    
    def _extract_frame_at_time(self, video_path: str, time_seconds: float) -> Optional[np.ndarray]:
        """在指定时间提取单帧（复制two_stage_processor的方法）"""
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                return None
            
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_index = int(time_seconds * fps)
            
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ret, frame = cap.read()
            cap.release()
            
            return frame if ret else None
            
        except Exception as e:
            self.logger.error(f"提取帧失败: {e}")
            return None
    
    def detect_frame(self, frame: np.ndarray) -> Dict[str, Any]:
        """
        检测单帧图像
        
        Args:
            frame: 输入帧图像
            
        Returns:
            检测结果字典
        """
        # 如果全局平均直方图还没有计算，先计算一个简单的参考直方图
        if self.global_avg_histogram is None:
            self.logger.info("全局平均直方图未初始化，使用当前帧作为参考")
            roi = self._extract_roi(frame)
            if roi is not None:
                self.global_avg_histogram = self._calculate_histogram(roi)
            else:
                return {'confidence': 0.0, 'is_lower_body_detected': False}
        
        # 1. 提取ROI区域
        roi = self._extract_roi(frame)
        if roi is None:
            return {'confidence': 0.0, 'is_lower_body_detected': False}
        
        # 2. 直方图检测
        histogram_score = self._detect_histogram_change(roi)
        
        # 3. 如果直方图检测通过，进行MediaPipe检测
        mediapipe_score = 0.0
        if histogram_score > self.lower_body_config.histogram_config.threshold:
            import time
            mediapipe_start = time.time()
            mediapipe_score = self._detect_buttocks_with_mediapipe(frame, roi)
            mediapipe_time = time.time() - mediapipe_start
            self.logger.debug(f"MediaPipe检测耗时: {mediapipe_time*1000:.1f}ms, 分数: {mediapipe_score:.3f}")
        else:
            self.logger.debug(f"直方图分数 {histogram_score:.3f} 未通过阈值 {self.lower_body_config.histogram_config.threshold}, 跳过MediaPipe")
        
        # 4. 综合评分
        confidence = (histogram_score + mediapipe_score) / 2.0
        is_detected = confidence > self.config.confidence_threshold
        
        return {
            'confidence': confidence,
            'is_lower_body_detected': is_detected,
            'histogram_score': histogram_score,
            'mediapipe_score': mediapipe_score
        }
    
    def _extract_roi(self, frame: np.ndarray, video_info: VideoInfo = None) -> Optional[np.ndarray]:
        """提取感兴趣区域"""
        height, width = frame.shape[:2]
        
        # 计算ROI边界
        roi_config = self.lower_body_config.roi_config
        x1 = int(width * roi_config.horizontal_start)
        x2 = int(width * roi_config.horizontal_end)
        y1 = int(height * roi_config.vertical_start)
        y2 = int(height * roi_config.vertical_end)
        
        # 提取ROI
        roi = frame[y1:y2, x1:x2]
        
        if roi.size == 0:
            return None
        
        return roi
    
    def _calculate_histogram(self, roi: np.ndarray) -> np.ndarray:
        """计算ROI区域的直方图"""
        hist_config = self.lower_body_config.histogram_config
        
        # 计算直方图
        hist = cv2.calcHist(
            [roi], 
            hist_config.channels, 
            None, 
            [hist_config.bins] * len(hist_config.channels), 
            [0, 256] * len(hist_config.channels)
        )
        
        # 归一化
        cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)
        
        return hist
    
    def _detect_histogram_change(self, roi: np.ndarray) -> float:
        """检测直方图变化"""
        if self.global_avg_histogram is None:
            return 0.0
        
        # 计算当前帧的直方图
        current_hist = self._calculate_histogram(roi)
        
        # 比较直方图
        method = self.lower_body_config.histogram_config.comparison_method
        if method == "correlation":
            similarity = cv2.compareHist(current_hist, self.global_avg_histogram, cv2.HISTCMP_CORREL)
        elif method == "chi_square":
            similarity = cv2.compareHist(current_hist, self.global_avg_histogram, cv2.HISTCMP_CHISQR)
        elif method == "intersection":
            similarity = cv2.compareHist(current_hist, self.global_avg_histogram, cv2.HISTCMP_INTERSECT)
        else:
            similarity = cv2.compareHist(current_hist, self.global_avg_histogram, cv2.HISTCMP_CORREL)
        
        # 转换为变化程度（0-1，1表示变化最大）
        if method == "correlation":
            change_degree = 1.0 - max(0, similarity)  # 相关性越高，变化越小
        elif method == "chi_square":
            change_degree = min(1.0, similarity / 1000.0)  # 卡方值越大，变化越大
        else:  # intersection
            change_degree = 1.0 - min(1.0, similarity / 100.0)  # 交集越小，变化越大
        
        return change_degree
    
    def _detect_buttocks_with_mediapipe(self, frame: np.ndarray, roi: np.ndarray) -> float:
        """使用MediaPipe检测臀部"""
        if self.mp_pose is None:
            return 0.0
        
        try:
            # 转换为RGB格式
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # MediaPipe检测
            results = self.mp_pose.process(frame_rgb)
            
            if results.pose_landmarks:
                # 获取臀部关键点（23: 左髋, 24: 右髋）
                landmarks = results.pose_landmarks.landmark
                
                # 检查臀部关键点是否在ROI区域内
                roi_config = self.lower_body_config.roi_config
                height, width = frame.shape[:2]
                
                # 计算ROI边界
                roi_x1 = width * roi_config.horizontal_start
                roi_x2 = width * roi_config.horizontal_end
                roi_y1 = height * roi_config.vertical_start
                roi_y2 = height * roi_config.vertical_end
                
                # 检查臀部关键点
                buttocks_in_roi = 0
                total_buttocks_points = 0
                
                # 臀部相关关键点：23(左髋), 24(右髋), 25(左膝), 26(右膝)
                buttocks_landmarks = [23, 24, 25, 26]
                
                for landmark_id in buttocks_landmarks:
                    if landmark_id < len(landmarks):
                        landmark = landmarks[landmark_id]
                        
                        # 检查可见性
                        if landmark.visibility > self.lower_body_config.mediapipe_config.confidence_threshold:
                            total_buttocks_points += 1
                            
                            # 检查是否在ROI区域内
                            x = landmark.x * width
                            y = landmark.y * height
                            
                            if (roi_x1 <= x <= roi_x2 and roi_y1 <= y <= roi_y2):
                                buttocks_in_roi += 1
                
                # 计算置信度
                if total_buttocks_points > 0:
                    confidence = buttocks_in_roi / total_buttocks_points
                    return confidence
                
        except Exception as e:
            self.logger.error(f"MediaPipe检测失败: {e}")
        
        return 0.0
    
    def is_split_screen(self, detection_result: Dict[str, Any]) -> bool:
        """
        判断是否为分屏场景（复用为下半身检测）
        
        Args:
            detection_result: 检测结果
            
        Returns:
            是否检测到下半身
        """
        return detection_result.get('is_lower_body_detected', False)
    
    def get_detector_info(self) -> Dict[str, Any]:
        """获取检测器信息"""
        info = super().get_detector_info()
        info.update({
            'roi_config': {
                'horizontal_start': self.lower_body_config.roi_config.horizontal_start,
                'horizontal_end': self.lower_body_config.roi_config.horizontal_end,
                'vertical_start': self.lower_body_config.roi_config.vertical_start,
                'vertical_end': self.lower_body_config.roi_config.vertical_end
            },
            'histogram_config': {
                'sample_interval': self.lower_body_config.histogram_config.sample_interval,
                'comparison_method': self.lower_body_config.histogram_config.comparison_method,
                'threshold': self.lower_body_config.histogram_config.threshold
            },
            'mediapipe_config': {
                'confidence_threshold': self.lower_body_config.mediapipe_config.confidence_threshold,
                'enable_buttocks_detection': self.lower_body_config.mediapipe_config.enable_buttocks_detection
            }
        })
        return info
