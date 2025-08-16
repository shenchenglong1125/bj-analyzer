"""
两阶段检测处理器
第一阶段：三联屏检测 + 自适应处理
第二阶段：对三联屏阴性片段进行下半身检测
"""
import time
import cv2
import numpy as np
from typing import List, Optional, Tuple
from pathlib import Path
from tqdm import tqdm

from ..models.data_models import (
    VideoInfo, ProcessingResult, DetectionResult, 
    ProcessingConfig, AdaptiveConfig
)
from ..detectors import create_detector
from ..utils.file_utils import FileUtils
from ..utils.logger import get_logger

logger = get_logger(__name__)


class EfficientFrameExtractor:
    """高效的帧提取器，避免重复打开视频文件"""
    
    def __init__(self, video_path: str):
        self.video_path = video_path
        self.cap = None
        self.fps = None
        self.total_frames = None
        self._open_video()
    
    def _open_video(self):
        """打开视频文件"""
        try:
            self.cap = cv2.VideoCapture(self.video_path)
            if not self.cap.isOpened():
                raise Exception(f"无法打开视频文件: {self.video_path}")
            
            self.fps = self.cap.get(cv2.CAP_PROP_FPS)
            self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
        except Exception as e:
            logger.error(f"打开视频文件失败: {e}")
            self.cap = None
    
    def extract_frame_at_time(self, time_seconds: float) -> Optional[np.ndarray]:
        """在指定时间提取单帧"""
        if self.cap is None:
            return None
        
        try:
            frame_index = int(time_seconds * self.fps)
            
            # 使用SEEK_SET模式直接跳转到指定帧
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ret, frame = self.cap.read()
            
            return frame if ret else None
            
        except Exception as e:
            logger.error(f"提取帧失败: {e}")
            return None
    
    def extract_frames_at_times(self, time_list: List[float]) -> List[Optional[np.ndarray]]:
        """批量提取多个时间点的帧，按时间排序以提高效率"""
        if self.cap is None:
            return [None] * len(time_list)
        
        # 按时间排序，减少跳转距离
        sorted_times = sorted(time_list)
        frames = []
        
        try:
            for time_seconds in sorted_times:
                frame_index = int(time_seconds * self.fps)
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
                ret, frame = self.cap.read()
                frames.append(frame if ret else None)
            
            # 按原始顺序返回
            time_to_frame = dict(zip(sorted_times, frames))
            return [time_to_frame.get(t, None) for t in time_list]
            
        except Exception as e:
            logger.error(f"批量提取帧失败: {e}")
            return [None] * len(time_list)
    
    def close(self):
        """关闭视频文件"""
        if self.cap is not None:
            self.cap.release()
            self.cap = None
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


class TwoStageProcessor:
    """两阶段检测处理器"""
    
    def __init__(self, config: ProcessingConfig):
        self.config = config
        self.adaptive_config = config.adaptive_config
        self.logger = logger
        
        # 第一阶段：三联屏检测器 + 自适应处理器
        self.triple_screen_config = self._create_triple_screen_config()
        self.triple_screen_detector = create_detector("triple_screen_detector", self.triple_screen_config)
        
        # 第二阶段：下半身检测器
        # 使用主配置中的下半身检测器配置
        self.lower_body_detector = create_detector("lower_body_detector", config.detector_config)
        
        # 生成抽帧间隔序列
        self.frame_intervals = self._generate_frame_intervals()
        
        self.logger.info("两阶段检测处理器初始化完成")
        self.logger.info(f"第一阶段：三联屏检测 + 自适应处理")
        self.logger.info(f"第二阶段：下半身检测")
        self.logger.info(f"自适应间隔序列: {self.frame_intervals}")
    
    def _create_triple_screen_config(self):
        """创建三联屏检测器配置"""
        from ..models.data_models import DetectorConfig
        
        # 使用配置中的三联屏检测器配置
        triple_config = DetectorConfig()
        triple_config.detector_type = "triple_screen_detector"
        triple_config.confidence_threshold = self.config.triple_screen_config.confidence_threshold
        triple_config.similarity_threshold = self.config.triple_screen_config.similarity_threshold
        triple_config.method = self.config.triple_screen_config.method
        triple_config.min_region_size = self.config.triple_screen_config.min_region_size
        
        return triple_config
    
    def _generate_frame_intervals(self) -> List[float]:
        """生成抽帧间隔序列"""
        intervals = []
        current_interval = self.adaptive_config.initial_interval
        
        while current_interval <= self.adaptive_config.max_interval:
            intervals.append(current_interval)
            current_interval *= self.adaptive_config.interval_multiplier
        
        return sorted(intervals)
    
    def get_video_info(self, video_path: str) -> Optional[VideoInfo]:
        """获取视频信息"""
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                self.logger.error(f"无法打开视频文件: {video_path}")
                return None
            
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            duration = total_frames / fps if fps > 0 else 0
            
            cap.release()
            
            return VideoInfo(
                file_path=video_path,
                file_name=Path(video_path).name,
                duration=duration,
                fps=fps,
                width=width,
                height=height,
                total_frames=total_frames
            )
            
        except Exception as e:
            self.logger.error(f"获取视频信息失败: {e}")
            return None
    
    def extract_frame_at_time(self, video_path: str, time_seconds: float) -> Optional[np.ndarray]:
        """在指定时间提取单帧（保持向后兼容）"""
        with EfficientFrameExtractor(video_path) as extractor:
            return extractor.extract_frame_at_time(time_seconds)
    
    def process_video(self, video_path: str) -> ProcessingResult:
        """两阶段处理视频"""
        start_time = time.time()
        
        # 获取视频信息
        video_info = self.get_video_info(video_path)
        if not video_info:
            return ProcessingResult(
                video_info=VideoInfo(file_path=video_path, file_name="", duration=0, fps=0, width=0, height=0, total_frames=0),
                processing_status="failed",
                error_message="无法获取视频信息"
            )
        
        self.logger.info(f"开始两阶段处理视频: {video_info.file_name}")
        self.logger.info(f"视频信息: 时长={video_info.duration:.2f}s, 帧率={video_info.fps:.2f}")
        
        # 第一阶段：三联屏检测 + 自适应处理
        triple_screen_segments = self._stage_one_triple_screen_detection(video_path, video_info)
        
        self.logger.info(f"第一阶段完成：检测到 {len(triple_screen_segments)} 个三联屏片段")
        
        # 第二阶段：下半身检测
        # 先预处理下半身检测器的全局平均直方图（只计算一阶段阴性时间段）
        self._preprocess_lower_body_histogram(video_path, video_info, triple_screen_segments)
        
        lower_body_segments = self._stage_two_lower_body_detection(video_path, video_info, triple_screen_segments)
        
        # 合并所有检测结果
        all_segments = triple_screen_segments + lower_body_segments
        
        # 按时间排序
        all_segments.sort(key=lambda x: x.segment_start)
        
        # 最终合并相邻片段（只在这里合并一次）
        merged_segments = self._merge_adjacent_segments(all_segments)
        
        # 统计结果
        triple_screen_count = sum(1 for seg in merged_segments if seg.is_split_screen and not getattr(seg, 'is_lower_body', False))
        lower_body_count = sum(1 for seg in merged_segments if getattr(seg, 'is_lower_body', False))
        total_processing_time = time.time() - start_time
        
        result = ProcessingResult(
            video_info=video_info,
            segments=merged_segments,
            total_processing_time=total_processing_time,
            total_segments=len(merged_segments),
            split_screen_segments=triple_screen_count + lower_body_count
        )
        
        self.logger.info(f"两阶段处理完成: {video_info.file_name}")
        self.logger.info(f"处理统计: 总片段={result.total_segments}, 三联屏片段={triple_screen_count}, 下半身片段={lower_body_count}, 处理时间={total_processing_time:.2f}s")
        
        return result
    
    def _stage_one_triple_screen_detection(self, video_path: str, video_info: VideoInfo) -> List[DetectionResult]:
        """第一阶段：三联屏检测 + 自适应处理"""
        self.logger.info("第一阶段：开始三联屏检测...")
        
        segments = []
        current_time = 0.0
        
        # 状态变化检测变量
        current_state = None  # 当前状态：True=三联屏, False=单屏
        state_count = 0  # 当前状态持续计数
        current_interval_index = 0  # 当前使用的间隔索引
        
        # 状态稳定性缓冲机制
        state_buffer = []  # 状态缓冲区，存储最近几次的检测结果
        buffer_size = 5  # 缓冲区大小，需要连续5次相同结果才确认状态变化
        min_stable_time = 10.0  # 最小稳定时间（秒），状态必须稳定至少10秒
        
        # 三联屏片段状态变量
        triple_screen_start = None  # 三联屏开始时间
        triple_screen_end = None    # 三联屏结束时间
        last_state_change_time = 0.0  # 上次状态变化时间
        
        # 创建进度条
        with tqdm(total=video_info.duration, desc="三联屏检测进度", unit="秒", 
                 bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt}s [{elapsed}<{remaining}, {rate_fmt}] {postfix}') as pbar:
            while current_time < video_info.duration:
                # 提取当前时间点的帧
                frame = self.extract_frame_at_time(video_path, current_time)
                if frame is None:
                    break
                
                # 检测当前帧
                result = self.triple_screen_detector.detect_frame(frame)
                if result is None:
                    break
                
                # 获取检测结果
                is_triple_screen = result.get('is_triple_screen', False)
                confidence = result.get('confidence', 0.0)
                
                # 更新状态缓冲区
                state_buffer.append(is_triple_screen)
                if len(state_buffer) > buffer_size:
                    state_buffer.pop(0)
                
                # 只有当缓冲区满了才进行状态判断
                if len(state_buffer) == buffer_size:
                    # 判断缓冲区中的状态是否一致
                    buffer_state = all(state_buffer) if state_buffer[0] else not any(state_buffer)
                    stable_state = state_buffer[0] if buffer_state else None
                    
                    # 状态变化检测
                    if current_state is None:
                        # 第一次检测，初始化状态
                        current_state = stable_state
                        state_count = 1
                        current_interval_index = 0
                        last_state_change_time = current_time
                        
                        # 如果第一次检测就是三联屏，记录开始时间
                        if stable_state:
                            triple_screen_start = current_time
                    elif current_state != stable_state and stable_state is not None:
                        # 状态发生变化，检查时间间隔
                        time_since_last_change = current_time - last_state_change_time
                        
                        if time_since_last_change >= min_stable_time:
                            # 状态变化有效
                            if current_state == False and stable_state:
                                # 单屏转三联屏：记录三联屏开始时间
                                triple_screen_start = current_time
                                self.logger.debug(f"三联屏开始: {triple_screen_start:.1f}s")
                            elif current_state == True and not stable_state:
                                # 三联屏转单屏：记录三联屏结束时间
                                triple_screen_end = current_time
                                self.logger.debug(f"三联屏结束: {triple_screen_end:.1f}s")
                                
                                # 创建三联屏片段
                                if triple_screen_start is not None and triple_screen_end is not None:
                                    segment_result = DetectionResult(
                                        segment_start=triple_screen_start,
                                        segment_end=triple_screen_end,
                                        is_split_screen=True,
                                        confidence=confidence,
                                        detection_frames=1,
                                        processing_time=0.0
                                    )
                                    segments.append(segment_result)
                                    
                                    # 重置三联屏状态
                                    triple_screen_start = None
                                    triple_screen_end = None
                            
                            # 更新状态
                            current_state = stable_state
                            state_count = 1
                            current_interval_index = 0
                            last_state_change_time = current_time
                        else:
                            # 状态变化时间太短，忽略
                            self.logger.debug(f"状态变化时间太短，忽略: {time_since_last_change:.1f}s < {min_stable_time}s")
                    elif current_state == stable_state:
                        # 状态相同，增加计数
                        state_count += 1
                        
                        # 根据持续次数调整间隔
                        if state_count >= 3 and current_interval_index < len(self.frame_intervals) - 1:
                            current_interval_index += 1
                            state_count = 0  # 重置计数
                
                # 根据当前间隔跳到下一个检测点
                next_time = current_time + self.frame_intervals[current_interval_index]
                
                # 更新进度条
                pbar.update(self.frame_intervals[current_interval_index])
                pbar.set_postfix({
                    '时间': f"{current_time:.1f}s",
                    '状态': "三联屏" if current_state else "单屏",
                    '计数': state_count,
                    '间隔': f"{self.frame_intervals[current_interval_index]:.1f}s",
                    '片段': len(segments),
                    '三联屏': sum(1 for seg in segments if seg.is_split_screen)
                }, refresh=True)
                
                current_time = next_time
                pbar.refresh()
        
        # 处理视频结束时的三联屏片段
        if triple_screen_start is not None and triple_screen_end is None:
            # 如果视频结束时还在三联屏状态，使用视频结束时间作为结束时间
            triple_screen_end = video_info.duration
            self.logger.debug(f"视频结束时的三联屏结束: {triple_screen_end:.1f}s")
            
            segment_result = DetectionResult(
                segment_start=triple_screen_start,
                segment_end=triple_screen_end,
                is_split_screen=True,
                confidence=confidence,
                detection_frames=1,
                processing_time=0.0
            )
            segments.append(segment_result)
        
        self.logger.info(f"第一阶段完成：检测到 {len(segments)} 个三联屏片段")
        return segments
    
    def _preprocess_lower_body_histogram(self, video_path: str, video_info: VideoInfo, triple_screen_segments: List[DetectionResult]):
        """预处理下半身检测器的全局平均直方图（只计算一阶段阴性时间段）"""
        self.logger.info("预处理下半身检测器的全局平均直方图（一阶段阴性时间段）...")
        
        # 找到三联屏阴性时间段
        negative_periods = self._find_negative_periods(video_info.duration, triple_screen_segments)
        
        if not negative_periods:
            self.logger.warning("没有找到三联屏阴性时间段，无法计算全局平均直方图")
            return
        
        # 计算总采样时长
        total_duration = sum(end - start for start, end in negative_periods)
        self.logger.info(f"阴性时间段总时长: {total_duration:.1f}s")
        
        # 调用下半身检测器的预处理方法，传入阴性时间段
        self.lower_body_detector._preprocess_negative_periods_histogram(video_path, video_info, negative_periods)
    
    def _stage_two_lower_body_detection(self, video_path: str, video_info: VideoInfo, 
                                      triple_screen_segments: List[DetectionResult]) -> List[DetectionResult]:
        """第二阶段：下半身检测"""
        self.logger.info("第二阶段：开始下半身检测...")
        
        # 找出三联屏检测为阴性的时间段
        negative_periods = self._find_negative_periods(video_info.duration, triple_screen_segments)
        
        if not negative_periods:
            self.logger.info("没有找到三联屏阴性时间段，跳过下半身检测")
            return []
        
        self.logger.info(f"找到 {len(negative_periods)} 个三联屏阴性时间段，开始下半身检测")
        
        # 对每个阴性时间段进行下半身检测
        lower_body_segments = []
        
        # 计算总检测时长
        total_detection_duration = sum(end - start for start, end in negative_periods)
        current_detection_duration = 0
        
        with tqdm(total=total_detection_duration, desc="下半身检测总体进度", 
                 unit="秒", bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt}s [{elapsed}<{remaining}, {rate_fmt}] {postfix}') as pbar:
            for i, (period_start, period_end) in enumerate(negative_periods):
                period_duration = period_end - period_start
                self.logger.info(f"检测时间段 {i+1}/{len(negative_periods)}: {period_start:.1f}s - {period_end:.1f}s (时长: {period_duration:.1f}s)")
                
                # 使用下半身检测器处理这个时间段
                segments = self._detect_lower_body_in_period(video_path, video_info, period_start, period_end)
                lower_body_segments.extend(segments)
                
                # 更新总体进度
                current_detection_duration += period_duration
                pbar.update(period_duration)
                pbar.set_postfix({
                    '时间段': f"{i+1}/{len(negative_periods)}",
                    '当前': f"{period_start:.1f}s-{period_end:.1f}s",
                    '总片段': len(lower_body_segments)
                }, refresh=True)
        
        self.logger.info(f"第二阶段完成：检测到 {len(lower_body_segments)} 个下半身片段")
        return lower_body_segments
    
    def _find_negative_periods(self, total_duration: float, triple_screen_segments: List[DetectionResult]) -> List[Tuple[float, float]]:
        """找出三联屏检测为阴性的时间段"""
        if not triple_screen_segments:
            # 如果没有三联屏片段，整个视频都是阴性
            return [(0.0, total_duration)]
        
        negative_periods = []
        current_time = 0.0
        
        for segment in triple_screen_segments:
            # 如果当前时间到片段开始之间有间隔，说明是阴性时间段
            if segment.segment_start > current_time:
                negative_periods.append((current_time, segment.segment_start))
            
            # 更新当前时间到片段结束
            current_time = segment.segment_end
        
        # 检查最后一个片段之后的时间
        if current_time < total_duration:
            negative_periods.append((current_time, total_duration))
        
        return negative_periods
    
    def _detect_lower_body_in_period(self, video_path: str, video_info: VideoInfo, 
                                   period_start: float, period_end: float) -> List[DetectionResult]:
        """在指定时间段内检测下半身"""
        segments = []
        current_time = period_start
        
        # 自适应间隔配置
        adaptive_intervals = [1.0, 2.0, 4.0, 8.0, 16.0]  # 自适应间隔序列
        
        # 状态变化检测变量
        current_state = None  # 当前状态：True=有臀部, False=无臀部
        state_count = 0  # 当前状态持续计数
        current_interval_index = 0  # 当前使用的间隔索引
        
        # 状态稳定性缓冲机制
        state_buffer = []  # 状态缓冲区，存储最近几次的检测结果
        buffer_size = 3  # 缓冲区大小，需要连续3次相同结果才确认状态变化
        min_stable_time = 5.0  # 最小稳定时间（秒），状态必须稳定至少5秒
        
        # 下半身片段状态变量
        lower_body_start = None  # 下半身开始时间
        lower_body_end = None    # 下半身结束时间
        last_state_change_time = 0.0  # 上次状态变化时间
        
        # 计算时间段总时长
        period_duration = period_end - period_start
        
        with tqdm(total=period_duration, desc=f"下半身检测进度 ({period_start:.1f}s-{period_end:.1f}s)", 
                 unit="秒", bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt}s [{elapsed}<{remaining}, {rate_fmt}] {postfix}') as pbar:
            while current_time < period_end:
                # 提取当前时间点的帧
                frame = self.extract_frame_at_time(video_path, current_time)
                if frame is None:
                    break
                
                # 检测当前帧
                result = self.lower_body_detector.detect_frame(frame)
                if result is None:
                    break
                
                # 获取检测结果
                is_lower_body = result.get('is_lower_body_detected', False)
                confidence = result.get('confidence', 0.0)
                histogram_score = result.get('histogram_score', 0.0)
                mediapipe_score = result.get('mediapipe_score', 0.0)
                
                # 更新状态缓冲区
                state_buffer.append(is_lower_body)
                if len(state_buffer) > buffer_size:
                    state_buffer.pop(0)
                
                # 只有当缓冲区满了才进行状态判断
                if len(state_buffer) == buffer_size:
                    # 判断缓冲区中的状态是否一致
                    buffer_state = all(state_buffer) if state_buffer[0] else not any(state_buffer)
                    stable_state = state_buffer[0] if buffer_state else None
                    
                    # 状态变化检测
                    if current_state is None:
                        # 第一次检测，初始化状态
                        current_state = stable_state
                        state_count = 1
                        current_interval_index = 0
                        last_state_change_time = current_time
                        
                        # 如果第一次检测就是下半身，记录开始时间
                        if stable_state:
                            lower_body_start = current_time
                    elif current_state != stable_state and stable_state is not None:
                        # 状态发生变化，检查时间间隔
                        time_since_last_change = current_time - last_state_change_time
                        
                        if time_since_last_change >= min_stable_time:
                            # 状态变化有效
                            if current_state == False and stable_state:
                                # 无臀部转有臀部：记录下半身开始时间
                                lower_body_start = current_time
                                self.logger.info(f"下半身开始: {lower_body_start:.1f}s (置信度={confidence:.3f})")
                            elif current_state == True and not stable_state:
                                # 有臀部转无臀部：记录下半身结束时间
                                lower_body_end = current_time
                                self.logger.info(f"下半身结束: {lower_body_end:.1f}s (置信度={confidence:.3f})")
                                
                                # 创建下半身检测片段
                                if lower_body_start is not None and lower_body_end is not None:
                                    segment_result = DetectionResult(
                                        segment_start=lower_body_start,
                                        segment_end=lower_body_end,
                                        is_split_screen=True,  # 复用字段，表示检测到目标
                                        confidence=confidence,
                                        detection_frames=1,
                                        processing_time=0.0
                                    )
                                    # 标记为下半身检测结果
                                    segment_result.is_lower_body = True
                                    segments.append(segment_result)
                                    
                                    # 重置下半身状态
                                    lower_body_start = None
                                    lower_body_end = None
                            
                            # 更新状态
                            current_state = stable_state
                            state_count = 1
                            current_interval_index = 0  # 状态变化时重置间隔
                            last_state_change_time = current_time
                        else:
                            # 状态变化时间太短，忽略
                            self.logger.debug(f"状态变化时间太短，忽略: {time_since_last_change:.1f}s < {min_stable_time}s")
                    elif current_state == stable_state:
                        # 状态相同，增加计数
                        state_count += 1
                        
                        # 根据持续次数调整间隔
                        if state_count >= 3 and current_interval_index < len(adaptive_intervals) - 1:
                            current_interval_index += 1
                            state_count = 0  # 重置计数
                
                # 根据当前间隔跳到下一个检测点
                next_time = current_time + adaptive_intervals[current_interval_index]
                
                # 更新进度条
                pbar.update(adaptive_intervals[current_interval_index])
                pbar.set_postfix({
                    '时间': f"{current_time:.1f}s",
                    '状态': "有臀部" if current_state else "无臀部",
                    '计数': state_count,
                    '间隔': f"{adaptive_intervals[current_interval_index]:.1f}s",
                    '片段': len(segments),
                    '置信度': f"{confidence:.3f}"
                }, refresh=True)
                
                current_time = next_time
                pbar.refresh()
        
        # 处理时间段结束时的下半身片段
        if lower_body_start is not None and lower_body_end is None:
            # 如果时间段结束时还在下半身状态，使用时间段结束时间作为结束时间
            lower_body_end = period_end
            self.logger.info(f"时间段结束时的下半身结束: {lower_body_end:.1f}s (置信度={confidence:.3f})")
            
            segment_result = DetectionResult(
                segment_start=lower_body_start,
                segment_end=lower_body_end,
                is_split_screen=True,  # 复用字段，表示检测到目标
                confidence=confidence,
                detection_frames=1,
                processing_time=0.0
            )
            # 标记为下半身检测结果
            segment_result.is_lower_body = True
            segments.append(segment_result)
        
        return segments
    
    def _merge_adjacent_segments(self, segments: List[DetectionResult]) -> List[DetectionResult]:
        """合并相邻的检测片段"""
        if not segments:
            return []
        
        merged = []
        current = segments[0]
        
        for next_segment in segments[1:]:
            # 如果当前片段和下一个片段都是检测结果，且时间间隔小于阈值
            if (current.is_split_screen == next_segment.is_split_screen and
                next_segment.segment_start - current.segment_end <= 5.0):  # 5秒合并阈值
                # 合并片段
                current = DetectionResult(
                    segment_start=current.segment_start,
                    segment_end=next_segment.segment_end,
                    is_split_screen=current.is_split_screen,
                    confidence=max(current.confidence, next_segment.confidence),
                    detection_frames=current.detection_frames + next_segment.detection_frames,
                    processing_time=current.processing_time + next_segment.processing_time
                )
                # 保持下半身标记
                if hasattr(current, 'is_lower_body') or hasattr(next_segment, 'is_lower_body'):
                    current.is_lower_body = True
            else:
                merged.append(current)
                current = next_segment
        
        merged.append(current)
        return merged
