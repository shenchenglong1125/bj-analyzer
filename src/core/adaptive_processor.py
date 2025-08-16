"""
自适应抽帧处理器
实现基于检测结果动态调整抽帧间隔的优化系统
"""
import time
import cv2
import numpy as np
from typing import List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from tqdm import tqdm

from ..models.data_models import (
    VideoInfo, ProcessingResult, DetectionResult, 
    ProcessingConfig, AdaptiveConfig
)
from ..detectors.triple_screen_detector import TripleScreenDetector
from ..utils.file_utils import FileUtils
from ..utils.logger import get_logger

logger = get_logger(__name__)


class AdaptiveProcessor:
    """自适应抽帧处理器"""
    
    def __init__(self, config: ProcessingConfig):
        self.config = config
        self.adaptive_config = config.adaptive_config
        self.detector = TripleScreenDetector(config.detector_config)
        self.logger = logger
        
        # 生成抽帧间隔序列
        self.frame_intervals = self._generate_frame_intervals()
        self.logger.info(f"抽帧间隔序列: {self.frame_intervals}")
    
    def _generate_frame_intervals(self) -> List[float]:
        """生成抽帧间隔序列"""
        intervals = []
        current_interval = self.adaptive_config.initial_interval
        
        while current_interval <= self.adaptive_config.max_interval:
            intervals.append(current_interval)
            current_interval *= self.adaptive_config.interval_multiplier
        
        # 添加精确模式间隔
        if self.adaptive_config.precision_mode_interval not in intervals:
            intervals.append(self.adaptive_config.precision_mode_interval)
        
        # 排序并去重
        intervals = sorted(list(set(intervals)))
        return intervals
    
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
        """在指定时间提取单帧"""
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
    
    def process_video_adaptive(self, video_path: str) -> ProcessingResult:
        """基于状态变化的自适应处理视频"""
        start_time = time.time()
        
        # 获取视频信息
        video_info = self.get_video_info(video_path)
        if not video_info:
            return ProcessingResult(
                video_info=VideoInfo(file_path=video_path, file_name="", duration=0, fps=0, width=0, height=0, total_frames=0),
                processing_status="failed",
                error_message="无法获取视频信息"
            )
        
        self.logger.info(f"开始自适应处理视频: {video_info.file_name}")
        self.logger.info(f"视频信息: 时长={video_info.duration:.2f}s, 帧率={video_info.fps:.2f}")
        self.logger.info(f"处理配置: 单线程, 抽帧间隔序列={self.frame_intervals}秒")
        
        segments = []
        current_time = 0.0
        
        # 状态变化检测变量
        current_state = None  # 当前状态：True=分屏, False=单屏
        state_count = 0  # 当前状态持续计数
        current_interval_index = 0  # 当前使用的间隔索引
        
        # 创建进度条 - 基于视频时长而不是预估检测次数
        with tqdm(total=video_info.duration, desc="状态检测进度", unit="秒", 
                 bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt}s [{elapsed}<{remaining}, {rate_fmt}] {postfix}') as pbar:
            while current_time < video_info.duration:
                # 提取当前时间点的帧
                frame = self.extract_frame_at_time(video_path, current_time)
                if frame is None:
                    break
                
                # 检测当前帧
                result = self.detector.detect_frame(frame)
                if result is None:
                    break
                
                # 获取检测结果
                is_split_screen = result.get('is_triple_screen', False)
                confidence = result.get('confidence', 0.0)
                
                # 状态变化检测
                if current_state is None:
                    # 第一次检测，初始化状态
                    current_state = is_split_screen
                    state_count = 1
                    current_interval_index = 0
                elif current_state == is_split_screen:
                    # 状态相同，增加计数
                    state_count += 1
                    
                    # 根据持续次数调整间隔
                    if state_count >= 2 and current_interval_index < len(self.frame_intervals) - 1:
                        current_interval_index += 1
                        state_count = 0  # 重置计数
                else:
                    # 状态发生变化，重置计数器
                    current_state = is_split_screen
                    state_count = 1
                    current_interval_index = 0
                
                # 记录检测结果
                if is_split_screen or confidence > 0.3:
                    segment_result = DetectionResult(
                        segment_start=current_time,
                        segment_end=current_time + self.frame_intervals[current_interval_index],
                        is_split_screen=is_split_screen,
                        confidence=confidence,
                        detection_frames=1,
                        processing_time=time.time() - start_time
                    )
                    segments.append(segment_result)
                
                # 根据当前间隔跳到下一个检测点
                next_time = current_time + self.frame_intervals[current_interval_index]
                
                # 更新进度条 - 基于实际处理的时间进度
                pbar.update(self.frame_intervals[current_interval_index])
                pbar.set_postfix({
                    '时间': f"{current_time:.1f}s",
                    '状态': "分屏" if current_state else "单屏",
                    '计数': state_count,
                    '间隔': f"{self.frame_intervals[current_interval_index]:.1f}s",
                    '片段': len(segments),
                    '分屏': sum(1 for seg in segments if seg.is_split_screen)
                }, refresh=True)
                
                current_time = next_time
                
                # 强制刷新进度条显示
                pbar.refresh()
        
        # 合并相邻的分屏片段
        merged_segments = self._merge_adjacent_segments(segments)
        
        # 统计结果
        split_screen_segments = sum(1 for seg in merged_segments if seg.is_split_screen)
        total_processing_time = time.time() - start_time
        
        result = ProcessingResult(
            video_info=video_info,
            segments=merged_segments,
            total_processing_time=total_processing_time,
            total_segments=len(merged_segments),
            split_screen_segments=split_screen_segments
        )
        
        self.logger.info(f"自适应处理完成: {video_info.file_name}")
        self.logger.info(f"处理统计: 总片段={result.total_segments}, 分屏片段={result.split_screen_segments}, 处理时间={total_processing_time:.2f}s")
        
        return result
    
    def _merge_adjacent_segments(self, segments: List[DetectionResult]) -> List[DetectionResult]:
        """合并相邻的分屏片段"""
        if not segments:
            return []
        
        merged = []
        current_segment = segments[0]
        
        for next_segment in segments[1:]:
            # 如果当前片段和下一个片段都是分屏，且时间间隔小于合并阈值，则合并
            if (current_segment.is_split_screen and next_segment.is_split_screen and
                next_segment.segment_start - current_segment.segment_end <= 5.0):  # 5秒合并阈值
                current_segment.segment_end = next_segment.segment_end
                current_segment.confidence = max(current_segment.confidence, next_segment.confidence)
                current_segment.detection_frames += next_segment.detection_frames
            else:
                merged.append(current_segment)
                current_segment = next_segment
        
        merged.append(current_segment)
        return merged
    
    def process_video_parallel(self, video_path: str) -> ProcessingResult:
        """并行处理视频（多线程版本）"""
        start_time = time.time()
        
        # 获取视频信息
        video_info = self.get_video_info(video_path)
        if not video_info:
            return ProcessingResult(
                video_info=VideoInfo(file_path=video_path, file_name="", duration=0, fps=0, width=0, height=0, total_frames=0),
                processing_status="failed",
                error_message="无法获取视频信息"
            )
        
        self.logger.info(f"开始并行自适应处理视频: {video_info.file_name}")
        self.logger.info(f"视频信息: 时长={video_info.duration:.2f}s, 帧率={video_info.fps:.2f}")
        self.logger.info(f"处理配置: {self.config.max_workers}线程, 抽帧间隔序列={self.frame_intervals}秒")
        
        # 将视频分割成多个片段进行并行处理
        segment_duration = self.config.detector_config.segment_duration
        segments = []
        
        # 计算总片段数
        total_segments = int(video_info.duration / segment_duration) + 1
        
        with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
            futures = []
            
            for start_time_seg in np.arange(0, video_info.duration, segment_duration):
                end_time_seg = min(start_time_seg + segment_duration, video_info.duration)
                future = executor.submit(
                    self._process_segment_parallel, 
                    video_path, start_time_seg, end_time_seg
                )
                futures.append(future)
            
            # 收集结果
            with tqdm(total=len(futures), desc="并行处理进度", unit="片段", 
                     bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] {postfix}') as pbar:
                for future in as_completed(futures):
                    try:
                        segment_result = future.result()
                        if segment_result:
                            segments.append(segment_result)
                    except Exception as e:
                        self.logger.error(f"处理片段失败: {e}")
                    pbar.update(1)
                    pbar.set_postfix({
                        '片段': len(segments),
                        '分屏': sum(1 for seg in segments if seg.is_split_screen),
                        '线程': self.config.max_workers
                    }, refresh=True)
                    pbar.refresh()
        
        # 合并相邻的分屏片段
        merged_segments = self._merge_adjacent_segments(segments)
        
        # 统计结果
        split_screen_segments = sum(1 for seg in merged_segments if seg.is_split_screen)
        total_processing_time = time.time() - start_time
        
        result = ProcessingResult(
            video_info=video_info,
            segments=merged_segments,
            total_processing_time=total_processing_time,
            total_segments=len(merged_segments),
            split_screen_segments=split_screen_segments
        )
        
        self.logger.info(f"并行自适应处理完成: {video_info.file_name}")
        self.logger.info(f"处理统计: 总片段={result.total_segments}, 分屏片段={result.split_screen_segments}, 处理时间={total_processing_time:.2f}s")
        
        return result
    
    def _process_segment_parallel(self, video_path: str, start_time: float, end_time: float) -> Optional[DetectionResult]:
        """并行处理单个片段"""
        try:
            # 使用自适应间隔检测
            for interval in self.frame_intervals:
                is_split_screen, confidence, detection_frames = self.adaptive_detect_segment(
                    video_path, start_time, end_time, interval
                )
                
                # 如果检测到分屏，使用更精确的间隔
                if is_split_screen and interval > self.adaptive_config.precision_mode_interval:
                    continue
                
                # 记录检测结果
                if is_split_screen or confidence > 0.3:
                    return DetectionResult(
                        segment_start=start_time,
                        segment_end=end_time,
                        is_split_screen=is_split_screen,
                        confidence=confidence,
                        detection_frames=detection_frames,
                        processing_time=0.0  # 并行处理中不计算单个片段时间
                    )
                
                break
            
            return None
            
        except Exception as e:
            self.logger.error(f"并行处理片段失败: {e}")
            return None
    
    def adaptive_detect_segment(self, video_path: str, start_time: float, 
                              end_time: float, current_interval: float) -> Tuple[bool, float, int]:
        """
        自适应检测片段
        
        Returns:
            (is_split_screen, confidence, detection_frames)
        """
        detection_frames = 0
        split_screen_count = 0
        total_confidence = 0.0
        
        current_time = start_time
        while current_time < end_time:
            frame = self.extract_frame_at_time(video_path, current_time)
            if frame is None:
                break
            
            # 检测当前帧
            result = self.detector.detect_frame(frame)
            detection_frames += 1
            
            if result and result.get('is_triple_screen', False):
                split_screen_count += 1
                total_confidence += result.get('confidence', 0.0)
            
            current_time += current_interval
        
        if detection_frames == 0:
            return False, 0.0, 0
        
        avg_confidence = total_confidence / detection_frames
        is_split_screen = (split_screen_count / detection_frames) >= 0.5  # 50%以上帧检测为分屏
        
        return is_split_screen, avg_confidence, detection_frames
    
    def process_video(self, video_path: str) -> ProcessingResult:
        """处理视频（根据配置选择单线程或并行模式）"""
        if self.config.enable_parallel:
            return self.process_video_parallel(video_path)
        else:
            return self.process_video_adaptive(video_path)
