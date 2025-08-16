"""
视频处理器核心引擎
"""
import cv2
import numpy as np
import time
from typing import List, Dict, Any, Optional, Generator
from pathlib import Path
from tqdm import tqdm

from ..models.data_models import VideoInfo, ProcessingResult, DetectionResult
from ..detectors.base_detector import DetectorBase
from ..utils.logger import get_logger
from ..utils.file_utils import FileUtils

logger = get_logger(__name__)


class VideoProcessor:
    """视频处理器"""
    
    def __init__(self, detector: DetectorBase):
        self.detector = detector
        self.logger = get_logger(__name__)
    
    def get_video_info(self, video_path: str) -> Optional[VideoInfo]:
        """
        获取视频信息
        
        Args:
            video_path: 视频文件路径
            
        Returns:
            视频信息
        """
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                self.logger.error(f"无法打开视频文件: {video_path}")
                return None
            
            # 获取视频属性
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
            self.logger.error(f"获取视频信息失败 {video_path}: {e}")
            return None
    
    def extract_frames(self, video_path: str, start_time: float, end_time: float) -> List[np.ndarray]:
        """
        提取指定时间段的帧
        
        Args:
            video_path: 视频文件路径
            start_time: 开始时间（秒）
            end_time: 结束时间（秒）
            
        Returns:
            帧列表
        """
        frames = []
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                self.logger.error(f"无法打开视频文件: {video_path}")
                return frames
            
            fps = cap.get(cv2.CAP_PROP_FPS)
            if fps <= 0:
                self.logger.error(f"无效的帧率: {fps}")
                cap.release()
                return frames
            
            # 计算帧索引
            start_frame = int(start_time * fps)
            end_frame = int(end_time * fps)
            
            # 设置起始位置
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            
            frame_count = start_frame
            while frame_count < end_frame:
                ret, frame = cap.read()
                if not ret:
                    break
                
                frames.append(frame)
                frame_count += 1
            
            cap.release()
            
        except Exception as e:
            self.logger.error(f"提取帧失败: {e}")
        
        return frames
    
    def process_video(self, video_path: str, show_progress: bool = True) -> ProcessingResult:
        """
        处理单个视频文件
        
        Args:
            video_path: 视频文件路径
            show_progress: 是否显示进度条
            
        Returns:
            处理结果
        """
        start_time = time.time()
        
        # 获取视频信息
        video_info = self.get_video_info(video_path)
        if not video_info:
            return ProcessingResult(
                video_info=VideoInfo(file_path=video_path, file_name="", duration=0, fps=0, width=0, height=0, total_frames=0),
                processing_status="failed",
                error_message="无法获取视频信息"
            )
        
        self.logger.info(f"开始处理视频: {video_info.file_name}")
        self.logger.info(f"视频信息: 时长={video_info.duration:.2f}s, 帧率={video_info.fps:.2f}, 分辨率={video_info.width}x{video_info.height}")
        
        # 计算片段
        segment_duration = self.detector.config.segment_duration
        segments = []
        
        current_time = 0.0
        total_segments = 0
        split_screen_segments = 0
        
        # 创建进度条
        if show_progress:
            total_segments_estimate = int(video_info.duration / segment_duration) + 1
            pbar = tqdm(total=total_segments_estimate, desc=f"处理 {video_info.file_name}")
        
        try:
            while current_time < video_info.duration:
                segment_start = current_time
                segment_end = min(current_time + segment_duration, video_info.duration)
                
                # 提取片段帧
                frames = self.extract_frames(video_path, segment_start, segment_end)
                
                if frames:
                    # 处理片段
                    result = self.detector.process_segment(frames, segment_start, segment_end)
                    segments.append(result)
                    
                    if result.is_split_screen:
                        split_screen_segments += 1
                        self.logger.debug(f"检测到分屏片段: {segment_start:.2f}s - {segment_end:.2f}s")
                
                total_segments += 1
                current_time = segment_end
                
                if show_progress:
                    pbar.update(1)
            
            if show_progress:
                pbar.close()
            
            processing_time = time.time() - start_time
            
            self.logger.info(f"视频处理完成: {video_info.file_name}")
            self.logger.info(f"处理统计: 总片段={total_segments}, 分屏片段={split_screen_segments}, 处理时间={processing_time:.2f}s")
            
            return ProcessingResult(
                video_info=video_info,
                segments=segments,
                total_processing_time=processing_time,
                total_segments=total_segments,
                split_screen_segments=split_screen_segments,
                processing_status="completed"
            )
        
        except Exception as e:
            self.logger.error(f"处理视频失败 {video_path}: {e}")
            if show_progress:
                pbar.close()
            
            return ProcessingResult(
                video_info=video_info,
                segments=segments,
                total_processing_time=time.time() - start_time,
                total_segments=total_segments,
                split_screen_segments=split_screen_segments,
                processing_status="failed",
                error_message=str(e)
            )
    
    def process_video_with_preview(self, video_path: str, output_path: str = "", 
                                 preview_interval: float = 5.0) -> ProcessingResult:
        """
        处理视频并生成预览（带检测结果的可视化）
        
        Args:
            video_path: 视频文件路径
            output_path: 输出视频路径
            preview_interval: 预览间隔（秒）
            
        Returns:
            处理结果
        """
        # 先进行正常处理
        result = self.process_video(video_path, show_progress=False)
        
        if result.processing_status != "completed":
            return result
        
        # 生成预览视频
        if output_path:
            self._generate_preview_video(video_path, output_path, result.segments, preview_interval)
        
        return result
    
    def _generate_preview_video(self, video_path: str, output_path: str, 
                              segments: List[DetectionResult], preview_interval: float):
        """
        生成预览视频
        
        Args:
            video_path: 原始视频路径
            output_path: 输出视频路径
            segments: 检测结果片段
            preview_interval: 预览间隔
        """
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                self.logger.error(f"无法打开视频文件: {video_path}")
                return
            
            # 获取视频属性
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            # 创建视频写入器
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            
            frame_count = 0
            current_time = 0.0
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                current_time = frame_count / fps
                
                # 查找当前时间对应的检测结果
                current_segment = None
                for segment in segments:
                    if segment.segment_start <= current_time <= segment.segment_end:
                        current_segment = segment
                        break
                
                # 如果当前时间需要预览，则绘制检测结果
                if current_segment and (frame_count % int(fps * preview_interval) == 0):
                    # 提取当前帧进行检测
                    detection_result = self.detector.detect_frame(frame)
                    
                    # 绘制检测结果
                    if hasattr(self.detector, 'draw_detection_result'):
                        frame = self.detector.draw_detection_result(frame, detection_result)
                
                out.write(frame)
                frame_count += 1
            
            cap.release()
            out.release()
            
            self.logger.info(f"预览视频生成完成: {output_path}")
        
        except Exception as e:
            self.logger.error(f"生成预览视频失败: {e}")
    
    def get_processing_summary(self, result: ProcessingResult) -> Dict[str, Any]:
        """
        获取处理摘要
        
        Args:
            result: 处理结果
            
        Returns:
            处理摘要
        """
        if not result.segments:
            return {}
        
        # 计算统计信息
        total_duration = result.video_info.duration
        split_screen_duration = sum(
            seg.segment_end - seg.segment_start 
            for seg in result.segments if seg.is_split_screen
        )
        
        split_screen_ratio = split_screen_duration / total_duration if total_duration > 0 else 0
        
        avg_confidence = sum(seg.confidence for seg in result.segments) / len(result.segments)
        avg_processing_time = sum(seg.processing_time for seg in result.segments) / len(result.segments)
        
        return {
            'file_name': result.video_info.file_name,
            'total_duration': total_duration,
            'total_segments': result.total_segments,
            'split_screen_segments': result.split_screen_segments,
            'split_screen_duration': split_screen_duration,
            'split_screen_ratio': split_screen_ratio,
            'avg_confidence': avg_confidence,
            'avg_processing_time': avg_processing_time,
            'total_processing_time': result.total_processing_time,
            'processing_status': result.processing_status
        }

    def process_video_streaming(self, video_path: str, show_progress: bool = True) -> ProcessingResult:
        """
        流式处理视频文件（优化版本）
        
        Args:
            video_path: 视频文件路径
            show_progress: 是否显示进度条
            
        Returns:
            处理结果
        """
        start_time = time.time()
        
        # 获取视频信息
        video_info = self.get_video_info(video_path)
        if not video_info:
            return ProcessingResult(
                video_info=VideoInfo(file_path=video_path, file_name="", duration=0, fps=0, width=0, height=0, total_frames=0),
                processing_status="failed",
                error_message="无法获取视频信息"
            )
        
        # 计算片段
        segment_duration = self.detector.config.segment_duration
        detection_interval = self.detector.config.detection_interval
        
        self.logger.info(f"开始流式处理视频: {video_info.file_name}")
        self.logger.info(f"视频信息: 时长={video_info.duration:.2f}s, 帧率={video_info.fps:.2f}, 分辨率={video_info.width}x{video_info.height}")
        self.logger.info(f"配置信息: 片段时长={segment_duration}s, 检测间隔={detection_interval}s, 跳帧间隔={int(video_info.fps * detection_interval)}帧")
        segments = []
        
        current_time = 0.0
        total_segments = 0
        split_screen_segments = 0
        
        # 创建进度条
        if show_progress:
            total_segments_estimate = int(video_info.duration / segment_duration) + 1
            pbar = tqdm(total=total_segments_estimate, desc=f"流式处理 {video_info.file_name}")
        
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise Exception("无法打开视频文件")
            
            fps = cap.get(cv2.CAP_PROP_FPS)
            if fps <= 0:
                raise Exception("无效的帧率")
            
            while current_time < video_info.duration:
                segment_start = current_time
                segment_end = min(current_time + segment_duration, video_info.duration)
                
                # 流式处理片段
                result = self._process_segment_streaming(cap, segment_start, segment_end, fps, detection_interval)
                segments.append(result)
                
                if result.is_split_screen:
                    split_screen_segments += 1
                    self.logger.debug(f"检测到分屏片段: {segment_start:.2f}s - {segment_end:.2f}s")
                
                total_segments += 1
                current_time = segment_end
                
                if show_progress:
                    pbar.update(1)
            
            cap.release()
            
            if show_progress:
                pbar.close()
            
            processing_time = time.time() - start_time
            
            self.logger.info(f"流式处理完成: {video_info.file_name}")
            self.logger.info(f"处理统计: 总片段={total_segments}, 分屏片段={split_screen_segments}, 处理时间={processing_time:.2f}s")
            
            return ProcessingResult(
                video_info=video_info,
                segments=segments,
                total_processing_time=processing_time,
                total_segments=total_segments,
                split_screen_segments=split_screen_segments,
                processing_status="completed"
            )
        
        except Exception as e:
            self.logger.error(f"流式处理视频失败 {video_path}: {e}")
            if show_progress:
                pbar.close()
            if 'cap' in locals():
                cap.release()
            
            return ProcessingResult(
                video_info=video_info,
                segments=segments,
                total_processing_time=time.time() - start_time,
                total_segments=total_segments,
                split_screen_segments=split_screen_segments,
                processing_status="failed",
                error_message=str(e)
            )
    
    def _process_segment_streaming(self, cap, segment_start: float, segment_end: float, fps: float, detection_interval: float) -> DetectionResult:
        """
        流式处理单个片段
        
        Args:
            cap: 视频捕获对象
            segment_start: 片段开始时间
            segment_end: 片段结束时间
            fps: 帧率
            detection_interval: 检测间隔
            
        Returns:
            检测结果
        """
        import time
        start_time = time.time()
        
        detection_frames = 0
        total_confidence = 0.0
        is_split = False
        
        # 计算检测帧的间隔（真正的跳帧处理）
        interval_frames = max(1, int(fps * detection_interval))
        
        # 设置起始帧
        start_frame = int(segment_start * fps)
        end_frame = int(segment_end * fps)
        
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        
        # 真正的跳帧处理：只读取需要检测的帧
        current_frame = start_frame
        while current_frame < end_frame:
            # 设置到指定帧位置
            cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
            ret, frame = cap.read()
            if not ret:
                break
            
            try:
                result = self.detector.detect_frame(frame)
                detection_frames += 1
                
                if result:
                    confidence = result.get('confidence', 0.0)
                    total_confidence += confidence
                    
                    # 判断是否为分屏
                    if self.detector.is_split_screen(result):
                        is_split = True
                        
                        # 如果启用提前终止，检测到分屏后立即停止
                        if self.detector.config.enable_early_stop:
                            self.logger.debug(f"检测到分屏，提前终止片段处理")
                            break
            
            except Exception as e:
                self.logger.error(f"处理帧失败: {e}")
                continue
            
            # 跳到下一帧
            current_frame += interval_frames
        
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
