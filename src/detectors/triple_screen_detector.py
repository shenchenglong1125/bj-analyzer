"""
三联屏检测器
通过比较左中右三个区域的相似度来检测三联屏
"""
import cv2
import numpy as np
import time
from typing import Dict, Any, Optional, List
from .base_detector import DetectorBase
from ..models.data_models import DetectorConfig, DetectionResult, VideoInfo
from ..utils.logger import get_logger

logger = get_logger(__name__)


class TripleScreenDetector(DetectorBase):
    """三联屏检测器"""
    
    def __init__(self, config: DetectorConfig):
        super().__init__(config)
        self.logger = get_logger(__name__)
        
        # 三联屏检测配置
        self.similarity_threshold = getattr(config, 'similarity_threshold', 0.85)
        self.method = getattr(config, 'method', 'histogram')  # 'histogram', 'orb', 'template'
        self.min_region_size = getattr(config, 'min_region_size', 100)  # 最小区域尺寸
        
        # 片段处理配置
        self.min_duration = getattr(config, 'min_duration', 5.0)  # 最小片段时长
        self.merge_threshold = getattr(config, 'merge_threshold', 0.0)  # 合并阈值
        
        # 缓存上一次的检测结果
        self.last_result = None
        self.last_frame_hash = None
        
        logger.info(f"三联屏检测器初始化完成: 阈值={self.similarity_threshold}, 方法={self.method}")
    
    def process_video_segments(self, video_path: str, video_info: VideoInfo) -> List[DetectionResult]:
        """
        处理整个视频，返回精确的片段列表
        
        Args:
            video_path: 视频文件路径
            video_info: 视频信息
            
        Returns:
            精确的检测结果片段列表
        """
        start_time = time.time()
        
        self.logger.info(f"开始处理视频片段: {video_info.file_name}")
        
        # 获取配置参数
        segment_duration = self.config.segment_duration
        detection_interval = self.config.detection_interval
        
        # 第一步：进行基础片段检测
        raw_segments = self._detect_raw_segments(video_path, video_info, segment_duration, detection_interval)
        
        # 第二步：合并相邻片段
        merged_segments = self._merge_adjacent_segments(raw_segments)
        
        # 第三步：过滤短片段
        final_segments = self._filter_short_segments(merged_segments)
        
        processing_time = time.time() - start_time
        
        self.logger.info(f"视频片段处理完成: {video_info.file_name}")
        self.logger.info(f"处理统计: 原始片段={len(raw_segments)}, 合并后={len(merged_segments)}, 最终={len(final_segments)}")
        self.logger.info(f"处理时间: {processing_time:.2f}s")
        
        return final_segments
    
    def _detect_raw_segments(self, video_path: str, video_info: VideoInfo, 
                           segment_duration: float, detection_interval: float) -> List[DetectionResult]:
        """
        检测原始片段
        
        Args:
            video_path: 视频文件路径
            video_info: 视频信息
            segment_duration: 片段时长
            detection_interval: 检测间隔
            
        Returns:
            原始检测结果片段列表
        """
        segments = []
        current_time = 0.0
        
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
                
                # 处理单个片段
                result = self._process_segment(cap, segment_start, segment_end, fps, detection_interval)
                segments.append(result)
                
                current_time = segment_end
            
            cap.release()
            
        except Exception as e:
            self.logger.error(f"检测原始片段失败: {e}")
            if 'cap' in locals():
                cap.release()
        
        return segments
    
    def _process_segment(self, cap, segment_start: float, segment_end: float, 
                        fps: float, detection_interval: float) -> DetectionResult:
        """
        处理单个片段
        
        Args:
            cap: 视频捕获对象
            segment_start: 片段开始时间
            segment_end: 片段结束时间
            fps: 帧率
            detection_interval: 检测间隔
            
        Returns:
            检测结果
        """
        start_time = time.time()
        
        detection_frames = 0
        total_confidence = 0.0
        is_split = False
        
        # 计算检测帧的间隔
        interval_frames = max(1, int(fps * detection_interval))
        
        # 设置起始帧
        start_frame = int(segment_start * fps)
        end_frame = int(segment_end * fps)
        
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        
        # 跳帧处理
        current_frame = start_frame
        while current_frame < end_frame:
            cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
            ret, frame = cap.read()
            if not ret:
                break
            
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
    
    def _merge_adjacent_segments(self, segments: List[DetectionResult]) -> List[DetectionResult]:
        """
        合并相邻的分屏片段
        
        Args:
            segments: 原始片段列表
            
        Returns:
            合并后的片段列表
        """
        if not segments:
            return []
        
        merged_segments = []
        current_segment = None
        
        for segment in segments:
            if not segment.is_split_screen:
                # 如果不是分屏片段，直接添加
                if current_segment:
                    merged_segments.append(current_segment)
                    current_segment = None
                continue
            
            if current_segment is None:
                # 开始新的分屏片段
                current_segment = DetectionResult(
                    segment_start=segment.segment_start,
                    segment_end=segment.segment_end,
                    is_split_screen=True,
                    confidence=segment.confidence,
                    detection_frames=segment.detection_frames,
                    processing_time=segment.processing_time
                )
            else:
                # 检查是否可以合并
                time_gap = segment.segment_start - current_segment.segment_end
                
                if time_gap <= self.merge_threshold:
                    # 合并片段
                    current_segment.segment_end = segment.segment_end
                    current_segment.confidence = max(current_segment.confidence, segment.confidence)
                    current_segment.detection_frames += segment.detection_frames
                    current_segment.processing_time += segment.processing_time
                else:
                    # 无法合并，保存当前片段并开始新片段
                    merged_segments.append(current_segment)
                    current_segment = DetectionResult(
                        segment_start=segment.segment_start,
                        segment_end=segment.segment_end,
                        is_split_screen=True,
                        confidence=segment.confidence,
                        detection_frames=segment.detection_frames,
                        processing_time=segment.processing_time
                    )
        
        # 添加最后一个片段
        if current_segment:
            merged_segments.append(current_segment)
        
        return merged_segments
    
    def _filter_short_segments(self, segments: List[DetectionResult]) -> List[DetectionResult]:
        """
        过滤掉太短的片段
        
        Args:
            segments: 片段列表
            
        Returns:
            过滤后的片段列表
        """
        filtered_segments = []
        
        for segment in segments:
            duration = segment.segment_end - segment.segment_start
            if duration >= self.min_duration:
                filtered_segments.append(segment)
            else:
                self.logger.debug(f"过滤掉短片段: {segment.segment_start:.2f}s - {segment.segment_end:.2f}s (时长: {duration:.2f}s)")
        
        return filtered_segments
    
    def detect_frame(self, frame: np.ndarray) -> Optional[Dict[str, Any]]:
        """
        检测单帧是否为三联屏
        
        Args:
            frame: 输入帧
            
        Returns:
            检测结果字典
        """
        start_time = time.time()
        
        try:
            height, width = frame.shape[:2]
            
            # 检查帧尺寸是否足够
            if width < self.min_region_size * 3:
                return {
                    'is_triple_screen': False,
                    'confidence': 0.0,
                    'similarity_scores': [0.0, 0.0],
                    'processing_time': time.time() - start_time
                }
            
            # 分割左中右三个区域
            region_width = width // 3
            left_region = frame[:, :region_width]
            center_region = frame[:, region_width:2*region_width]
            right_region = frame[:, 2*region_width:]
            
            # 计算区域相似度
            if self.method == 'histogram':
                similarity_left_center = self._compare_histogram(left_region, center_region)
                similarity_center_right = self._compare_histogram(center_region, right_region)
            elif self.method == 'orb':
                similarity_left_center = self._compare_orb(left_region, center_region)
                similarity_center_right = self._compare_orb(center_region, right_region)
            elif self.method == 'template':
                similarity_left_center = self._compare_template(left_region, center_region)
                similarity_center_right = self._compare_template(center_region, right_region)
            else:
                # 默认使用直方图方法
                similarity_left_center = self._compare_histogram(left_region, center_region)
                similarity_center_right = self._compare_histogram(center_region, right_region)
            
            # 判断是否为三联屏：左右两块都要和中间相似才是三联屏
            is_triple_screen = (similarity_left_center >= self.similarity_threshold and 
                              similarity_center_right >= self.similarity_threshold)
            
            # 计算置信度（取两个相似度的最小值）
            confidence = min(similarity_left_center, similarity_center_right)
            
            result = {
                'is_triple_screen': is_triple_screen,
                'confidence': confidence,
                'similarity_scores': [similarity_left_center, similarity_center_right],
                'processing_time': time.time() - start_time
            }
            
            return result
            
        except Exception as e:
            self.logger.error(f"三联屏检测失败: {e}")
            return {
                'is_triple_screen': False,
                'confidence': 0.0,
                'similarity_scores': [0.0, 0.0],
                'processing_time': time.time() - start_time
            }
    
    def is_split_screen(self, detection_result: Dict[str, Any]) -> bool:
        """
        判断是否为分屏
        
        Args:
            detection_result: 检测结果
            
        Returns:
            是否为分屏
        """
        return detection_result.get('is_triple_screen', False)
    
    def _compare_histogram(self, region1: np.ndarray, region2: np.ndarray) -> float:
        """
        使用直方图比较两个区域的相似度
        
        Args:
            region1: 区域1
            region2: 区域2
            
        Returns:
            相似度分数 (0-1)
        """
        try:
            # 转换为灰度图
            if len(region1.shape) == 3:
                gray1 = cv2.cvtColor(region1, cv2.COLOR_BGR2GRAY)
                gray2 = cv2.cvtColor(region2, cv2.COLOR_BGR2GRAY)
            else:
                gray1, gray2 = region1, region2
            
            # 计算直方图
            hist1 = cv2.calcHist([gray1], [0], None, [256], [0, 256])
            hist2 = cv2.calcHist([gray2], [0], None, [256], [0, 256])
            
            # 归一化直方图
            cv2.normalize(hist1, hist1, 0, 1, cv2.NORM_MINMAX)
            cv2.normalize(hist2, hist2, 0, 1, cv2.NORM_MINMAX)
            
            # 计算相似度（使用相关系数）
            similarity = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
            
            # 将相关系数转换为0-1范围
            return max(0, similarity)
            
        except Exception as e:
            self.logger.error(f"直方图比较失败: {e}")
            return 0.0
    
    def _compare_orb(self, region1: np.ndarray, region2: np.ndarray) -> float:
        """
        使用ORB特征点比较两个区域的相似度
        
        Args:
            region1: 区域1
            region2: 区域2
            
        Returns:
            相似度分数 (0-1)
        """
        try:
            # 转换为灰度图
            if len(region1.shape) == 3:
                gray1 = cv2.cvtColor(region1, cv2.COLOR_BGR2GRAY)
                gray2 = cv2.cvtColor(region2, cv2.COLOR_BGR2GRAY)
            else:
                gray1, gray2 = region1, region2
            
            # 创建ORB检测器
            orb = cv2.ORB_create()
            
            # 检测关键点和描述符
            kp1, des1 = orb.detectAndCompute(gray1, None)
            kp2, des2 = orb.detectAndCompute(gray2, None)
            
            if des1 is None or des2 is None:
                return 0.0
            
            # 创建BF匹配器
            bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
            
            # 匹配特征点
            matches = bf.match(des1, des2)
            
            # 按距离排序
            matches = sorted(matches, key=lambda x: x.distance)
            
            # 计算相似度（基于匹配点数量和距离）
            if len(matches) > 0:
                # 取前50%的匹配点计算平均距离
                good_matches = matches[:len(matches)//2]
                avg_distance = sum(m.distance for m in good_matches) / len(good_matches)
                
                # 将距离转换为相似度（距离越小，相似度越高）
                similarity = max(0, 1 - avg_distance / 100)
                return similarity
            else:
                return 0.0
                
        except Exception as e:
            self.logger.error(f"ORB比较失败: {e}")
            return 0.0
    
    def _compare_template(self, region1: np.ndarray, region2: np.ndarray) -> float:
        """
        使用模板匹配比较两个区域的相似度
        
        Args:
            region1: 区域1
            region2: 区域2
            
        Returns:
            相似度分数 (0-1)
        """
        try:
            # 转换为灰度图
            if len(region1.shape) == 3:
                gray1 = cv2.cvtColor(region1, cv2.COLOR_BGR2GRAY)
                gray2 = cv2.cvtColor(region2, cv2.COLOR_BGR2GRAY)
            else:
                gray1, gray2 = region1, region2
            
            # 使用模板匹配
            result = cv2.matchTemplate(gray1, gray2, cv2.TM_CCOEFF_NORMED)
            similarity = np.max(result)
            
            return max(0, similarity)
            
        except Exception as e:
            self.logger.error(f"模板匹配失败: {e}")
            return 0.0
    
    def draw_detection_result(self, frame: np.ndarray, detection_result: Dict[str, Any]) -> np.ndarray:
        """
        在帧上绘制检测结果
        
        Args:
            frame: 输入帧
            detection_result: 检测结果
            
        Returns:
            绘制了结果的帧
        """
        try:
            height, width = frame.shape[:2]
            region_width = width // 3
            
            # 绘制分割线
            cv2.line(frame, (region_width, 0), (region_width, height), (0, 255, 0), 2)
            cv2.line(frame, (2*region_width, 0), (2*region_width, height), (0, 255, 0), 2)
            
            # 添加标签
            cv2.putText(frame, 'Left', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, 'Center', (region_width + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, 'Right', (2*region_width + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # 显示检测结果
            is_triple = detection_result.get('is_triple_screen', False)
            confidence = detection_result.get('confidence', 0.0)
            scores = detection_result.get('similarity_scores', [0.0, 0.0])
            
            status = "Triple Screen" if is_triple else "Single Screen"
            color = (0, 255, 0) if is_triple else (0, 0, 255)
            
            cv2.putText(frame, f"{status} ({confidence:.2f})", (10, height - 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
            cv2.putText(frame, f"L-C: {scores[0]:.2f}, C-R: {scores[1]:.2f}", (10, height - 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            return frame
            
        except Exception as e:
            self.logger.error(f"绘制检测结果失败: {e}")
            return frame
