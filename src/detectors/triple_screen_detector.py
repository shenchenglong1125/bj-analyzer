"""
三联屏检测器
通过比较左中右三个区域的相似度来检测三联屏
"""
import cv2
import numpy as np
import time
from typing import Dict, Any, Optional
from .base_detector import DetectorBase
from ..models.data_models import DetectorConfig
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
        
        # 缓存上一次的检测结果
        self.last_result = None
        self.last_frame_hash = None
        
        logger.info(f"三联屏检测器初始化完成: 阈值={self.similarity_threshold}, 方法={self.method}")
    
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
            
            # 判断是否为三联屏
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
