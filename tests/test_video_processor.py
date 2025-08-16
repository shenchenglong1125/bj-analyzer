"""
视频处理器测试
"""
import unittest
import numpy as np
import tempfile
import os
import sys
from pathlib import Path

# 添加项目根目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.core.video_processor import VideoProcessor
from src.detectors.face_detector import FaceDetector
from src.models.data_models import DetectorConfig, VideoInfo


class TestVideoProcessor(unittest.TestCase):
    """视频处理器测试类"""
    
    def setUp(self):
        """测试前准备"""
        self.config = DetectorConfig(
            detector_type="face_detector",
            segment_duration=15.0,
            detection_interval=1.0,
            face_threshold=2,
            confidence_threshold=0.5,
            max_faces=10,
            enable_early_stop=True
        )
        self.detector = FaceDetector(self.config)
        self.processor = VideoProcessor(self.detector)
    
    def test_processor_initialization(self):
        """测试处理器初始化"""
        self.assertIsNotNone(self.processor)
        self.assertIsNotNone(self.processor.detector)
    
    def test_get_processing_summary_empty_result(self):
        """测试空结果的处理摘要"""
        # 创建空的处理结果
        empty_result = type('obj', (object,), {
            'segments': [],
            'video_info': type('obj', (object,), {
                'file_name': 'test.mp4',
                'duration': 0
            })(),
            'total_processing_time': 0,
            'processing_status': 'completed'
        })()
        
        summary = self.processor.get_processing_summary(empty_result)
        self.assertEqual(summary, {})
    
    def test_get_processing_summary_with_data(self):
        """测试有数据的处理摘要"""
        # 创建模拟的处理结果
        from src.models.data_models import ProcessingResult, VideoInfo, DetectionResult
        
        video_info = VideoInfo(
            file_path="test.mp4",
            file_name="test.mp4",
            duration=30.0,
            fps=30.0,
            width=1920,
            height=1080,
            total_frames=900
        )
        
        segments = [
            DetectionResult(
                segment_start=0.0,
                segment_end=15.0,
                is_split_screen=True,
                confidence=0.8,
                face_count=3,
                detection_frames=5,
                processing_time=1.0
            ),
            DetectionResult(
                segment_start=15.0,
                segment_end=30.0,
                is_split_screen=False,
                confidence=0.6,
                face_count=1,
                detection_frames=5,
                processing_time=1.0
            )
        ]
        
        result = ProcessingResult(
            video_info=video_info,
            segments=segments,
            total_processing_time=2.0,
            total_segments=2,
            split_screen_segments=1,
            processing_status="completed"
        )
        
        summary = self.processor.get_processing_summary(result)
        
        self.assertIsInstance(summary, dict)
        self.assertIn('file_name', summary)
        self.assertIn('total_duration', summary)
        self.assertIn('split_screen_ratio', summary)
        self.assertEqual(summary['file_name'], 'test.mp4')
        self.assertEqual(summary['total_duration'], 30.0)
        self.assertEqual(summary['split_screen_ratio'], 0.5)  # 15秒/30秒 = 0.5
    
    def test_extract_frames_invalid_path(self):
        """测试提取无效路径的帧"""
        frames = self.processor.extract_frames("invalid_path.mp4", 0.0, 15.0)
        self.assertEqual(frames, [])
    
    def test_get_video_info_invalid_path(self):
        """测试获取无效路径的视频信息"""
        info = self.processor.get_video_info("invalid_path.mp4")
        self.assertIsNone(info)


if __name__ == '__main__':
    unittest.main()
