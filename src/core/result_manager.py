"""
结果管理器
"""
import json
import time
from datetime import datetime
from typing import Dict, Any, List, Optional
from pathlib import Path

from ..models.data_models import ProcessingResult, BatchResult, DetectionResult
from ..utils.file_utils import FileUtils
from ..utils.logger import get_logger

logger = get_logger(__name__)


class ResultManager:
    """结果管理器"""
    
    def __init__(self, output_dir: str = "output"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.logger = get_logger(__name__)
    
    def save_processing_result(self, result: ProcessingResult, output_file: str = "") -> bool:
        """
        保存单个处理结果
        
        Args:
            result: 处理结果
            output_file: 输出文件路径
            
        Returns:
            是否保存成功
        """
        try:
            if not output_file:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"{Path(result.video_info.file_name).stem}_{timestamp}.json"
                output_file = str(self.output_dir / filename)
            
            # 转换为可序列化的字典
            result_dict = self._result_to_dict(result)
            
            # 保存到文件
            success = FileUtils.save_json(result_dict, output_file)
            
            if success:
                self.logger.info(f"处理结果保存成功: {output_file}")
            
            return success
        
        except Exception as e:
            self.logger.error(f"保存处理结果失败: {e}")
            return False
    
    def save_batch_result(self, batch_result: BatchResult, output_file: str = "") -> bool:
        """
        保存批量处理结果
        
        Args:
            batch_result: 批量处理结果
            output_file: 输出文件路径
            
        Returns:
            是否保存成功
        """
        try:
            if not output_file:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_file = str(self.output_dir / f"batch_result_{timestamp}.json")
            
            # 转换为可序列化的字典
            batch_dict = self._batch_result_to_dict(batch_result)
            
            # 保存到文件
            success = FileUtils.save_json(batch_dict, output_file)
            
            if success:
                self.logger.info(f"批量处理结果保存成功: {output_file}")
            
            return success
        
        except Exception as e:
            self.logger.error(f"保存批量处理结果失败: {e}")
            return False
    
    def _result_to_dict(self, result: ProcessingResult) -> Dict[str, Any]:
        """将处理结果转换为字典"""
        return {
            'video_info': {
                'file_path': result.video_info.file_path,
                'file_name': result.video_info.file_name,
                'duration': result.video_info.duration,
                'fps': result.video_info.fps,
                'width': result.video_info.width,
                'height': result.video_info.height,
                'total_frames': result.video_info.total_frames
            },
            'segments': [
                {
                    'segment_start': seg.segment_start,
                    'segment_end': seg.segment_end,
                    'is_split_screen': seg.is_split_screen,
                    'confidence': seg.confidence,
                    'detection_frames': seg.detection_frames,
                    'processing_time': seg.processing_time,
                    'similarity_scores': seg.similarity_scores
                }
                for seg in result.segments
            ],
            'total_processing_time': result.total_processing_time,
            'total_segments': result.total_segments,
            'split_screen_segments': result.split_screen_segments,
            'processing_status': result.processing_status,
            'error_message': result.error_message,
            'summary': self._generate_summary(result)
        }
    
    def _batch_result_to_dict(self, batch_result: BatchResult) -> Dict[str, Any]:
        """将批量处理结果转换为字典"""
        return {
            'processed_files': [
                self._result_to_dict(result) for result in batch_result.processed_files
            ],
            'total_files': batch_result.total_files,
            'successful_files': batch_result.successful_files,
            'failed_files': batch_result.failed_files,
            'start_time': batch_result.start_time.isoformat() if batch_result.start_time else None,
            'end_time': batch_result.end_time.isoformat() if batch_result.end_time else None,
            'total_processing_time': batch_result.total_processing_time,
            'summary': self._generate_batch_summary(batch_result)
        }
    
    def _generate_summary(self, result: ProcessingResult) -> Dict[str, Any]:
        """生成处理摘要"""
        if not result.segments:
            return {}
        
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
    
    def _generate_batch_summary(self, batch_result: BatchResult) -> Dict[str, Any]:
        """生成批量处理摘要"""
        if not batch_result.processed_files:
            return {}
        
        total_duration = sum(
            result.video_info.duration for result in batch_result.processed_files
        )
        
        total_split_screen_duration = sum(
            sum(seg.segment_end - seg.segment_start 
                for seg in result.segments if seg.is_split_screen)
            for result in batch_result.processed_files
        )
        
        total_split_screen_ratio = total_split_screen_duration / total_duration if total_duration > 0 else 0
        
        avg_processing_time = sum(
            result.total_processing_time for result in batch_result.processed_files
        ) / len(batch_result.processed_files)
        
        return {
            'total_files': batch_result.total_files,
            'successful_files': batch_result.successful_files,
            'failed_files': batch_result.failed_files,
            'total_duration': total_duration,
            'total_split_screen_duration': total_split_screen_duration,
            'total_split_screen_ratio': total_split_screen_ratio,
            'avg_processing_time': avg_processing_time,
            'total_processing_time': batch_result.total_processing_time
        }
    
    def load_result(self, file_path: str) -> Optional[ProcessingResult]:
        """
        加载处理结果
        
        Args:
            file_path: 结果文件路径
            
        Returns:
            处理结果
        """
        try:
            data = FileUtils.load_json(file_path)
            if not data:
                return None
            
            # 这里可以添加从字典恢复ProcessingResult的逻辑
            # 由于比较复杂，暂时返回None
            self.logger.warning("加载结果功能暂未实现")
            return None
        
        except Exception as e:
            self.logger.error(f"加载结果失败 {file_path}: {e}")
            return None
    
    def generate_report(self, batch_result: BatchResult, report_file: str = "") -> bool:
        """
        生成处理报告
        
        Args:
            batch_result: 批量处理结果
            report_file: 报告文件路径
            
        Returns:
            是否生成成功
        """
        try:
            if not report_file:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                report_file = str(self.output_dir / f"report_{timestamp}.txt")
            
            # 生成报告内容
            report_content = self._generate_report_content(batch_result)
            
            # 保存报告
            with open(report_file, 'w', encoding='utf-8') as f:
                f.write(report_content)
            
            self.logger.info(f"处理报告生成成功: {report_file}")
            return True
        
        except Exception as e:
            self.logger.error(f"生成报告失败: {e}")
            return False
    
    def _generate_report_content(self, batch_result: BatchResult) -> str:
        """生成报告内容"""
        lines = []
        lines.append("=" * 60)
        lines.append("视频分片检测处理报告")
        lines.append("=" * 60)
        lines.append(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append("")
        
        # 总体统计
        lines.append("总体统计:")
        lines.append(f"  总文件数: {batch_result.total_files}")
        lines.append(f"  成功处理: {batch_result.successful_files}")
        lines.append(f"  处理失败: {batch_result.failed_files}")
        lines.append(f"  总处理时间: {batch_result.total_processing_time:.2f}秒")
        lines.append("")
        
        # 详细结果
        lines.append("详细结果:")
        for i, result in enumerate(batch_result.processed_files, 1):
            lines.append(f"  {i}. {result.video_info.file_name}")
            lines.append(f"     状态: {result.processing_status}")
            lines.append(f"     时长: {result.video_info.duration:.2f}秒")
            lines.append(f"     分屏片段: {result.split_screen_segments}/{result.total_segments}")
            lines.append(f"     处理时间: {result.total_processing_time:.2f}秒")
            
            if result.error_message:
                lines.append(f"     错误: {result.error_message}")
            lines.append("")
        
        return "\n".join(lines)
