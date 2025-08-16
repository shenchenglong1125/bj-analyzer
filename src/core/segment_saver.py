"""
视频切片保存器
"""
import os
import subprocess
import time
from typing import List, Dict, Any
from pathlib import Path

from ..models.data_models import DetectionResult, VideoInfo
from ..utils.logger import get_logger
from ..utils.file_utils import FileUtils

logger = get_logger(__name__)


class SegmentSaver:
    """视频切片保存器"""
    
    def __init__(self, output_dir: str = "output"):
        self.output_dir = output_dir
        self.logger = get_logger(__name__)
        
        # 确保输出目录存在
        FileUtils.ensure_directory(output_dir)
    
    def save_split_screen_segments(self, video_path: str, video_info: VideoInfo, 
                                  segments: List[DetectionResult], 
                                  min_duration: float = 1.0) -> Dict[str, Any]:
        """
        保存分屏片段（重构版本 - 只负责保存）
        
        Args:
            video_path: 原始视频路径
            video_info: 视频信息
            segments: 检测结果片段列表（已经过检测器处理）
            min_duration: 最小保存时长（秒）
            
        Returns:
            保存结果统计
        """
        start_time = time.time()
        
        # 过滤出分屏片段
        split_screen_segments = [s for s in segments if s.is_split_screen]
        
        if not split_screen_segments:
            self.logger.info("没有找到分屏片段")
            return {
                'total_segments': 0,
                'saved_segments': 0,
                'failed_segments': 0,
                'total_duration': 0.0,
                'processing_time': time.time() - start_time
            }
        
        # 过滤掉太短的片段
        valid_segments = [s for s in split_screen_segments 
                         if (s.segment_end - s.segment_start) >= min_duration]
        
        if not valid_segments:
            self.logger.info("没有找到符合条件的分屏片段")
            return {
                'total_segments': 0,
                'saved_segments': 0,
                'failed_segments': 0,
                'total_duration': 0.0,
                'processing_time': time.time() - start_time
            }
        
        # 创建视频专属输出目录
        video_name = Path(video_info.file_name).stem
        video_output_dir = os.path.join(self.output_dir, video_name)
        
        # 暂时关闭重复检测功能
        # 检查是否已经处理过（输出目录存在且包含片段文件）
        # if os.path.exists(video_output_dir):
        #     existing_files = [f for f in os.listdir(video_output_dir) if f.endswith('.mp4') and f.startswith('ss_')]
        #     if existing_files:
        #         self.logger.info(f"检测到已处理的文件: {video_info.file_name}，跳过处理")
        #         self.logger.info(f"已存在的片段: {len(existing_files)} 个")
        #         return {
        #             'total_segments': len(valid_segments),
        #             'saved_segments': len(existing_files),
        #             'failed_segments': 0,
        #             'total_duration': sum(s.segment_end - s.segment_start for s in valid_segments),
        #             'processing_time': time.time() - start_time,
        #             'output_directory': video_output_dir,
        #             'skipped': True
        #         }
        
        FileUtils.ensure_directory(video_output_dir)
        
        self.logger.info(f"开始保存 {len(valid_segments)} 个分屏片段")
        
        saved_count = 0
        failed_count = 0
        total_duration = 0.0
        
        for i, segment in enumerate(valid_segments):
            try:
                # 生成输出文件名
                output_filename = f"ss_{i+1:03d}_{segment.segment_start:.1f}s-{segment.segment_end:.1f}s.mp4"
                output_path = os.path.join(video_output_dir, output_filename)
                
                # 使用ffmpeg提取片段
                success = self._extract_segment_with_ffmpeg(
                    video_path, output_path, 
                    segment.segment_start, segment.segment_end
                )
                
                if success:
                    saved_count += 1
                    total_duration += (segment.segment_end - segment.segment_start)
                    self.logger.info(f"保存片段 {i+1}: {output_filename} (时长: {segment.segment_end - segment.segment_start:.1f}s)")
                else:
                    failed_count += 1
                    self.logger.error(f"保存片段 {i+1} 失败")
                    
            except Exception as e:
                failed_count += 1
                self.logger.error(f"保存片段 {i+1} 异常: {e}")
        
        processing_time = time.time() - start_time
        
        result = {
            'total_segments': len(valid_segments),
            'saved_segments': saved_count,
            'failed_segments': failed_count,
            'total_duration': total_duration,
            'processing_time': processing_time,
            'output_directory': video_output_dir
        }
        
        self.logger.info(f"切片保存完成: 成功 {saved_count}/{len(valid_segments)}, "
                        f"总时长 {total_duration:.2f}s, 处理时间 {processing_time:.2f}s")
        
        return result
    

    
    def _extract_segment_with_ffmpeg(self, input_path: str, output_path: str, 
                                   start_time: float, end_time: float) -> bool:
        """
        使用ffmpeg提取视频片段（优化复制模式）
        
        Args:
            input_path: 输入视频路径
            output_path: 输出视频路径
            start_time: 开始时间（秒）
            end_time: 结束时间（秒）
            
        Returns:
            是否成功
        """
        try:
            # 计算持续时间
            duration = end_time - start_time
            
            # 构建优化的ffmpeg命令（确保使用复制模式）
            cmd = [
                'ffmpeg',
                '-i', input_path,
                '-ss', str(start_time),
                '-t', str(duration),
                '-c:v', 'copy',  # 复制视频流
                '-c:a', 'copy',  # 复制音频流
                '-avoid_negative_ts', 'make_zero',
                '-fflags', '+genpts',  # 生成新的时间戳
                '-y',  # 覆盖输出文件
                output_path
            ]
            
            self.logger.debug(f"执行FFmpeg命令: {' '.join(cmd)}")
            
            # 执行ffmpeg命令
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300  # 5分钟超时
            )
            
            if result.returncode == 0:
                # 检查输出文件是否存在且大小合理
                if os.path.exists(output_path) and os.path.getsize(output_path) > 1024:
                    file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
                    self.logger.debug(f"成功保存片段: {output_path} (大小: {file_size_mb:.2f}MB)")
                    return True
                else:
                    self.logger.warning(f"输出文件异常: {output_path}")
                    return False
            else:
                self.logger.error(f"ffmpeg执行失败: {result.stderr}")
                # 如果复制模式失败，尝试重新编码模式
                return self._extract_segment_with_reencode(input_path, output_path, start_time, end_time)
                
        except subprocess.TimeoutExpired:
            self.logger.error(f"ffmpeg执行超时: {start_time}s-{end_time}s")
            return False
        except Exception as e:
            self.logger.error(f"ffmpeg执行异常: {e}")
            return False
    
    def _extract_segment_with_reencode(self, input_path: str, output_path: str, 
                                     start_time: float, end_time: float) -> bool:
        """
        使用ffmpeg重新编码模式提取视频片段（备用方案）
        
        Args:
            input_path: 输入视频路径
            output_path: 输出视频路径
            start_time: 开始时间（秒）
            end_time: 结束时间（秒）
            
        Returns:
            是否成功
        """
        try:
            duration = end_time - start_time
            
            # 使用重新编码模式（较慢但兼容性更好）
            cmd = [
                'ffmpeg',
                '-i', input_path,
                '-ss', str(start_time),
                '-t', str(duration),
                '-c:v', 'libx264',  # 使用H.264编码
                '-c:a', 'aac',      # 使用AAC音频编码
                '-preset', 'fast',  # 快速编码预设
                '-crf', '23',       # 质量设置
                '-avoid_negative_ts', 'make_zero',
                '-y',
                output_path
            ]
            
            self.logger.debug(f"使用重新编码模式: {' '.join(cmd)}")
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=600  # 10分钟超时（重新编码需要更多时间）
            )
            
            if result.returncode == 0:
                if os.path.exists(output_path) and os.path.getsize(output_path) > 1024:
                    file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
                    self.logger.debug(f"重新编码成功: {output_path} (大小: {file_size_mb:.2f}MB)")
                    return True
                else:
                    self.logger.warning(f"重新编码输出文件异常: {output_path}")
                    return False
            else:
                self.logger.error(f"重新编码失败: {result.stderr}")
                return False
                
        except Exception as e:
            self.logger.error(f"重新编码异常: {e}")
            return False
    
    def create_summary_report(self, video_info: VideoInfo, segments: List[DetectionResult], 
                            save_result: Dict[str, Any]) -> str:
        """
        创建切片保存摘要报告
        
        Args:
            video_info: 视频信息
            segments: 检测结果片段列表
            save_result: 保存结果
            
        Returns:
            报告内容
        """
        report_lines = [
            "=" * 60,
            "视频分屏片段切片报告",
            "=" * 60,
            f"视频文件: {video_info.file_name}",
            f"视频时长: {video_info.duration:.2f}秒",
            f"视频分辨率: {video_info.width}x{video_info.height}",
            f"视频帧率: {video_info.fps:.2f}fps",
            "",
            "检测结果:",
            f"  总片段数: {len(segments)}",
            f"  分屏片段数: {len([s for s in segments if s.is_split_screen])}",
            f"  分屏时长: {sum(s.segment_end - s.segment_start for s in segments if s.is_split_screen):.2f}秒",
            f"  分屏比例: {sum(s.segment_end - s.segment_start for s in segments if s.is_split_screen) / video_info.duration * 100:.2f}%",
            "",
            "切片保存结果:",
            f"  有效片段数: {save_result['total_segments']}",
            f"  成功保存: {save_result['saved_segments']}",
            f"  保存失败: {save_result['failed_segments']}",
            f"  保存总时长: {save_result['total_duration']:.2f}秒",
            f"  处理时间: {save_result['processing_time']:.2f}秒",
            f"  输出目录: {save_result.get('output_directory', 'N/A')}",
            "",
            "分屏片段详情:"
        ]
        
        # 添加分屏片段详情
        split_segments = [s for s in segments if s.is_split_screen]
        for i, segment in enumerate(split_segments):
            duration = segment.segment_end - segment.segment_start
            report_lines.append(
                f"  片段{i+1}: {segment.segment_start:.2f}s - {segment.segment_end:.2f}s "
                f"(时长: {duration:.2f}s)"
            )
        
        report_lines.extend([
            "",
            "=" * 60
        ])
        
        return "\n".join(report_lines)
