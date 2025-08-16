"""
视频分片检测工具主程序
"""
import argparse
import sys
import time
from pathlib import Path
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

from .core.config_manager import ConfigManager
from .core.video_processor import VideoProcessor
from .core.adaptive_processor import AdaptiveProcessor
from .core.result_manager import ResultManager
from .core.segment_saver import SegmentSaver
from .detectors.triple_screen_detector import TripleScreenDetector
from .models.data_models import BatchResult
from .utils.file_utils import FileUtils
from .utils.logger import get_logger

logger = get_logger(__name__)


class VideoAnalyzer:
    """视频分析器主类"""
    
    def __init__(self, config_file: str = ""):
        self.config_manager = ConfigManager(config_file)
        self.config = self.config_manager.get_config()
        self.result_manager = ResultManager(self.config.output_path)
        
        # 初始化检测器
        self.detector = self._create_detector()
        self.video_processor = VideoProcessor(self.detector)
        
        # 初始化自适应处理器
        self.adaptive_processor = AdaptiveProcessor(self.config)
        
        # 初始化切片保存器
        self.segment_saver = SegmentSaver(self.config.output_path)
        
        logger.info("视频分析器初始化完成")
    
    def _create_detector(self):
        """创建检测器实例"""
        detector_type = self.config.detector_config.detector_type
        
        if detector_type == "triple_screen_detector":
            return TripleScreenDetector(self.config.detector_config)
        else:
            logger.error(f"不支持的检测器类型: {detector_type}")
            raise ValueError(f"不支持的检测器类型: {detector_type}")
    
    def process_single_file(self, video_path: str, output_file: str = "", 
                           use_adaptive: bool = False, save_segments: bool = False) -> bool:
        """
        处理单个视频文件
        
        Args:
            video_path: 视频文件路径
            output_file: 输出文件路径
            use_adaptive: 是否使用自适应处理
            save_segments: 是否保存分屏片段
            
        Returns:
            是否处理成功
        """
        try:
            logger.info(f"开始处理单个文件: {video_path}")
            
            # 处理视频
            if use_adaptive:
                # 使用自适应处理器
                result = self.adaptive_processor.process_video(video_path)
            else:
                result = self.video_processor.process_video(video_path)
            
            # 处理结果
            if result.processing_status == "completed":
                success = True
                if success:
                    logger.info(f"文件处理完成: {video_path}")
                    
                    # 保存分屏片段
                    if save_segments and result.segments:
                        logger.info("开始保存分屏片段...")
                        
                        # 获取保存配置
                        save_config = getattr(self.config, 'save_config', {})
                        min_save_duration = save_config.get('min_save_duration', 10.0)
                        
                        save_result = self.segment_saver.save_split_screen_segments(
                            video_path, result.video_info, result.segments,
                            min_duration=min_save_duration
                        )
                        
                        logger.info(f"切片保存完成，共保存 {save_result['saved_segments']} 个片段")
                    
                    return True
                else:
                    logger.error(f"保存结果失败: {video_path}")
                    return False
            else:
                logger.error(f"文件处理失败: {video_path}, 错误: {result.error_message}")
                return False
        
        except Exception as e:
            logger.error(f"处理文件异常: {video_path}, 错误: {e}")
            return False
    
    def process_directory(self, directory: str, output_file: str = "", use_adaptive: bool = False, save_segments: bool = False) -> bool:
        """
        批量处理目录中的视频文件
        
        Args:
            directory: 目录路径
            output_file: 输出文件路径
            use_adaptive: 是否使用自适应处理
            save_segments: 是否保存分屏片段
            
        Returns:
            是否处理成功
        """
        try:
            logger.info(f"开始批量处理目录: {directory}")
            
            # 获取视频文件列表
            video_files = FileUtils.get_video_files(directory)
            if not video_files:
                logger.warning(f"目录中没有找到视频文件: {directory}")
                return False
            
            logger.info(f"找到 {len(video_files)} 个视频文件")
            
            # 创建批量结果
            batch_result = BatchResult()
            batch_result.total_files = len(video_files)
            
            start_time = time.time()
            
            if self.config.enable_parallel and len(video_files) > 1:
                # 并行处理
                batch_result = self._process_parallel(video_files, use_adaptive, save_segments)
            else:
                # 串行处理
                batch_result = self._process_sequential(video_files, use_adaptive, save_segments)
            
            batch_result.total_processing_time = time.time() - start_time
            batch_result.end_time = datetime.now()
            
            # 批量处理完成，不保存JSON和报告文件
            success = True
            
            if success:
                logger.info(f"批量处理完成: 成功 {batch_result.successful_files}/{batch_result.total_files}")
                return True
            else:
                logger.error("保存批量结果失败")
                return False
        
        except Exception as e:
            logger.error(f"批量处理异常: {e}")
            return False
    
    def _process_sequential(self, video_files: list, use_adaptive: bool = False, save_segments: bool = False) -> BatchResult:
        """串行处理视频文件"""
        batch_result = BatchResult()
        batch_result.total_files = len(video_files)
        
        for video_file in video_files:
            try:
                if use_adaptive:
                    result = self.adaptive_processor.process_video(video_file)
                else:
                    result = self.video_processor.process_video(video_file)
                
                batch_result.processed_files.append(result)
                
                if result.processing_status == "completed":
                    batch_result.successful_files += 1
                    logger.info(f"处理成功: {result.video_info.file_name}")
                    
                    # 保存分屏片段
                    if save_segments and result.segments:
                        logger.info(f"开始保存 {result.video_info.file_name} 的分屏片段...")
                        
                        # 获取保存配置
                        save_config = getattr(self.config, 'save_config', {})
                        min_save_duration = save_config.get('min_save_duration', 10.0)
                        
                        save_result = self.segment_saver.save_split_screen_segments(
                            video_file, result.video_info, result.segments,
                            min_duration=min_save_duration
                        )
                        
                        logger.info(f"{result.video_info.file_name} 切片保存完成，共保存 {save_result['saved_segments']} 个片段")
                else:
                    batch_result.failed_files += 1
                    logger.error(f"处理失败: {result.video_info.file_name}")
            
            except Exception as e:
                batch_result.failed_files += 1
                logger.error(f"处理异常: {video_file}, 错误: {e}")
        
        return batch_result
    
    def _process_parallel(self, video_files: list, use_adaptive: bool = False, save_segments: bool = False) -> BatchResult:
        """并行处理视频文件"""
        batch_result = BatchResult()
        batch_result.total_files = len(video_files)
        
        with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
            # 提交任务
            future_to_file = {}
            for video_file in video_files:
                if use_adaptive:
                    future = executor.submit(self.adaptive_processor.process_video, video_file)
                else:
                    future = executor.submit(self.video_processor.process_video, video_file)
                future_to_file[future] = video_file
            
            # 收集结果
            for future in as_completed(future_to_file):
                video_file = future_to_file[future]
                try:
                    result = future.result()
                    batch_result.processed_files.append(result)
                    
                    if result.processing_status == "completed":
                        batch_result.successful_files += 1
                        logger.info(f"处理成功: {result.video_info.file_name}")
                        
                        # 注意：并行处理时不保存分屏片段，因为可能有线程安全问题
                        if save_segments:
                            logger.warning(f"并行处理模式下跳过 {result.video_info.file_name} 的分屏片段保存")
                    else:
                        batch_result.failed_files += 1
                        logger.error(f"处理失败: {result.video_info.file_name}")
                
                except Exception as e:
                    batch_result.failed_files += 1
                    logger.error(f"处理异常: {video_file}, 错误: {e}")
        
        return batch_result
    
    def validate_input(self, input_path: str) -> bool:
        """验证输入路径"""
        if not input_path:
            logger.error("输入路径不能为空")
            return False
        
        path = Path(input_path)
        if not path.exists():
            logger.error(f"输入路径不存在: {input_path}")
            return False
        
        if path.is_file():
            if not FileUtils.is_video_file(input_path):
                logger.error(f"不是支持的视频文件: {input_path}")
                return False
        elif path.is_dir():
            video_files = FileUtils.get_video_files(input_path)
            if not video_files:
                logger.error(f"目录中没有找到视频文件: {input_path}")
                return False
        else:
            logger.error(f"无效的输入路径: {input_path}")
            return False
        
        return True


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="视频分片检测工具")
    parser.add_argument("--input", "-i", required=True, help="输入视频文件或目录路径")
    parser.add_argument("--output", "-o", help="输出文件路径")
    parser.add_argument("--config", "-c", help="配置文件路径")
    parser.add_argument("--segment-duration", type=float, help="片段时长（秒）")
    parser.add_argument("--detection-interval", type=float, help="检测间隔（秒）")
    parser.add_argument("--face-threshold", type=int, help="人脸数量阈值")
    parser.add_argument("--confidence-threshold", type=float, help="置信度阈值")
    parser.add_argument("--enable-parallel", action="store_true", help="启用并行处理")
    parser.add_argument("--max-workers", type=int, help="最大工作线程数")
    parser.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR"], help="日志级别")
    parser.add_argument("--adaptive", action="store_true", help="使用自适应抽帧处理（推荐用于优化速度）")
    
    parser.add_argument("--save-segments", action="store_true", help="保存检测到的分屏片段到output目录")
    
    args = parser.parse_args()
    
    try:
        # 创建分析器
        analyzer = VideoAnalyzer(args.config)
        
        # 更新配置
        if args.segment_duration:
            analyzer.config.detector_config.segment_duration = args.segment_duration
        if args.detection_interval:
            analyzer.config.detector_config.detection_interval = args.detection_interval
        if args.face_threshold:
            analyzer.config.detector_config.face_threshold = args.face_threshold
        if args.confidence_threshold:
            analyzer.config.detector_config.confidence_threshold = args.confidence_threshold
        if args.enable_parallel:
            analyzer.config.enable_parallel = True
        if args.max_workers:
            analyzer.config.max_workers = args.max_workers
        if args.log_level:
            analyzer.config.log_level = args.log_level
        
        # 验证配置
        if not analyzer.config_manager.validate_config():
            logger.error("配置验证失败")
            sys.exit(1)
        
        # 验证输入
        if not analyzer.validate_input(args.input):
            logger.error("输入验证失败")
            sys.exit(1)
        
        # 处理视频
        input_path = Path(args.input)
        success = False
        
        if input_path.is_file():
            # 处理单个文件
            success = analyzer.process_single_file(args.input, args.output, args.adaptive, args.save_segments)
        else:
            # 处理目录
            success = analyzer.process_directory(args.input, args.output, args.adaptive, args.save_segments)
        
        if success:
            logger.info("处理完成")
            sys.exit(0)
        else:
            logger.error("处理失败")
            sys.exit(1)
    
    except KeyboardInterrupt:
        logger.info("用户中断处理")
        sys.exit(1)
    except Exception as e:
        logger.error(f"程序异常: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
