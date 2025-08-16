"""
文件工具模块
"""
import os
import json
from pathlib import Path
from typing import List, Dict, Any, Optional
import yaml
from .logger import get_logger

logger = get_logger(__name__)


class FileUtils:
    """文件工具类"""
    
    # 支持的视频格式
    SUPPORTED_VIDEO_FORMATS = {
        '.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', 
        '.webm', '.m4v', '.3gp', '.ts', '.mts', '.m2ts'
    }
    
    @staticmethod
    def is_video_file(file_path: str) -> bool:
        """判断是否为支持的视频文件"""
        # 首先检查扩展名
        if Path(file_path).suffix.lower() not in FileUtils.SUPPORTED_VIDEO_FORMATS:
            return False
        
        # 然后检查实际内容（使用OpenCV验证）
        try:
            import cv2
            cap = cv2.VideoCapture(file_path)
            if not cap.isOpened():
                return False
            
            # 检查是否有视频流
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            cap.release()
            
            # 如果宽度、高度、帧率都为0，或者总帧数为0，可能是纯音频文件
            if width <= 0 or height <= 0 or fps <= 0 or total_frames <= 0:
                logger.warning(f"文件可能是纯音频文件: {file_path}")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"检查视频文件失败 {file_path}: {e}")
            return False
    
    @staticmethod
    def get_video_files(directory: str) -> List[str]:
        """获取目录下所有视频文件"""
        video_files = []
        try:
            for file_path in Path(directory).rglob('*'):
                if file_path.is_file() and FileUtils.is_video_file(str(file_path)):
                    video_files.append(str(file_path))
        except Exception as e:
            logger.error(f"获取视频文件列表失败: {e}")
        
        return sorted(video_files)
    
    @staticmethod
    def ensure_directory(directory: str) -> bool:
        """确保目录存在，不存在则创建"""
        try:
            Path(directory).mkdir(parents=True, exist_ok=True)
            return True
        except Exception as e:
            logger.error(f"创建目录失败 {directory}: {e}")
            return False
    
    @staticmethod
    def save_json(data: Dict[str, Any], file_path: str) -> bool:
        """保存JSON文件"""
        try:
            # 确保输出目录存在
            output_dir = Path(file_path).parent
            FileUtils.ensure_directory(str(output_dir))
            
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            logger.info(f"JSON文件保存成功: {file_path}")
            return True
        except Exception as e:
            logger.error(f"保存JSON文件失败 {file_path}: {e}")
            return False
    
    @staticmethod
    def load_json(file_path: str) -> Optional[Dict[str, Any]]:
        """加载JSON文件"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"加载JSON文件失败 {file_path}: {e}")
            return None
    
    @staticmethod
    def save_yaml(data: Dict[str, Any], file_path: str) -> bool:
        """保存YAML文件"""
        try:
            # 确保输出目录存在
            output_dir = Path(file_path).parent
            FileUtils.ensure_directory(str(output_dir))
            
            with open(file_path, 'w', encoding='utf-8') as f:
                yaml.dump(data, f, default_flow_style=False, allow_unicode=True)
            logger.info(f"YAML文件保存成功: {file_path}")
            return True
        except Exception as e:
            logger.error(f"保存YAML文件失败 {file_path}: {e}")
            return False
    
    @staticmethod
    def load_yaml(file_path: str) -> Optional[Dict[str, Any]]:
        """加载YAML文件"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        except Exception as e:
            logger.error(f"加载YAML文件失败 {file_path}: {e}")
            return None
    
    @staticmethod
    def get_file_size(file_path: str) -> int:
        """获取文件大小（字节）"""
        try:
            return Path(file_path).stat().st_size
        except Exception as e:
            logger.error(f"获取文件大小失败 {file_path}: {e}")
            return 0
    
    @staticmethod
    def format_file_size(size_bytes: int) -> str:
        """格式化文件大小"""
        if size_bytes == 0:
            return "0B"
        
        size_names = ["B", "KB", "MB", "GB", "TB"]
        i = 0
        while size_bytes >= 1024 and i < len(size_names) - 1:
            size_bytes /= 1024.0
            i += 1
        
        return f"{size_bytes:.2f}{size_names[i]}"
    
    @staticmethod
    def get_file_info(file_path: str) -> Dict[str, Any]:
        """获取文件信息"""
        try:
            path = Path(file_path)
            stat = path.stat()
            return {
                'name': path.name,
                'path': str(path),
                'size': stat.st_size,
                'size_formatted': FileUtils.format_file_size(stat.st_size),
                'modified_time': stat.st_mtime,
                'is_video': FileUtils.is_video_file(file_path)
            }
        except Exception as e:
            logger.error(f"获取文件信息失败 {file_path}: {e}")
            return {}
