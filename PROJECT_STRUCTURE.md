# 项目结构说明

## 目录结构

```
video_analyzer/
├── src/                          # 源代码目录
│   ├── __init__.py
│   ├── main.py                   # 主程序入口
│   ├── core/                     # 核心引擎
│   │   ├── __init__.py
│   │   ├── config_manager.py     # 配置管理器
│   │   ├── video_processor.py    # 视频处理器
│   │   └── result_manager.py     # 结果管理器
│   ├── detectors/                # 检测器插件
│   │   ├── __init__.py
│   │   ├── base_detector.py      # 检测器基类
│   │   └── face_detector.py      # 人脸检测器实现
│   ├── models/                   # 数据模型
│   │   ├── __init__.py
│   │   └── data_models.py        # 数据模型定义
│   └── utils/                    # 工具类
│       ├── __init__.py
│       ├── logger.py             # 日志工具
│       └── file_utils.py         # 文件工具
├── config/                       # 配置文件
│   ├── default.yaml              # 默认配置
│   ├── high_accuracy.yaml        # 高精度配置
│   └── fast_processing.yaml      # 快速处理配置
├── tests/                        # 测试代码
│   ├── __init__.py
│   ├── test_face_detector.py     # 人脸检测器测试
│   └── test_video_processor.py   # 视频处理器测试
├── examples/                     # 使用示例
│   └── basic_usage.py            # 基本使用示例
├── data/                         # 测试数据（用户创建）
├── output/                       # 输出结果（自动创建）
├── logs/                         # 日志文件（自动创建）
├── requirements.txt              # 项目依赖
├── README.md                     # 项目说明
├── PROJECT_STRUCTURE.md          # 项目结构说明
├── run.py                        # 启动脚本
└── test_installation.py          # 安装测试脚本
```

## 核心组件说明

### 1. 数据模型 (src/models/)
- **DetectorConfig**: 检测器配置
- **ProcessingConfig**: 处理配置
- **VideoInfo**: 视频信息
- **DetectionResult**: 检测结果
- **ProcessingResult**: 处理结果
- **BatchResult**: 批量处理结果

### 2. 检测器系统 (src/detectors/)
- **DetectorBase**: 检测器基类，定义接口
- **FaceDetector**: 人脸检测器实现
  - 支持MediaPipe和OpenCV两种检测方法
  - 自动降级机制
  - 可配置的检测参数

### 3. 核心引擎 (src/core/)
- **ConfigManager**: 配置管理
  - 支持YAML配置文件
  - 配置验证
  - 默认值处理
- **VideoProcessor**: 视频处理引擎
  - 视频信息提取
  - 帧提取和处理
  - 分段检测逻辑
- **ResultManager**: 结果管理
  - JSON结果保存
  - 报告生成
  - 批量结果处理

### 4. 工具类 (src/utils/)
- **Logger**: 日志管理
  - 控制台和文件输出
  - 可配置日志级别
- **FileUtils**: 文件操作
  - 视频文件识别
  - 批量文件处理
  - 文件格式转换

## 配置文件说明

### default.yaml (默认配置)
- 平衡性能和精度的配置
- 15秒片段，1秒检测间隔
- 2个人脸阈值，0.5置信度

### high_accuracy.yaml (高精度配置)
- 更短的片段和更频繁的检测
- 更高的置信度阈值
- 禁用提前终止以确保准确性

### fast_processing.yaml (快速处理配置)
- 更长的片段和更少的检测
- 更低的置信度阈值
- 启用并行处理

## 使用方法

### 1. 安装依赖
```bash
pip install -r requirements.txt
```

### 2. 测试安装
```bash
python test_installation.py
```

### 3. 基本使用
```bash
# 处理单个文件
python run.py --input video.mp4 --output result.json

# 批量处理目录
python run.py --input /path/to/videos/ --output batch_result.json

# 使用配置文件
python run.py --config config/high_accuracy.yaml --input video.mp4
```

### 4. 高级选项
```bash
# 自定义参数
python run.py --input video.mp4 \
    --segment-duration 10.0 \
    --detection-interval 0.5 \
    --face-threshold 3 \
    --confidence-threshold 0.7 \
    --enable-parallel \
    --max-workers 4
```

## 扩展开发

### 添加新的检测器
1. 继承 `DetectorBase` 类
2. 实现 `detect_frame()` 和 `is_split_screen()` 方法
3. 在 `VideoAnalyzer._create_detector()` 中添加新检测器类型

### 添加新的配置选项
1. 在 `data_models.py` 中添加新的配置字段
2. 在 `ConfigManager` 中添加配置解析逻辑
3. 更新配置文件模板

### 添加新的输出格式
1. 在 `ResultManager` 中添加新的输出方法
2. 实现相应的格式转换逻辑

## 性能优化

### 当前优化策略
- **分段处理**: 15秒片段减少内存占用
- **智能检测**: 1秒间隔减少计算量
- **提前终止**: 检测到分屏后跳过后续检测
- **并行处理**: 支持多文件并行处理

### 进一步优化建议
- **GPU加速**: 使用CUDA加速OpenCV操作
- **模型优化**: 使用更轻量级的人脸检测模型
- **缓存机制**: 缓存检测结果避免重复计算
- **流式处理**: 实现真正的流式视频处理

## 测试策略

### 单元测试
- 检测器功能测试
- 配置管理测试
- 文件操作测试

### 集成测试
- 完整处理流程测试
- 批量处理测试
- 错误处理测试

### 性能测试
- 处理速度测试
- 内存使用测试
- 准确性测试
