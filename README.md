# 视频分屏检测工具 (BJ Analyzer)

## 项目概述
这是一个高性能的视频分屏检测工具，专门用于自动识别视频文件中的分屏场景（如三联屏、多画面等），并支持精确的片段提取和保存。

## 核心功能
- **自适应抽帧检测**：智能调整检测间隔，大幅提升处理速度（20-50倍加速）
- **精确片段提取**：基于状态转换的精确分屏片段边界识别
- **视频切片保存**：自动提取并保存分屏片段为独立视频文件
- **批量处理能力**：支持指定目录或单个文件处理
- **配置驱动**：通过YAML配置文件灵活设置参数
- **实时进度显示**：处理过程中显示实时进度条

## 安装依赖
```bash
pip install -r requirements.txt
```

## 使用方法

### 基本用法
```bash
# 处理单个视频文件
python -m src.main --input video.mp4 --config config/triple_screen.yaml --adaptive --save-segments

# 批量处理目录
python -m src.main --input /path/to/videos/ --config config/triple_screen.yaml --adaptive --save-segments
```

### 参数说明
- `--input, -i`: 输入视频文件或目录路径
- `--config, -c`: 配置文件路径
- `--adaptive`: 启用自适应抽帧处理（推荐）
- `--save-segments`: 保存检测到的分屏片段

## 项目结构
```
bj-analyzer/
├── src/
│   ├── core/              # 核心引擎
│   │   ├── config_manager.py      # 配置管理
│   │   ├── adaptive_processor.py  # 自适应处理器
│   │   ├── segment_saver.py       # 片段保存器
│   │   └── video_processor.py     # 视频处理器
│   ├── detectors/         # 检测器插件
│   │   └── triple_screen_detector.py  # 三联屏检测器
│   ├── utils/             # 工具类
│   ├── models/            # 数据模型
│   └── main.py            # 主程序入口
├── config/                # 配置文件
│   └── triple_screen.yaml # 三联屏检测配置
├── tests/                 # 测试代码
└── output/                # 结果输出（自动创建）
```

## 核心算法

### 自适应抽帧检测
- **状态跟踪**：跟踪当前画面状态（单屏/分屏）
- **智能跳帧**：状态不变时增加跳帧间隔（0.5s → 1.0s → 2.0s → 4.0s → 8.0s → 16.0s）
- **状态重置**：状态变化时重置跳帧间隔
- **性能提升**：相比逐帧检测提升20-50倍速度

### 精确片段提取
- **状态转换识别**：基于单屏↔分屏状态转换确定片段边界
- **精确边界**：单转分时开始片段，分转单时结束片段
- **智能合并**：自动合并间隔很小的相邻片段
- **无上下文扩展**：避免固定时间扩展导致的精度损失

### 三联屏检测
- **直方图分析**：基于颜色直方图相似度判断分屏
- **阈值控制**：可配置的相似度阈值（默认0.85）
- **多区域检测**：检测视频画面的多个区域

## 输出格式

### 文件命名规则
- **目录名**：原始视频文件名（不含扩展名）
- **文件名**：`ss_序号_开始时间s-结束时间s.mp4`

### 示例输出
```
output/
├── [2025-06-26 22-01-08][Minana呀][宝宝 晚上见～～～]/
│   ├── ss_001_0.0s-799.0s.mp4
│   └── ss_002_813.0s-2825.5s.mp4
```

## 配置说明

### 自适应配置
```yaml
adaptive:
  frame_intervals: [0.5, 1.0, 2.0, 4.0, 8.0, 16.0]  # 跳帧间隔序列
  max_interval: 16.0                                 # 最大跳帧间隔
```

### 检测器配置
```yaml
detector_config:
  detector_type: "triple_screen_detector"
  similarity_threshold: 0.85                        # 相似度阈值
  detection_method: "histogram"                     # 检测方法
```

### 保存配置
```yaml
save_config:
  min_save_duration: 10.0                           # 最小保存时长
  merge_threshold: 15.0                             # 合并间隔阈值
```

## 性能特点
- **处理速度**：34秒处理2825秒视频（约83倍实时速度）
- **检测精度**：基于状态转换的精确边界识别
- **内存效率**：流式处理，支持大文件
- **输出质量**：使用ffmpeg复制模式，保持原视频质量

## 系统要求
- Python 3.7+
- FFmpeg（用于视频切片）
- OpenCV
- NumPy
- PyYAML

## 许可证
MIT License
