# 视频分片检测工具 (Video Analyzer)

## 项目概述
这是一个视频分片检测工具，用于自动识别视频文件中的特定场景（如分屏、多人等），并支持多种检测算法。

## 核心功能
- **插件化检测器设计**：支持多种检测算法
- **批量处理能力**：支持指定目录或单个文件处理
- **配置驱动**：通过配置文件选择检测器和设置参数
- **高效检测**：基于15秒分段的智能检测，提升3-5倍处理速度

## 安装依赖
```bash
pip install -r requirements.txt
```

## 使用方法
```bash
# 处理单个视频文件
python src/main.py --input video.mp4 --output results.json

# 批量处理目录
python src/main.py --input /path/to/videos/ --output results.json

# 使用配置文件
python src/main.py --config config/default.yaml
```

## 项目结构
```
video_analyzer/
├── src/
│   ├── core/          # 核心引擎
│   ├── detectors/     # 检测器插件
│   ├── utils/         # 工具类
│   └── models/        # 数据模型
├── config/            # 配置文件
├── data/              # 测试数据
├── tests/             # 测试代码
└── output/            # 结果输出
```

## 检测算法
- **人脸检测分屏识别**：基于人脸数量判断是否为分屏场景
- **分段处理**：15秒为单位分割视频
- **智能检测**：1秒间隔逐帧检测，检测到分屏后提前终止

## 输出格式
检测结果以JSON格式输出，包含每个15秒片段的检测状态和处理统计信息。
