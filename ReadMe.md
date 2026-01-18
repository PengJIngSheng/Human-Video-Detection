# Human Detection & Tracking System (人物检测与追踪系统)

这是一个基于深度学习的计算机视觉项目，旨在自动识别并追踪视频监控中佩戴名牌的员工。该系统结合了 **YOLO** (用于目标检测) 和 **Kalman Filter / SORT** (用于目标追踪) 技术，即使在检测暂时丢失的情况下也能通过预测算法保持对目标的持续跟踪。

## 🚀 主要功能

- **自定义目标检测**：针对“佩戴名牌的员工”进行微调训练的 YOLO 模型，能区分员工与普通行人。
- **强健的目标追踪**：集成 Kalman Filter (卡尔曼滤波) 算法，实现“死死咬住”目标的追踪效果。
  - 支持**预测跟踪**：当检测器在某些帧漏检时，利用运动惯性预测目标位置，防止轨迹中断。
  - 支持**遮挡处理**：通过延长的记忆时间 (`track_buffer`)，允许目标在被短暂遮挡后重新被识别时保持 ID 不变。
- **全流程工具链**：包含从视频抽帧、模型训练到推理检测的完整脚本。
- **详细的数据报告**：输出带有边界框的视频 (`.mp4`) 以及包含逐帧坐标数据的 JSON 报告 (`.json`)。

## 📂 项目结构

```
.
├── config.yaml              # 核心配置文件 (模型路径、阈值、追踪参数)
├── detect.py                # 推理主程序 (用于运行检测和追踪)
├── extract_frames.py        # 数据预处理工具 (从视频提取图片)
├── tain_staff.py            # 模型训练脚本 (微调Yolov13x)
├── tracker.py               # 追踪算法实现 (Kalman Filter + SORT)
├── configs/
    └── data.yaml            # 数据集根目以及
├── models/
    └── yolov13x.pt          # 预训练底模
├── src/
    └── yolov13x.pt  
    └── detector.py          # YOLO 检测器封装类
    └── utils.py             # 通用工具函数 (绘图、IO、日志)
└── data/
    ├── input/               # 输入视频 (如 sample.mp4)
    ├── output/              # 输出结果 (视频和JSON)
    ├── dataset/
        └── images/
        	└── train
        	└── val 
        └── labels/
        	└── train
        	└── val          # 训练数据集 (images/labels)
    └── staff_dataset.yaml   # YOLO 数据集配置
```

## 🛠️ 环境依赖

请确保安装以下 Python 库：

Bash

```
pip install ultralytics opencv-python numpy tqdm pyyaml filterpy scipy
```

- **Ultralytics**: 用于 YOLO 模型训练和推理。
- **Filterpy**: 用于实现卡尔曼滤波追踪。
- **OpenCV**: 用于图像处理和视频读写。

## ⚡ 使用指南

### 第一步：准备数据 (Data Preparation)

如果需要训练自己的模型，首先需要从视频中提取图片进行标注。

1. 把视频放入 `data/input/sample.mp4`。

2. 运行提取脚本：

   ```
python extract_frames.py
   ```
   
   *这会在 `data/dataset/images/` 下生成训练集和验证集图片。*

3. 使用 **LabelImg** 工具对图片进行标注：

   - **标注对象**：佩戴名牌的员工（建议框选上半身或全身）。
   - **标签名称**：`Staff`
   - **格式**：选择 **YOLO** 格式。

### 第二步：模型训练 (Training)

使用标注好的数据微调 YOLO 模型。

1. 确认 `data/staff_dataset.yaml` 配置正确（路径需为绝对路径）。

2. 运行训练脚本：

   Bash

   ```
   python tain_staff.py
   ```

3. 训练完成后，最佳模型将保存在 `staff_project/yolov13_finetune/weights/best.pt`。

### 第三步：运行检测与追踪 (Detection & Tracking)

使用训练好的模型对视频进行推理。

1. 在 `config.yaml` 中确认模型路径指向你的 `best.pt`。

2. 运行检测命令：

   Bash

   ```
   python detect.py --video data/input/sample.mp4
   ```

**核心参数调整 (`config.yaml`)**: 如果你发现追踪效果不理想，可以调整以下参数：

- `conf_threshold`: **0.05 - 0.1** (建议设低，提高召回率，让检测器更敏感)
- `track_buffer`: **60 - 100** (增大此值可以让模型在目标消失后“记忆”更久)
- `min_hits`: **1** (代码已硬编码为1，确保只要检测到一次就开始追踪)

## 📊 输出结果

运行完成后，请查看 `data/output/` 目录：

1. **`result_video.mp4`**:
   - 绿色框：表示检测到的员工。
   - ID 标签：表示每个员工的唯一身份编号。
   - *注：即使某些帧检测器漏检，绿框也会根据预测算法平滑补齐。*

![](result1.bmp)

![](result2.bmp)

1. **`detection_results.json`**:
   - 包含每一帧的详细数据：是否有人、员工数量、坐标位置 (BBox center)、置信度等。
