# 基于 YOLO 与 OpenCV 的工业表面缺陷检测系统

这是一个面向机器视觉岗位面试的端到端项目：使用 Python 训练 YOLO，使用
OpenCV 完成 ROI、图像增强、尺寸换算与结果可视化，并将模型导出为 ONNX，
最后由 C++/OpenCV DNN 完成部署推理。

项目默认使用东北大学 **NEU-DET** 钢材表面缺陷数据集，适合作为激光加工后
金属表面质量检测的公开数据基线。实际落地时，只需换成现场相机采集并重新标注
的数据即可。

## 已实现功能

- 官方 NEU-DET 数据自动下载、解压和完整性检查
- Pascal VOC XML 到 YOLO 标签转换
- 按类别分层划分 train/val/test，固定随机种子保证可复现
- YOLOv8n 训练、测试集评估、混淆矩阵和指标导出
- 图片、目录、视频、摄像头推理
- OpenCV ROI 裁剪、可选 CLAHE 光照增强和结果绘制
- 像素到毫米标定、缺陷宽高/面积计算、OK/NG 判定
- JSON/JSONL 质检报告
- ONNX 导出
- C++17 + OpenCV DNN 图片/视频/摄像头部署
- OpenCV 黑帽变换与轮廓检测传统基线，便于面试时解释方案对比
- 核心工具单元测试

## 数据集

NEU-DET 有 1,800 张 200×200 灰度图，共 6 类：

| ID | 英文类别 | 中文含义 |
|---:|---|---|
| 0 | crazing | 龟裂 |
| 1 | inclusion | 夹杂 |
| 2 | patches | 斑块 |
| 3 | pitted_surface | 麻点表面 |
| 4 | rolled-in_scale | 压入氧化皮 |
| 5 | scratches | 划痕 |

数据集来源：

- [东北大学 NEU 表面缺陷数据库官方页面](https://faculty.neu.edu.cn/songkc/en/zdylm/263270/list/index.htm)
- [官方 Google Drive：NEU-DET.zip](https://drive.google.com/file/d/1qrdZlaDi272eA79b0uCwwqPrm2Q_WI3k/view)

本机项目内已经下载并转换完成，当前划分为 1,260 张训练图、270 张验证图、
270 张测试图。转换时去除了原始 XML 中 3 个完全重复的框，最终共 4,186 个目标，
标签审计未发现越界框、空标签或图文缺失。数据集版权和引用要求归原作者所有，
不属于本项目 MIT 许可证。

如果以后需要重新生成：

```powershell
python scripts\download_neu_det.py --prepare
```

若 Google Drive 临时限制自动下载，可从上面的官方链接手动下载 ZIP，放到
`datasets\downloads\NEU-DET.zip`，再运行：

```powershell
python scripts\prepare_dataset.py --overwrite
```

## 环境准备

建议 Python 3.10～3.12。PowerShell 中执行：

```powershell
cd E:\desk\laser_defect_inspection
Set-ExecutionPolicy -Scope Process Bypass
.\scripts\setup.ps1
.\.venv\Scripts\Activate.ps1
```

如果已有环境，也可以直接：

```powershell
pip install -r requirements.txt
```

本机当前安装的是 CPU 版 PyTorch。如有 NVIDIA 显卡，请先根据
[PyTorch 官方安装页](https://pytorch.org/get-started/locally/)安装匹配 CUDA 的版本，
训练速度会明显提高。

## 训练与评估

正式训练：

```powershell
python scripts\train.py --epochs 100 --imgsz 640 --batch 16
```

训练前可先审计标签和类别分布：

```powershell
python scripts\audit_dataset.py
```

只验证流程是否能跑通：

```powershell
python scripts\train.py --quick --imgsz 160 --batch 1 --no-val
```

内存较小的 CPU 电脑可进一步用 `--batch 2 --fraction 0.1 --mosaic 0`；正式结果
仍应在完整训练集上训练。

训练产物位于 `outputs\train\neu_det_yolov8n`，最佳权重会自动复制到
`models\best.pt`。

项目内的 `models\smoke_test.pt` 只用 5% 数据训练 1 个 epoch，用于证明训练、
推理和导出链路能运行，**不能当成正式模型或面试精度结果**。正式训练完成后才会
生成 `models\best.pt`。

如需立刻验证现成烟雾模型：

```powershell
python scripts\infer.py `
  --weights models\smoke_test.pt `
  --source datasets\neu_det\images\test\crazing_101.jpg `
  --imgsz 160 `
  --output outputs\smoke_demo
```

在独立测试集上评估：

```powershell
python scripts\evaluate.py --weights models\best.pt --device cpu
```

指标保存在 `outputs\evaluation\metrics.json`，训练/评估过程还会生成
PR 曲线、混淆矩阵和样例预测图。

## Python 推理

单张图片：

```powershell
python scripts\infer.py `
  --weights models\best.pt `
  --source datasets\neu_det\images\test\crazing_101.jpg `
  --output outputs\demo
```

处理整个目录：

```powershell
python scripts\infer.py --source datasets\neu_det\images\test --device cpu
```

摄像头实时检测：

```powershell
python scripts\infer.py --source 0 --show
```

只检测工件区域，并加入物理尺寸：

```powershell
python scripts\infer.py `
  --source sample.jpg `
  --roi 100,80,800,600 `
  --mm-per-pixel 0.025 `
  --min-area-mm2 0.10 `
  --max-defects 0
```

开启 CLAHE 光照增强需加 `--clahe`。注意：若训练图片没有做相同增强，CLAHE
不一定提高准确率，应通过测试集或现场验证后再启用。

输出包括标注图片和 JSON 报告。报告包含类别、置信度、像素坐标、中心点、宽高、
面积、推理耗时和 OK/NG 结论。

## 像素尺寸标定

准备一张包含已知长度标尺的图，点击该长度的两个端点：

```powershell
python scripts\calibrate.py --image calibration.jpg --known-distance-mm 10
```

也可以直接指定端点：

```powershell
python scripts\calibrate.py `
  --image calibration.jpg `
  --known-distance-mm 10 `
  --points 120,200,520,200
```

脚本输出 `mm_per_pixel`，把它传给推理命令。这个方法适合相机、镜头、工作距离
固定且工件近似在同一平面的场景；更严格的测量应使用棋盘格完成畸变校正与平面标定。

## OpenCV 传统算法基线

```powershell
python scripts\classical_baseline.py `
  --source datasets\neu_det\images\test\crazing_101.jpg `
  --kernel 15 `
  --min-area 20
```

该流程使用灰度化、CLAHE、高斯滤波、黑帽变换、Otsu 二值化、形态学开运算和
轮廓筛选。它速度快、容易解释，但对材质纹理、光照和不同缺陷外观的泛化通常弱于
经过良好训练的检测模型。

## ONNX 与 C++ 部署

先导出 ONNX：

```powershell
python scripts\export_onnx.py --weights models\best.pt --output models\neu_det.onnx
```

C++ 工程要求 CMake 3.20+、支持 DNN 模块的 OpenCV 4.8+ 和 C++17 编译器。
安装好 OpenCV/CMake 后：

```powershell
$env:OpenCV_DIR = 'D:\opencv\build'
.\scripts\build_cpp.ps1 -Configuration Release
```

图片推理：

```powershell
.\cpp\build\Release\laser_inspection.exe `
  --model models\neu_det.onnx `
  --classes cpp\config\classes.txt `
  --source datasets\neu_det\images\test\crazing_101.jpg `
  --output outputs\cpp_result.jpg
```

摄像头推理把 `--source` 改为 `0`，视频推理传入视频路径。C++ 端自行完成
letterbox、blob 构建、YOLOv8 输出解析、逐类别 NMS、坐标反变换、尺寸显示和
OK/NG 判定。

## 测试

```powershell
python -m unittest discover -s tests -v
python -m compileall src scripts tests
```

## 项目结构

```text
laser_defect_inspection/
├── configs/                  # 模型和数据配置
├── cpp/                      # C++17 + OpenCV DNN 部署
├── datasets/
│   ├── downloads/            # 官方 ZIP（已下载）
│   ├── raw/                  # 原始 VOC 数据（已解压）
│   └── neu_det/              # 转换后的 YOLO train/val/test（已生成）
├── models/                   # best.pt 与导出的 ONNX
├── outputs/                  # 训练、评估、推理输出
├── scripts/                  # 数据、训练、推理、标定脚本
├── src/laser_inspection/     # 可复用 Python 模块
└── tests/                    # 单元测试
```

## 面试讲解主线

1. 先做数据审查和分层划分，避免同类样本分布不均，并保留独立测试集。
2. 用 YOLO 负责复杂缺陷的定位和分类，用 OpenCV 负责 ROI、光照处理、几何计算、
   结果绘制和工业判定。
3. 对比传统视觉基线，说明固定阈值在光照和纹理变化下的局限。
4. Python 用于训练和快速实验，ONNX + C++ OpenCV DNN 用于工程部署。
5. 指标不能只讲 mAP，还要讲每类召回率、误检/漏检、推理延迟和现场数据漂移。
6. NEU-DET 只是公开基线；真正上线要重新采集目标材料、镜头、光源、曝光和节拍下
   的数据，并建立困难样本回流机制。

## 下一步可扩展项

- 相机畸变标定和单应性变换，实现更可靠的毫米测量
- 滑窗或切片推理，提升微小缺陷召回率
- TensorRT/OpenVINO 加速和 FP16/INT8 量化
- Qt 操作界面、相机 SDK、PLC/Modbus 通信
- 保存批次号、时间、结果图到 SQLite，形成质量追溯闭环
