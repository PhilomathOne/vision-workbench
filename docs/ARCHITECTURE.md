# Vision Workbench — 视觉模型工程化平台架构设计

> **定位**：面向计算机视觉的端到端 MLOps 平台，覆盖从数据管理到边缘部署的完整模型生命周期。
> **设计哲学**：模块化、可复现、框架无关、边缘优先。

---

## 目录

1. [项目愿景与范围边界](#1-项目愿景与范围边界)
2. [架构设计原则](#2-架构设计原则)
3. [总体架构概览](#3-总体架构概览)
4. [目录结构与模块职责](#4-目录结构与模块职责)
5. [核心抽象层设计](#5-核心抽象层设计)
6. [数据管理层设计](#6-数据管理层设计)
7. [流水线阶段详细规格](#7-流水线阶段详细规格)
8. [配置体系设计](#8-配置体系设计)
9. [实验追踪与可复现性](#9-实验追踪与可复现性)
10. [模型存储与生命周期](#10-模型存储与生命周期)
11. [自定义架构与算法开发平台](#11-自定义架构与算法开发平台)
12. [模型注册中心](#12-模型注册中心)
13. [边缘部署体系](#13-边缘部署体系)
14. [CLI 与 API 设计](#14-cli-与-api-设计)
15. [可观测性体系](#15-可观测性体系)
16. [测试策略](#16-测试策略)
17. [安全与合规](#17-安全与合规)
18. [依赖管理策略](#18-依赖管理策略)
19. [实现路线图](#19-实现路线图)
20. [风险与缓解](#20-风险与缓解)

---

## 1. 项目愿景与范围边界

### 1.1 核心愿景

构建一个**生产级视觉 AI 工程化平台**，使得：
- 数据科学家可以**快速迭代实验**（数据→标注→训练→评估）
- ML 工程师可以**一键导出部署**（量化→剪枝→导出→边缘端上线）
- 整个流程**可复现、可追溯、可对比**

### 1.2 范围边界

| 维度 | 包含 | 不包含（但预留接口） |
|------|------|---------------------|
| 任务类型 | 分类、检测、分割、姿态、OCR、跟踪 | 视频理解、3D 视觉、生成模型 |
| 训练框架 | PyTorch/MMDet/Ultralytics/HuggingFace | TensorFlow/Keras（二期） |
| 边缘平台 | TensorRT/OpenVINO/TFLite/CoreML/ONNX Runtime | 自研推理引擎 |
| 数据管理 | 图像数据集版本化、标注管理 | 视频标注、点云标注 |
| 部署模式 | 本地单机、边缘设备推送 | Kubernetes 集群调度（预留） |

---

## 2. 架构设计原则

1. **契约优先 (Contract-First)** — 每个模块通过 Protocol/ABC 定义接口，模块间仅依赖接口
2. **配置驱动 (Config-Driven)** — 一切行为通过 YAML 配置声明，实现可复现实验
3. **不可变上下文 (Immutable Context)** — 阶段间数据传递使用不可变快照，保证追溯性
4. **优雅降级 (Graceful Degradation)** — 可选依赖缺失时给出明确指引，不阻塞核心功能
5. **可观测性内置 (Observability by Default)** — 每个阶段自动记录指标、日志和产物路径
6. **边缘优先 (Edge-First)** — 训练可发生在云端，但导出和部署原生面向边缘推理场景
7. **数据血统追踪 (Data Lineage)** — 每个模型/产物都携带完整的输入来源和处理历史

---

## 3. 总体架构概览

```
┌────────────────────────────────────────────────────────────────────┐
│                         CLI / Python API / Web UI                  │
│                     (Typer)    (Programmatic)   (Gradio)           │
├────────────────────────────────────────────────────────────────────┤
│                      Pipeline Orchestrator                         │
│  ┌─────────┐   ┌─────────┐   ┌─────────┐   ┌──────────┐          │
│  │ Scheduler│   │ Context │   │ Resumer │   │ Parallel │          │
│  │  (DAG)   │   │ Manager │   │  (容错)  │   │ Executor │          │
│  └─────────┘   └─────────┘   └─────────┘   └──────────┘          │
├────────────────────────────────────────────────────────────────────┤
│                         Pipeline Stages                            │
│  ┌──────┐  ┌────────┐  ┌───────┐  ┌────────┐  ┌──────────┐      │
│  │ Data │→ │Annotate│→ │ Train │→ │Validate│→ │ Evaluate │      │
│  └──────┘  └────────┘  └───────┘  └────────┘  └──────────┘      │
│                                                ↓                  │
│  ┌────────┐  ┌────────┐  ┌──────────┐  ┌────────┐               │
│  │ Deploy │← │ Export │← │ Optimize │  │  Report │               │
│  └────────┘  └────────┘  └──────────┘  └────────┘               │
├────────────────────────────────────────────────────────────────────┤
│                      Shared Infrastructure                         │
│  ┌──────────┐  ┌───────────┐  ┌────────────┐  ┌──────────────┐  │
│  │  Model   │  │  Dataset  │  │ Experiment  │  │  Metric /    │  │
│  │ Registry │  │  Catalog  │  │  Tracker    │  │  Artifact    │  │
│  │          │  │  (DVC)    │  │  (MLflow)   │  │  Store       │  │
│  └──────────┘  └───────────┘  └────────────┘  └──────────────┘  │
├────────────────────────────────────────────────────────────────────┤
│                      Core Primitives                               │
│  ┌──────────┐  ┌───────────┐  ┌──────────┐  ┌──────────────┐    │
│  │  Types   │  │  Registry │  │  Config  │  │  Exceptions  │    │
│  │ (Pydantic)│  │ (Plugin)  │  │ (YAML)   │  │  Hierarchy   │    │
│  └──────────┘  └───────────┘  └──────────┘  └──────────────┘    │
└────────────────────────────────────────────────────────────────────┘
```

---

## 4. 目录结构与模块职责

```
vision-workbench/
├── pyproject.toml
├── README.md
├── Makefile                        # 常用开发命令快捷方式
├── .pre-commit-config.yaml
├── .markdownlint.yaml
│
├── src/vision_workbench/
│   ├── __init__.py                 # from vision_workbench import __version__
│   │
│   ├── core/                       # ===== 核心基础层 =====
│   │   ├── __init__.py
│   │   ├── types.py                # 领域原语：ImageTensor, BBox, Keypoint, Mask, ClassLabel, DatasetSplit
│   │   ├── base.py                 # BaseStage ABC, BaseDetector ABC, BaseExporter ABC
│   │   ├── result.py               # StageResult 代数类型（Union[DataResult, TrainResult, ...]）
│   │   ├── registry.py             # 泛型注册表：DetectorRegistry, FormatRegistry, FrameworkRegistry, ExporterRegistry
│   │   ├── config.py               # PipelineConfig 根模型 + 各阶段配置子模型
│   │   ├── context.py              # PipelineContext（不可变快照 + 版本向量）
│   │   ├── orchestrator.py         # PipelineOrchestrator（DAG 调度 + 并行执行 + 断点恢复）
│   │   ├── artifacts.py            # ArtifactStore — 统一产物存储与索引
│   │   └── exceptions.py           # 分层异常体系
│   │
│   ├── data/                       # ===== 数据管理 =====
│   │   ├── __init__.py
│   │   ├── catalog.py              # DatasetCatalog — 数据集注册、版本控制、搜索
│   │   ├── schema.py               # DatasetSchema — 约定目录结构
│   │   ├── versioning.py           # DVC 集成 — 数据版本追踪
│   │   ├── downloader.py           # 公开数据集下载（COCO, VOC, ImageNet 子集等）
│   │   └── provenance.py           # 数据血统追踪
│   │
│   ├── pipeline/                   # ===== 流水线阶段 =====
│   │   ├── __init__.py
│   │   │
│   │   ├── data/                   # ① 数据清洗
│   │   │   ├── __init__.py
│   │   │   ├── stage.py            # DataStage
│   │   │   ├── config.py           # DataStageConfig
│   │   │   ├── validator.py        # 格式校验引擎
│   │   │   ├── cleaner.py          # 去重（pHash/dHash/SSIM）+ 质量过滤
│   │   │   ├── augmentor.py        # Albumentations 增强管道
│   │   │   ├── splitter.py         # 分层/分组数据集划分
│   │   │   ├── analyzer.py         # EDA 统计分析 + 自动报告
│   │   │   └── balancer.py         # 类别平衡（过采样/欠采样/合成）
│   │   │
│   │   ├── annotate/               # ② 特征标注
│   │   │   ├── __init__.py
│   │   │   ├── stage.py            # AnnotateStage
│   │   │   ├── config.py
│   │   │   ├── formats.py          # 标注格式注册表 + 双向转换器
│   │   │   ├── converters/         # 各格式转换器实现
│   │   │   │   ├── __init__.py
│   │   │   │   ├── coco.py         # COCO ↔ 内部格式
│   │   │   │   ├── yolo.py         # YOLO ↔ 内部格式
│   │   │   │   ├── voc.py          # Pascal VOC ↔ 内部格式
│   │   │   │   ├── labelme.py      # LabelMe ↔ 内部格式
│   │   │   │   └── cvat.py         # CVAT ↔ 内部格式
│   │   │   ├── pre_annotator.py    # AI 辅助预标注引擎
│   │   │   ├── validator.py        # 标注质量审查（规则 + 统计）
│   │   │   └── visualizer.py       # 标注预览与交互式审查
│   │   │
│   │   ├── train/                  # ③ 模型训练
│   │   │   ├── __init__.py
│   │   │   ├── stage.py            # TrainStage
│   │   │   ├── config.py           # 含超参空间定义
│   │   │   ├── adapter.py          # TrainFrameworkAdapter ABC
│   │   │   ├── adapters/
│   │   │   │   ├── __init__.py
│   │   │   │   ├── torch_.py       # 原生 PyTorch
│   │   │   │   ├── mmdet_.py       # MMDetection 3.x
│   │   │   │   ├── ultralytics_.py # Ultralytics YOLO
│   │   │   │   └── hf_.py          # HuggingFace Transformers/Timm
│   │   │   ├── hyperparams.py      # 超参搜索（Optuna 集成）
│   │   │   ├── callbacks.py        # 训练回调系统
│   │   │   ├── checkpoint.py       # Checkpoint 管理 + 云端同步
│   │   │   ├── scheduler.py        # 学习率调度策略库
│   │   │   └── distributed.py      # DDP/FSDP 分布式训练配置
│   │   │
│   │   ├── validate/               # ④ 验证测试
│   │   │   ├── __init__.py
│   │   │   ├── stage.py            # ValidateStage
│   │   │   ├── config.py
│   │   │   ├── inferencer.py       # 批量推理引擎（支持多 GPU）
│   │   │   ├── metrics.py          # 指标计算库
│   │   │   │   ├── detection.py    # mAP, AP@[.5:.95], AR
│   │   │   │   ├── classification.py # Acc, F1, AUC-ROC, Top-K
│   │   │   │   ├── segmentation.py # mIoU, Dice, Pixel Acc
│   │   │   │   └── regression.py   # MAE, RMSE（关键点/姿态）
│   │   │   ├── comparator.py       # 与基线模型对比
│   │   │   └── error_analysis.py   # FP/FN 错误案例分析
│   │   │
│   │   ├── evaluate/               # ⑤ 模型评估
│   │   │   ├── __init__.py
│   │   │   ├── stage.py            # EvaluateStage
│   │   │   ├── config.py
│   │   │   ├── comparator.py       # 多模型横向对比矩阵
│   │   │   ├── curves.py           # PR/ROC/Confusion Matrix/F1-Curve
│   │   │   ├── profiler.py         # 推理性能剖析（延迟/吞吐/显存/FLOPs）
│   │   │   ├── robustness.py       # 鲁棒性测试（噪声/模糊/光照/遮挡）
│   │   │   └── reporter.py         # 多格式报告（MD/HTML/PDF/JSON）
│   │   │
│   │   ├── optimize/               # ⑥ 量化剪枝
│   │   │   ├── __init__.py
│   │   │   ├── stage.py            # OptimizeStage
│   │   │   ├── config.py
│   │   │   ├── quantizer.py        # 量化引擎
│   │   │   │   ├── ptq.py          # 训练后量化（PTQ）
│   │   │   │   ├── qat.py          # 量化感知训练（QAT）
│   │   │   │   └── fp16.py         # FP16/BF16 半精度
│   │   │   ├── pruner.py           # 剪枝引擎
│   │   │   │   ├── structured.py   # 结构化剪枝（通道/层）
│   │   │   │   ├── unstructured.py # 非结构化剪枝
│   │   │   │   └── search.py       # 剪枝率自动搜索
│   │   │   ├── distill.py          # 知识蒸馏框架
│   │   │   └── calibrator.py       # 量化校准数据生成
│   │   │
│   │   ├── export/                 # ⑦ 模型导出
│   │   │   ├── __init__.py
│   │   │   ├── stage.py            # ExportStage
│   │   │   ├── config.py
│   │   │   ├── onnx_.py            # PyTorch → ONNX（核心中间表示）
│   │   │   ├── tensorrt_.py        # ONNX → TensorRT（Jetson/NVIDIA GPU）
│   │   │   ├── openvino_.py        # ONNX → OpenVINO IR（Intel x86）
│   │   │   ├── tflite_.py          # ONNX → TFLite/Float16/INT8
│   │   │   ├── coreml_.py          # ONNX → CoreML（Apple）
│   │   │   ├── rknn_.py            # ONNX → RKNN（瑞芯微）
│   │   │   ├── validator.py        # 导出模型正确性验证（输出对比）
│   │   │   └── ops.py              # 自定义算子注册与兼容性检查
│   │   │
│   │   └── deploy/                 # ⑧ 边缘部署
│   │       ├── __init__.py
│   │       ├── stage.py            # DeployStage
│   │       ├── config.py
│   │       ├── pusher.py           # 模型推送引擎
│   │       │   ├── ssh_.py         # SSH/SCP 推送
│   │       │   ├── http_.py        # HTTP API 推送
│   │       │   └── mqtt_.py        # MQTT 通知 + OTA
│   │       ├── benchmark.py        # 边缘端推理基准测试
│   │       ├── server.py           # 轻量推理微服务（FastAPI + ONNX Runtime）
│   │       ├── monitor.py          # 部署后监控（数据漂移/精度衰减检测）
│   │       └── rollback.py         # 版本回滚管理
│   │
│   ├── detectors/                  # ===== 预训练检测器（快速推理） =====
│   │   ├── __init__.py
│   │   ├── base.py                 # BaseDetector ABC
│   │   ├── opencv/
│   │   │   ├── haar_face.py
│   │   │   ├── dnn_.py             # OpenCV DNN 通用检测器
│   │   │   └── feature_match.py    # SIFT/ORB 特征匹配
│   │   ├── yolo/
│   │   │   └── yolo_.py
│   │   ├── mediapipe/
│   │   │   ├── face.py
│   │   │   ├── pose.py
│   │   │   └── hands.py
│   │   └── huggingface/
│   │       └── hf_.py
│   │
│   ├── models/                     # ===== 模型注册中心 =====
│   │   ├── __init__.py
│   │   ├── registry.py             # ModelRegistry — 模型版本管理
│   │   ├── zoo.py                  # ModelZoo — 预训练模型库
│   │   ├── card.py                 # ModelCard — 模型文档生成
│   │   └── comparator.py           # 模型对比查询接口
│   │
│   ├── tracking/                   # ===== 实验追踪 =====
│   │   ├── __init__.py
│   │   ├── tracker.py              # ExperimentTracker ABC
│   │   ├── mlflow_.py              # MLflow 集成
│   │   ├── wandb_.py               # Weights & Biases 集成
│   │   ├── local_.py               # 本地文件系统追踪（默认）
│   │   └── schema.py               # 追踪数据标准化 schema
│   │
│   ├── viz/                        # ===== 可视化 =====
│   │   ├── __init__.py
│   │   ├── annotate.py             # 图像标注绘制
│   │   ├── dashboard.py            # 训练监控仪表板
│   │   ├── grid.py                 # 多图对比网格
│   │   ├── curves.py               # 评估曲线（PR/ROC/CM/F1）
│   │   ├── eda.py                  # 数据探索可视化
│   │   ├── profiling.py            # 性能剖析火焰图/瀑布图
│   │   └── export.py               # 图表/视频/GIF 导出
│   │
│   ├── serve/                      # ===== 推理服务 =====
│   │   ├── __init__.py
│   │   ├── app.py                  # FastAPI 推理应用
│   │   ├── routes.py               # API 路由定义
│   │   ├── middleware.py           # 速率限制、认证、日志
│   │   └── client.py               # Python SDK 客户端
│   │
│   └── cli/                        # ===== CLI =====
│       ├── __init__.py
│       ├── app.py                  # Typer 主应用 + 全局选项
│       ├── cmd_run.py              # vw run — 执行流水线
│       ├── cmd_detect.py           # vw detect — 快速推理
│       ├── cmd_list.py             # vw list — 列出注册项
│       ├── cmd_data.py             # vw data — 数据集管理
│       ├── cmd_model.py            # vw model — 模型管理
│       ├── cmd_serve.py            # vw serve — 启动服务
│       └── cmd_export.py           # vw export — 单步导出
│
├── configs/                        # 流水线配置示例
│   ├── full_pipeline.yaml          # 全流程
│   ├── data_prepare.yaml           # 纯数据准备
│   ├── train_detect.yaml           # 训练检测模型
│   ├── optimize_export.yaml        # 优化+导出
│   └── deploy_edge.yaml            # 边缘部署
│
├── templates/                      # 可复用的流水线模板/配方
│   ├── object_detection.yaml       # 目标检测配方
│   ├── face_recognition.yaml       # 人脸识别配方
│   ├── pose_estimation.yaml        # 姿态估计配方
│   └── instance_segmentation.yaml  # 实例分割配方
│
├── tests/
│   ├── conftest.py                 # 共享 fixtures + mock 工厂
│   ├── test_core/
│   │   ├── test_registry.py
│   │   ├── test_config.py
│   │   ├── test_context.py
│   │   └── test_orchestrator.py
│   ├── test_pipeline/
│   │   ├── test_data_stage.py
│   │   ├── test_annotate_stage.py
│   │   ├── test_train_stage.py
│   │   ├── test_validate_stage.py
│   │   ├── test_evaluate_stage.py
│   │   ├── test_optimize_stage.py
│   │   ├── test_export_stage.py
│   │   └── test_deploy_stage.py
│   ├── test_detectors/
│   ├── test_viz/
│   └── fixtures/
│       ├── mini_coco/              # 微型 COCO 数据集（5 图 + 标注）
│       ├── sample_images/          # 各类测试图片
│       └── mock_models/            # 小型 PT/ONNX 模型用于测试
│
└── notebooks/
    ├── 01_data_exploration.ipynb
    ├── 02_annotation_workflow.ipynb
    ├── 03_training_experiment.ipynb
    ├── 04_model_evaluation.ipynb
    ├── 05_optimization.ipynb
    └── 06_edge_deployment.ipynb
```

---

## 5. 核心抽象层设计

### 5.1 领域原语类型 (`core/types.py`)

所有模块共享的不可变数据结构，使用 Pydantic v2 + `frozen=True`：

```python
from pydantic import BaseModel
from typing import Literal, Optional
import numpy as np

class BoundingBox(BaseModel, frozen=True):
    """归一化边界框 [0,1] 或像素坐标"""
    x1: float
    y1: float
    x2: float
    y2: float
    coord_type: Literal["pixel", "normalized"] = "pixel"
    confidence: Optional[float] = None
    class_id: Optional[int] = None
    class_name: Optional[str] = None

class Keypoint(BaseModel, frozen=True):
    x: float
    y: float
    visibility: Literal["visible", "occluded", "not_present"] = "visible"
    name: Optional[str] = None  # e.g., "left_eye", "right_shoulder"

class SegmentationMask(BaseModel, frozen=True):
    """编码后的分割掩码引用（掩码数据存文件，这里存元数据）"""
    mask_path: str           # 相对 artifact 根路径
    format: Literal["png", "rle", "polygon"]
    height: int
    width: int
    class_id: int

class ImageMetadata(BaseModel, frozen=True):
    path: str
    width: int
    height: int
    channels: int
    format: str              # jpg/png/bmp
    file_size_bytes: int
    md5_hash: str            # 文件完整性校验
    exif: dict = {}

class DatasetSplit(BaseModel, frozen=True):
    """数据集划分记录"""
    train: list[str]         # 相对路径列表
    val: list[str]
    test: list[str]
    split_method: str        # random/stratified/group
    random_seed: int
    split_timestamp: str     # ISO 8601
```

### 5.2 阶段基类 (`core/base.py`)

```python
from abc import ABC, abstractmethod
from typing import Generic, TypeVar
from .context import PipelineContext
from .result import StageResult

TConfig = TypeVar("TConfig")
TResult = TypeVar("TResult", bound=StageResult)

class BaseStage(ABC, Generic[TConfig, TResult]):
    """流水线阶段抽象基类。

    设计要点：
    - 阶段内部状态不可变（所有参数通过 config 传入）
    - run() 的输入输出仅通过 PipelineContext 传递
    - dry_run() 用于配置校验和副作用预览
    - 阶段可以声明前置依赖阶段（用于 DAG 调度）
    """

    name: str                # 阶段标识符
    description: str         # 人类可读描述
    depends_on: list[str] = []  # 前置阶段名列表

    @abstractmethod
    def validate_inputs(self, ctx: PipelineContext) -> tuple[bool, list[str]]:
        """验证 ctx 是否包含本阶段所需输入。
        Returns: (is_valid, missing_keys)
        """
        ...

    @abstractmethod
    def run(self, config: TConfig, ctx: PipelineContext) -> tuple[TResult, PipelineContext]:
        """执行阶段逻辑。
        Returns: (stage_result, updated_context)
        PipelineContext 在此处是不可变的——返回新的 ctx 快照。
        """
        ...

    @abstractmethod
    def dry_run(self, config: TConfig, ctx: PipelineContext) -> dict:
        """预览将执行的操作，返回操作摘要字典。
        不产生副作用。
        """
        ...
```

### 5.3 流水线上下文 (`core/context.py`)

上下文设计要点：
- **不可变性**：每次更新生产新快照，旧快照可追溯
- **类型安全**：键名标准化为 `ContextKey` 常量
- **版本追踪**：每个快照携带版本号和父快照引用

```python
from datetime import datetime
from pathlib import Path
from typing import Any, Optional
from pydantic import BaseModel, Field

class ContextKey:
    """上下文键名常量，防止字符串拼写错误"""
    DATASET_DIR     = "artifacts.dataset_dir"
    ANNOTATIONS     = "artifacts.annotations"
    CHECKPOINT_PATH = "artifacts.checkpoint_path"
    VAL_METRICS     = "metrics.validation"
    EVAL_REPORT     = "artifacts.evaluation_report"
    OPTIMIZED_MODEL = "artifacts.optimized_model"
    EXPORTED_MODELS = "artifacts.exports"       # Dict[platform, Path]
    DEPLOY_STATUS   = "metadata.deploy_status"
    DATA_PROVENANCE = "metadata.data_provenance"
    STAGE_HISTORY   = "metadata.stage_history"

class PipelineContext(BaseModel, frozen=True):
    """不可变上下文快照"""

    # 工件路径索引
    artifacts: dict[str, Any] = Field(default_factory=dict)
    # 数值指标
    metrics: dict[str, Any] = Field(default_factory=dict)
    # 文本/结构化元数据
    metadata: dict[str, Any] = Field(default_factory=dict)
    # 各阶段执行记录
    stage_history: list[dict] = Field(default_factory=list)

    # 版本信息
    version: int = 0
    parent_version: Optional[int] = None
    created_at: str = Field(default_factory=lambda: datetime.now().isoformat())
    created_by: str = ""  # stage name

    def evolve(self, **updates) -> "PipelineContext":
        """生成新快照（不可变更新）"""
        new_data = self.model_dump()
        for key, value in updates.items():
            # 支持点分隔路径: "artifacts.dataset_dir"
            self._set_nested(new_data, key, value)
        new_data["version"] = self.version + 1
        new_data["parent_version"] = self.version
        return PipelineContext(**new_data)

    def get(self, key: str, default: Any = None) -> Any:
        """通过点分隔路径读取嵌套值"""
        ...

    @staticmethod
    def _set_nested(data: dict, key: str, value: Any) -> None:
        """设置嵌套字典值: "a.b.c" -> data["a"]["b"]["c"]"""
        ...
```

### 5.4 泛型注册表 (`core/registry.py`)

统一插件注册机制，所有组件类型共用同一模式：

```python
from typing import TypeVar, Generic, ClassVar, Callable

T = TypeVar("T")

class Registry(Generic[T]):
    """泛型注册表基类。

    用法：
        detector_registry = Registry[BaseDetector]("detector")

        @detector_registry.register("yolov8")
        class YOLOv8Detector(BaseDetector): ...

        cls = detector_registry.get("yolov8")
    """

    def __init__(self, kind: str):
        self.kind = kind            # "detector", "format_converter", "framework", "exporter"
        self._entries: dict[str, type[T]] = {}

    def register(self, name: str, **metadata) -> Callable:
        """装饰器：注册组件并附带元数据"""
        def decorator(cls: type[T]) -> type[T]:
            self._entries[name] = cls
            cls._registry_metadata = metadata  # type: ignore
            return cls
        return decorator

    def register_direct(self, name: str, cls: type[T], **metadata) -> None:
        """编程式注册"""
        self._entries[name] = cls
        cls._registry_metadata = metadata  # type: ignore

    def get(self, name: str) -> type[T]:
        if name not in self._entries:
            raise RegistryError(self.kind, name, list(self._entries.keys()))
        return self._entries[name]

    def list(self) -> dict[str, type[T]]:
        return dict(self._entries)

    def list_by(self, **filters) -> dict[str, type[T]]:
        """按元数据过滤"""
        result = {}
        for name, cls in self._entries.items():
            meta = getattr(cls, "_registry_metadata", {})
            if all(meta.get(k) == v for k, v in filters.items()):
                result[name] = cls
        return result

    def discover_entry_points(self, group: str) -> None:
        """从 pyproject.toml entry_points 发现插件"""
        import importlib.metadata as md
        for ep in md.entry_points(group=group):
            ep.load()  # 触发装饰器注册


# 全局注册表实例
detector_registry   = Registry("detector")
format_registry     = Registry("format_converter")
framework_registry  = Registry("framework")
exporter_registry   = Registry("exporter")
```

### 5.5 分层异常体系 (`core/exceptions.py`)

```python
class VisionWorkbenchError(Exception):
    """所有异常的基类"""
    ...

# 配置层
class ConfigError(VisionWorkbenchError): ...
class ConfigValidationError(ConfigError): ...
class ConfigMissingKeyError(ConfigError): ...

# 注册表层
class RegistryError(VisionWorkbenchError): ...

# 依赖层
class DependencyError(VisionWorkbenchError): ...
class MissingDependency(DependencyError): ...
class VersionConflict(DependencyError): ...

# 流水线层
class PipelineError(VisionWorkbenchError): ...
class StageInputError(PipelineError): ...
class StageExecutionError(PipelineError): ...
class StageNotFoundError(PipelineError): ...

# 数据层
class DataError(VisionWorkbenchError): ...
class DataValidationError(DataError): ...
class AnnotationFormatError(DataError): ...

# 模型层
class ModelError(VisionWorkbenchError): ...
class ModelNotFoundError(ModelError): ...
class ExportError(ModelError): ...
class InferenceError(ModelError): ...

# 部署层
class DeployError(VisionWorkbenchError): ...
class ConnectionError(DeployError): ...
class BenchmarkError(DeployError): ...
```

---

## 6. 数据管理层设计

### 6.1 数据集目录结构约定 (`data/schema.py`)

标准化数据集目录结构，确保所有阶段使用统一布局：

```
<dataset_root>/
├── dataset.yaml              # 数据集描述文件（必须）
├── images/
│   ├── train/
│   │   ├── 000001.jpg
│   │   └── ...
│   ├── val/
│   │   └── ...
│   └── test/
│       └── ...
├── annotations/
│   ├── instances_train.json  # COCO 格式标注
│   ├── instances_val.json
│   └── instances_test.json
├── splits/
│   └── split_20240101.json   # 划分记录
└── .dvc/                     # DVC 版本追踪（可选）
```

### 6.2 数据集描述文件 (`dataset.yaml`)

```yaml
# dataset.yaml — 每个数据集的元数据文件
name: "street-scenes-v1"
version: "1.0.0"
created: "2024-01-01T00:00:00Z"
description: "Urban street scenes for object detection"
license: "CC-BY-4.0"
tasks:
  - object_detection
  - instance_segmentation

categories:
  - id: 0
    name: "pedestrian"
    supercategory: "person"
  - id: 1
    name: "vehicle"
    supercategory: "vehicle"

image_count:
  train: 5000
  val: 1000
  test: 500

image_stats:
  avg_width: 1920
  avg_height: 1080
  total_size_gb: 2.3

annotation_stats:
  total_boxes: 45000
  boxes_per_image_avg: 6.9
  class_distribution:
    pedestrian: 25000
    vehicle: 20000

provenance:
  source: "custom_collection"
  collection_location: "Beijing, China"
  collection_period: "2023-06 to 2023-12"
  preprocessing: ["dedup_phash", "blur_filter"]

checksums:
  sha256: "abc123..."   # 整个数据集的内容校验
```

### 6.3 数据版本控制 (`data/versioning.py`)

```python
class DataVersionControl:
    """DVC 集成封装。

    职责：
    - 数据集版本快照创建与切换
    - 与 Git 分支的联动
    - 远程存储（S3/MinIO/NAS）同步
    """

    def init(self, dataset_path: Path) -> None: ...
    def snapshot(self, dataset_path: Path, message: str) -> str: ...  # 返回版本号
    def checkout(self, dataset_path: Path, version: str) -> None: ...
    def push(self, dataset_path: Path, remote: str) -> None: ...
    def pull(self, dataset_path: Path, remote: str) -> None: ...
    def log(self, dataset_path: Path) -> list[dict]: ...
    def diff(self, dataset_path: Path, v1: str, v2: str) -> dict: ...
```

### 6.4 数据血统追踪 (`data/provenance.py`)

```python
@dataclass
class DataLineage:
    """追踪数据从源头到模型的完整路径"""
    dataset_id: str
    source_path: str
    transformations: list[dict]   # [{stage: "data", action: "dedup", params: {...}}, ...]
    model_checkpoints: list[str]  # 使用此数据的模型版本
    annotations_origin: str       # 标注来源（manual/model_assisted/external）
    created_at: str
```

---

## 7. 流水线阶段详细规格

### 7.1 阶段①：数据清洗 (`pipeline/data/`)

**输入**：原始数据目录 / 公开数据集名称
**输出**：符合 `DatasetSchema` 的标准化数据集 + `dataset.yaml`

| 子组件 | 功能 | 关键技术 |
|--------|------|---------|
| `validator.py` | 图片可读性、尺寸范围、通道数、格式白名单 | PIL/cv2 双重验证 |
| `cleaner.py` | 去重（pHash/dHash/SSIM）、模糊检测（Laplacian）、曝光检测 | imagehash, cv2.Laplacian |
| `augmentor.py` | 配置驱动的数据增强管道 | Albumentations 组合管道 |
| `splitter.py` | 分层/分组/时间序列划分 | sklearn StratifiedShuffleSplit |
| `analyzer.py` | 类别分布、尺寸热力图、通道统计 | matplotlib + seaborn |
| `balancer.py` | 类别平衡（过采样/欠采样/合成少数类） | imbalanced-learn |

**配置示例**：

```yaml
data:
  source:
    type: "directory"          # directory | coco | voc | image_folder
    path: "data/raw/"
  target: "data/processed/"
  dataset_name: "street-scenes-v1"

  validation:
    min_resolution: [32, 32]
    max_resolution: [4096, 4096]
    allowed_formats: [".jpg", ".jpeg", ".png", ".bmp"]
    check_corruption: true
    remove_truncated: true

  dedup:
    enabled: true
    method: "phash"            # phash | dhash | ssim | combined
    phash_threshold: 5         # Hamming distance
    ssim_threshold: 0.95

  quality:
    blur_threshold: 100        # Laplacian variance
    overexposure_threshold: 0.95  # 过曝像素比例
    underexposure_threshold: 0.05

  split:
    train: 0.7
    val: 0.15
    test: 0.15
    method: "stratified"       # random | stratified | group | time_series
    stratify_by: "category"
    seed: 42

  balance:
    enabled: false
    method: "oversample"       # oversample | undersample | smote
    target_ratio: 1.0

  augment:
    enabled: true
    preset: "detection_light"  # detection_light | detection_heavy | classification | custom
    custom_pipeline:
      - {type: "HorizontalFlip", p: 0.5}
      - {type: "RandomBrightnessContrast", p: 0.3}
      - {type: "RandomResizedCrop", height: 640, width: 640, p: 0.5}
```

---

### 7.2 阶段②：特征标注 (`pipeline/annotate/`)

**核心流程**：

```
外部标注 (LabelMe/VOC/CVAT)
          │
          ▼
    [Format Converter]  ← FormatRegistry 查找
          │
          ▼
    COCO JSON (内部统一格式)
          │                         │
          ▼                         ▼
    [Quality Validator]     [AI Pre-Annotator]
          │                    (用已有模型预标注新数据)
          ▼                         │
    标注质量报告              人工校正后合并
```

**标注格式转换矩阵**：

```
        COCO   YOLO   VOC   LabelMe  CVAT
COCO     -      →      →      →       →
YOLO     →      -      →      →       →
VOC      →      →      -      →       →
LabelMe  →      →      →      -       →
CVAT     →      →      →      →       -
```

**配置示例**：

```yaml
annotate:
  input:
    format: "cvat"             # 源标注格式
    path: "data/external/cvat_export.xml"
  output:
    format: "coco"             # 目标格式（默认 COCO）
    path: "data/processed/annotations/"

  pre_annotation:
    enabled: true
    model: "yolov8x"           # 用于预标注的模型
    confidence_threshold: 0.5
    review_mode: true          # 仅标注置信度 < threshold 的样本

  quality:
    checks:
      - check_empty_annotations
      - check_box_size: {min_area: 25, max_area_ratio: 0.8}
      - check_class_balance: {min_samples_per_class: 10}
      - check_cross_boundary: true
    generate_report: true
```

---

### 7.3 阶段③：模型训练 (`pipeline/train/`)

**框架适配器接口**：

```python
class TrainFrameworkAdapter(ABC, Generic[TConfig]):
    """训练框架适配器抽象。

    每个训练框架（PyTorch, MMDetection, Ultralytics, HuggingFace）
    实现此接口，使得 TrainStage 可以框架无关地编排训练。
    """

    name: str
    supports_tasks: list[str]   # 框架支持的任务类型

    @abstractmethod
    def create_model(self, config: TConfig) -> Any: ...
    @abstractmethod
    def create_dataloaders(self, dataset_path: Path, config: TConfig) -> dict: ...
    @abstractmethod
    def create_optimizer(self, model: Any, config: TConfig) -> Any: ...
    @abstractmethod
    def create_scheduler(self, optimizer: Any, config: TConfig) -> Any: ...
    @abstractmethod
    def train(self, model, dataloaders, optimizer, scheduler, callbacks, config) -> Any: ...
    @abstractmethod
    def save_checkpoint(self, trainer_state: Any, path: Path) -> Path: ...
    @abstractmethod
    def load_checkpoint(self, path: Path) -> Any: ...
    @abstractmethod
    def export_to_onnx(self, model: Any, input_shape: tuple, path: Path) -> Path: ...
```

**训练回调系统**：

```python
class TrainingCallback(ABC):
    """训练生命周期钩子"""
    def on_train_start(self, ctx: dict) -> None: ...
    def on_epoch_start(self, epoch: int, ctx: dict) -> None: ...
    def on_batch_end(self, batch: int, loss: float, ctx: dict) -> None: ...
    def on_epoch_end(self, epoch: int, metrics: dict, ctx: dict) -> None: ...
    def on_validation_end(self, metrics: dict, ctx: dict) -> None: ...
    def on_train_end(self, ctx: dict) -> None: ...

# 内置回调
class EarlyStopping(TrainingCallback): ...
class ModelCheckpoint(TrainingCallback): ...
class LearningRateMonitor(TrainingCallback): ...
class MetricsLogger(TrainingCallback): ...
class GPUStatsMonitor(TrainingCallback): ...
class WandbLogger(TrainingCallback): ...
```

**超参搜索**：

```python
class HyperparameterSearch:
    """Optuna 集成超参搜索"""
    def search(
        self,
        objective: Callable[[dict], float],  # 目标函数（返回验证指标）
        space: dict,                          # 超参空间定义
        n_trials: int = 50,
        method: Literal["tpe", "grid", "random"] = "tpe",
        direction: Literal["maximize", "minimize"] = "maximize",
        pruner: Optional[str] = "median",     # MedianPruner
    ) -> dict:                                 # 最佳超参组合
        ...
```

**配置示例**：

```yaml
train:
  framework: "ultralytics"
  task: "object_detection"

  model:
    architecture: "yolov8n"       # 预定义架构或自定义配置路径
    pretrained: "yolov8n.pt"     # None = from scratch
    num_classes: 10

  data:
    dataset_path: "data/processed/"
    image_size: [640, 640]
    batch_size: 16
    num_workers: 4

  training:
    epochs: 100
    optimizer: "adamw"
    lr: 0.001
    lr_scheduler: "cosine"
    warmup_epochs: 5
    weight_decay: 0.0005
    mixed_precision: "fp16"      # fp32 | fp16 | bf16
    gradient_clip: 1.0
    ema: true                     # 指数移动平均

  augmentation:
    mosaic: 1.0
    mixup: 0.1
    copy_paste: 0.1

  distributed:
    strategy: "ddp"              # null | ddp | fsdp
    devices: [0, 1]              # GPU IDs 或 "auto"

  callbacks:
    early_stopping:
      patience: 20
      metric: "val/mAP_0.5:0.95"
    model_checkpoint:
      monitor: "val/mAP_0.5:0.95"
      mode: "max"
      save_top_k: 3
    lr_monitor: true

  hyperparameter_search:
    enabled: false
    n_trials: 30
    space:
      lr: {type: "loguniform", low: 1e-4, high: 1e-2}
      batch_size: {type: "categorical", choices: [8, 16, 32]}
      weight_decay: {type: "uniform", low: 1e-5, high: 1e-3}
```

---

### 7.4 阶段④：验证测试 (`pipeline/validate/`)

**输出**：`val_metrics.json`

```yaml
validate:
  checkpoint: "artifacts/checkpoints/best.pt"  # 或从 ctx 读取
  data: "data/processed/"
  image_size: [640, 640]
  batch_size: 32
  device: "cuda:0"

  metrics:
    - "mAP"           # mAP@[.5:.95]
    - "mAP_50"        # mAP@0.5
    - "mAP_75"        # mAP@0.75
    - "AP_per_class"  # 每类别 AP
    - "AR"            # Average Recall
    - "F1"
    - "confusion_matrix"

  regression_test:
    enabled: true
    baseline_checkpoint: "s3://models/baseline/yolov8n.pt"
    tolerance:
      mAP_drop_max: 0.02         # 允许的最大 mAP 下降

  error_analysis:
    enabled: true
    top_k_fp: 20                  # 输出 Top-20 假阳性样例
    top_k_fn: 20                  # 输出 Top-20 假阴性样例
    output_dir: "artifacts/error_analysis/"
```

---

### 7.5 阶段⑤：模型评估 (`pipeline/evaluate/`)

**核心输出**：多模型对比报告 + 性能剖析

```yaml
evaluate:
  models:
    - name: "baseline_yolov8n"
      checkpoint: "artifacts/checkpoints/baseline.pt"
    - name: "candidate_v2"
      checkpoint: "artifacts/checkpoints/candidate_v2.pt"

  comparison:
    metrics: ["mAP", "mAP_50", "mAP_75", "F1", "params", "FLOPs", "latency_ms", "throughput_fps"]
    output: "artifacts/reports/comparison_report.html"

  curves:
    - type: "pr_curve"
      per_class: true
    - type: "roc_curve"
    - type: "confusion_matrix"
      normalize: true
    - type: "f1_confidence"
    - type: "precision_recall_by_threshold"

  profiling:
    batch_sizes: [1, 4, 8, 16, 32]
    iterations: 100              # 预热后测量迭代次数
    measure:
      - "latency_p50"
      - "latency_p95"
      - "latency_p99"
      - "throughput_fps"
      - "gpu_memory_mb"
      - "cpu_memory_mb"
      - "flops"

  robustness:
    enabled: false
    perturbations:
      gaussian_noise: {std: [0.01, 0.05, 0.1]}
      gaussian_blur: {kernel: [3, 5, 7]}
      brightness: {factor: [0.5, 0.8, 1.2, 1.5]}
      rotation: {degrees: [5, 10, 15]}
```

---

### 7.6 阶段⑥：量化剪枝 (`pipeline/optimize/`)

**优化策略管道**（可按顺序组合多个方法）：

```yaml
optimize:
  input: "artifacts/checkpoints/best.pt"
  output_dir: "artifacts/optimized/"

  # 优化目标优先级
  objective:
    primary: "latency"           # latency | size | accuracy
    max_accuracy_drop: 0.02      # 精度损失上限

  # 策略管道（按顺序执行）
  pipeline:
    - type: "prune"
      method: "l1_structured"    # l1_unstructured | l1_structured | l2_structured | random
      amount: 0.3                # 剪枝比例
      layers: ["conv"]           # 目标层类型
      iterative: true            # 迭代式剪枝
      steps: 5
      fine_tune_epochs: 2

    - type: "quantize"
      method: "ptq"              # ptq | qat
      precision: "int8"          # int8 | fp16 | bf16
      backend: "fbgemm"          # fbgemm | qnnpack | tensorrt
      calibration:
        method: "percentile"     # percentile | histogram | entropy
        num_samples: 500

    - type: "distill"           # 知识蒸馏（可选）
      teacher_checkpoint: "artifacts/checkpoints/teacher_large.pt"
      temperature: 3.0
      alpha: 0.7                 # 蒸馏损失权重
```

---

### 7.7 阶段⑦：模型导出 (`pipeline/export/`)

**ONNX 作为核心中间表示 (IR)**，所有框架先导出为 ONNX，再从 ONNX 转换到目标平台：

```yaml
export:
  input: "artifacts/optimized/model_int8.pt"
  output_dir: "artifacts/exports/"

  onnx:
    opset_version: 17
    input_shape: [1, 3, 640, 640]
    dynamic_batch: true          # 支持动态 batch size
    simplify: true               # onnx-simplifier 优化图结构
    validate: true               # 导出后推理验证

  targets:
    - platform: "tensorrt"
      precision: "fp16"          # fp32 | fp16 | int8
      workspace_size_gb: 2
      max_batch_size: 16
      dynamic_shapes: true

    - platform: "openvino"
      precision: "fp16"
      optimize_for: "latency"    # latency | throughput

    - platform: "tflite"
      precision: "int8"
      represent_dataset: "data/processed/images/val/"  # 量化校准数据
      optimization: ["latency", "size"]

    - platform: "coreml"
      precision: "fp16"
      compute_units: "all"       # all | cpu_and_gpu | cpu_only | neural_engine

    - platform: "rknn"
      target: "rk3588"           # 瑞芯微芯片型号
      precision: "int8"
```

**导出验证**：

```python
class ExportValidator:
    """导出后验证：对比原始模型与导出模型的输出一致性"""
    def validate(
        self,
        original_model: Any,
        exported_model_path: Path,
        sample_inputs: list[np.ndarray],
        tolerance: float = 1e-3,
    ) -> tuple[bool, dict]:  # (is_valid, per_sample_diffs)
        ...
```

---

### 7.8 阶段⑧：边缘部署 (`pipeline/deploy/`)

```yaml
deploy:
  model: "artifacts/exports/model_fp16.tflite"
  platform: "tflite"

  devices:
    - name: "edge-camera-01"
      type: "raspberry_pi_5"
      address: "192.168.1.101"
      auth:
        method: "ssh_key"
        key_path: "~/.ssh/id_rsa_edge"
      target_path: "/opt/models/detector_v2.tflite"

    - name: "edge-gateway"
      type: "jetson_orin_nx"
      address: "192.168.1.200"
      auth:
        method: "ssh_password"  # 仅开发阶段
      target_path: "/home/nvidia/models/"

  benchmark:
    enabled: true
    warmup_runs: 50
    measure_runs: 1000
    batch_sizes: [1, 4]
    report_path: "artifacts/reports/edge_benchmark.json"

  monitoring:
    enabled: true
    metrics_endpoint: "http://192.168.1.200:9090/metrics"
    drift_detection:
      reference_dataset: "data/processed/images/val/"
      interval_hours: 24

  rollout:
    strategy: "canary"          # direct | canary | blue_green
    canary_percent: 10
    auto_rollback: true
    rollback_metric: "inference_latency_p99"
    rollback_threshold: 1.5     # 超过基线 1.5x 则回滚
```

**推理微服务**（`serve/app.py`）：

```python
# FastAPI 推理服务，可部署到边缘设备
from fastapi import FastAPI, File, UploadFile
from vision_workbench.serve.middleware import RateLimiter, RequestLogger

app = FastAPI(title="Vision Inference Service")

@app.post("/v1/detect")
async def detect(image: UploadFile = File(...)):
    ...

@app.get("/v1/health")
async def health():
    return {"status": "ok", "model_version": MODEL_VERSION}

@app.get("/metrics")
async def metrics():  # Prometheus 格式
    ...
```

---

## 8. 配置体系设计

### 8.1 配置层次结构

```
全局默认值 (src/vision_workbench/config/defaults.yaml)
    ↓ 覆盖
流水线模板 (templates/object_detection.yaml)
    ↓ 覆盖
用户配置 (configs/my_experiment.yaml)
    ↓ 覆盖
CLI 参数 (--train.epochs 200)
    ↓ 覆盖
环境变量 (VW_DATA_SOURCE=...)
```

### 8.2 根配置模型

```python
class PipelineConfig(BaseModel):
    """流水线根配置"""
    # 元信息
    name: str
    description: str = ""
    version: str = "1.0"

    # 运行时配置
    runtime: RuntimeConfig = RuntimeConfig()

    # 阶段配置（按需启用）
    stages: list[str] = []
    # 各阶段的具体配置
    data: Optional[DataStageConfig] = None
    annotate: Optional[AnnotateStageConfig] = None
    train: Optional[TrainStageConfig] = None
    validate: Optional[ValidateStageConfig] = None
    evaluate: Optional[EvaluateStageConfig] = None
    optimize: Optional[OptimizeStageConfig] = None
    export: Optional[ExportStageConfig] = None
    deploy: Optional[DeployStageConfig] = None

class RuntimeConfig(BaseModel):
    """全局运行时配置"""
    workspace: Path = Path("./vw_workspace")      # 工作目录
    artifacts_dir: Path = Path("./vw_workspace/artifacts")
    device: Literal["cpu", "cuda", "mps", "auto"] = "auto"
    seed: int = 42
    log_level: Literal["DEBUG", "INFO", "WARN", "ERROR"] = "INFO"
    log_file: Optional[Path] = None
    tracking:
        backend: Literal["local", "mlflow", "wandb"] = "local"
        experiment_name: Optional[str] = None
        tags: dict[str, str] = {}
    cache:
        enabled: bool = True
        ttl_hours: int = 24
    parallel:
        max_workers: int = 4
```

### 8.3 配置继承与模板

```yaml
# templates/object_detection.yaml — 可复用的配方模板
_base_: "vision_workbench/config/defaults.yaml"  # 继承基础配置

name: "object_detection_recipe"
runtime:
  device: "auto"
  seed: 42

data:
  validation:
    min_resolution: [32, 32]
  split:
    train: 0.7; val: 0.15; test: 0.15
  augment:
    preset: "detection_light"

train:
  task: "object_detection"
  training:
    optimizer: "adamw"
    mixed_precision: "fp16"

# 此模板不指定 framework 和 model，由用户配置覆盖
```

```yaml
# configs/my_detector.yaml — 用户实验配置
_base_: "templates/object_detection.yaml"

name: "traffic-detector-v3"
stages: ["data", "annotate", "train", "validate", "evaluate", "optimize", "export"]

data:
  source:
    path: "data/traffic_cam/"
  dataset_name: "traffic-v1"

train:
  framework: "ultralytics"
  model:
    architecture: "yolov8m"
    pretrained: "yolov8m.pt"
    num_classes: 5
  training:
    epochs: 200
    batch_size: 32
```

---

## 9. 实验追踪与可复现性

### 9.1 实验追踪器接口

```python
class ExperimentTracker(ABC):
    """实验追踪抽象。默认实现：本地文件系统。

    实现：LocalTracker, MLflowTracker, WandbTracker
    """

    @abstractmethod
    def init(self, experiment_name: str, tags: dict = {}) -> str: ...    # 返回 run_id

    @abstractmethod
    def log_params(self, params: dict) -> None: ...

    @abstractmethod
    def log_metrics(self, metrics: dict, step: int = 0) -> None: ...

    @abstractmethod
    def log_artifact(self, path: Path, artifact_name: str) -> None: ...

    @abstractmethod
    def log_model(self, path: Path, model_name: str, framework: str) -> None: ...

    @abstractmethod
    def log_figure(self, figure: Any, name: str) -> None: ...

    @abstractmethod
    def set_status(self, status: Literal["running", "completed", "failed"]) -> None: ...

    @abstractmethod
    def end(self) -> None: ...
```

### 9.2 可复现性保证

```yaml
# 每个实验自动生成的 manifest.yaml
experiment:
  id: "exp_20240101_120000_a1b2c3"
  name: "traffic-detector-v3"
  status: "completed"

environment:
  vision_workbench_version: "0.1.0"
  python_version: "3.12.1"
  platform: "linux-x86_64"
  gpu: "NVIDIA RTX 4090"
  cuda_version: "12.1"
  dependencies:
    torch: "2.1.0"
    ultralytics: "8.0.200"
    # ... 完整 pip freeze

pipeline:
  stages_executed: ["data", "train", "validate"]
  total_duration_seconds: 1423.5
  config_snapshot: "configs/my_detector.yaml"
  config_hash: "sha256:def789..."

artifacts:
  dataset: "s3://datasets/traffic-v1/"
  checkpoint: "vw_workspace/artifacts/checkpoints/best.pt"
  metrics: "vw_workspace/artifacts/metrics.json"

data_lineage:
  dataset_version: "traffic-v1@dvc:abc123"
  preprocessing_hash: "sha256:ghi456..."
```

---

## 10. 模型存储与生命周期

### 10.1 三句话回答核心问题

> **Q1: YOLO 预训练模型放哪？**
> → `vw_workspace/models/zoo/yolov8n.pt`（通过 `vw model pull yolov8n` 自动下载到此处）
>
> **Q2: ResNet / ViT / EfficientDet 等其他架构的预训练模型放哪？**
> → 同样放在 `vw_workspace/models/zoo/`。zoo 是**框架无关**的预训练模型缓存，不论什么架构都统一存放，差异通过 `zoo_index.yaml` 记录。
>
> **Q3: 训练得到的最优模型放哪？**
> → `vw_workspace/models/checkpoints/<实验名>/<运行时间戳>/best.pt`

**核心原则：zoo 是"原料库"，checkpoints 是"工厂产线"，目录结构按模型来源阶段划分，不按架构或框架划分。**

### 10.2 五种模型形态与存储位置

| 模型形态 | 来源 | 存放位置 | 典型文件 |
|---------|------|---------|---------|
| ① 预训练模型 | 下载 / 拷贝 | `zoo/` | `yolov8n.pt`, `resnet50.pth` |
| ② 训练 Checkpoint | TrainStage 产出 | `checkpoints/<experiment>/<run>/` | `best.pt`, `last.pt`, `epoch_050.pt` |
| ③ 优化后模型 | OptimizeStage 产出 | `optimized/<experiment>/<strategy>/` | `model_int8.pt`, `model_pruned.pt` |
| ④ 导出模型 | ExportStage 产出 | `exports/<experiment>/<platform>/` | `model.onnx`, `model_fp16.engine` |
| ⑤ 部署包 | DeployStage 组装 | `deployments/<experiment>/<version>/` | `model.tflite` + `Dockerfile` |

### 10.3 工作区完整物理布局

所有工件（数据、模型、标注、报告）统一存储在工作区目录下，默认路径为项目根目录的 `vw_workspace/`：

```
<project_root>/vw_workspace/               # VW_WORKSPACE 环境变量可覆盖
│
├── workspace.yaml                          # 工作区元数据（创建时间、版本、全局配置引用）
│
├── models/                                 # ===== 模型存储根目录 =====
│   │
│   ├── zoo/                                # ① 预训练模型缓存（Model Zoo）
│   │   ├── zoo_index.yaml                  #    模型索引 + 校验和
│   │   ├── yolov8n.pt                      #    Ultralytics 官方权重
│   │   ├── yolov8m.pt
│   │   ├── yolov8x.pt
│   │   ├── mobilenet_v2_ssd/               #    目录形式（多文件模型）
│   │   │   ├── model.pth
│   │   │   └── config.json
│   │   ├── resnet50-imagenet.pth
│   │   ├── efficientdet_d0.pth
│   │   └── vit_base_patch16_224.pth
│   │
│   ├── checkpoints/                        # ② 训练产出（每个实验一个子目录）
│   │   ├── traffic-detector-v3/            #    experiment_name/
│   │   │   ├── run_20240601_120000/        #      run_timestamp/
│   │   │   │   ├── best.pt                 #        最佳模型
│   │   │   │   ├── last.pt                 #        最后一个 epoch
│   │   │   │   ├── epoch_050.pt            #        定期存档
│   │   │   │   ├── epoch_100.pt
│   │   │   │   ├── checkpoint_manifest.yaml#        存档清单
│   │   │   │   └── train_config_snapshot.yaml #     训练配置快照
│   │   │   └── run_20240615_080000/
│   │   │       └── ...
│   │   ├── face-detector-v1/
│   │   │   └── run_20240520_090000/
│   │   └── ...
│   │
│   ├── optimized/                          # ③ 优化后模型（量化/剪枝/蒸馏产出）
│   │   ├── traffic-detector-v3/
│   │   │   ├── pruned_l1_0.3/
│   │   │   │   ├── model_pruned.pt
│   │   │   │   └── optimize_manifest.yaml  #    优化操作记录
│   │   │   ├── quantized_int8/
│   │   │   │   ├── model_int8.pt
│   │   │   │   └── optimize_manifest.yaml
│   │   │   └── pruned_quantized/
│   │   │       ├── model_int8_pruned.pt
│   │   │       └── optimize_manifest.yaml
│   │   └── ...
│   │
│   └── exports/                            # ④ 导出模型（平台原生格式）
│       ├── traffic-detector-v3/
│       │   ├── onnx/
│       │   │   ├── model.onnx
│       │   │   └── export_manifest.yaml
│       │   ├── tensorrt/
│       │   │   ├── model_fp16.engine
│       │   │   ├── model_int8.engine
│       │   │   └── export_manifest.yaml
│       │   ├── tflite/
│       │   │   ├── model_int8.tflite
│       │   │   └── export_manifest.yaml
│       │   ├── openvino/
│       │   │   ├── model.xml
│       │   │   ├── model.bin
│       │   │   └── export_manifest.yaml
│       │   └── coreml/
│       │       ├── model.mlpackage/
│       │       └── export_manifest.yaml
│       └── ...
│
├── data/                                   # ===== 数据集存储 =====
│   ├── raw/                                #   原始数据
│   ├── processed/                          #   清洗后数据集（符合 DatasetSchema）
│   │   ├── street-scenes-v1/
│   │   │   ├── dataset.yaml
│   │   │   ├── images/
│   │   │   ├── annotations/
│   │   │   └── splits/
│   │   └── traffic-v1/
│   │       └── ...
│   └── external/                           #   外部导入的原始标注
│
├── experiments/                            # ===== 实验记录 =====
│   ├── traffic-detector-v3/
│   │   ├── run_20240601_120000/
│   │   │   ├── manifest.yaml               #   实验完整快照
│   │   │   ├── config_snapshot.yaml        #   配置快照
│   │   │   ├── metrics.json                #   指标时间序列
│   │   │   ├── pipeline.log                #   结构化日志
│   │   │   └── lineage.yaml                #   数据/模型血统
│   │   └── run_20240615_080000/
│   └── ...
│
├── reports/                                # ===== 报告输出 =====
│   ├── comparison_reports/
│   ├── evaluation_reports/
│   ├── data_analysis/
│   └── benchmark_reports/
│
├── cache/                                  # ===== 临时缓存 =====
│   ├── preprocessed/                       #   预处理缓存（增强后的临时数据）
│   ├── calibration/                        #   量化校准数据缓存
│   └── inference/                          #   批量推理中间结果
│
└── deployments/                            # ===== 部署包 =====
    ├── traffic-detector-v3/
    │   ├── v1.0.0/
    │   │   ├── model.tflite
    │   │   ├── inference_service.py        #   生成的推理服务
    │   │   ├── config.yaml
    │   │   ├── Dockerfile
    │   │   └── deploy_manifest.yaml
    │   └── v1.1.0/
    └── ...

# 注意：vw_workspace/ 整体加入 .gitignore
# 但各子目录的 manifest.yaml 可通过 DVC 或 Git LFS 版本控制
```

### 10.4 模型生命周期状态机

每个模型在平台上经历明确的状态流转，每一步都记录在 `manifest.yaml` 中：

```
                    ┌──────────────┐
                    │  DOWNLOADED  │  预训练模型下载到 zoo/
                    └──────┬───────┘
                           │  TrainStage
                           ▼
                    ┌──────────────┐
                    │   TRAINED    │  Checkpoint 保存到 checkpoints/
                    └──────┬───────┘
                           │  ValidateStage + EvaluateStage
                           ▼
                    ┌──────────────┐
                    │  VALIDATED   │  指标达标，进入优化队列
                    └──────┬───────┘
                           │  OptimizeStage
                           ▼
                    ┌──────────────┐
                    │  OPTIMIZED   │  量化/剪枝后存入 optimized/
                    └──────┬───────┘
                           │  ExportStage
                           ▼
                    ┌──────────────┐
                    │  EXPORTED    │  平台格式存入 exports/
                    └──────┬───────┘
                           │  DeployStage
                           ▼
                    ┌──────────────┐
                    │  DEPLOYED    │  推送至边缘设备
                    └──────┬───────┘
                           │  (可选) Monitor 触发回滚
                           ▼
                    ┌──────────────┐
                    │  ARCHIVED    │  退役模型，保留历史
                    └──────────────┘
```

### 10.5 每类模型的 Manifest 结构

每个模型目录下都有一个 `*_manifest.yaml`，记录模型元数据和血统：

```yaml
# checkpoints/traffic-detector-v3/run_20240601_120000/checkpoint_manifest.yaml
model:
  name: "traffic-detector-v3"
  version: "1.0.0"
  stage: "trained"              # downloaded | trained | validated | optimized | exported | deployed | archived
  framework: "ultralytics"
  task: "object_detection"
  architecture: "yolov8m"
  num_parameters: 25800000
  input_shape: [3, 640, 640]

training:
  experiment_id: "exp_20240601_120000_a1b2c3"
  base_model: "zoo/yolov8m.pt"  # 预训练权重来源
  epochs: 200
  final_epoch: 200
  optimizer: "adamw"
  initial_lr: 0.001
  mixed_precision: "fp16"
  dataset:
    name: "traffic-v1"
    version: "dvc:abc123"
    train_samples: 7000
    val_samples: 1500

metrics:
  mAP_0.5:0.95: 0.782
  mAP_0.5: 0.923
  mAP_0.75: 0.801

files:
  - path: "best.pt"
    role: "primary"
    sha256: "abc123..."
    size_mb: 52.3
  - path: "last.pt"
    role: "backup"
    sha256: "def456..."
    size_mb: 52.3
  - path: "epoch_050.pt"
    role: "intermediate"
  - path: "epoch_100.pt"
    role: "intermediate"
```

```yaml
# exports/traffic-detector-v3/onnx/export_manifest.yaml
model:
  name: "traffic-detector-v3"
  version: "1.0.0"
  stage: "exported"
  platform: "onnx"
  precision: "fp32"

source:
  checkpoint: "checkpoints/traffic-detector-v3/run_20240601_120000/best.pt"
  optimized_from: "optimized/traffic-detector-v3/pruned_l1_0.3/model_pruned.pt"  # 或 null

export:
  opset_version: 17
  dynamic_batch: true
  simplified: true
  input_shape: [1, 3, 640, 640]
  output_names: ["output0"]

validation:
  output_match: true            # 与原始模型输出对比通过
  max_diff: 1.2e-5
  onnx_checker: "PASS"

files:
  - path: "model.onnx"
    sha256: "xyz789..."
    size_mb: 49.8
```

### 10.6 Model Zoo 索引

`zoo/zoo_index.yaml` 是可用预训练模型的目录：

```yaml
# zoo_index.yaml — 自动维护，可通过 vw model pull <name> 更新
entries:
  - name: "yolov8n"
    task: "object_detection"
    framework: "ultralytics"
    source: "ultralytics"               # ultralytics | torchvision | huggingface | url
    url: "https://github.com/ultralytics/assets/releases/download/v8.0.0/yolov8n.pt"
    file: "yolov8n.pt"
    sha256: "abc123..."
    size_mb: 6.2
    input_shape: [3, 640, 640]
    num_classes: 80
    license: "AGPL-3.0"

  - name: "yolov8m"
    task: "object_detection"
    framework: "ultralytics"
    source: "ultralytics"
    file: "yolov8m.pt"
    sha256: "def456..."
    size_mb: 52.0

  - name: "mobilenet_v2"
    task: "classification"
    framework: "torch"
    source: "torchvision"
    file: "mobilenet_v2.pth"
    sha256: "ghi789..."
    input_shape: [3, 224, 224]
    num_classes: 1000

  - name: "resnet50"
    task: "classification"
    framework: "torch"
    source: "torchvision"
    file: "resnet50-imagenet.pth"

  - name: "faster_rcnn_resnet50"
    task: "object_detection"
    framework: "torch"
    source: "torchvision"
    file: "fasterrcnn_resnet50_fpn.pth"

  - name: "vit_base_patch16_224"
    task: "classification"
    framework: "huggingface"
    source: "huggingface"
    repo: "google/vit-base-patch16-224"
    file: "vit_base_patch16_224.pth"

  - name: "mediapipe_face"
    task: "face_detection"
    framework: "mediapipe"
    source: "mediapipe"
    file: "face_detection_short_range.tflite"
```

### 10.7 与 PipelineContext 的对应关系

PipelineContext 中的键映射到物理路径：

```python
# ContextKey 常量 → 实际物理路径映射
ContextKey.DATASET_DIR       → "vw_workspace/data/processed/<name>/"
ContextKey.ANNOTATIONS       → "vw_workspace/data/processed/<name>/annotations/"
ContextKey.CHECKPOINT_PATH   → "vw_workspace/models/checkpoints/<experiment>/<run>/best.pt"
ContextKey.OPTIMIZED_MODEL   → "vw_workspace/models/optimized/<experiment>/<strategy>/model_*.pt"
ContextKey.EXPORTED_MODELS   → {
    "onnx":    "vw_workspace/models/exports/<experiment>/onnx/model.onnx",
    "tensorrt": "vw_workspace/models/exports/<experiment>/tensorrt/model_fp16.engine",
    "tflite":  "vw_workspace/models/exports/<experiment>/tflite/model_int8.tflite",
    ...
}
ContextKey.ZOO_MODEL         → "vw_workspace/models/zoo/<model_name>.pt"
ContextKey.DEPLOY_PACKAGE    → "vw_workspace/deployments/<experiment>/<version>/"
```

### 10.8 模型相关的 CLI 命令

```bash
# 预训练模型管理
vw model pull yolov8n                    # 下载预训练模型到 zoo/
vw model pull --from huggingface vit_base_patch16_224
vw model list zoo                        # 列出所有缓存的预训练模型
vw model info yolov8n                    # 显示模型详细信息
vw model remove yolov8n                  # 从 zoo 中删除

# 模型注册与版本管理（对应 ModelRegistry）
vw model register checkpoints/traffic-detector-v3/run_xxx/best.pt \
    --name traffic-detector \
    --task object_detection \
    --framework ultralytics \
    --tags "production-ready"
vw model list registered                 # 列出已注册模型
vw model promote traffic-detector v1.0.0 --stage production
vw model compare traffic-detector v1.0.0 v1.1.0

# 导出管理
vw model export traffic-detector \
    --from optimized/traffic-detector-v3/quantized_int8/ \
    --to onnx tensorrt tflite
```

### 10.9 模型存储策略

| 策略 | 说明 |
|------|------|
| **预训练模型 (zoo/)** | 持久保存，不自动删除。可手动管理。类似 HuggingFace cache。 |
| **Checkpoint (checkpoints/)** | 每个实验保留 best.pt + last.pt + 每 N epoch 的存档。可配置 `keep_top_k` 自动淘汰旧实验。 |
| **优化模型 (optimized/)** | 保留所有优化策略的产出用于对比。可手动清理中间产物。 |
| **导出模型 (exports/)** | 按平台分别保存，直到部署成功后至少保留一个版本。 |
| **部署包 (deployments/)** | 作为最终交付物，支持版本化，不自动删除。 |

### 10.10 平台代码中的模型模块

与物理存储对应的代码模块（`src/vision_workbench/models/`）提供编程访问接口：

```python
from vision_workbench.models.zoo import ModelZoo
from vision_workbench.models.registry import ModelRegistry
from vision_workbench.models.card import ModelCard
from pathlib import Path

# Zoo: 管理预训练模型下载和缓存
zoo = ModelZoo(workspace=Path("vw_workspace"))
zoo.pull("yolov8n")                        # 下载 + 注册到 zoo_index
zoo.list()                                 # 列出所有已缓存的预训练模型
zoo.resolve("yolov8n")                     # 返回模型文件路径
zoo.pull_from_huggingface("google/vit-base-patch16-224")

# Registry: 管理训练产出的模型版本
registry = ModelRegistry(workspace=Path("vw_workspace"))
registry.register(
    name="traffic-detector",
    checkpoint="models/checkpoints/traffic-detector-v3/run_xxx/best.pt",
    metrics={"mAP": 0.782},
    tags={"dataset": "traffic-v1"},
)
registry.list_versions("traffic-detector")
registry.promote("traffic-detector", "v1.0.0", stage="production")
registry.get_production("traffic-detector")  # 获取生产环境模型路径

# Card: 自动生成模型文档
card = ModelCard()
md = card.generate(
    model_name="traffic-detector",
    checkpoint_path=Path("vw_workspace/models/checkpoints/traffic-detector-v3/run_xxx/"),
    task="object_detection",
    architecture="yolov8m",
    metrics={"mAP": 0.782, "F1": 0.85},
    dataset_info={"name": "traffic-v1", "samples": 8500},
    intended_use="Traffic surveillance object detection",
)
```

---

## 11. 自定义架构与算法开发平台

> **定位**：让平台不仅能"用"模型，更能"造"模型——支持修改现有架构、从零设计新架构、以及多模态融合算法的开发。

### 11.1 三种场景的覆盖方案

| 场景 | 当前支持 | 增强后方案 |
|------|:---:|-----------|
| 微调 YOLO / 调整检测头 | ✓ | 通过 `TrainStage` 原生支持 |
| 替换模型 Backbone（如 YOLO + MobileNetV3） | ✗ | 架构组件注册表 + `ModelBuilder` |
| 在 Neck 中插入 CBAM/SE 注意力 | ✗ | `models/modules/attention.py` 库 |
| 从零构建全新检测架构 | ✗ | `BaseArchitecture` + 组件组装 |
| RGB + 深度图双模态融合 | ✗ | `models/fusion/` 融合引擎 |
| FPN → BiFPN 一键切换 | ✗ | Neck 注册表 + 配置驱动 |

### 11.2 新增代码模块

```
src/vision_workbench/models/
│
├── architectures/                    # ===== 架构定义系统（新增） =====
│   ├── __init__.py
│   ├── base.py                      # BaseArchitecture, BaseBackbone, BaseNeck, BaseHead ABC
│   ├── builder.py                   # ModelBuilder — 配置驱动的模型组装引擎
│   ├── registry.py                  # architecture/backbone/neck/head/attention 注册表
│   ├── composer.py                  # ArchitectureComposer — 组件热替换/热插拔
│   └── custom/                      # 用户自定义架构存放目录
│       └── .gitkeep
│
├── modules/                         # ===== 可复用模块库（新增） =====
│   ├── __init__.py
│   ├── attention.py                # CBAM, SE, ECA, CA, MHSA, GAM, SimAM
│   ├── fusion.py                   # ConcatFusion, SumFusion, GatedFusion, CrossAttentionFusion
│   ├── normalization.py            # GroupNorm, LayerNorm, FilterResponseNorm 等
│   ├── activation.py               # SiLU, HardSwish, FReLU, GELU 变体
│   ├── convolution.py              # 深度可分离卷积、空洞卷积、Deformable Conv
│   └── blocks.py                   # InvertedResidual, CSPBlock, FPNBlock 等组合块
│
├── fusion/                          # ===== 融合算法引擎（新增） =====
│   ├── __init__.py
│   ├── base.py                     # BaseFusionModule, FusionStage 枚举
│   ├── early.py                    # 早融合（输入级多模态通道拼接）
│   ├── late.py                     # 晚融合（决策级投票/加权/NMS-Fusion）
│   ├── intermediate.py             # 中间融合（特征级 Concat/Sum/Cross-Attention）
│   ├── fpn.py                      # FPN 家族（FPN, PANet, BiFPN, NAS-FPN, AFPN）
│   ├── multimodal.py               # 跨模态融合（RGB-D, RGB-LiDAR）
│   └── config.py                   # FusionConfig Pydantic 模型
│
├── zoo.py                          # 已有：预训练模型缓存
├── registry.py                     # 已有：模型版本管理
├── card.py                         # 已有：模型卡片生成
└── comparator.py                   # 已有：模型对比查询
```

### 11.3 核心抽象：可组合的架构组件

```python
# models/architectures/base.py

class BaseBackbone(ABC):
    """特征提取骨干。
    输入: (B, C_in, H, W)  输出: dict[stage_name, Tensor] 如 {"s2": ..., "s3": ..., "s4": ...}"""
    @abstractmethod
    def forward(self, x: Tensor) -> dict[str, Tensor]: ...
    @abstractmethod
    def out_channels(self) -> dict[str, int]: ...

class BaseNeck(ABC):
    """特征金字塔/融合颈。
    输入: 骨干多尺度特征  输出: 增强多尺度特征"""
    @abstractmethod
    def forward(self, feats: dict[str, Tensor]) -> dict[str, Tensor]: ...
    @abstractmethod
    def out_channels(self) -> dict[str, int]: ...

class BaseHead(ABC):
    """任务头（检测/分类/分割/姿态）。
    输入: 增强特征  输出: 原始预测"""
    @abstractmethod
    def forward(self, feats: dict[str, Tensor]) -> Any: ...
    @abstractmethod
    def loss(self, preds: Any, targets: Any) -> dict[str, Tensor]: ...

class BaseArchitecture(ABC):
    """完整架构 = Backbone + Neck + Head (+ 可选 Attention 模块)"""
    backbone: BaseBackbone
    neck: BaseNeck
    head: BaseHead
    attention_modules: dict[str, nn.Module] = {}

    @classmethod
    @abstractmethod
    def from_config(cls, cfg: dict, registries: dict) -> "BaseArchitecture": ...
```

### 11.4 配置驱动的架构定义

**关键设计：配置支持两种模式——简单字符串（向后兼容）和嵌套组件定义（新能力）**

```yaml
# === 模式一：简单字符串（向后兼容，不改现有行为）===
train:
  model:
    architecture: "yolov8n"           # 直接使用框架原生模型

# === 模式二：组件级定义（替换 Backbone）===
train:
  model:
    architecture:
      type: "composite"              # 触发 ModelBuilder
      backbone:
        type: "mobilenet_v3_large"   # 从 backbone_registry 选取
        pretrained: "zoo/mobilenet_v3_large.pth"
        freeze_stages: [0, 1]        # 冻结浅层
      neck:
        type: "bifpn"                # 从 neck_registry 选取
        in_channels: [80, 160, 320]
        out_channels: 256
      head:
        type: "yolov8_detect"        # 从 head_registry 选取
        num_classes: 10
      attention:                      # 可选：插入注意力模块
        - target: "neck.bifpn.layers.2"
          type: "cbam"
          reduction: 16

# === 模式三：多模态融合架构 ===
train:
  model:
    architecture:
      type: "fusion"
      streams:
        rgb:
          backbone: { type: "resnet50", pretrained: true }
        depth:
          backbone: { type: "mobilenet_v2", pretrained: false, in_channels: 1 }
      fusion:
        stage: "intermediate"         # early | intermediate | late
        strategy: "cross_attention"   # concat | sum | gated | cross_attention
        fused_channels: 512
      head:
        type: "retinanet_head"
        num_classes: 10
```

### 11.5 模型构建器

```python
# models/architectures/builder.py

class ModelBuilder:
    """配置驱动的递归模型组装引擎"""

    def __init__(self, registries: dict):
        self.backbone_reg = registries["backbone"]
        self.neck_reg = registries["neck"]
        self.head_reg = registries["head"]
        self.attention_reg = registries["attention"]
        self.fusion_reg = registries["fusion"]

    def build(self, config) -> nn.Module:
        if isinstance(config, str):
            return self._build_native(config)     # 向后兼容：框架原生模型
        if config.type == "composite":
            return self._build_composite(config)  # 组件组装
        if config.type == "fusion":
            return self._build_fusion(config)     # 多模态融合
        raise ValueError(f"Unknown architecture type: {config.type}")

    def _build_composite(self, cfg) -> BaseArchitecture:
        backbone = self.backbone_reg.get(cfg.backbone.type)(**cfg.backbone.__dict__)
        neck = self.neck_reg.get(cfg.neck.type)(**cfg.neck.__dict__)
        head = self.head_reg.get(cfg.head.type)(**cfg.head.__dict__)
        model = BaseArchitecture(backbone, neck, head)
        for attn in cfg.get("attention", []):
            ArchitectureComposer.insert_attention(model, attn)
        return model
```

### 11.6 架构修改器

```python
# models/architectures/composer.py

class ArchitectureComposer:
    """对已有架构进行外科手术式修改"""

    @staticmethod
    def replace_backbone(model, new_name: str, **overrides): ...
    @staticmethod
    def replace_neck(model, new_name: str, **overrides): ...
    @staticmethod
    def replace_head(model, new_name: str, **overrides): ...
    @staticmethod
    def insert_attention(model, target_path: str, attn_type: str, **params): ...
    @staticmethod
    def add_modality_stream(model, name: str, backbone_cfg, neck_cfg): ...
    @staticmethod
    def freeze_component(model, component_path: str): ...
```

### 11.7 组件注册表

```python
# core/registry.py 中新增全局注册表实例
backbone_registry    = Registry("backbone")
neck_registry        = Registry("neck")
head_registry        = Registry("head")
attention_registry   = Registry("attention")
fusion_registry      = Registry("fusion")
architecture_registry = Registry("architecture")

# 注册示例
@backbone_registry.register("cspdarknet", source="ultralytics", task="detection")
class CSPDarknetBackbone(BaseBackbone): ...

@backbone_registry.register("mobilenet_v3_large", source="torchvision")
class MobileNetV3Backbone(BaseBackbone): ...

@neck_registry.register("fpn", source="torchvision")
class FPNNeck(BaseNeck): ...

@neck_registry.register("bifpn", source="custom")
class BiFPNNeck(BaseNeck): ...

@head_registry.register("yolov8_detect", source="ultralytics")
class YOLOv8DetectHead(BaseHead): ...

@attention_registry.register("cbam", source="custom")
class CBAM(nn.Module): ...
```

### 11.8 融合算法库

```python
# models/fusion/base.py

class FusionStage(str, Enum):
    EARLY = "early"               # 输入级：多模态通道拼接 → Backbone
    INTERMEDIATE = "intermediate" # 特征级：Backbone 后融合 → Neck
    LATE = "late"                 # 决策级：各模态独立推理 → 融合结果

class BaseFusionModule(ABC):
    @abstractmethod
    def forward(self, *modality_features: Tensor) -> Tensor: ...
```

| 融合策略 | 位置 | 适用场景 |
|---------|------|---------|
| ConcatFusion | `modules/fusion.py` | 通道拼接 + 1x1 卷积降维 |
| SumFusion | `modules/fusion.py` | 等通道数特征按元素相加 |
| GatedFusion | `modules/fusion.py` | 门控学习每个通道的保留权重 |
| CrossAttentionFusion | `modules/fusion.py` | 模态间互注意力对齐 |
| FPN / PANet / BiFPN / NAS-FPN | `fusion/fpn.py` | 特征金字塔：多尺度特征增强 |

### 11.9 多模态数据轻量扩展

```python
# core/types.py 新增

class ModalityType(str, Enum):
    RGB = "rgb"
    DEPTH = "depth"
    INFRARED = "infrared"

class MultiModalSample(BaseModel, frozen=True):
    modalities: dict[ModalityType, np.ndarray]  # {"rgb": (H,W,3), "depth": (H,W,1)}
    annotations: Optional[list] = None
```

```yaml
# 多模态数据集配置示例
data:
  source:
    type: "multimodal"
    modalities:
      rgb:  { path: "data/synced/rgb/",   transforms: ["imagenet_norm"] }
      depth: { path: "data/synced/depth/", transforms: ["clip_0_10m", "minmax_norm"] }
    alignment: "timestamp"       # timestamp | filename | calibration
```

### 11.10 与现有框架适配器的无缝集成

**关键：自定义架构产出标准 `nn.Module`，兼容所有现有框架适配器，无需修改任何适配器代码。**

```
TrainStage
    │
    ▼
TrainFrameworkAdapter.create_model(config)
    │
    ├─ config.architecture 是字符串（如 "yolov8n"）
    │   └─ 走原有路径：框架原生模型工厂（行为不变）
    │
    └─ config.architecture 是嵌套对象（有 type/backbone/neck/head）
        └─ 走新增路径：
            ModelBuilder.build(config.architecture)
                ├── 从各 registry 获取组件
                ├── 组装为 nn.Module（可训练对象）
                └── 传给框架适配器的 create_trainer() → 正常训练
```

**导出路径完全不受影响**：自定义架构 → `nn.Module` → ONNX → TensorRT/OpenVINO/TFLite（链路不变）

### 11.11 用户工作流示例

```bash
# 1. 用熟悉的 YAML 配置设计架构（无需写 Python）
vim configs/my_custom_detector.yaml
#   backbone: mobilenet_v3_large (从 zoo 获取预训练权重)
#   neck: bifpn
#   head: yolov8_detect (num_classes=10)
#   attention: cbam 插入到 neck 第2层

# 2. 直接训练，和用 YOLO 一样
vw run configs/my_custom_detector.yaml

# 3. 注册为新架构预设，后面反复引用
vw model register-architecture my_detector \
    --from-config configs/my_custom_detector.yaml

# 4. 下次实验只需一行
vw run --train.model.architecture my_detector \
    --train.training.epochs 300

# 5. 修改组件只需覆盖参数
vw run --train.model.architecture my_detector \
    --train.model.backbone.type efficientnet_b4 \
    --train.model.head.num_classes 20
```

### 11.12 与模型存储的对应

自定义架构的产物同样落入现有存储体系，无需额外目录：

| 产物 | 存放位置 |
|------|---------|
| 自定义架构配置 | `configs/<name>.yaml`（可版本控制） |
| 训练出的 Checkpoint | `vw_workspace/models/checkpoints/<name>/run_xxx/best.pt` |
| 优化/导出 | `vw_workspace/models/optimized/...` 和 `exports/...`（路径不变） |
| 注册为预设 | ModelRegistry 记录，可通过 `vw list architectures` 查询 |


---

## 12. 模型注册中心

### 12.1 模型注册表

```python
class ModelRegistry:
    """模型版本管理。

    功能：
    - 注册模型 → 版本递增
    - 阶段标记（staging → production → archived）
    - 模型对比查询
    """

    def register(
        self,
        name: str,
        checkpoint_path: Path,
        framework: str,
        task: str,
        metrics: dict,
        tags: dict = {},
    ) -> str:  # 返回 version_id
        ...

    def promote(self, name: str, version: str, stage: Literal["staging","production","archived"]) -> None: ...

    def get_production(self, name: str) -> dict: ...
    def list_versions(self, name: str) -> list[dict]: ...
    def compare(self, name: str, v1: str, v2: str) -> dict: ...
    def delete(self, name: str, version: str) -> None: ...
```

### 12.2 模型卡片

```python
class ModelCard:
    """自动生成模型文档"""

    def generate(
        self,
        model_name: str,
        checkpoint_path: Path,
        task: str,
        architecture: str,
        metrics: dict,
        dataset_info: dict,
        intended_use: str,
        limitations: str = "",
    ) -> str:  # 返回 Markdown 文档
        ...
```

---

## 13. 边缘部署体系

### 13.1 部署拓扑

```
                    Vision Workbench (工作站)
                           │
              ┌────────────┼────────────┐
              ▼            ▼            ▼
         SSH/SCP      HTTP API     MQTT OTA
              │            │            │
    ┌─────────▼──┐  ┌─────▼─────┐  ┌───▼──────────┐
    │ Jetson     │  │ x86 Edge  │  │ ARM/Embedded │
    │ TensorRT   │  │ OpenVINO  │  │ TFLite/RKNN  │
    └────────────┘  └───────────┘  └──────────────┘
```

### 13.2 设备管理

```yaml
# ~/.vision_workbench/devices.yaml — 设备注册表
devices:
  edge-camera-01:
    type: "raspberry_pi_5"
    address: "192.168.1.101"
    auth:
      method: "ssh_key"
      user: "pi"
      key_path: "~/.ssh/id_rsa_edge"
    specs:
      cpu: "ARM Cortex-A76"
      ram: "8GB"
      accelerator: null
    runtime: "tflite"

  jetson-orin-01:
    type: "jetson_orin_nx"
    address: "192.168.1.200"
    auth:
      method: "ssh_key"
      user: "nvidia"
      key_path: "~/.ssh/id_rsa_edge"
    specs:
      gpu: "1024-core NVIDIA Ampere"
      ram: "16GB"
      cuda: "11.4"
      tensorrt: "8.5"
      jp_version: "5.1.2"
    runtime: "tensorrt"

  edge-server-01:
    type: "x86_edge"
    address: "192.168.1.50"
    auth:
      method: "ssh_key"
      user: "admin"
    specs:
      cpu: "Intel Core i7-13700"
      ram: "32GB"
      accelerator: "Intel Iris Xe"
    runtime: "openvino"
```

---

## 14. CLI 与 API 设计

### 14.1 CLI 命令树

```
vw
├── run <config>                     # 执行流水线
│   ├── --stage <name>               #   仅执行指定阶段
│   ├── --from <name>                #   从指定阶段恢复
│   ├── --until <name>               #   执行到指定阶段
│   ├── --dry-run                    #   预览执行计划
│   ├── --resume <run_id>            #   恢复中断的运行
│   └── --parallel                  #   并行执行无依赖的阶段
│
├── detect <source>                  # 快速推理
│   ├── --detector <name>            #   选择检测器
│   ├── --show / --no-show           #   实时显示
│   ├── --save <dir>                 #   保存结果
│   └── --format <json|image|video>  #   输出格式
│
├── data                             # 数据集管理
│   ├── status <path>                #   数据集统计分析
│   ├── clean <src> <dst>            #   数据清洗
│   │   └── --config <yaml>
│   ├── split <dataset>              #   数据集划分
│   ├── convert <src> <dst>          #   标注格式转换
│   │   └── --from <fmt> --to <fmt>
│   ├── download <dataset_name>      #   下载公开数据集
│   └── version                      #   数据版本管理
│       ├── snapshot <path>          #   创建版本快照
│       ├── list <path>              #   版本历史
│       └── diff <path> <v1> <v2>    #   版本差异
│
├── model                            # 模型管理
│   ├── list                         #   列出已注册模型
│   ├── register <checkpoint>        #   注册模型
│   ├── promote <name> <version> <stage>  # 变更模型阶段
│   ├── compare <name> <v1> <v2>     #   模型对比
│   └── card <name> <version>        #   生成模型卡片
│
├── list                             # 列出注册项
│   ├── detectors [--task <t>]       #   检测器
│   ├── formats                      #   标注格式
│   ├── frameworks [--task <t>]      #   训练框架
│   ├── platforms                    #   导出平台
│   ├── templates                    #   流水线模板
│   └── devices                      #   已注册边缘设备
│
├── export <checkpoint>              # 单步导出
│   ├── --to <platform>              #   目标平台
│   └── --config <yaml>
│
└── serve                            # 启动服务
    ├── --host <addr>
    ├── --port <int>
    ├── --model <checkpoint>
    └── --platform <onnx|tensorrt|...>
```

### 14.2 Python API（编程式调用）

```python
from vision_workbench.core.config import PipelineConfig
from vision_workbench.core.orchestrator import PipelineOrchestrator
from vision_workbench.data.catalog import DatasetCatalog

# 完整流水线
config = PipelineConfig.from_yaml("configs/my_experiment.yaml")
orchestrator = PipelineOrchestrator(config)
ctx = orchestrator.run()

# 编程式阶段调用
from vision_workbench.pipeline.data.stage import DataStage
stage = DataStage()
ok, missing = stage.validate_inputs(ctx)

# 单步检测
from vision_workbench.detectors.yolo.yolo_ import YOLODetector
detector = YOLODetector()
detector.initialize(model="yolov8n.pt")
result = detector.process(image)
```

---

## 15. 可观测性体系

### 15.1 日志规范

```python
# 使用 structlog 结构化日志
import structlog
logger = structlog.get_logger()

# 每个阶段入口/出口
logger.info("stage.start", stage="data", config_hash="abc123")
logger.info("stage.complete", stage="data", duration_ms=1234, artifacts=3)

# 关键操作
logger.info("dataset.split", train=700, val=150, test=150, method="stratified")
logger.warning("data.validation", corrupted_files=5, truncated=2)
logger.error("export.failed", platform="tensorrt", error="ONNX opset mismatch")
```

### 15.2 指标采集

自动采集的指标：
- **系统**：CPU/GPU 使用率、内存、磁盘 I/O
- **流水线**：各阶段耗时、吞吐量
- **模型**：参数量、FLOPs、推理延迟（p50/p95/p99）、显存占用
- **数据**：图片数量、标注框数量、类别分布

---

## 16. 测试策略

### 16.1 测试分层

| 层级 | 范围 | 执行时间 | 覆盖目标 |
|------|------|---------|---------|
| **Unit** | 单个函数/类 | < 1min | 核心逻辑、边界条件 |
| **Integration** | 单阶段端到端 | < 5min | 阶段输入→输出正确性 |
| **E2E** | 完整流水线 | < 30min | 多阶段串联、context 传递 |
| **Smoke** | 安装+CLI | < 2min | 安装可用、命令可达 |

### 16.2 测试基础设施

```python
# conftest.py 提供的关键 fixtures

@pytest.fixture
def sample_image() -> np.ndarray:
    """640x480 BGR 测试图（含简单几何形状）"""
    ...

@pytest.fixture
def mini_coco_dataset(tmp_path) -> Path:
    """5 张图 + COCO 标注的微型数据集"""
    ...

@pytest.fixture
def mock_detector() -> BaseDetector:
    """返回固定 DetectionResult 的 Mock 检测器"""
    ...

@pytest.fixture
def empty_context() -> PipelineContext:
    """空白 PipelineContext"""
    ...

@pytest.fixture
def mock_onnx_model(tmp_path) -> Path:
    """生成一个最小的合法 ONNX 模型文件"""
    ...
```

### 16.3 可选依赖测试处理

```python
# 利用 pytest skipif 处理可选依赖
@pytest.mark.skipif(
    not importlib.util.find_spec("ultralytics"),
    reason="ultralytics not installed"
)
def test_yolo_detector(): ...

# 利用 marker 分组
# pytest -m "not slow and not gpu"  快速测试
# pytest -m "gpu"                    需要 GPU 的测试
```

---

## 17. 安全与合规

- **模型签名**：导出模型附带数字签名，防止篡改
- **敏感数据脱敏**：可视化输出可配置模糊人脸/车牌
- **SSH 密钥管理**：部署阶段使用 ssh-agent 或密钥文件，禁止硬编码密码
- **模型加密**：可选 AES 加密导出模型，边缘端解密加载

---

## 18. 依赖管理策略

```toml
[project]
name = "vision-workbench"
version = "0.1.0"
description = "End-to-end Computer Vision MLOps Workbench"
requires-python = ">=3.12"

dependencies = [
    # 运行时核心（总是安装，保持轻量）
    "numpy>=1.26,<3.0",
    "opencv-python>=4.9,<5.0",
    "PyYAML>=6.0",
    "pydantic>=2.0,<3.0",
    "typer>=0.12,<1.0",
    "Pillow>=10.0,<12.0",
    "structlog>=24.0",          # 结构化日志
    "rich>=13.0",               # CLI 美化输出
    "omegaconf>=2.3",           # 配置合并（支持 _base_ 继承）

    # 数据
    "albumentations>=1.4",
    "scikit-learn>=1.4",
    "imagehash>=4.3",           # 图片去重
    "pandas>=2.0",

    # 可视化
    "matplotlib>=3.8",
    "seaborn>=0.13",
]

[project.optional-dependencies]
# --- 训练框架 ---
torch       = ["torch>=2.0", "torchvision>=0.15", "pytorch-lightning>=2.0"]
yolo        = ["ultralytics>=8.0"]
mmdet       = ["mmdet>=3.0", "mmengine>=0.10", "mmcv>=2.0"]
huggingface = ["transformers>=4.36", "timm>=0.9", "datasets>=2.0"]

# --- 媒体处理 ---
mediapipe   = ["mediapipe>=0.10"]

# --- 实验追踪 ---
mlflow      = ["mlflow>=2.10"]
wandb       = ["wandb>=0.16"]

# --- 优化/导出（ONNX 生态） ---
optimize    = ["onnx>=1.15", "onnxruntime>=1.17", "onnx-simplifier>=0.4"]
tensorrt    = ["onnx-tensorrt", "cuda-python"]
openvino    = ["openvino>=2024.0"]
tflite      = ["tensorflow>=2.15"]      # 仅转换，不训练
coremltools = ["coremltools>=7.0"]

# --- 超参搜索 ---
hpo         = ["optuna>=3.0"]

# --- 数据版本 ---
dvc         = ["dvc>=3.0"]

# --- 服务 ---
serve       = ["fastapi>=0.110", "uvicorn>=0.27", "gradio>=4.0", "prometheus-client>=0.19"]

# --- 开发 ---
dev         = ["pytest>=8.0", "pytest-cov>=4.0", "pytest-xdist>=3.0",
               "ruff>=0.1", "mypy>=1.7", "pre-commit>=3.0"]

# --- 完整安装 ---
all = ["vision-workbench[torch,yolo,mmdet,huggingface,mediapipe,mlflow,optimize,serve,hpo,dvc]"]
```

---

## 19. 实现路线图

| 阶段 | 内容 | 里程碑验证 |
|------|------|-----------|
| **Phase 1** | 项目骨架 + 核心基础层 | `pip install -e .` + `vw --help` |
| **Phase 2** | PipelineContext + Orchestrator + DAG 调度 | DAG 拓扑排序测试通过 |
| **Phase 3** | Data 管理 + Data Stage | 数据集清洗/划分/分析全链路 |
| **Phase 4** | Annotation Stage + 格式转换矩阵 | COCO↔YOLO↔VOC 双向转换 |
| **Phase 5** | Train Stage + PyTorch/Ultralytics Adapter | 配置驱动的单卡训练 |
| **Phase 6** | Validate + Evaluate Stage | 指标计算 + 对比报告生成 |
| **Phase 7** | Optimize Stage（量化+剪枝） | PTQ INT8 量化 + L1 剪枝 |
| **Phase 8** | Export Stage（ONNX + 多平台） | ONNX→TensorRT/OpenVINO/TFLite |
| **Phase 9** | Deploy Stage + 推理微服务 | SSH 推送 + FastAPI 服务 |
| **Phase 10** | CLI 全命令 + Web UI + Notebooks | `vw run full_pipeline.yaml` 端到端 |
| **Phase 11** | 实验追踪集成 + 模型注册中心 | MLflow/W&B + ModelRegistry |
| **Phase 12** | 边缘设备管理 + 监控 + OTA | 边缘基准测试 + 漂移检测 |

---

## 20. 风险与缓解

| 风险 | 影响 | 缓解措施 |
|------|------|---------|
| ONNX 算子兼容性不足 | 导出失败 | 自定义算子注册机制 + 算子兼容性预检查 |
| 框架版本 API 不稳定 | 适配器失效 | 每个框架适配器独立测试 + 版本 pin |
| 边缘设备异构性 | 部署困难 | ONNX 统一 IR + 设备能力注册表 |
| 大规模数据集处理 | 内存溢出 | 流式处理 + 分片 + 进度持久化 |
| 训练/量化精度下降超出预期 | 模型不可用 | 自动精度阈值检查 + 回滚机制 |
| 依赖冲突（torch/onnx/tf） | 安装失败 | 全部 ML 依赖可选分组 + 严格版本约束 |

---

## 附录 A：内部统一标注格式（COCO 扩展）

```json
{
  "info": {
    "description": "vision-workbench internal format",
    "version": "1.0",
    "year": 2024,
    "contributor": "",
    "date_created": "2024-01-01T00:00:00"
  },
  "licenses": [],
  "images": [
    {
      "id": 1,
      "file_name": "images/train/000001.jpg",
      "width": 1920,
      "height": 1080,
      "md5": "abc123..."
    }
  ],
  "annotations": [
    {
      "id": 1,
      "image_id": 1,
      "category_id": 1,
      "bbox": [100.0, 200.0, 50.0, 80.0],
      "area": 4000.0,
      "segmentation": [[100,200,150,200,150,280,100,280]],
      "iscrowd": 0,
      "attributes": {
        "occluded": false,
        "truncated": false,
        "difficult": false
      }
    }
  ],
  "categories": [
    {
      "id": 1,
      "name": "pedestrian",
      "supercategory": "person"
    }
  ]
}
```

## 附录 B：阶段间数据流图

```
[Data Stage]
  produces: artifacts.dataset_dir, metadata.data_provenance
      │
      ▼
[Annotate Stage]
  produces: artifacts.annotations, metadata.class_distribution
      │
      ▼
[Train Stage]
  produces: artifacts.checkpoint_path, metrics.train_loss, metrics.val_loss
      │
      ├──────────────────┐
      ▼                  ▼
[Validate Stage]   [Evaluate Stage]
  produces:            produces:
  metrics.mAP,         artifacts.evaluation_report,
  metrics.F1           artifacts.comparison_grid
      │
      ▼
[Optimize Stage]
  produces: artifacts.optimized_model, metrics.optimized_accuracy
      │
      ▼
[Export Stage]
  produces: artifacts.exports = {
      "onnx": Path,
      "tensorrt": Path,
      "tflite": Path, ...
  }
      │
      ▼
[Deploy Stage]
  produces: metadata.deploy_status, metadata.benchmark_results
```
