# Vision Workbench — 项目架构定义

> 本文档定义项目的完整文件结构、模块职责、依赖关系和实现状态。
> 架构设计理念参见 `docs/ARCHITECTURE.md`。

---

## 目录

1. [项目概览](#1-项目概览)
2. [根目录文件](#2-根目录文件)
3. [核心层 `core/`](#3-核心层-core)
4. [数据层 `data/`](#4-数据层-data)
5. [流水线 `pipeline/`](#5-流水线-pipeline)
6. [检测器 `detectors/`](#6-检测器-detectors)
7. [模型管理 `models/`](#7-模型管理-models)
8. [可视化 `viz/`](#8-可视化-viz)
9. [推理服务 `serve/`](#9-推理服务-serve)
10. [CLI 命令 `cli/`](#10-cli-命令-cli)
11. [实验追踪 `tracking/`](#11-实验追踪-tracking)
12. [测试体系 `tests/`](#12-测试体系-tests)
13. [配置与模板 `configs/` `templates/`](#13-配置与模板)
14. [模块依赖关系图](#14-模块依赖关系图)
15. [实现状态矩阵](#15-实现状态矩阵)
16. [扩展开发指南](#16-扩展开发指南)

---

## 1. 项目概览

```
vision-workbench/                    # 项目根目录
├── src/vision_workbench/            # 主包（56 个 .py 文件）
│   ├── core/          (8 files)     # 核心抽象层 — 平台基石
│   ├── pipeline/      (8 files)     # 8 大流水线阶段 — 业务逻辑核心
│   ├── detectors/     (2 files)     # 预训练检测器 — 快速推理
│   ├── models/        (1 file)      # 模型管理 — Zoo + 注册 + 架构
│   ├── data/          (2 files)     # 数据管理 — 目录约定 + 索引
│   ├── viz/           (1 file)      # 可视化 — 标注绘制
│   ├── serve/         (2 files)     # 推理服务 — FastAPI + Gradio
│   ├── cli/           (1 file)      # CLI — Typer 命令入口
│   └── tracking/      (1 file)      # 实验追踪 — MLflow/W&B 集成
├── tests/             (4 files)     # 测试套件 — 26 个用例
├── configs/           (2 files)     # 流水线配置示例
├── notebooks/         (1 file)      # Jupyter 教程
├── docs/              (2 files)     # 设计文档
└── templates/                       # 流水线模板（预留）
```

**指标**：56 个 Python 源文件 | 26 个测试用例 | 77 个 Git 文件 | Python 3.12+

---

## 2. 根目录文件

| 文件 | 作用 |
|------|------|
| `pyproject.toml` | 项目元数据、15 个核心依赖、18 个可选依赖组、CLI 入口点（`vw` / `vision-workbench`）、pytest/ruff/mypy 配置 |
| `main.py` | 便捷入口：`python main.py` 等价于 `vw`，内部直接调用 `cli.app:app` |
| `README.md` | 项目简介、快速开始、命令列表、项目结构概览 |
| `LICENSE` | MIT 许可证 |
| `.gitignore` | 忽略 `__pycache__/`、`.venv/`、`vw_workspace/`、`.pytest_cache/` 等 |
| `.python-version` | Python 3.12 版本锁定（pyenv 兼容） |
| `vision-workbench.code-workspace` | VS Code 工作区配置（120 字符标尺） |

---

## 3. 核心层 `core/`

> **依赖层级**：Level 0 — 无项目内依赖，被所有其他模块依赖。

```
src/vision_workbench/core/
├── __init__.py              # 公共 API 导出（BaseDetector, BaseStage, Registry, PipelineConfig 等）
├── types.py                 # 领域原语类型（Pydantic frozen models）
├── base.py                  # 抽象基类（ABC）
├── result.py                # 统一结果类型
├── registry.py              # 泛型注册表 + 10 个全局实例
├── config.py                # 流水线配置模型（Pydantic v2）
├── context.py               # 不可变流水线上下文
├── orchestrator.py          # 流水线编排器
└── exceptions.py            # 分层异常体系（20+ 类）
```

### 3.1 `types.py` — 领域原语

| 类 | 作用 | 实现 |
|----|------|:----:|
| `TaskType` (StrEnum) | 9 种视觉任务枚举：object_detection, classification, segmentation, face_detection, pose_estimation, hand_tracking, ocr, feature_matching | ✅ |
| `BoundingBox` (frozen) | 归一化/像素坐标边界框：x1, y1, x2, y2, confidence, class_id, class_name | ✅ |
| `Keypoint` (frozen) | 关键点：x, y, visibility (visible/occluded/not_present), confidence, name | ✅ |
| `SegmentationMask` | 分割掩码引用：mask_path, format (png/rle/polygon), height, width, class_id | ✅ |
| `ImageMetadata` (frozen) | 图片元数据：path, width, height, channels, format, file_size_bytes, md5_hash, exif | ✅ |
| `DatasetSplit` (frozen) | 数据集划分记录：train/val/test 路径列表, split_method, random_seed, timestamp | ✅ |
| `ModalityType` (StrEnum) | 多模态输入类型：rgb, depth, infrared, lidar_projection | ✅ |
| `MultiModalSample` | 多模态数据样本：modalities dict + annotations + metadata | ✅ |

### 3.2 `base.py` — 抽象基类

| 类 | 作用 | 关键方法 | 实现 |
|----|------|---------|:----:|
| `BaseStage[TConfig, TResult]` | 流水线阶段基类（泛型） | `validate_inputs(ctx)` → `(bool, list[str])`; `run(config, ctx)` → `(TResult, PipelineContext)`; `dry_run(config, ctx)` → `dict` | ✅ |
| `BaseDetector` | 视觉检测器基类 | `initialize(**params)`; `process(image: np.ndarray)` → `DetectionResult`; `cleanup()` | ✅ |
| `BaseExporter` | 模型导出器基类 | `export(model, output_path, input_shape)`; `validate(original, exported, sample)` | ✅ |

### 3.3 `result.py` — 统一结果

| 类 | 作用 | 实现 |
|----|------|:----:|
| `DetectionResult` | 检测器输出：boxes, keypoints, masks, classifications, text + 来源溯源（detector_name, framework, task_type, processing_time_ms） | ✅ |
| `StageResult` | 流水线阶段产出：stage_name, success, error_message, duration_seconds, artifacts dict, metrics dict | ✅ |

### 3.4 `registry.py` — 注册表系统

| 组件 | 作用 | 实现 |
|------|------|:----:|
| `Registry[T]` (泛型类) | 泛型注册表：`register(name, **metadata)` 装饰器, `get(name)`, `list()`, `list_by(**filters)`, `discover_entry_points(group)` | ✅ |
| `detector_registry` | 检测器注册表实例 | ✅ |
| `format_registry` | 标注格式转换器注册表 | ✅ |
| `framework_registry` | 训练框架适配器注册表 | ✅ |
| `exporter_registry` | 模型导出器注册表 | ✅ |
| `backbone_registry` | Backbone 组件注册表 | ✅ |
| `neck_registry` | Neck 组件注册表 | ✅ |
| `head_registry` | Head 组件注册表 | ✅ |
| `attention_registry` | 注意力模块注册表 | ✅ |
| `fusion_registry` | 融合模块注册表 | ✅ |
| `architecture_registry` | 完整架构注册表 | ✅ |

**设计亮点**：所有 10 个全局注册表共享同一泛型实现，装饰器注册 + entry_points 插件发现。

### 3.5 `config.py` — 配置模型

| 类 | 作用 | 实现 |
|----|------|:----:|
| `RuntimeConfig` | 全局运行时：workspace, artifacts_dir, device, seed, log_level, cache | ✅ |
| `DataStageConfig` | 阶段①数据配置：source, target, validation, dedup, split, augment | ✅ |
| `AnnotateStageConfig` | 阶段②标注配置：input/output format, source_path, pre_annotation, quality_checks | ✅ |
| `TrainStageConfig` | 阶段③训练配置：framework, task, model, data, training, callbacks | ✅ |
| `ValidateStageConfig` | 阶段④验证配置：batch_size, metrics, regression_test | ✅ |
| `EvaluateStageConfig` | 阶段⑤评估配置：curves, profiling | ✅ |
| `OptimizeStageConfig` | 阶段⑥优化配置：methods (quantize/prune/distill) | ✅ |
| `ExportStageConfig` | 阶段⑦导出配置：onnx opset, targets (tensorrt/openvino/tflite/coreml/rknn) | ✅ |
| `DeployStageConfig` | 阶段⑧部署配置：devices (name/address/auth) | ✅ |
| `PipelineConfig` | 根配置：name, runtime, stages 列表 + 8 个子配置；`from_yaml(path)` 工厂方法 | ✅ |

### 3.6 `context.py` — 流程上下文

| 组件 | 作用 | 实现 |
|------|------|:----:|
| `ContextKey` | 标准化键名常量：`DATASET_DIR`, `CHECKPOINT_PATH`, `EXPORTED_MODELS`, `VAL_METRICS` 等 | ✅ |
| `PipelineContext` (frozen) | 不可变上下文：artifacts, metrics, metadata, stage_history; `evolve(**updates)` 创建新快照; `get(key, default)` 点号路径读取; `record_stage(name, success)` 审计记录; version/parent_version 版本链 | ✅ |

### 3.7 `orchestrator.py` — 流水线编排

| 组件 | 作用 | 实现 |
|------|------|:----:|
| `PipelineOrchestrator` | 执行引擎：`register_stage(stage)`, `resolve_stages()`, `run(ctx)` 顺序执行, `run_from(stage, ctx)` 断点恢复, `dry_run()` 预览计划 | ✅ |

### 3.8 `exceptions.py` — 异常体系

```
VisionWorkbenchError          # 根基类
├── ConfigError               # 配置类（ValidationError, MissingKeyError）
├── RegistryError             # 注册表查找失败
├── DependencyError           # 依赖类（MissingDependency, VersionConflict）
├── PipelineError             # 流水线类（StageInputError, StageExecutionError, StageNotFoundError）
├── DataError                 # 数据类（DataValidationError, AnnotationFormatError）
├── ModelError                # 模型类（ModelNotFoundError, ExportError, InferenceError）
└── DeployError               # 部署类（ConnectionError, BenchmarkError）
```

---

## 4. 数据层 `data/`

> **依赖**：`core/types.py`, `core/exceptions.py`

```
src/vision_workbench/data/
├── __init__.py              # 导出 DatasetCatalog, DatasetSchema
├── catalog.py               # 数据集目录索引
└── schema.py                # 数据集标准结构定义
```

### 4.1 `schema.py`

| 组件 | 作用 | 实现 |
|------|------|:----:|
| `CategoryInfo` | 类别定义：id, name, supercategory | ✅ |
| `ImageStats` | 图像聚合统计：avg_width, avg_height, total_size_gb | ✅ |
| `AnnotationStats` | 标注聚合统计：total_boxes, boxes_per_image_avg, class_distribution | ✅ |
| `ProvenanceInfo` | 数据溯源：source, collection_location, collection_period, preprocessing | ✅ |
| `DatasetManifest` | `dataset.yaml` 完整内容模型：name, version, tasks, categories, image_count, stats, provenance | ✅ |
| `DatasetSchema` | 标准目录约定：`validate(root)`, `create(root)` 按规范创建目录结构 | ✅ |

**标准数据集布局**：
```
<dataset_root>/
├── dataset.yaml
├── images/{train,val,test}/
├── annotations/
└── splits/
```

### 4.2 `catalog.py`

| 组件 | 作用 | 实现 |
|------|------|:----:|
| `DatasetCatalog` | 数据集注册/查找/扫描/删除：`register(name, path)`, `get(name)`, `list()`, `load_manifest(name)`, `scan(directory)`, `remove(name)` | ✅ |

---

## 5. 流水线 `pipeline/`

> **依赖**：`core/base.py`, `core/config.py`, `core/context.py`, `core/result.py`, `data/schema.py`

### 5.1 阶段总览

| # | 阶段 | 文件 | 职责 | 输入(从ctx) | 输出(写入ctx) | 实现 |
|---|------|------|------|------------|--------------|:----:|
| ① | Data | `data/stage.py` | 图片发现→验证→去重→质量过滤→划分→复制→生成manifest | `(config.source)` | `artifacts.dataset_dir` | ✅ |
| ② | Annotate | `annotate/stage.py` | 标注格式转换→质量检查 | `artifacts.dataset_dir` | `artifacts.annotations` | ✅ |
| ③ | Train | `train/stage.py` | 框架适配器解析→训练→checkpoint保存 | `artifacts.dataset_dir` | `artifacts.checkpoint_path` | ✅ |
| ④ | Validate | `validate/stage.py` | 批量推理→指标计算→回归测试→指标JSON | `artifacts.checkpoint_path` | `metrics.validation` | ✅ |
| ⑤ | Evaluate | `evaluate/stage.py` | 多模型对比→曲线→剖析→报告生成(MD) | `metrics.validation` | `artifacts.evaluation_report` | ✅ |
| ⑥ | Optimize | `optimize/stage.py` | 量化(INT8/FP16)→剪枝(L1/L2)→蒸馏→产出优化模型 | `artifacts.checkpoint_path` | `artifacts.optimized_model` | ✅ |
| ⑦ | Export | `export/stage.py` | PyTorch→ONNX→TensorRT/OpenVINO/TFLite/CoreML/RKNN | `artifacts.optimized_model` | `artifacts.exports` | ✅ |
| ⑧ | Deploy | `deploy/stage.py` | SSH/HTTP/MQTT推送→边缘基准测试→部署状态报告 | `artifacts.exports` | `metadata.deploy_status` | ✅ |

### 5.2 各阶段详细模块

#### ① DataStage (`pipeline/data/stage.py`)

```
DataStage(BaseStage)
├── validate_inputs(ctx) → (True, [])           # 自包含，不检查ctx
├── run(config, ctx) → (StageResult, ctx)        # 完整数据准备流程
├── dry_run(config, ctx) → dict                  # 预览
├── _discover_images(root, formats) → list[Path] # 递归发现图片
├── _validate(images, cfg) → (valid, issues)     # 分辨率/格式检查
├── _deduplicate(images, cfg) → (unique, count)  # pHash 去重
├── _quality_filter(images, cfg) → (valid, n)    # Laplacian 模糊检测
├── _split(images, cfg) → {train, val, test}     # 随机/分层划分
├── _copy_images(dst, splits)                    # 复制到标准目录
└── _build_manifest(cfg, total, splits)          # 生成 dataset.yaml
```

#### ③ TrainStage (`pipeline/train/stage.py`)

```
TrainStage(BaseStage)
├── depends_on = ["data", "annotate"]
├── validate_inputs(ctx) → (bool, [])            # 检查 DATASET_DIR
├── run(config, ctx) → (StageResult, ctx)        # 解析框架适配器→训练
└── dry_run(config, ctx) → dict
```

**预留扩展**：`pipeline/train/adapters/` — 各框架训练适配器（torch, mmdet, ultralytics, huggingface）

### 5.3 流水线 `__init__.py` — 自动发现

`discover_stages()` 函数自动导入 8 个阶段模块并返回 `name → class` 映射。

---

## 6. 检测器 `detectors/`

> **依赖**：`core/base.py`, `core/registry.py`, `core/result.py`

```
src/vision_workbench/detectors/
├── __init__.py              # 空文件
├── base.py                  # BaseDetector ABC + discover_detectors()
├── opencv/
│   ├── __init__.py          # 空文件
│   └── haar_face.py         # ✅ Haar Cascade 人脸检测（内置，零额外依赖）
├── yolo/
│   └── __init__.py          # 空（预留 Ultralytics YOLO）
├── mediapipe/
│   └── __init__.py          # 空（预留 MediaPipe 检测器）
└── huggingface/
    └── __init__.py          # 空（预留 HuggingFace 检测器）
```

### 6.1 `opencv/haar_face.py`

| 组件 | 作用 | 实现 |
|------|------|:----:|
| `HaarFaceDetector` | OpenCV Haar Cascade 人脸检测：`initialize(cascade_path?)` → `process(image)` → `cleanup()`；通过 `@detector_registry.register("opencv_haar_face")` 自动注册 | ✅ |

### 6.2 `base.py`

`discover_detectors()` — 使用 `pkgutil.iter_modules` 自动导入所有检测器子包，触发 `@register` 装饰器注册。

---

## 7. 模型管理 `models/`

> **依赖**：`core/registry.py`

```
src/vision_workbench/models/
├── __init__.py              # 空文件
├── zoo.py                   # ✅ Model Zoo — 预训练模型下载与缓存
├── architectures/
│   ├── __init__.py          # 空文件
│   └── custom/.gitkeep      # 用户自定义架构目录
├── modules/
│   └── __init__.py          # 空（预留注意力/卷积/融合模块库）
└── fusion/
    └── __init__.py          # 空（预留多模态融合引擎）
```

### 7.1 `zoo.py`

| 组件 | 作用 | 实现 |
|------|------|:----:|
| `KNOWN_MODELS` | 已知预训练模型字典：yolov8n, yolov8m, yolov8x (含 url, task, framework, size_mb, input_shape) | ✅ |
| `ModelZoo` | 模型缓存管理：`list()`, `resolve(name)`, `pull(name)`, `info(name)`, `remove(name)`；下载后自动更新 `zoo_index.yaml` | ✅ |

---

## 8. 可视化 `viz/`

> **依赖**：`core/result.py`, `core/types.py`

```
src/vision_workbench/viz/
├── __init__.py              # 空文件
└── annotate.py              # ✅ 标注绘制引擎
```

### 8.1 `annotate.py`

| 函数 | 作用 | 实现 |
|------|------|:----:|
| `annotate(image, result)` | 核心绘制：边界框（含标签+置信度）→ 关键点（圆点）→ 骨骼连接线（pose/hand）→ 分割掩码（半透明叠加）→ 返回 BGR 图像 | ✅ |
| `draw_comparison_grid(images, titles, columns)` | 多检测器并排对比网格：自动排列、居中标题 | ✅ |
| `DEFAULT_COLORS` | 20 色预设调色板 | ✅ |
| `SKELETON_CONNECTIONS` | 姿态/手部骨骼连接定义 | ✅ |

---

## 9. 推理服务 `serve/`

> **依赖**：`detectors/`, `viz/`, `core/registry.py`

```
src/vision_workbench/serve/
├── __init__.py              # 空文件
├── app.py                   # ✅ FastAPI 推理微服务
└── gradio_ui.py             # ✅ Gradio Web UI
```

### 9.1 `app.py`

| 端点 | 方法 | 作用 | 实现 |
|------|------|------|:----:|
| `/v1/health` | GET | 健康检查：返回 status + model_version | ✅ |
| `/metrics` | GET | Prometheus 指标端点 | ✅ |
| `/v1/detect` | POST | 目标检测：上传图片 → 返回 JSON 结果 | ✅ |
| `/v1/classify` | POST | 图像分类：上传图片 → 返回分类预测 | ✅ |

### 9.2 `gradio_ui.py`

| 组件 | 作用 | 实现 |
|------|------|:----:|
| `create_ui()` | 构建 Gradio Blocks：Detection Tab（图片上传 + 检测器选择 + 结果展示）, Pipeline Tab（配置上传 + 运行）, Model Zoo Tab（预训练模型浏览表） | ✅ |
| `run_detection(image, detector_name)` | 执行检测并返回标注图片 + JSON | ✅ |
| `launch(host, port)` | 启动 Web UI 服务 | ✅ |

---

## 10. CLI 命令 `cli/`

> **依赖**：`core/`, `models/zoo.py`, `serve/`

```
src/vision_workbench/cli/
├── __init__.py              # 空文件
└── app.py                   # ✅ Typer 应用 — 6 个命令
```

### 10.1 命令矩阵

| 命令 | 参数 | 功能 | 实现 |
|------|------|------|:----:|
| `vw run` | `<config>` `--stage` `--dry-run` `--resume` | 执行流水线：加载 YAML → PipelineConfig 验证 → Orchestrator 调度 | ✅ |
| `vw list` | `detectors\|tasks\|formats\|frameworks\|platforms` `--task` | 列出已注册组件（Rich Table 渲染） | ✅ |
| `vw detect` | `<source>` `--detector` `--show` `--save` `--format` | 快速单次推理（图/视频/摄像头） | ✅ |
| `vw data` | `status\|clean\|split\|convert` `--source` `--target` | 数据集管理 | ✅ |
| `vw model` | `list\|pull\|info\|register` `--name` `--checkpoint` | 模型 Zoo：下载/查询/注册预训练模型 | ✅ |
| `vw serve` | `--host` `--port` `--model` `--api` | 启动 Gradio UI（`--api` 仅启动 FastAPI） | ✅ |

---

## 11. 实验追踪 `tracking/`

```
src/vision_workbench/tracking/
└── __init__.py              # 空（预留 MLflow / W&B 集成适配器）
```

---

## 12. 测试体系 `tests/`

> 26 个用例 | 全部通过 | pytest + fixtures

```
tests/
├── conftest.py              # 共享 fixtures：sample_image(640x480 BGR), empty_context, tmp_workspace
├── __init__.py              # 空文件
├── test_core/
│   ├── __init__.py
│   ├── test_context.py      # 9 个用例：PipelineContext 初始状态/evolve/嵌套读取/默认值/审计/版本链/ContextKey
│   └── test_registry.py     # 8 个用例：注册/查重/未找到异常/列表/元数据过滤/装饰器/包含/长度
├── test_detectors/
│   ├── __init__.py
│   └── test_haar_face.py    # 4 个用例：初始化/空检测/注册表注册/未初始化异常
├── test_pipeline/
│   ├── __init__.py
│   └── test_pipeline_integration.py  # 5 个用例：阶段发现/数据阶段端到端/编排器/上下文往返/配置YAML加载
├── test_viz/
│   └── __init__.py          # 预留
└── fixtures/
    ├── mini_coco/           # 微型 COCO 数据集（空目录）
    ├── sample_images/       # 测试图片
    └── mock_models/         # 模拟模型文件
```

### 12.1 测试覆盖矩阵

| 模块 | 用例数 | 覆盖类型 |
|------|:------:|---------|
| `core/context.py` | 9 | 单元测试（完整覆盖） |
| `core/registry.py` | 8 | 单元测试（完整覆盖） |
| `detectors/opencv/haar_face.py` | 4 | 集成测试（完整覆盖） |
| `pipeline/data/stage.py` | 2 | 端到端集成 |
| `core/orchestrator.py` | 1 | 编排器集成 |
| `core/config.py` | 1 | 配置加载 |
| `pipeline/__init__.py` | 1 | 阶段发现 |

---

## 13. 配置与模板

```
configs/
├── full_pipeline.yaml       # 全流程示例：8 阶段完整配置（YOLOv8 检测）
└── quick_detect.yaml        # 快速检测示例：单图推理

templates/                   # 可复用流水线模板（预留）
```

---

## 14. 模块依赖关系图

```
                    ┌──────────────────────────────────────┐
                    │              CLI (app.py)             │
                    │  vw run/list/detect/data/model/serve │
                    └──┬──────┬──────┬──────┬──────┬──────┘
                       │      │      │      │      │
            ┌──────────┘      │      │      │      └──────────┐
            ▼                 ▼      ▼      ▼                 ▼
    ┌───────────────┐  ┌────────┐ ┌──────┐ ┌──────────┐ ┌─────────┐
    │  orchestrator │  │registry│ │ viz  │ │models/zoo│ │ serve/  │
    └───────┬───────┘  └───┬────┘ └──┬───┘ └────┬─────┘ └────┬────┘
            │              │         │          │            │
            ▼              │         │          │            │
    ┌───────────────┐      │         │          │            │
    │   pipeline/   │      │         │          │            │
    │  (8 stages)   │      │         │          │            │
    └───┬───┬───┬───┘      │         │          │            │
        │   │   │          │         │          │            │
        ▼   ▼   ▼          ▼         ▼          ▼            ▼
    ┌──────────────────────────────────────────────────────────┐
    │                     core/  (Level 0)                      │
    │  types.py  base.py  result.py  config.py  context.py     │
    │  registry.py  orchestrator.py  exceptions.py             │
    └──────────────────────────────────────────────────────────┘
```

**依赖规则**：
- **Level 0** (`core/`)：无项目内依赖，被所有人依赖
- **Level 1** (`data/`, `pipeline/`, `detectors/`, `models/`, `viz/`)：依赖 core，彼此独立
- **Level 2** (`serve/`)：依赖 detectors + viz + core
- **Level 3** (`cli/`)：依赖所有其他模块

---

## 15. 实现状态矩阵

### 15.1 核心模块

| 模块 | 文件 | 状态 | 测试 |
|------|------|:----:|:----:|
| types | `core/types.py` | ✅ 完成 (8 个类型) | — |
| base | `core/base.py` | ✅ 完成 (3 个 ABC) | — |
| result | `core/result.py` | ✅ 完成 (2 个模型) | — |
| registry | `core/registry.py` | ✅ 完成 (10 个全局实例) | 8/8 |
| config | `core/config.py` | ✅ 完成 (10 个配置模型) | 集成 |
| context | `core/context.py` | ✅ 完成 (不可变 + 版本链) | 9/9 |
| orchestrator | `core/orchestrator.py` | ✅ 完成 (DAG + 恢复) | 集成 |
| exceptions | `core/exceptions.py` | ✅ 完成 (20+ 类) | — |

### 15.2 流水线阶段

| 阶段 | 文件 | 核心逻辑 | 状态 |
|------|------|---------|:----:|
| ① Data | `pipeline/data/stage.py` | 发现/验证/去重/质量过滤/划分/复制/manifest | ✅ |
| ② Annotate | `pipeline/annotate/stage.py` | 格式转换/质量检查框架 | ✅ |
| ③ Train | `pipeline/train/stage.py` | 框架适配器解析/训练流程 | ✅ |
| ④ Validate | `pipeline/validate/stage.py` | 指标计算/回归测试/JSON 导出 | ✅ |
| ⑤ Evaluate | `pipeline/evaluate/stage.py` | 多模型对比/MD 报告生成 | ✅ |
| ⑥ Optimize | `pipeline/optimize/stage.py` | 量化/剪枝/蒸馏框架 | ✅ |
| ⑦ Export | `pipeline/export/stage.py` | ONNX/多平台导出框架 | ✅ |
| ⑧ Deploy | `pipeline/deploy/stage.py` | SSH/HTTP/MQTT 推送框架 | ✅ |

> **注**：阶段 ②-⑧ 的深层实现（格式转换器、框架适配器、量化引擎、平台导出器）预留了目录结构和注册入口，具体算法实现标记为 TODO。

### 15.3 检测器

| 检测器 | 文件 | 状态 |
|--------|------|:----:|
| Haar Cascade (face) | `detectors/opencv/haar_face.py` | ✅ 完成（零依赖） |
| YOLO (Ultralytics) | `detectors/yolo/` | 📋 预留 |
| MediaPipe | `detectors/mediapipe/` | 📋 预留 |
| HuggingFace | `detectors/huggingface/` | 📋 预留 |

### 15.4 其他模块

| 模块 | 文件 | 状态 |
|------|------|:----:|
| Model Zoo | `models/zoo.py` | ✅ 完成（3 个预训练模型定义 + 下载缓存） |
| Dataset Schema | `data/schema.py` | ✅ 完成 |
| Dataset Catalog | `data/catalog.py` | ✅ 完成 |
| Visualization | `viz/annotate.py` | ✅ 完成（框/关键点/骨骼/遮罩/对比网格） |
| FastAPI Serve | `serve/app.py` | ✅ 完成（4 个端点） |
| Gradio UI | `serve/gradio_ui.py` | ✅ 完成（3 个 Tab） |
| CLI | `cli/app.py` | ✅ 完成（6 个命令） |

### 15.5 预留目录（骨架已建）

```
pipeline/annotate/converters/    # COCO↔YOLO↔VOC↔LabelMe↔CVAT 转换器
pipeline/train/adapters/         # PyTorch/MMDet/Ultralytics/HF 适配器
pipeline/validate/metrics/       # mAP/IoU/F1/AUC-ROC 等指标实现
pipeline/optimize/quantizer/     # PTQ/QAT/FP16 量化实现
pipeline/optimize/pruner/        # L1/L2/结构化剪枝实现
pipeline/deploy/pusher/          # SSH/HTTP/MQTT 推送实现
models/architectures/            # 自定义架构定义系统
models/modules/                  # 注意力/卷积/融合模块库
models/fusion/                   # 多模态融合引擎
tracking/                        # MLflow/W&B 追踪适配器
```

---

## 16. 扩展开发指南

### 如何添加新的检测器

1. 在 `detectors/<framework>/` 下创建 `my_detector.py`
2. 继承 `BaseDetector`，实现 `initialize()` / `process()` / `cleanup()`
3. 用 `@detector_registry.register("my_detector")` 装饰类
4. 在 `detectors/<framework>/__init__.py` 中导入

### 如何添加新的流水线阶段

1. 创建 `pipeline/<stage_name>/stage.py`
2. 继承 `BaseStage[TConfig, TResult]`，定义 `name` / `description` / `depends_on`
3. 实现 `validate_inputs()` / `run()` / `dry_run()`
4. 将模块路径加入 `pipeline/__init__.py` 的 `_STAGE_MODULES` 列表

### 如何添加新的导出目标

1. 在 `pipeline/export/` 下创建 `tvm_exporter.py`
2. 继承 `BaseExporter`，实现 `export()` 和 `validate()`
3. 用 `@exporter_registry.register("tvm")` 注册

### 如何注册自定义架构组件

1. 在 `models/architectures/custom/` 下创建 backbone/neck/head 定义
2. 使用 `@backbone_registry.register("my_backbone")` 注册
3. 在训练配置 YAML 中通过嵌套结构引用

---

*最后更新：2026-06-25*
*项目版本：0.1.0*
