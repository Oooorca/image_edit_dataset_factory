# SPEC Mapping

## 目录与架构

- Core: `src/image_edit_dataset_factory/core/`
- Utils: `src/image_edit_dataset_factory/utils/`
- Backends: `src/image_edit_dataset_factory/backends/`
- Pipeline: `src/image_edit_dataset_factory/pipeline/`
- QA: `src/image_edit_dataset_factory/qa/`
- Scripts: `src/image_edit_dataset_factory/scripts/`

## 只读数据输入

- 输入扫描：`pipeline/ingest.py`
- 不复制、不移动原图：直接保存路径到 `outputs/manifests/source_manifest.jsonl`

## 模型后端（ModelScope + lazy + optional）

- Layered: `backends/qwen_layered_modelscope.py`
- Edit: `backends/qwen_image_edit_modelscope.py`
- 不自动下载：缺模型目录时明确报错
- Mock/Fallback：`backends/mock_backend.py`, `backends/opencv_fallback.py`

## 生成任务映射

- 中文类别枚举：`core/enums.py`
- 配置映射：`generate.category_to_task`
- 生成器实现：
  - `generate/structural.py`
  - `generate/semantic.py`
  - `generate/consistency.py`

## 导出与命名

- 导出：`pipeline/export.py`
- 命名规则：`utils/naming.py`

## QA

- Linter：`qa/linter.py`
- 非编辑区一致性：`qa/consistency.py`
- 报告：`qa/report.py`
