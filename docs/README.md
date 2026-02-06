# Image Edit Dataset Factory

该项目用于从 `./data/*` 读取图片，构建图像编辑数据集，并执行严格命名与一致性 QA。

## 当前支持的数据源目录（只读）

- `data/人物物体一致性`
- `data/物体一致性`
- `data/物理变化`

说明：流水线不会修改上述目录，不会移动/删除原图；中间产物和结果全部写入 `outputs/`。

## 依赖安装

项目只使用一个依赖入口文件：`pyproject.toml`。

```bash
python -m pip install -e .
```

开发依赖：

```bash
python -m pip install -e .[dev]
```

## 运行 Mock/Fallback 全流程（不下载模型）

```bash
python -m image_edit_dataset_factory.scripts.run_all --config configs/default.yaml
```

输出位置：

- 清单：`outputs/manifests/`
- 导出数据集：`outputs/dataset/`
- 报告：`outputs/reports/`
- 日志：`logs/`

## 分步运行

```bash
python -m image_edit_dataset_factory.scripts.run_ingest --config configs/default.yaml
python -m image_edit_dataset_factory.scripts.run_generate --config configs/default.yaml
python -m image_edit_dataset_factory.scripts.run_qa --config configs/default.yaml
```

## 通过配置映射中文类别到编辑任务

`configs/default.yaml` 中：

- `人物物体一致性 -> consistency_edit`
- `物体一致性 -> semantic_edit`
- `物理变化 -> structural_edit`

可通过 `--set` 覆盖：

```bash
python -m image_edit_dataset_factory.scripts.run_all \
  --config configs/default.yaml \
  --set generate.category_to_task."物体一致性"=structural_edit
```

## 后续启用 ModelScope Qwen（仅当模型已提前下载）

本项目不会自动下载模型。需要先在服务器手工下载，再配置本地目录：

- `qwen/Qwen-Image-Layered`
- `Qwen/Qwen-Image-Edit`

示例配置见：`configs/server_qwen.yaml`

```bash
python -m image_edit_dataset_factory.scripts.run_all --config configs/server_qwen.yaml
```

如果模型目录不存在，程序会报清晰错误并退出，不会触发下载。
