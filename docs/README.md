# Image Edit Dataset Factory

本项目用于从本地只读数据目录读取图片，生成图像编辑数据集，并执行命名规范检查与一致性 QA。

- 输入目录（只读，不会被修改）：
  - `data/人物物体一致性`
  - `data/物体一致性`
  - `data/物理变化`
- 输出目录（流水线产物）：
  - `outputs/`（manifests / staging / dataset / reports）
  - `logs/`

## 1. 项目逻辑（先理解）

工作流分三类进程：

1. Orchestrator（主流程）
- 负责 ingest -> decompose -> generate -> export -> qa 的调度。
- 入口：`python -m image_edit_dataset_factory.scripts.run_all`

2. Layered Service（分层服务）
- 提供 `FastAPI /infer`，输出图层和 alpha/mask。
- 入口：`services/layered_service/app.py`

3. Edit Service（编辑服务）
- 提供 `FastAPI /infer`，执行编辑/修复。
- 入口：`services/edit_service/app.py`

在 API mode 下，Orchestrator 通过 HTTP 调用两个服务，不在主进程加载重模型依赖，避免环境冲突。

## 2. 从零开始：完整使用流程

## 2.1 拉代码并进入项目

```bash
git clone https://github.com/Oooorca/image_edit_dataset_factory.git
cd image_edit_dataset_factory
```

## 2.2 创建 3 个隔离 conda 环境

```bash
conda env create -f envs/orchestrator_env.yml
conda env create -f envs/layered_env.yml
conda env create -f envs/edit_env.yml
```

说明：
- `iedf-orchestrator`：轻量环境，只跑 workflow 与 QA。
- `iedf-layered`：只跑 Layered FastAPI。
- `iedf-edit`：只跑 Edit FastAPI。

## 2.3 准备模型（手动下载，不自动触发）

示例下载目录（建议数据盘）：`/root/autodl-tmp/models/qwen/`

```bash
mkdir -p /root/autodl-tmp/models/qwen
cd /root/autodl-tmp/models/qwen

# 在你自己的模型下载流程中，确保最终存在以下两个目录：
# /root/autodl-tmp/models/qwen/Qwen-Image-Layered
# /root/autodl-tmp/models/qwen/Qwen-Image-Edit
```

项目代码不会自动下载权重；目录不存在会直接报错退出。

## 2.4 用 screen 启动两个模型服务（推荐）

### 启动 Layered 服务（占用 2 张卡）

```bash
screen -dmS layered_service bash -lc '
cd ~/image_edit_dataset_factory && \
CUDA_VISIBLE_DEVICES=0,1 \
CONDA_ENV=iedf-layered \
LAYERED_BACKEND=qwen \
LAYERED_MODEL_DIR=/root/autodl-tmp/models/qwen/Qwen-Image-Layered \
bash scripts/start_layered_service.sh
'
```

### 启动 Edit 服务（占用 2 张卡）

```bash
screen -dmS edit_service bash -lc '
cd ~/image_edit_dataset_factory && \
CUDA_VISIBLE_DEVICES=2,3 \
CONDA_ENV=iedf-edit \
EDIT_BACKEND=qwen \
EDIT_MODEL_DIR=/root/autodl-tmp/models/qwen/Qwen-Image-Edit \
bash scripts/start_edit_service.sh
'
```

### 查看 screen 会话与日志

```bash
screen -ls
screen -r layered_service
screen -r edit_service
# 退出当前 screen 会话不杀进程：Ctrl+A 然后 D
```

### 停止服务

```bash
screen -S layered_service -X quit
screen -S edit_service -X quit
```

## 2.5 健康检查（必须先通过）

```bash
curl -s http://127.0.0.1:8101/healthz
curl -s http://127.0.0.1:8101/readyz
curl -s http://127.0.0.1:8102/healthz
curl -s http://127.0.0.1:8102/readyz
```

## 2.6 跑 API 编排全流程

```bash
conda run -n iedf-orchestrator python -m image_edit_dataset_factory.scripts.run_all \
  --config configs/server_qwen.yaml \
  --set services.api_mode=true \
  --set backends.layered_backend=api \
  --set backends.edit_backend=api \
  --set services.layered.endpoint=http://127.0.0.1:8101 \
  --set services.edit.endpoint=http://127.0.0.1:8102
```

## 2.7 查看输出

- 数据集：`outputs/dataset/`
- 清单：`outputs/manifests/`
- QA 报告：`outputs/reports/qa/`
- 日志：`logs/`

## 3. 快速测试命令

## 3.1 本地 mock 全流程（不依赖模型）

```bash
python -m image_edit_dataset_factory.scripts.run_all --config configs/default.yaml
```

## 3.2 API mode + mock 服务（联调用）

把两个服务环境变量改成 `*_BACKEND=mock` 后再运行第 2.6 节命令。

## 3.3 单样本编辑测试（当前调试脚本）

```bash
python -m image_edit_dataset_factory.scripts.test_qwen_single \
  --image "data/物体一致性/0004_image_129/0004_image_129.png" \
  --output-dir outputs/qwen_single \
  --layered-model-dir "/root/autodl-tmp/models/qwen/Qwen-Image-Layered" \
  --edit-model-dir "/root/autodl-tmp/models/qwen/Qwen-Image-Edit" \
  --device cuda \
  --skip-layered
```

## 4. 常见问题

1. `ModuleNotFoundError`
- 基本是环境装错。先确认当前命令运行在哪个 conda 环境。

2. `CUDA out of memory`
- 先确认是否有其他进程占卡：`nvidia-smi`
- 先用 API mode + mock 验证链路，再逐步上真实模型。

3. 服务 readyz 一直 false
- 检查模型目录是否存在、是否可读、依赖是否匹配。

## 5. 参考文档

- `docs/SERVICE_ARCHITECTURE.md`
- `docs/DEPLOY_MULTI_ENV.md`
- `docs/API_CONTRACT.md`
- `docs/ASSUMPTIONS.md`
