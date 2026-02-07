# ASSUMPTIONS

1. `data/人物物体一致性`、`data/物体一致性`、`data/物理变化` 视为只读输入目录，流水线不修改其内容。
2. 默认只做最小可运行编辑：
   - structural: move
   - semantic: delete
   - consistency: identity/mild consistency shift
3. ModelScope 后端仅支持“本地已存在模型目录”场景；不执行任何下载逻辑。
4. Qwen ModelScope 后端当前是安全骨架（lazy init + clear error），便于后续在服务器补齐真实调用适配。
5. 输出目录统一在 `outputs/`，日志在 `logs/`，便于部署机清理与归档。
6. 导出命名沿用严格规则：`00001.jpg`, `00001_result.jpg`, `00001_CH.txt`, `00001_EN.txt`, `00001_mask.png`, `00001_mask-1.png`。
7. QA 中“非编辑区不变”使用 `allowed_region_mask_path` 或主 mask 膨胀区域作为允许编辑区域。
8. API mode 下默认假设 orchestrator 与服务进程可访问同一文件系统；若跨机部署请改为 `send_mode=base64`。
9. 分层与编辑服务使用独立 conda 环境，依赖冲突在服务边界隔离；主流程环境保持轻量。
