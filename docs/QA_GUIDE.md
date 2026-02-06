# QA Guide

## 1) Linter

检查项：

- 文件命名格式
- 必需文件是否齐全
- 图片损坏
- source/result 分辨率一致性

报告：`outputs/reports/lint_issues.json`

## 2) 非编辑区域不变

流程：

1. 获取允许编辑区域（`allowed_region_mask_path` 或主 mask 膨胀）
2. 在允许区域外比较 source/result
3. 计算 MSE、SSIM、像素变更比例
4. 按阈值判定 pass/fail

报告：

- `outputs/reports/qa/qa_scores.csv`
- `outputs/reports/qa/qa_summary.json`
