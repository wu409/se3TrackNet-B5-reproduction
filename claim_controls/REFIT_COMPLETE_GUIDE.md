# 第二项补全：fixed-policy 与 rollout-data 归因

旧脚本只接受外部 fixed-policy CSV，没有生成程序，因此原 `02_refit` 的
`NOT RUN` 记录是正确的，不会回写修改。新版 `02_rollout_refit_attribution.sh`
默认转入完整流程；`REFIT_SAME_CANDIDATE_ONLY=1` 可复现旧的局部诊断。

## 数据如何生成

仅使用冻结 release 的开发名单（当前 4 序列 × 9 条件，22,770 帧）。
固定策略始终采用 manifest 中预先计算的 tracking observation，不由质量预测器
决定输出，不触发 registration 或 observer reseeding。前两帧 prior 与当前
observation 相同，之后严格调用冻结的 `compute_se3_prior`，从前两次 observation
构造 prior。每个候选重新调用冻结 `2-risk_label.py` 的特征提取与 ADD-S 标签代码。
GT 只用于生成误差标签，不输入候选生成或历史更新。

这是一个不由质量预测驱动的候选数据对照；与 q0-policy 的差别包含选择、融合、
恢复和 observer restart 的整体作用，不能再单独归因于其中某一机制。

## 拟合的控制组

所有新模型复用冻结的 `fit_shared_quality_model`，保持四特征、Huber 参数、
按 episode 前 70% 拟合/后 30% 校准、isotonic 与风险阈值选择方法。

- `fixed_obs_only`：固定观测 N 个样本。
- `fixed_obs_duplicated_count_control`：同一批观测复制一遍，2N 个样本，控制样本数量/正则项影响。
- `fixed_obs_prior`：固定策略 observation＋prior，2N 个样本。
- `q0_policy_obs_prior_refit`：已有 q0 闭环 observation＋prior，2N 个样本。
- 原始冻结 `q0`、`q1` 作为参考，不覆盖它们。

重点比较：`fixed_obs_prior` 对 `fixed_obs_duplicated_count_control`，以及
`q0_policy_obs_prior_refit` 对 `fixed_obs_prior`。两组预测都在同一份指定候选表上
比较。缓存 q0 rollout 来自派生 q0 release 的诊断轨迹，可能与原 q1 拟合时的
随机轨迹有差异，所以重拟合控制不会被冒充为原冻结 q1。

## 在原服务器运行

```bash
cd /root/autodl-tmp/se3TrackNet-B5-reproduction
export TEST_PYTHON=/root/autodl-tmp/conda-envs/b5-main/bin/python3.8
# 小规模真实渲染检查；只生成 SMOKE.json，不拟合、不声称完成
bash claim_controls/02_rollout_refit_attribution.sh --smoke
# 完整数据采集和拟合，默认使用独立的新目录
bash claim_controls/02_rollout_refit_attribution.sh
```

可用 `--check-only` 只验证开发数据输入。可设置 `CLAIM_OUTPUT` 指向不存在的新目录。
再次运行时，设置 `FIXED_POLICY_CACHE=/上一轮/fixed_policy_cache` 可复用已完成缓存；
程序核对收据、manifest/source/sample 哈希与全部开发 frame/source 键。
裸 CSV、缺帧、重复帧、测试序列或 smoke 缓存均不会作为完整对照输入。

输出 `fixed_policy_cache/`、`models/`、`training_counts.csv`、`metrics.csv`、
`sequence_metrics.csv`、输入/结果哈希和 `COMPLETE.json`。无需重新训练 tracker，
但全量几何特征提取仍需 GPU 渲染与时间。不会重新运行已完成的第 1、3、4、5 项。

这些指标仍是开发数据上的离线归因诊断。`cal` 部分用过校准标签，不能称为独立
held-out test；也不能把离线预测误差差异当作新模型的闭环 tracking AUC 增益。
