# Phase-2 Student 学不会 Teacher 的系统性排查结论

## 结论（一句话）
你现在的 Phase-2 失败，**首要根因不是 teacher 坏了，而是你把 student 的辨识任务从“慢变量辨识（m, l）”悄悄变成了“慢变量 + 快时变风场在线估计（m, l, wind_x, wind_y, wind_z）”**，同时还叠加了观测噪声、动作延迟和低通；在 3500 steps 预算下，任务难度发生阶跃式提升。

---

## 证据链（从代码直接可验证）

1. **privileged label 已经是 5 维：`[m_norm, l_norm, wind_norm(3)]`**。
   - `privileged_obs_dim = 5`，注释明确写了 e_t = [m_norm, l_norm, wind_norm(3)]。
   - 这意味着 student 监督目标不再是纯静态辨识。

2. **student 训练标签直接用 `z_teacher = mu(priv)`，而 priv 包含风**。
   - 数据采集脚本直接从 obs tail 取 `21:26` 做 priv，再过 `mu` 得 z_teacher。
   - student 被强制拟合包含快时变风信息的 z。

3. **wind 是快时变随机过程，甚至允许极短持续时间**。
   - `wind_gust_dt_min = 0.0`，`wind_gust_dt_max = 2.0`。
   - OU + gust 叠加后，wind 分量可快速变化；teacher 有 oracle priv，student 只有 noisy history。

4. **观测噪声已启用**。
   - `enable_obs_noise = True`，并对 e_load、tilt、线速度、角速度加噪。
   - student 输入噪声上升，标签仍是“真值风+参数”映射后的 z_teacher。

5. **动作链路增加了不可忽略的动态：延迟 + LPF**。
   - `action_delay_steps = 2`，`action_lpf_alpha = 0.2`。
   - 从 student 角度看，系统可观测性下降（更多隐状态）。

6. **teacher policy 更依赖 wind 分量是合理结果**。
   - 训练/部署里 teacher 使用 `use_mu=True` 时直接获得 priv→z；
   - student `use_mu=False` 只能靠 history 估计 z_hat，一旦 z2~z4（风相关）偏差大，动作就会明显掉性能。

---

## 为什么你会有“teacher 的 z0 z1 对，student 还是学不动”的体感

这是符合机制的：

- z0,z1（质量/绳长锚定）可能确实是对的；
- 但 policy 输入的是 **完整 z(5)**，不是只看 z0,z1；
- 当前环境又强化了风扰补偿需求（wind + lag + noise），所以 z2~z4 学不好就会拖垮闭环。

换句话说：你验证“decoupled 在 phase2 更快”的实验命题，已经被现在的任务定义污染了。

---

## 终极答案（Root Cause 排名）

### P0（决定性）
**任务目标漂移**：Phase-2 student 的监督目标从“慢变量辨识”漂移为“慢变量 + 快时变风估计”。

### P1（放大器）
**可观测性恶化**：观测噪声 + 动作延迟 + LPF 让 history->wind 的估计更难，且 3500 steps 不再是原先等价难度。

### P2（实验设计偏差）
你要证明的是 decoupled 让 phase2 更快（理论依据是 z0,z1 辨识），但当前 loss/标签并未聚焦 z0,z1，导致实验结论被“风估计难度”主导。

---

## 最小改动的修复路线（建议按顺序）

1. **先把 phase2 命题对齐**：做一个“仅静态参数辨识”的 student 版本。
   - 选项 A：把 privileged 先改为 2 维（m_norm, l_norm），phase2 全链路只喂这 2 维（或其映射 z_exp）。
   - 选项 B：保留 z_dim=5，但 student loss 主监督 z0,z1，z2~z4 弱监督或不监督。

2. **冻结风复杂度做对照**（用于找回可学习基线）。
   - 临时关掉 obs noise / action delay / LPF / gust（至少先关 gust 和 delay）。
   - 确认 3500 steps 能恢复“student 追 teacher”。

3. **再逐项加回干扰（真正消融）**。
   - +obs noise
   - +delay+lpf
   - +slow wind
   - +full wind (gust+OU)
   - 每次只开一项，记录 z0,z1 RMSE 与闭环指标。

4. **评估指标要拆分**。
   - 现在只看总表现会误判。
   - 至少拆：z0,z1 RMSE；z2~z4 RMSE；payload swing RMS；到点误差；耗时。

---

## 你现在最该立刻做的两件事

1. 跑一次“静态任务” sanity check（只保留 m,l，不要 wind 动态）；
2. 在现有任务上单独打印/画 z0,z1 与 z2~z4 的误差曲线，确认失败主要来自风相关维度。

只要这两步结果符合预期，你就能把“student 学不好”的锅从“模型坏了”转移到“任务定义变了且超预算”。
