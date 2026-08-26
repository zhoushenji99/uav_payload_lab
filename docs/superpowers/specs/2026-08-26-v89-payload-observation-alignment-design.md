# V8.9 Payload观测语义对齐设计

## 目标

在不改变质量、CTBR、Reward、风扰、动作延迟及电池效率设置的前提下，使仿真Policy 21维观测中的Payload视觉链与当前Jetson真机生成器一致，然后从零重训Teacher。

## 已锁定边界

- 保持`uav_bare_mass_kg = 3.100`、`uav_fixed_gimbal_mass_kg = 0.03175`和二者求和逻辑；3.100 kg明确不含固定侧半个万向节。
- 保持`collective_efficiency_range = (1.0, 1.0)`；本轮不增加电池效率训练随机化或OOD场景。
- 不修改CTBR绝对边界、slew limit、动作delay/LPF、Body-rate模型、Reward、Position前缀、DAgger或Student结构。
- 已停止的V8.9 Teacher只作smoke记录，禁止resume或用于后续Student。

## 方案选择

### 采用：仿真匹配现有真机观测生成器

仿真延迟相对Payload测量，再与当前UAV位姿组合；同时只在有效的新视觉测量上更新摆速低通。这是改动面最小、不会增加真机运行复杂度的方案。

### 不采用：真机回溯PX4位姿到相机时间戳

该方案在理论时间对齐上更完整，但需要Jetson缓存、插值和异常处理，扩大首次真机飞行风险。

### 不采用：保留当前仿真语义

当前实现延迟完整世界系`e_load + theta`，与真机“当前机体位姿 + 延迟相对测量”不一致，不能作为最终重训基线。

## 数据流设计

每个仿真控制周期：

1. 由当前真实UAV与Payload位姿计算Payload相对UAV的机体系向量。
2. 将该相对向量写入延迟环，而不是写入世界系`e_load`。
3. 到达视觉更新周期时，从延迟环读取相对向量；丢帧时保持上一有效观测。
4. 使用当前UAV世界位姿和姿态将延迟相对向量重新组成测量Payload世界位置。
5. 从该测量位置计算Policy侧`e_load`和摆角；Critic和Reward继续使用无延迟真实状态。
6. 仅在有效的新视觉测量更新时，由测量摆角差分得到raw摆速，再执行：

   `filtered = alpha * previous_filtered + (1 - alpha) * raw`

7. 首次测量摆速为0；dropout/hold期间保持上一滤波值；不得按60 Hz tick重复滤波。

固定相机外参在这条相对位置链中属于常量变换。当前仿真未建独立相机刚体，因此使用等价的UAV机体系相对向量，不新增未经测量的外参参数。

## 代码范围

- `meta_uav_env.py`
  - 调整Payload传感器延迟环的数据语义和观测重建。
  - 使用现有`obs_theta_dot_lpf_alpha`实现有效测量驱动的摆速LPF。
- `regression_tests/test_real_hover_gap.py`
  - 增加LPF首次更新、后续更新、dropout保持测试。
  - 增加“UAV移动、相对Payload固定、已知延迟”坐标组合测试。
- `03_V8.9训练评估与导出命令.md`及审核说明
  - Torch脚本统一使用IsaacLab Python。
  - 验收计数改为`5 seeds × 8 simulation scenarios + 1 fixed real replay`。
  - 记录质量与电池边界，避免后续误改。

## 验收标准

- 新回归测试先在旧实现上以预期原因失败，再在修改后通过。
- `alpha=0.5`时滤波数值与真机递推公式一致。
- 首次有效测量摆速为0；无新有效测量时角度和摆速均保持。
- UAV发生平移/转动而延迟相对Payload向量固定时，重建的Payload世界位置随当前UAV位姿变化，不冻结在历史世界坐标。
- Policy使用传感器Gap观测，Critic/Reward仍使用clean state。
- 现有V8.9回归测试全部通过。
- 从零启动新的Teacher；不得resume已停止的smoke Teacher。

## 非目标

- 不以本次修改宣称已经通过最终Teacher、Student或真机放行。
- 不修改或重新解释Hard-explicit z0/z1物理语义。
- 不新增电池压降模型。
