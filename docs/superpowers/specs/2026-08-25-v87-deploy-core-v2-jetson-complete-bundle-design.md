# V8.7 DeployCoreV2 Jetson 完整部署包设计

## 1. 目标

将当前候选悬停策略导出为可复制到 NVIDIA
Jetson、无需 Isaac Sim 的 ROS 2 Humble 部署包。部署包必须同时满足：

1. 在 Jetson 使用 ONNX Runtime 以 60 Hz 运行 Fast-Slow Student 与 Actor；
2. 保留 TorchScript 模型作为一致性检查和备用后端；
3. 记录可用于事后重建 50H、上下文估计和 CTBR 控制过程的完整时序数据；
4. 推理节点只发布候选 CTBR，不直接取得 PX4 控制权；
5. 使用现有权限网关完成 Offboard 接管、退出和实际 CTBR 记录。

## 2. 固定模型谱系

### 2.1 Teacher/Actor

路径：

`logs/rsl_rl/uav_payload_sim2real_hover_deploy_core_v2/2026-08-25_01-14-54_hardexplicit_teacher_hover_deploy_core_v2_seed42/model_1500.pt`

Teacher checkpoint 仅用于导出 Actor、保存模型谱系和离线复核，不在真机
运行时读取特权上下文。

### 2.2 Student

路径：

`logs/rsl_rl/uav_payload_sim2real_hover_deploy_core_v2/2026-08-25_01-14-54_hardexplicit_teacher_hover_deploy_core_v2_seed42/StudentFastSlow_hover_deploy_core_v2_model1500_seed42_noprobe/best_fast_slow_student_encoder_z.pth`

仅导出并保存该 best Student。不得把 `last_checkpoint.pth`、中断轮次权重或
其他实验 Student 放入部署包。Manifest 必须记录 checkpoint 内部 epoch、
best validation loss、seed、Teacher context mode 和 Student context mode。

## 3. 模型文件

部署包包含三个独立推理图：

1. `slow_encoder.onnx/.ts`：`[B, 50, 21] -> [B, 2]`；
2. `fast_encoder.onnx/.ts`：`[B, 50, 21] -> [B, 3]`；
3. `actor.onnx/.ts`：`[B, 26] -> [B, 4]`，包含训练时 Actor observation
   normalizer。

ONNX 是 Jetson 主运行后端；TorchScript 仅作为校验与备用，不生成与具体
JetPack/TensorRT 版本绑定的 engine。

## 4. ROS 2 推理数据流

推理节点执行以下数据流：

```text
/uav_payload/observation21
    -> 合法性、时效性和单位检查
    -> 50x21 FIFO历史
    -> Slow/Fast ONNX
    -> [当前21维, z_slow_cache, z_fast]
    -> Actor ONNX
    -> CTBR裁剪
    -> /uav_payload/rl_ctbr_candidate
```

固定 CTBR 顺序：

```text
[thrust_body_z, roll_rate, pitch_rate, yaw_rate]
```

固定裁剪范围：

```text
[-1, 0], [-2.5, 2.5], [-2.5, 2.5], [-1.5, 1.5]
```

推理节点不得发布 `VehicleCommand`、`OffboardControlMode` 或
`VehicleRatesSetpoint`。这些消息继续由权限网关统一管理。

## 5. 历史与快慢调度

- policy：60 Hz；
- history：50 帧；
- Fast Encoder：每个控制周期更新；
- Slow Encoder：启动前 3 s 以 60 Hz 更新；
- 启动 3 s 后：Slow Encoder 以 1 Hz 更新；
- Slow cache：使用 0.25 s 因果 EMA；
- Position 阶段允许推理节点后台 shadow 运行；
- 未收满 50 个连续有效观测或未完成 3 s shadow 时，不允许发布
  `candidate_ready=true`。

50H 中最后四维必须来自真实上一帧执行/Position CTBR 数据链，不使用固定
占位值伪造有效历史。

## 6. 真机日志

推理节点每个有效 60 Hz tick 写入一行 CSV，至少包含：

- wall/monotonic/PX4/observation 时间戳；
- 原始 21 维观测；
- history fill count；
- z0-z4 raw、clamped和越界标志；
- slow raw、slow target、slow cache；
- slow/fast refresh 标志；
- Actor raw CTBR；
- clamped candidate CTBR；
- 21维中携带的上一帧实际 CTBR；
- Slow、Fast、Actor和端到端推理延迟；
- observation age、有效性、candidate ready 和 shadow 状态；
- 异常/丢帧/拒绝发布原因。

完整 `50x21` 矩阵不在每个 tick 重复写入。连续 21 维观测、reset 事件与
history fill count 足以严格重建任意时刻的 50H，同时避免阻塞控制线程。

权限网关继续单独记录候选CTBR、实际发送CTBR、PX4模式、退出原因和ACK。
两个CSV使用同一运行目录和单调时钟，可离线对齐。

CSV 使用缓冲写入并定期 flush；节点正常退出时必须关闭文件。推理异常、
输入过期、非有限值或上下文严重越界时记录原因并停止发布 candidate，不得发送旧动作。

部署包本身不证明 Student 已满足实飞条件。Position真实CTBR前缀与Student训练
分布的一致性、多seed闭环、持续掉高和上下文越界必须在仿真预导出验收报告中
单独给出；证据不足时Manifest保持 `flight_approved=false`，部署包仅允许shadow。

## 7. 部署包结构

```text
V8.7_DeployCoreV2_model1500_StudentBest_Jetson完整部署包/
  models/
    actor.onnx
    actor.ts
    slow_encoder.onnx
    slow_encoder.ts
    fast_encoder.onnx
    fast_encoder.ts
    source_teacher_model_1500.pt
    source_student_best.pth
  runtime/
    rl_fastslow_inference_onnx_v1.py
    jetson_reference_runtime.py
    rl_ctbr_handover_core_v1.py
    rl_ctbr_offboard_handover_v1.py
  verification/
    verify_jetson_bundle.py
    parity_vectors.npz
    parity_report.json
  config/
    manifest.json
    training_config_snapshot/
  README.md
  sha256sums.txt
```

最终压缩文件：

`/home/shenji/桌面/V8.7_DeployCoreV2_model1500_StudentBest_Jetson完整部署包.zip`

## 8. 验证门槛

导出完成必须通过：

1. Teacher/Student 谱系、维度和 context mode 检查；
2. 三个 ONNX 模型通过 `onnx.checker`；
3. PyTorch vs TorchScript 最大绝对误差不超过 `1e-6`；
4. PyTorch vs ONNX 最大绝对误差不超过 `1e-5`；
5. 真实训练数据黄金向量一致性；
6. 185步状态化调度、缓存、滤波和动作裁剪一致性；
7. ROS 2 推理节点纯逻辑测试：历史填充、过期输入、非有限输入、reset、
   candidate ready 和CSV字段；
8. 压缩前后 SHA256 与必需文件清单一致；
9. 在目标 Jetson 上运行 `verify_jetson_bundle.py` 后才允许启动 shadow。

## 9. 安全边界

部署包证明模型格式和数值一致性，不自动批准自由飞行。首次运行必须保持：

- Position起飞；
- 推理节点先shadow；
- 权限网关和飞手拥有退出权；
- RL推理节点不Arm、不Disarm、不切模式；
- ULog和Jetson日志同时记录。
