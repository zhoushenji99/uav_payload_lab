# 1.venv

venv中，（px4）是用来专门build的环境，外面的普通环境为isaacsim专用。
ubuntu24.04新增了环境保护，不让px4的下载环境污染总环境所以推荐下载en
激活 env
source ~/.venvs/px4/bin/activate
关闭
deactivate

# 2.训练

默认仓库路径：
cd ~/uav_payload_lab/uav_payload_lab

当前 Sim2Real 任务名：
Isaac-Uav-Sim2Real-v0

当前版本：
V7.4 sim2real measured-physics + warm start + raw-action reward

当前核心设置：
- 起点 `(0, 0, 2)`
- 目标点 `(4, 0, 2)`
- `action_delay_steps=0`
- `action_lpf_alpha=1.0`
- `action_smooth_penalty_scale=0.03`
- `action_l2_penalty_scale=0.03`
- `action_raw_excess_penalty_scale=1e-3`
- `spin_weight=0.15`
- `moment_scale_xy=1.0`
- `moment_scale_z=0.25`

注意：
开局掉高是当前 reward 下的自然策略行为，不作为这个版本的 bug 处理。
这个版本重点是把动作惩罚改到真正的 policy raw action 上，避免 a3 在 clamp 后长期顶格但 raw action 爆炸。

## Teacher训练

#### V7.4 PA teacher

默认 2w iterations。

```
~/IsaacLab/isaaclab.sh -p scripts/rsl_rl/train.py \
  --task=Isaac-Uav-Sim2Real-v0 \
  --seed 42 \
  --max_iterations 20000 \
  --headless \
  agent.experiment_name=uav_payload_sim2real_rl \
  agent.run_name=measured_phys_goal4_warm_pa_rawreward003_rawexcess1e-3_seed42 \
  env.action_smooth_penalty_scale=0.03 \
  env.action_l2_penalty_scale=0.03 \
  env.action_raw_excess_penalty_scale=1e-3 \
  env.spin_weight=0.15 \
  env.action_lpf_alpha=1.0 \
  env.action_delay_steps=0
```

训练结束后设置：

```
SIM2REAL_RUN=/home/shenji/uav_payload_lab/uav_payload_lab/logs/rsl_rl/uav_payload_sim2real_rl/替换成新的measured_phys_goal4_warm_pa_rawreward003_rawexcess1e-3_seed42日志目录
SIM2REAL_CKPT=$SIM2REAL_RUN/model_19999.pt
```

如果日志里保存的是 `model_20000.pt`，就把上面的 checkpoint 改成 `model_20000.pt`。

#### Black-box RMA teacher消融

这组用于和 PA teacher 对比。默认也是 2w iterations。

```
~/IsaacLab/isaaclab.sh -p scripts/rsl_rl/train.py \
  --task=Isaac-Uav-Sim2Real-v0 \
  --seed 42 \
  --max_iterations 20000 \
  --headless \
  --black_box_rma \
  --rma_z_exp_dim 2 \
  --rma_phys_anchor_coef 0.0 \
  agent.experiment_name=uav_payload_sim2real_rl \
  agent.run_name=measured_phys_goal4_warm_blackbox_rma_seed42 \
  env.action_smooth_penalty_scale=0.03 \
  env.action_l2_penalty_scale=0.03 \
  env.action_raw_excess_penalty_scale=1e-3 \
  env.spin_weight=0.15 \
  env.action_lpf_alpha=1.0 \
  env.action_delay_steps=0
```

## Teacher Eval

#### Headless play生成 CSV

```
~/IsaacLab/isaaclab.sh -p scripts/rsl_rl/play.py \
  --task Isaac-Uav-Sim2Real-v0 \
  --num_envs 1 \
  --seed 42 \
  --checkpoint "$SIM2REAL_CKPT" \
  --headless \
  agent.experiment_name=uav_payload_sim2real_rl \
  env.action_lpf_alpha=1.0 \
  env.action_delay_steps=0 \
  env.action_smooth_penalty_scale=0.03 \
  env.action_l2_penalty_scale=0.03 \
  env.action_raw_excess_penalty_scale=1e-3 \
  env.spin_weight=0.15 \
  env.moment_scale_xy=1.0 \
  env.moment_scale_z=0.25
```

如果要打开可视化窗口，去掉 `--headless`。

#### 保存 play CSV

```
mkdir -p "$SIM2REAL_RUN/analysis_latest"
cp "$SIM2REAL_RUN/payload_data.csv" "$SIM2REAL_RUN/analysis_latest/payload_data_model_19999.csv"
```

## 继续训练continue

把 `--load_run` 和 `--checkpoint` 换成要继续的 run。

```
~/IsaacLab/isaaclab.sh -p scripts/rsl_rl/train.py \
  --task=Isaac-Uav-Sim2Real-v0 \
  --resume \
  --load_run 替换成已有run目录名 \
  --checkpoint model_10000.pt \
  --seed 42 \
  --max_iterations 20000 \
  --headless \
  agent.experiment_name=uav_payload_sim2real_rl \
  agent.run_name=measured_phys_goal4_warm_pa_continue_seed42 \
  env.action_smooth_penalty_scale=0.03 \
  env.action_l2_penalty_scale=0.03 \
  env.action_raw_excess_penalty_scale=1e-3 \
  env.spin_weight=0.15 \
  env.action_lpf_alpha=1.0 \
  env.action_delay_steps=0
```

## Collect data

当前 V7.4 用 Sim2Real PA teacher collect。

```
~/IsaacLab/isaaclab.sh -p scripts/rsl_rl/collect_z_dataset.py \
  --task Isaac-Uav-Sim2Real-v0 \
  --checkpoint "$SIM2REAL_CKPT" \
  --num_envs 4096 \
  --steps 4200 \
  --save_every 25 \
  --sample_stride 5 \
  --out_name DecoderAddedDataset_sim2real_pa_v74_noprobe \
  --trace_csv \
  --trace_env 0 \
  --probe_sec 0 \
  --headless
```

设置数据路径：

```
SIM2REAL_DATA=$SIM2REAL_RUN/DecoderAddedDataset_sim2real_pa_v74_noprobe
SIM2REAL_STUDENT=$SIM2REAL_DATA/student_train_sched_2e-4_500_to_3e-5_1000
```

## train encoder

第一段：2e-4, 500 epochs

```
~/IsaacLab/isaaclab.sh -p scripts/rsl_rl/train_student_z.py \
  --data_dir "$SIM2REAL_DATA" \
  --out_dir "$SIM2REAL_STUDENT" \
  --epochs 500 \
  --batch_size 4096 \
  --lr 2e-4 \
  --num_workers 4 \
  --use_weighted_mse \
  --aux_ml_coef 0.5
```

第二段：3e-5, 1000 epochs

```
~/IsaacLab/isaaclab.sh -p scripts/rsl_rl/train_student_z.py \
  --data_dir "$SIM2REAL_DATA" \
  --out_dir "$SIM2REAL_STUDENT" \
  --resume \
  --resume_path "$SIM2REAL_STUDENT/last_checkpoint.pth" \
  --epochs 1000 \
  --batch_size 4096 \
  --lr 3e-5 \
  --num_workers 4 \
  --use_weighted_mse \
  --aux_ml_coef 0.5
```

设置 student encoder：

```
SIM2REAL_ENCODER=$SIM2REAL_STUDENT/best_student_encoder_z.pth
```

## Play phase2

**Play Teacher：**

```
~/IsaacLab/isaaclab.sh -p scripts/rsl_rl/play_student_phase2.py \
  --task Isaac-Uav-Sim2Real-v0 \
  --mode teacher \
  --checkpoint "$SIM2REAL_CKPT" \
  --seed 42 \
  --num_envs 1 \
  --max_steps 2100 \
  --headless
```

**Play Student：**

```
~/IsaacLab/isaaclab.sh -p scripts/rsl_rl/play_student_phase2.py \
  --task Isaac-Uav-Sim2Real-v0 \
  --mode student \
  --checkpoint "$SIM2REAL_CKPT" \
  --encoder "$SIM2REAL_ENCODER" \
  --seed 42 \
  --num_envs 1 \
  --max_steps 2100 \
  --headless
```

三 seed teacher/student CSV：

```
for SEED in 38 40 42; do
  ~/IsaacLab/isaaclab.sh -p scripts/rsl_rl/play_student_phase2.py \
    --task Isaac-Uav-Sim2Real-v0 \
    --mode teacher \
    --checkpoint "$SIM2REAL_CKPT" \
    --csv "$SIM2REAL_RUN/phase2_teacher_seed${SEED}.csv" \
    --seed "$SEED" \
    --num_envs 1 \
    --max_steps 2100 \
    --headless

  ~/IsaacLab/isaaclab.sh -p scripts/rsl_rl/play_student_phase2.py \
    --task Isaac-Uav-Sim2Real-v0 \
    --mode student \
    --checkpoint "$SIM2REAL_CKPT" \
    --encoder "$SIM2REAL_ENCODER" \
    --csv "$SIM2REAL_RUN/phase2_student_seed${SEED}.csv" \
    --seed "$SEED" \
    --num_envs 1 \
    --max_steps 2100 \
    --headless
done
```

## 分析和画图

单 seed teacher/student 分析：

```
~/IsaacLab/isaaclab.sh -p scripts/rsl_rl/analyze_phase2_csv.py \
  --teacher "$SIM2REAL_RUN/phase2_teacher_seed42.csv" \
  --student "$SIM2REAL_RUN/phase2_student_seed42.csv" \
  --out_dir "$SIM2REAL_DATA/analysis_seed42" \
  --time_window 35
```

多 seed 对比：

```
~/IsaacLab/isaaclab.sh -p scripts/rsl_rl/analyze_phase2_csv.py \
  --dec_teachers "$SIM2REAL_RUN/phase2_teacher_seed38.csv" "$SIM2REAL_RUN/phase2_teacher_seed40.csv" "$SIM2REAL_RUN/phase2_teacher_seed42.csv" \
  --dec_students "$SIM2REAL_RUN/phase2_student_seed38.csv" "$SIM2REAL_RUN/phase2_student_seed40.csv" "$SIM2REAL_RUN/phase2_student_seed42.csv" \
  --coup_teachers 替换成RMA_teacher_seed38.csv 替换成RMA_teacher_seed40.csv 替换成RMA_teacher_seed42.csv \
  --coup_students 替换成RMA_student_seed38.csv 替换成RMA_student_seed40.csv 替换成RMA_student_seed42.csv \
  --seed_labels seed38 seed40 seed42 \
  --compare_labels Decoupled Coupled \
  --out_dir "$SIM2REAL_RUN/phase2_multiseed_compare" \
  --time_window 35
```

## Phase1 / teacher 画图

单 CSV：

```
~/IsaacLab/isaaclab.sh -p source/uav_payload_lab/uav_payload_lab/tasks/direct/uav_payload_lab/plot/IsaaclabPlot12.5.py \
  --csv "$SIM2REAL_RUN/payload_data.csv" \
  --time_window 5
```

多 CSV 对比：

```
~/IsaacLab/isaaclab.sh -p source/uav_payload_lab/uav_payload_lab/tasks/direct/uav_payload_lab/plot/IsaaclabPlot12.5.py \
  --csv \
  "$SIM2REAL_RUN/payload_data.csv" \
  替换成RMA_or_PPO_payload_data.csv \
  --labels PA RMA \
  --out_dir "$SIM2REAL_RUN/compare_teacher" \
  --time_window 35
```

# 3.Github

以后固定流程：

看改了什么

##### git status

##### git diff

加到暂存区

##### git add <相关文件>

提交

##### git commit -m "V7.4: sim2real measured physics and raw-action reward"

打 tag（可选，但对阶段版本很有用）

##### git tag -a v7.4_sim2real_measured_physics_raw_action_reward -m "V7.4: sim2real measured physics and raw-action reward"

推上去

##### git push origin master

##### git push origin v7.4_sim2real_measured_physics_raw_action_reward
