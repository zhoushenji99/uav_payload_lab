# collect_data.py
import torch
import os
import hydra
from isaaclab.envs import DirectRLEnv
from rsl_rl.runners import OnPolicyRunner

# 导入你的环境配置
import uav_payload_lab.tasks.direct.uav_payload_meta.meta_uav_env_cfg as cfg_entry

@hydra.main(config_path=".", config_name="config", version_base=None)
def main(cfg):
    # 1. 设置配置：强制开启 Headless 和 4096 并行
    env_cfg = cfg_entry.UavPayloadMetaEnvCfg()
    env_cfg.scene.num_envs = 4096  # 并行采集，速度极快
    env_cfg.sim.render_interval = 2 # 不渲染
    
    # 2. 强制开启 Oracle 模式 (因为我们要用 Teacher 跑数据)
    env_cfg.use_oracle_mass_obs = True
    env_cfg.observation_space = 19
    env_cfg.rope_length_range = (0.3, 0.8) # 确保随机化开启
    
    # 3. 初始化环境
    env = DirectRLEnv(env_cfg, render_mode=None)
    
    # 4. 加载你训练好的 Oracle Policy
    # 修改这里的路径指向你最好的那个 .pt 模型
    policy_path = "/home/shenji/uav_payload_lab/uav_payload_lab/logs/rsl_rl/uav_payload_meta_massL_baseline/2026-01-05_22-24-58/model_4000.pt"
    loaded_dict = torch.load(policy_path, map_location="cuda:0")
    
    # 提取 Actor 网络权重
    actor_state_dict = loaded_dict['model_state_dict']
    
    # 简单的 MLP 加载逻辑 (根据 RSL-RL 结构重建网络)
    # 注意：这里我们不需要完整的 Runner，只需要一个前向传播函数
    actor = torch.nn.Sequential(
        torch.nn.Linear(19, 128), torch.nn.ELU(),
        torch.nn.Linear(128, 128), torch.nn.ELU(),
        torch.nn.Linear(128, 4)
    ).to("cuda:0")
    
    # 这里的 key 名字可能需要根据 saved model 微调，通常是 'actor.0.weight' 等
    # 如果直接 load 不成功，可能需要手动对应层名字
    # 为了演示简单，这里假设你能成功加载 (通常 rsl_rl保存的是整个 runner state)
    # 建议：直接实例化一个 OnPolicyRunner 来加载，最稳妥
    runner = OnPolicyRunner(env, {"policy": {"actor_hidden_dims": [128, 128], "activation": "elu"}}, log_dir=None, device="cuda:0")
    runner.load(policy_path)
    policy = runner.alg.actor_critic.actor # 获取 actor

    # 5. 准备数据容器
    # 采集 1000 步 x 4096 环境 = 400万条数据 (足够了)
    num_steps = 1000 
    
    # 预分配内存 (在 GPU 上)
    data_buffer = {
        "obs_blind": torch.zeros((num_steps, 4096, 17), device="cuda:0"), # 盲观测
        "actions":   torch.zeros((num_steps, 4096, 4),  device="cuda:0"), # 老师的动作
        "physics":   torch.zeros((num_steps, 4096, 2),  device="cuda:0"), # 真值 [Mass, Length]
    }

    obs, _ = env.reset()
    
    print("开始采集数据...")
    with torch.inference_mode():
        for t in range(num_steps):
            # A. Oracle 决策
            actions = policy(obs) # obs 是 19 维
            
            # B. 拆分 Obs：把 Oracle 的特权信息剥离，只存 Blind 部分给 Student
            # 你的 obs 结构是 [Blind(17), Mass(1), Length(1)]
            obs_blind = obs[:, :17] 
            
            # C. 获取真实的物理参数 (用于 Decoder 监督)
            # 你的代码里 m_norm 是倒数第2个，l_norm 是倒数第1个
            # 建议直接存 obs 里的最后两维，或者存 env._payload_mass 和 env._rope_lengths
            # 这里直接存 obs 里的归一化值，方便 Decoder 训练
            physics_params = obs[:, 17:19] 

            # D. 存入 Buffer
            data_buffer["obs_blind"][t] = obs_blind
            data_buffer["actions"][t]   = actions
            data_buffer["physics"][t]   = physics_params
            
            # E. 物理步进
            obs, rewards, dones, extras = env.step(actions)
            
            if t % 100 == 0:
                print(f"Step {t}/{num_steps}")

    # 6. 保存数据
    print("正在保存数据...")
    torch.save(data_buffer, "dataset_oracle.pt")
    print("采集完成！保存为 dataset_oracle.pt")

if __name__ == "__main__":
    main()