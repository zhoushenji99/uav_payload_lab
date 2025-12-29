# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym

from . import agents


# 注册新的 Meta-RL 环境
gym.register(
    id="Isaac-Uav-Meta-v0",  # 这是你训练时 task=... 用的名字
    entry_point=f"{__name__}.meta_uav_env:UavPayloadMetaEnv", # 指向 meta_uav_env.py 里的类
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.meta_uav_env_cfg:UavPayloadMetaEnvCfg", # 指向 meta_uav_env_cfg.py 里的类
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.meta_ppo_cfg:MetaPPORunnerCfg",       # 指向 meta_ppo_cfg.py 里的类
    },
)
