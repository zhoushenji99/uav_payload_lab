# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym

from . import agents


# Register the measured-parameter Sim2Real environment.
gym.register(
    id="Isaac-Uav-Sim2Real-v0",
    entry_point=f"{__name__}.meta_uav_env:UavPayloadMetaEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.meta_uav_env_cfg:UavPayloadMetaEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.meta_ppo_cfg:MetaPPORunnerCfg",
    },
)
