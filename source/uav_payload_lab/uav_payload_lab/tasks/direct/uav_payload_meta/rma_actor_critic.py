# rma_actor_critic.py
# Phase-1 RMA: jointly train base policy π and env-factor encoder μ using PPO.

from __future__ import annotations

import torch
import torch.nn as nn
from tensordict import TensorDict
from torch.distributions import Normal
from typing import Any, NoReturn

from rsl_rl.networks import MLP, EmpiricalNormalization


class RMAActorCritic(nn.Module):
    """RMA Phase-1 ActorCritic.

    Expected env output:
      obs["policy"] shape: (num_envs, proprio_dim + privileged_dim)
        - proprio = obs[:proprio_dim]           (e.g., 17 dims)
        - privileged e = obs[proprio_dim:]      (e.g., 5 dims: m,l,w)
    Phase-1 (teacher): use_mu=True -> z = μ(e), π takes [proprio, z]
    Phase-2 (deployment): use_mu=False -> tail is treated as z directly (from φ(history))
    """
    is_recurrent: bool = False

    def __init__(
        self,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        num_actions: int,
        # keep same knobs as original ActorCritic
        actor_obs_normalization: bool = False,
        critic_obs_normalization: bool = False,
        actor_hidden_dims: tuple[int] | list[int] = [256, 256, 256],
        critic_hidden_dims: tuple[int] | list[int] = [256, 256, 256],
        activation: str = "elu",
        init_noise_std: float = 1.0,
        noise_std_type: str = "log",          # IMPORTANT: "log" guarantees std>0
        state_dependent_std: bool = False,
        # -------- RMA knobs --------
        proprio_obs_dim: int = 17,
        privileged_obs_dim: int = 5,
        z_dim: int = 5,
        z_exp_dim: int = 2,                   # optional split for logging
        use_mu: bool = True,                  # Phase-1 True, Phase-2 False
        mu_hidden_dims: tuple[int] | list[int] = [64, 64],
        mu_activation: str | None = None,     # default use `activation`
        **kwargs: dict[str, Any],
    ) -> None:
        if kwargs:
            # keep it strict: surface unexpected args early
            print(f"[RMA] Ignoring unexpected kwargs: {list(kwargs.keys())}")

        super().__init__()
        self.obs_groups = obs_groups
        self.num_actions = num_actions
        self.state_dependent_std = state_dependent_std

        self.proprio_dim = int(proprio_obs_dim)
        self.priv_dim = int(privileged_obs_dim)
        self.z_dim = int(z_dim)
        self.z_exp_dim = int(z_exp_dim)
        self.use_mu = bool(use_mu)
        self.probe = nn.Linear(self.z_dim, self.priv_dim, bias=True)

        # -------- infer raw obs dim from obs_groups (same as original ActorCritic) --------
        num_actor_obs_raw = 0
        for g in obs_groups["policy"]:
            assert len(obs[g].shape) == 2, "Only supports 1D observations."
            num_actor_obs_raw += obs[g].shape[-1]
        num_critic_obs_raw = 0
        for g in obs_groups["critic"]:
            assert len(obs[g].shape) == 2, "Only supports 1D observations."
            num_critic_obs_raw += obs[g].shape[-1]

        expected_raw = self.proprio_dim + self.priv_dim
        if num_actor_obs_raw != expected_raw or num_critic_obs_raw != expected_raw:
            raise ValueError(
                f"Obs dim mismatch: expected raw={expected_raw} (proprio {self.proprio_dim}+priv {self.priv_dim}), "
                f"got actor_raw={num_actor_obs_raw}, critic_raw={num_critic_obs_raw}. "
                f"Check env obs concat order & cfg observation_space."
            )

        # -------- μ: e -> z --------
        mu_act = mu_activation if mu_activation is not None else activation
        self.mu = MLP(self.priv_dim, self.z_dim, mu_hidden_dims, mu_act)

        # -------- actor/critic take [proprio, z] --------
        num_actor_obs = self.proprio_dim + self.z_dim
        num_critic_obs = self.proprio_dim + self.z_dim

        # Actor
        if self.state_dependent_std:
            # keep compatibility if you ever enable it
            self.actor = MLP(num_actor_obs, [2, num_actions], actor_hidden_dims, activation)
        else:
            self.actor = MLP(num_actor_obs, num_actions, actor_hidden_dims, activation)
        print(f"[RMA] Actor MLP: {self.actor}")

        # Critic
        self.critic = MLP(num_critic_obs, 1, critic_hidden_dims, activation)
        print(f"[RMA] Critic MLP: {self.critic}")

        # Obs normalization (on transformed obs, not raw)
        self.actor_obs_normalization = actor_obs_normalization
        self.critic_obs_normalization = critic_obs_normalization
        self.actor_obs_normalizer = EmpiricalNormalization(num_actor_obs) if actor_obs_normalization else nn.Identity()
        self.critic_obs_normalizer = EmpiricalNormalization(num_critic_obs) if critic_obs_normalization else nn.Identity()

        # Action noise: MUST be log to guarantee std>0
        self.noise_std_type = noise_std_type
        if self.state_dependent_std:
            # same as original behavior
            torch.nn.init.zeros_(self.actor[-2].weight[num_actions:])
            if self.noise_std_type == "scalar":
                torch.nn.init.constant_(self.actor[-2].bias[num_actions:], init_noise_std)
            elif self.noise_std_type == "log":
                torch.nn.init.constant_(self.actor[-2].bias[num_actions:], torch.log(torch.tensor(init_noise_std + 1e-7)))
            else:
                raise ValueError("noise_std_type must be 'scalar' or 'log'")
        else:
            if self.noise_std_type == "scalar":
                # NOTE: this is unconstrained -> can go negative and crash Normal(...)
                self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
            elif self.noise_std_type == "log":
                self.log_std = nn.Parameter(torch.log(init_noise_std * torch.ones(num_actions)))
            else:
                raise ValueError("noise_std_type must be 'scalar' or 'log'")

        self.distribution = None
        Normal.set_default_validate_args(False)

        # cached for debugging / Phase-2 distillation later
        self.last_z: torch.Tensor | None = None
        self.last_z_exp: torch.Tensor | None = None
        self.last_z_imp: torch.Tensor | None = None
        self.last_e: torch.Tensor | None = None  # <--- NEW: cache privileged e for phys anchor

    def reset(self, dones: torch.Tensor | None = None) -> None:
        pass

    def forward(self) -> NoReturn:
        raise NotImplementedError

    @property
    def action_mean(self) -> torch.Tensor:
        return self.distribution.mean

    @property
    def action_std(self) -> torch.Tensor:
        return self.distribution.stddev

    @property
    def entropy(self) -> torch.Tensor:
        return self.distribution.entropy().sum(dim=-1)

    # ---- core: build transformed obs = [proprio, z] ----
    def _split_raw(self, raw: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        proprio = raw[..., : self.proprio_dim]
        tail = raw[..., self.proprio_dim : self.proprio_dim + self.priv_dim]  # e (Phase1) or z (Phase2)
        return proprio, tail

    def _compute_z(self, tail: torch.Tensor) -> torch.Tensor:
        if self.use_mu:
            self.last_e = tail
            z = self.mu(tail)
        else:
            # Phase2: tail is already z_hat from φ(history)
            self.last_e = None
            if tail.shape[-1] != self.z_dim:
                raise ValueError(f"Phase2 expects tail dim={self.z_dim}, got {tail.shape[-1]}")
            z = tail
        self.last_z = z
        self.last_z_exp = z[..., : self.z_exp_dim]
        self.last_z_imp = z[..., self.z_exp_dim :]
        return z

    def _transform(self, raw: torch.Tensor) -> torch.Tensor:
        proprio, tail = self._split_raw(raw)
        z = self._compute_z(tail)
        return torch.cat([proprio, z], dim=-1)

    # ---- override: actor/critic obs come from transformed obs ----
    def get_actor_obs(self, obs: TensorDict) -> torch.Tensor:
        obs_list = [obs[g] for g in self.obs_groups["policy"]]
        raw = torch.cat(obs_list, dim=-1)
        return self._transform(raw)

    def get_critic_obs(self, obs: TensorDict) -> torch.Tensor:
        obs_list = [obs[g] for g in self.obs_groups["critic"]]
        raw = torch.cat(obs_list, dim=-1)
        return self._transform(raw)

    def _update_distribution(self, obs: TensorDict) -> None:
        x = self.get_actor_obs(obs)
        x = self.actor_obs_normalizer(x)

        if self.state_dependent_std:
            mean_and_std = self.actor(x)
            if self.noise_std_type == "scalar":
                mean, std = torch.unbind(mean_and_std, dim=-2)
            elif self.noise_std_type == "log":
                mean, log_std = torch.unbind(mean_and_std, dim=-2)
                std = torch.exp(log_std)
            else:
                raise ValueError("noise_std_type must be 'scalar' or 'log'")
        else:
            mean = self.actor(x)
            if self.noise_std_type == "scalar":
                std = self.std.expand_as(mean)
            elif self.noise_std_type == "log":
                std = torch.exp(self.log_std).expand_as(mean)
            else:
                raise ValueError("noise_std_type must be 'scalar' or 'log'")

        # hard safety clamp (prevents numeric blow-ups)
        std = torch.nan_to_num(std, nan=1e-3, posinf=10.0, neginf=1e-3).clamp(min=1e-6, max=10.0)
        self.distribution = Normal(mean, std)

    # rma_actor_critic.py

    def act(self, obs: TensorDict, **kwargs):
        self._update_distribution(obs)
        actions = self.distribution.sample()

        # --- debug cache: DO NOT mutate obs tensordict ---
        if self.last_z is not None:
            with torch.no_grad():
                z = self.last_z.detach()
                self.debug_z_mean = z.mean(dim=0).cpu()          # (z_dim,)
                self.debug_z_std  = z.std(dim=0).cpu()           # (z_dim,)
                self.debug_z_absmax = float(z.abs().max().cpu()) # scalar

                # teacher(Phase-1) only: probe MSE from z -> privileged e
                if self.use_mu:
                    raw = torch.cat([obs[g] for g in self.obs_groups["policy"]], dim=-1)
                    _, e = self._split_raw(raw)          # e: (N, priv_dim)
                    e_hat = self.probe(z)                # (N, priv_dim)
                    self.debug_probe_mse = float(((e_hat - e).pow(2).mean()).cpu())

        return actions



    def act_inference(self, obs: TensorDict) -> torch.Tensor:
        x = self.get_actor_obs(obs)
        x = self.actor_obs_normalizer(x)
        if self.state_dependent_std:
            return self.actor(x)[..., 0, :]
        else:
            return self.actor(x)

    def evaluate(self, obs: TensorDict, **kwargs: dict[str, Any]) -> torch.Tensor:
        x = self.get_critic_obs(obs)
        x = self.critic_obs_normalizer(x)
        return self.critic(x)

    def get_actions_log_prob(self, actions: torch.Tensor) -> torch.Tensor:
        return self.distribution.log_prob(actions).sum(dim=-1)

    def update_normalization(self, obs: TensorDict) -> None:
        if self.actor_obs_normalization:
            self.actor_obs_normalizer.update(self.get_actor_obs(obs))
        if self.critic_obs_normalization:
            self.critic_obs_normalizer.update(self.get_critic_obs(obs))
    # --- in rma_actor_critic.py ---
    def aux_phys_loss(self) -> torch.Tensor | None:
        """Anchor z_exp to privileged slow vars: ||z_exp - e[:z_exp_dim]||^2."""
        if (not self.use_mu) or (self.last_z_exp is None) or (self.last_e is None):
            return None
        # e 的前两维就是 m_norm, l_norm（你的 env 就是这么拼的）
        target = self.last_e[..., : self.z_exp_dim]
        return (self.last_z_exp - target).pow(2).mean()

    def load_state_dict(self, state_dict, strict: bool = True):
        # Allow older checkpoints without probe.*
        allowed_missing = {"probe.weight", "probe.bias"}

        res = super().load_state_dict(state_dict, strict=False)

        # If missing keys beyond probe.*, treat as real error
        bad_missing = [k for k in res.missing_keys if k not in allowed_missing]
        if strict and (len(bad_missing) > 0):
            raise RuntimeError(f"Missing keys not allowed: {bad_missing}")

        # Unexpected keys are usually a real mismatch
        if strict and (len(res.unexpected_keys) > 0):
            raise RuntimeError(f"Unexpected keys: {res.unexpected_keys}")

        return res
