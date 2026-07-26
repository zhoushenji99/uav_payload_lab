import importlib.util
from pathlib import Path
import sys
import unittest

import torch
from tensordict import TensorDict


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "source/uav_payload_lab/uav_payload_lab/tasks/direct/uav_payload_sim2real/rma_actor_critic.py"
)


def _load_module():
    spec = importlib.util.spec_from_file_location("sim2real_rma_actor_critic", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _make_policy(module, context_mode: str):
    obs = TensorDict(
        {
            "policy": torch.zeros(4, 26),
            "critic": torch.zeros(4, 26),
        },
        batch_size=[4],
    )
    return module.RMAActorCritic(
        obs=obs,
        obs_groups={"policy": ["policy"], "critic": ["critic"]},
        num_actions=4,
        actor_hidden_dims=[16],
        critic_hidden_dims=[16],
        mu_hidden_dims=[8],
        context_mode=context_mode,
        proprio_obs_dim=21,
        privileged_obs_dim=5,
        z_dim=5,
        z_exp_dim=2,
    )


class RMAContextModeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.module = _load_module()

    def test_split_hard_structural_context_is_exact_identity(self):
        policy = _make_policy(self.module, "split_hard")
        privileged = torch.tensor(
            [
                [0.0, 1.0, -0.3, 0.2, 0.7],
                [0.12345679, 0.8765432, 0.0, -1.0, 1.0],
            ],
            dtype=torch.float32,
        )

        context = policy.mu(privileged)

        self.assertTrue(torch.equal(context[:, :2], privileged[:, :2]))
        self.assertFalse(hasattr(policy, "mu_exp"))
        self.assertTrue(hasattr(policy, "mu_imp"))
        self.assertEqual(policy.context_mode, "split_hard")

    def test_split_soft_has_two_independent_learned_branches(self):
        policy = _make_policy(self.module, "split_soft")

        self.assertTrue(hasattr(policy, "mu_exp"))
        self.assertTrue(hasattr(policy, "mu_imp"))
        self.assertFalse(hasattr(policy, "context_encoder"))
        self.assertEqual(policy.context_mode, "split_soft")

    def test_monolithic_has_one_joint_context_encoder(self):
        policy = _make_policy(self.module, "monolithic")

        self.assertTrue(hasattr(policy, "context_encoder"))
        self.assertFalse(hasattr(policy, "mu_exp"))
        self.assertFalse(hasattr(policy, "mu_imp"))
        self.assertEqual(policy.context_mode, "monolithic")
        context = policy.mu(torch.zeros(3, 5))
        self.assertEqual(tuple(context.shape), (3, 5))

    def test_checkpoint_from_another_context_mode_is_rejected(self):
        hard = _make_policy(self.module, "split_hard")
        soft = _make_policy(self.module, "split_soft")

        with self.assertRaisesRegex(RuntimeError, "context mode|architecture|checkpoint"):
            hard.load_state_dict(soft.state_dict(), strict=True)

    def test_hard_mode_physics_loss_is_exact_zero(self):
        policy = _make_policy(self.module, "split_hard")
        obs = TensorDict(
            {"policy": torch.randn(8, 26), "critic": torch.randn(8, 26)},
            batch_size=[8],
        )

        loss = policy.compute_physics_loss(obs)

        self.assertEqual(float(loss), 0.0)


if __name__ == "__main__":
    unittest.main()
