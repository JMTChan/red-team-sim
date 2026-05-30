"""
Train / resume the Red Team agent and export a browser-deployable ONNX policy.

Algorithm: MaskablePPO (sb3-contrib). The action space is Discrete(N) but only a
handful of nodes are legal moves at any step (adjacent, non-firewalled). Action
masking removes the illegal actions from the policy distribution during training,
so the agent stops wasting capacity on ~20 invalid actions per step -- this is the
key to it actually learning on dense topologies. The exported ONNX model still
outputs raw [batch, N] logits; the frontend already masks illegal moves before
argmax, so the inference contract is unchanged.

Behaviour:
  * If training/latest_model.zip exists AND is compatible -> load and CONTINUE.
  * Otherwise (missing or incompatible, e.g. switching algorithms) -> fresh start.
  * Each env is wrapped in Monitor, so rollout/ep_rew_mean (average score) and
    rollout/ep_len_mean (average steps per episode) show up in the logs.
  * Training is wall-clock bounded (default 5.5h) with periodic checkpoints, so it
    finishes inside GitHub's 6h runner limit and never loses progress.

Run from the repository root:  python training/train.py
"""

from __future__ import annotations

import os
import time

import numpy as np
import torch as th

from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback

from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker

from env import RedTeamNetworkEnv

# --------------------------------------------------------------------- config
N_NODES = 24                      # MUST match lib/constants.ts NODE_COUNT on the frontend
N_ENVS = 8                        # parallel rollout environments
MAX_SECONDS = float(os.environ.get("TRAIN_MAX_SECONDS", 5.5 * 3600))  # 5.5 hours
CHECKPOINT_EVERY = 50_000         # timesteps between safety saves
TOTAL_TIMESTEPS = 50_000_000      # an upper bound; the time callback stops us first
# Curriculum: ramp the env difficulty ceiling from 0 -> 1 over this many steps, so the
# agent learns on easy boards first and graduates to dense boards + a smart defender.
CURRICULUM_RAMP_STEPS = 4_000_000

# Larger network to handle the high-dimensional graph observation (node states +
# the full N*N adjacency matrix). Masking matters more than raw size, but a roomy
# net helps the value function fit the returns.
NET_ARCH = dict(pi=[512, 512, 256], vf=[512, 512, 256])

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
MODEL_ZIP = os.path.join(HERE, "latest_model.zip")
ONNX_OUT = os.path.join(REPO, "public", "red_team_agent.onnx")


def _mask_fn(env):
    return env.unwrapped.action_masks()


def make_env():
    env = RedTeamNetworkEnv(n_nodes=N_NODES)
    env = Monitor(env)                 # -> rollout/ep_rew_mean + ep_len_mean
    env = ActionMasker(env, _mask_fn)  # -> legal-move masking for MaskablePPO
    return env


class TimeAndCheckpointCallback(BaseCallback):
    """Stop training once MAX_SECONDS elapses; checkpoint periodically en route."""

    def __init__(self, max_seconds: float, checkpoint_every: int, model_path: str):
        super().__init__()
        self.max_seconds = max_seconds
        self.checkpoint_every = checkpoint_every
        self.model_path = model_path
        self.start = None
        self.last_ckpt = 0

    def _on_training_start(self) -> None:
        self.start = time.time()

    def _on_step(self) -> bool:
        if self.num_timesteps - self.last_ckpt >= self.checkpoint_every:
            self.last_ckpt = self.num_timesteps
            self.model.save(self.model_path)
            elapsed = time.time() - self.start
            print(f"[checkpoint] {self.num_timesteps:,} steps | {elapsed/60:.1f} min", flush=True)
        if time.time() - self.start >= self.max_seconds:
            print(f"[time-limit] {self.max_seconds/3600:.2f}h reached -> stopping", flush=True)
            return False  # ends learn() gracefully
        return True


class CurriculumCallback(BaseCallback):
    """Ramp the env difficulty ceiling from 0 -> 1 over `ramp_steps` timesteps and push
    it to every (vectorized) env, so training starts on easy/sparse boards and graduates
    to dense boards with a smart, route-targeting defender. The env samples a difficulty
    window below this ceiling each episode, so it always sees some spread."""

    def __init__(self, ramp_steps: int, update_every: int = 2048):
        super().__init__()
        self.ramp_steps = max(1, int(ramp_steps))
        self.update_every = int(update_every)
        self._last_d = -1.0

    def _on_step(self) -> bool:
        if self.n_calls % self.update_every == 0:
            d = min(1.0, self.num_timesteps / self.ramp_steps)
            # Only broadcast on a meaningful change to avoid needless env calls.
            if abs(d - self._last_d) >= 0.01 or d >= 1.0:
                self.training_env.env_method("set_difficulty", d)
                self._last_d = d
        return True


class OnnxableSB3Policy(th.nn.Module):
    """Wraps the SB3 policy so ONNX receives the four named Dict tensors and
    returns the raw action logits (shape [batch, N]). Computed straight through
    the feature extractor + actor head so it works for the maskable policy and
    never applies masks (the frontend masks at inference)."""

    def __init__(self, policy):
        super().__init__()
        self.policy = policy

    def forward(self, node_states, adjacency_matrix, target_node, current_position):
        obs = {
            "node_states": node_states,
            "adjacency_matrix": adjacency_matrix,
            "target_node": target_node,
            "current_position": current_position,
        }
        features = self.policy.extract_features(obs)
        latent_pi, _ = self.policy.mlp_extractor(features)
        return self.policy.action_net(latent_pi)  # [B, N] raw logits


def export_onnx(model) -> None:
    os.makedirs(os.path.dirname(ONNX_OUT), exist_ok=True)
    model.policy.set_training_mode(False)
    onnxable = OnnxableSB3Policy(model.policy).eval()

    dummy = (
        th.zeros(1, N_NODES, dtype=th.float32),            # node_states
        th.zeros(1, N_NODES * N_NODES, dtype=th.float32),  # adjacency_matrix
        th.zeros(1, N_NODES, dtype=th.float32),            # target_node
        th.zeros(1, N_NODES, dtype=th.float32),            # current_position
    )
    th.onnx.export(
        onnxable,
        dummy,
        ONNX_OUT,
        input_names=["node_states", "adjacency_matrix", "target_node", "current_position"],
        output_names=["logits"],
        dynamic_axes={
            "node_states": {0: "batch"},
            "adjacency_matrix": {0: "batch"},
            "target_node": {0: "batch"},
            "current_position": {0: "batch"},
            "logits": {0: "batch"},
        },
        opset_version=17,
    )
    print(f"[export] wrote {ONNX_OUT}", flush=True)


def build_fresh(venv):
    print("[init] creating fresh MaskablePPO model", flush=True)
    return MaskablePPO(
        "MultiInputPolicy",
        venv,
        n_steps=1024,
        batch_size=1024,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        ent_coef=0.01,
        learning_rate=3e-4,
        policy_kwargs=dict(net_arch=NET_ARCH),
        verbose=1,
        device="cpu",
    )


def main():
    th.manual_seed(0)
    np.random.seed(0)

    venv = DummyVecEnv([make_env for _ in range(N_ENVS)])

    model = None
    if os.path.exists(MODEL_ZIP):
        try:
            print(f"[resume] loading {MODEL_ZIP}", flush=True)
            model = MaskablePPO.load(MODEL_ZIP, env=venv, device="cpu")
        except Exception as exc:  # incompatible checkpoint (old algo / net) -> fresh
            print(f"[resume-failed] {exc}\n[resume-failed] starting fresh instead", flush=True)
            model = None
    if model is None:
        model = build_fresh(venv)

    callbacks = [
        TimeAndCheckpointCallback(MAX_SECONDS, CHECKPOINT_EVERY, MODEL_ZIP),
        CurriculumCallback(CURRICULUM_RAMP_STEPS),
    ]
    model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=callbacks, reset_num_timesteps=False)

    model.save(MODEL_ZIP)
    print(f"[save] overwrote {MODEL_ZIP}", flush=True)
    export_onnx(model)
    print("[done]", flush=True)


if __name__ == "__main__":
    main()
