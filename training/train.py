"""
Train / resume the Red Team PPO agent and export a browser-deployable ONNX policy.

Behaviour:
  * If training/latest_model.zip exists  -> load and CONTINUE training.
  * Otherwise                            -> initialize a fresh PPO model.
  * Training is bounded by wall-clock time (default 5.5h) via a callback so it
    always finishes well inside GitHub's 6h runner limit, regardless of CPU speed.
    Checkpoints are written periodically, so a killed run never loses progress.
  * On exit it overwrites training/latest_model.zip and exports
    public/red_team_agent.onnx (raw action logits, so the frontend can mask
    illegal moves before choosing the agent's next node).

Run from the repository root:  python training/train.py
"""

from __future__ import annotations

import os
import time

import numpy as np
import torch as th

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback

from env import RedTeamNetworkEnv, OPEN  # noqa: F401  (run with cwd=training or via sys.path)

# --------------------------------------------------------------------- config
N_NODES = 24                      # MUST match lib/constants.ts NODE_COUNT on the frontend
N_ENVS = 8                        # parallel rollout environments
MAX_SECONDS = float(os.environ.get("TRAIN_MAX_SECONDS", 5.5 * 3600))  # 5.5 hours
CHECKPOINT_EVERY = 50_000         # timesteps between safety saves
TOTAL_TIMESTEPS = 50_000_000      # an upper bound; the time callback stops us first

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
MODEL_ZIP = os.path.join(HERE, "latest_model.zip")
ONNX_OUT = os.path.join(REPO, "public", "red_team_agent.onnx")


def make_env():
    return RedTeamNetworkEnv(n_nodes=N_NODES)


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


class OnnxableSB3Policy(th.nn.Module):
    """Wraps the SB3 policy so ONNX receives the four named Dict tensors and
    returns the raw action logits (shape [batch, N])."""

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
        distribution = self.policy.get_distribution(obs)
        return distribution.distribution.logits  # [B, N]


def export_onnx(model: PPO) -> None:
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


def main():
    th.manual_seed(0)
    np.random.seed(0)

    venv = DummyVecEnv([make_env for _ in range(N_ENVS)])

    if os.path.exists(MODEL_ZIP):
        print(f"[resume] loading {MODEL_ZIP}", flush=True)
        model = PPO.load(MODEL_ZIP, env=venv, device="cpu")
    else:
        print("[init] creating fresh PPO model", flush=True)
        model = PPO(
            "MultiInputPolicy",
            venv,
            n_steps=1024,
            batch_size=1024,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            ent_coef=0.01,
            learning_rate=3e-4,
            policy_kwargs=dict(net_arch=[512, 512, 256]),
            verbose=1,
            device="cpu",
        )

    callback = TimeAndCheckpointCallback(MAX_SECONDS, CHECKPOINT_EVERY, MODEL_ZIP)
    model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=callback, reset_num_timesteps=False)

    model.save(MODEL_ZIP)
    print(f"[save] overwrote {MODEL_ZIP}", flush=True)
    export_onnx(model)
    print("[done]", flush=True)


if __name__ == "__main__":
    main()
