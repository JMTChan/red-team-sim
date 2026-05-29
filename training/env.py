"""
Red Team Network Simulator -- custom Gymnasium environment (OVERLOAD build).

The Red Team agent pathfinds across a procedurally generated network from a random
start toward a high-value target, under randomized Blue-Team defenses. Domain
randomization (topology + defenses every episode) forces a general policy.

This build layers FIVE training mechanics on top of basic pathfinding, while
keeping the observation/action SHAPES identical to the original (so the exported
ONNX contract is unchanged and the model stays a drop-in replacement):

  1. Penetrable firewalls   -- a firewall is attackable: each attempt to step onto
                               it breaks through with probability FIREWALL_BREACH;
                               otherwise the agent is rebuffed (wastes the tick) and
                               makes noise (raises detection).
  2. Leaky honeypots        -- stepping onto a honeypot traps the agent only with
                               probability HONEYPOT_TRAP; otherwise it slips past.
  3. Tarpit / slow nodes    -- a new node state SLOW that passes only with
                               probability SLOW_PASS, otherwise stalls a tick.
  4. Multi-objective        -- the agent must reach a FOOTHOLD node first, then the
                               DATABASE. The active goal is exposed via target_node
                               (sequenced by the env), so a memoryless policy can
                               still follow it; no extra observation needed.
  5. Detection pressure     -- an internal "noise" meter rises each step (and extra
                               on failed firewall attempts). If it caps, the agent
                               is caught. It is NOT in the observation, so the agent
                               can't read it directly -- it instead learns a quieter,
                               faster policy via the reward. (Agent-observable stealth
                               would require a new input = a contract change; left for
                               a later phase.)

Observation (Dict) -- SHAPE-IDENTICAL to the original:
    node_states        Box(N,)   0=Open 1=Firewall 2=Honeypot 3=Corrupted 4=Slow
    adjacency_matrix   Box(N*N,) flattened NxN, 1=connected 0=severed
    target_node        Box(N,)   one-hot of the CURRENT active goal (foothold->db)
    current_position   Box(N,)   one-hot of the agent's current node
Action: Discrete(N) -- node id the agent attempts to move to.
"""

from __future__ import annotations

import numpy as np
import gymnasium as gym
from gymnasium import spaces

# Node-state constants (kept in sync with lib/constants.ts on the frontend).
OPEN = 0
FIREWALL = 1
HONEYPOT = 2
CORRUPTED = 3
SLOW = 4  # tarpit: passable but slow


def _one_hot(idx: int, n: int) -> np.ndarray:
    v = np.zeros(n, dtype=np.float32)
    v[idx] = 1.0
    return v


class RedTeamNetworkEnv(gym.Env):
    """Goal-conditioned pathfinding under adversarial, randomized defenses."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        n_nodes: int = 24,
        max_steps: int = 90,
        extra_edge_prob: float = 0.18,
        # Defense spawn probabilities per non-anchor node.
        firewall_prob: float = 0.12,
        honeypot_prob: float = 0.10,
        corrupt_prob: float = 0.08,
        slow_prob: float = 0.10,
        shaping_coef: float = 1.0,
        # Mechanic tunables.
        firewall_breach: float = 0.35,   # P(break through a firewall per attempt)
        slow_pass: float = 0.60,         # P(cross a tarpit per attempt)
        honeypot_trap: float = 0.70,     # P(a honeypot traps you when stepped on)
        detection_per_step: float = 0.014,
        detection_on_fail: float = 0.05,  # extra noise on a failed firewall attempt
        foothold_bonus: float = 30.0,
    ):
        super().__init__()
        self.n = int(n_nodes)
        self.max_steps = int(max_steps)
        self.extra_edge_prob = float(extra_edge_prob)
        self.firewall_prob = float(firewall_prob)
        self.honeypot_prob = float(honeypot_prob)
        self.corrupt_prob = float(corrupt_prob)
        self.slow_prob = float(slow_prob)
        self.shaping_coef = float(shaping_coef)
        self.firewall_breach = float(firewall_breach)
        self.slow_pass = float(slow_pass)
        self.honeypot_trap = float(honeypot_trap)
        self.detection_per_step = float(detection_per_step)
        self.detection_on_fail = float(detection_on_fail)
        self.foothold_bonus = float(foothold_bonus)

        n = self.n
        self.observation_space = spaces.Dict(
            {
                "node_states": spaces.Box(low=0.0, high=4.0, shape=(n,), dtype=np.float32),
                "adjacency_matrix": spaces.Box(low=0.0, high=1.0, shape=(n * n,), dtype=np.float32),
                "target_node": spaces.Box(low=0.0, high=1.0, shape=(n,), dtype=np.float32),
                "current_position": spaces.Box(low=0.0, high=1.0, shape=(n,), dtype=np.float32),
            }
        )
        self.action_space = spaces.Discrete(n)

        # Episode state (populated in reset()).
        self.adj = np.zeros((n, n), dtype=np.float32)
        self.node_states = np.zeros(n, dtype=np.float32)
        self.current = 0
        self.foothold = 0
        self.final_target = 0
        self.target = 0          # current active goal (foothold, then final_target)
        self.reached_foothold = False
        self.detection = 0.0
        self.steps = 0
        self._corrupt_next = False

    # ------------------------------------------------------------------ utils
    def _random_connected_graph(self) -> np.ndarray:
        n = self.n
        adj = np.zeros((n, n), dtype=np.float32)
        nodes = self.np_random.permutation(n)
        for i in range(1, n):
            a = nodes[i]
            b = nodes[self.np_random.integers(0, i)]
            adj[a, b] = adj[b, a] = 1.0
        for a in range(n):
            for b in range(a + 1, n):
                if adj[a, b] == 0.0 and self.np_random.random() < self.extra_edge_prob:
                    adj[a, b] = adj[b, a] = 1.0
        return adj

    def _bfs_distance(self, src: int, dst: int) -> int:
        """Pure hop distance over the graph. Firewalls/tarpits are penetrable now,
        so they don't block reachability -- the agent learns to avoid hazards via
        rewards, not via the shaping distance."""
        if src == dst:
            return 0
        n = self.n
        visited = np.zeros(n, dtype=bool)
        visited[src] = True
        frontier = [src]
        dist = 0
        while frontier:
            dist += 1
            nxt = []
            for u in frontier:
                for v in range(n):
                    if self.adj[u, v] == 1.0 and not visited[v]:
                        if v == dst:
                            return dist
                        visited[v] = True
                        nxt.append(v)
            frontier = nxt
        return n + 1

    def _neighbors(self, node: int) -> list[int]:
        return [v for v in range(self.n) if self.adj[node, v] == 1.0]

    def action_masks(self) -> np.ndarray:
        """Legal moves = every adjacent node (firewalls are now attackable, tarpits
        passable, honeypots steppable). Staying is masked unless fully isolated."""
        mask = np.zeros(self.n, dtype=bool)
        for v in range(self.n):
            if v != self.current and self.adj[self.current, v] == 1.0:
                mask[v] = True
        if not mask.any():
            mask[self.current] = True
        return mask

    def _obs(self) -> dict:
        return {
            "node_states": self.node_states.copy(),
            "adjacency_matrix": self.adj.flatten().copy(),
            "target_node": _one_hot(self.target, self.n),
            "current_position": _one_hot(self.current, self.n),
        }

    # ----------------------------------------------------------------- reset
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        n = self.n
        self.adj = self._random_connected_graph()

        # Three distinct anchor nodes: start -> foothold -> database.
        picks = self.np_random.permutation(n)[:3]
        self.current = int(picks[0])
        self.foothold = int(picks[1])
        self.final_target = int(picks[2])
        self.reached_foothold = False
        self.target = self.foothold  # first goal

        anchors = {self.current, self.foothold, self.final_target}
        self.node_states = np.zeros(n, dtype=np.float32)
        for node in range(n):
            if node in anchors:
                continue
            r = self.np_random.random()
            t = self.firewall_prob
            if r < t:
                self.node_states[node] = FIREWALL
                continue
            t += self.honeypot_prob
            if r < t:
                self.node_states[node] = HONEYPOT
                continue
            t += self.corrupt_prob
            if r < t:
                self.node_states[node] = CORRUPTED
                continue
            t += self.slow_prob
            if r < t:
                self.node_states[node] = SLOW

        self.detection = 0.0
        self.steps = 0
        self._corrupt_next = False
        return self._obs(), {}

    # ------------------------------------------------------------------ step
    def step(self, action: int):
        self.steps += 1
        action = int(action)
        reward = -1.0
        terminated = False
        truncated = False
        info = {}

        self.detection += self.detection_per_step

        if self._corrupt_next:
            self._corrupt_next = False
            neigh = self._neighbors(self.current)
            if neigh:
                action = int(self.np_random.choice(neigh))
            info["corrupted_move"] = True

        dist_before = self._bfs_distance(self.current, self.target)
        goal_advanced = False

        if action == self.current:
            reward -= 1.0  # standing still
        elif self.adj[self.current, action] != 1.0:
            reward -= 5.0  # illegal (non-adjacent) -- masked in training, rare
        else:
            state = self.node_states[action]
            moved = True

            # Resolve hazard-on-entry that can STOP the move.
            if state == FIREWALL:
                if self.np_random.random() < self.firewall_breach:
                    moved = True
                else:
                    moved = False
                    reward -= 1.0  # rebuffed, wasted the tick
                    self.detection += self.detection_on_fail  # noisy
            elif state == SLOW:
                if self.np_random.random() < self.slow_pass:
                    moved = True
                else:
                    moved = False
                    reward -= 0.5  # stalled in the tarpit

            if moved:
                self.current = action
                if self.current == self.target:
                    if not self.reached_foothold and self.target == self.foothold:
                        # Stage 1 complete: foothold taken, advance to the database.
                        self.reached_foothold = True
                        self.target = self.final_target
                        reward += self.foothold_bonus
                        goal_advanced = True
                        info["result"] = "foothold"
                    else:
                        reward += 100.0  # database breached
                        terminated = True
                        info["result"] = "breach"
                elif state == HONEYPOT:
                    if self.np_random.random() < self.honeypot_trap:
                        reward -= 100.0
                        terminated = True
                        info["result"] = "honeypot"
                    # else: honeypot missed, agent slips past
                elif state == CORRUPTED:
                    self._corrupt_next = True

        # Potential-based shaping toward the CURRENT goal (skip on the step the
        # goal switches, since dist_before/after refer to different targets).
        if not terminated and not goal_advanced:
            dist_after = self._bfs_distance(self.current, self.target)
            reward += self.shaping_coef * (dist_before - dist_after)

        # Detection cap: the agent is spotted and evicted.
        if not terminated and self.detection >= 1.0:
            reward -= 50.0
            terminated = True
            info["result"] = "detected"

        if not terminated and self.steps >= self.max_steps:
            truncated = True
            info["result"] = info.get("result", "timeout")

        return self._obs(), float(reward), terminated, truncated, info
