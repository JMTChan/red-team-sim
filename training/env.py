"""
Red Team Network Simulator -- custom Gymnasium environment.

The Red Team agent must pathfind across a procedurally generated network from a
random start node to a dynamic target ("database") node, while avoiding the Blue
Team's defenses. During training we randomize the topology AND the defenses every
episode (domain randomization) so the agent cannot memorize a fixed solution and
learns a general policy that the live, human-driven defenses can challenge.

Observation (Dict) -- mirrors the frontend state exactly so the exported ONNX
model receives identically-shaped tensors at inference time:

    node_states        Box(N,)   float32   0=Open 1=Firewall 2=Honeypot 3=Corrupted
    adjacency_matrix   Box(N*N,) float32   flattened NxN, 1=connected 0=severed
    target_node        Box(N,)   float32   one-hot of the target node id
    current_position   Box(N,)   float32   one-hot of the agent's current node id

Action: Discrete(N) -- the node id the agent attempts to move to.
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
        # Probability that a non-start / non-target node carries a defense during
        # training. This teaches the agent to route around the Blue Team.
        firewall_prob: float = 0.12,
        honeypot_prob: float = 0.10,
        corrupt_prob: float = 0.08,
        shaping_coef: float = 1.0,
    ):
        super().__init__()
        self.n = int(n_nodes)
        self.max_steps = int(max_steps)
        self.extra_edge_prob = float(extra_edge_prob)
        self.firewall_prob = float(firewall_prob)
        self.honeypot_prob = float(honeypot_prob)
        self.corrupt_prob = float(corrupt_prob)
        self.shaping_coef = float(shaping_coef)

        n = self.n
        self.observation_space = spaces.Dict(
            {
                "node_states": spaces.Box(low=0.0, high=3.0, shape=(n,), dtype=np.float32),
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
        self.target = 0
        self.steps = 0
        self._corrupt_next = False  # set when the agent lands on a CORRUPTED node

    # ------------------------------------------------------------------ utils
    def _random_connected_graph(self) -> np.ndarray:
        """Random connected undirected graph: a random spanning tree + extra edges."""
        n = self.n
        adj = np.zeros((n, n), dtype=np.float32)
        nodes = self.np_random.permutation(n)
        # Spanning tree guarantees connectivity.
        for i in range(1, n):
            a = nodes[i]
            b = nodes[self.np_random.integers(0, i)]
            adj[a, b] = adj[b, a] = 1.0
        # Sprinkle extra edges for alternative routes.
        for a in range(n):
            for b in range(a + 1, n):
                if adj[a, b] == 0.0 and self.np_random.random() < self.extra_edge_prob:
                    adj[a, b] = adj[b, a] = 1.0
        return adj

    def _bfs_distance(self, src: int, dst: int) -> int:
        """Shortest hop count from src to dst over passable nodes (firewalls block)."""
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
                        if self.node_states[v] == FIREWALL and v != dst:
                            continue  # cannot pass through firewalls
                        if v == dst:
                            return dist
                        visited[v] = True
                        nxt.append(v)
            frontier = nxt
        return n + 1  # unreachable -> large finite penalty proxy

    def _neighbors(self, node: int) -> list[int]:
        return [v for v in range(self.n) if self.adj[node, v] == 1.0]

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

        # Pick distinct start + target nodes.
        self.current = int(self.np_random.integers(0, n))
        self.target = int(self.np_random.integers(0, n))
        while self.target == self.current:
            self.target = int(self.np_random.integers(0, n))

        # Randomize defenses everywhere except start and target.
        self.node_states = np.zeros(n, dtype=np.float32)
        for node in range(n):
            if node in (self.current, self.target):
                continue
            r = self.np_random.random()
            if r < self.firewall_prob:
                self.node_states[node] = FIREWALL
            elif r < self.firewall_prob + self.honeypot_prob:
                self.node_states[node] = HONEYPOT
            elif r < self.firewall_prob + self.honeypot_prob + self.corrupt_prob:
                self.node_states[node] = CORRUPTED

        # Guarantee at least one open route exists; if firewalls fully wall off the
        # target, clear firewalls until a path opens.
        guard = 0
        while self._bfs_distance(self.current, self.target) > n and guard < n:
            fw = np.where(self.node_states == FIREWALL)[0]
            if len(fw) == 0:
                break
            self.node_states[fw[0]] = OPEN
            guard += 1

        self.steps = 0
        self._corrupt_next = False
        return self._obs(), {}

    # ------------------------------------------------------------------ step
    def step(self, action: int):
        self.steps += 1
        action = int(action)
        reward = -1.0  # time penalty: shorter breaches are better
        terminated = False
        truncated = False
        info = {}

        # A corrupted node hijacks the NEXT move: replace the chosen action with a
        # random valid neighbor of the current node.
        if self._corrupt_next:
            self._corrupt_next = False
            neigh = self._neighbors(self.current)
            if neigh:
                action = int(self.np_random.choice(neigh))
            info["corrupted_move"] = True

        dist_before = self._bfs_distance(self.current, self.target)

        # Validate the move.
        if action == self.current:
            reward -= 1.0  # wasting a tick standing still
        elif self.adj[self.current, action] != 1.0:
            reward -= 5.0  # illegal (non-adjacent / severed) move, agent stays put
        elif self.node_states[action] == FIREWALL:
            reward -= 5.0  # firewalls are impassable, agent is blocked
        else:
            # Valid traversal.
            self.current = action
            state = self.node_states[self.current]

            if self.current == self.target:
                reward += 100.0  # database breached
                terminated = True
                info["result"] = "breach"
            elif state == HONEYPOT:
                reward -= 100.0  # trapped: massive penalty, episode ends
                terminated = True
                info["result"] = "honeypot"
            elif state == CORRUPTED:
                self._corrupt_next = True  # next move will be randomized

        # Potential-based shaping toward the target (speeds up learning a lot).
        if not terminated:
            dist_after = self._bfs_distance(self.current, self.target)
            reward += self.shaping_coef * (dist_before - dist_after)

        if not terminated and self.steps >= self.max_steps:
            truncated = True
            info["result"] = info.get("result", "timeout")

        return self._obs(), float(reward), terminated, truncated, info
