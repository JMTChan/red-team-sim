import { NODE_COUNT, OPEN, FIREWALL, type NodeState } from "./constants";

export interface NodePos {
  id: number;
  x: number; // 0..1 normalized layout coordinate
  y: number;
}

export interface NetworkState {
  adjacency: number[][]; // NxN, 1=connected 0=severed
  positions: NodePos[];
  nodeStates: NodeState[];
  start: number;
  target: number;
}

function mulberry32(seed: number) {
  return function () {
    seed |= 0;
    seed = (seed + 0x6d2b79f5) | 0;
    let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** Random connected undirected graph: spanning tree + extra edges. */
export function generateNetwork(seed = Date.now(), extraEdgeProb = 0.18): NetworkState {
  const rand = mulberry32(seed);
  const n = NODE_COUNT;
  const adjacency: number[][] = Array.from({ length: n }, () => Array(n).fill(0));

  // Shuffle node order for the spanning tree.
  const order = Array.from({ length: n }, (_, i) => i);
  for (let i = n - 1; i > 0; i--) {
    const j = Math.floor(rand() * (i + 1));
    [order[i], order[j]] = [order[j], order[i]];
  }
  for (let i = 1; i < n; i++) {
    const a = order[i];
    const b = order[Math.floor(rand() * i)];
    adjacency[a][b] = adjacency[b][a] = 1;
  }
  for (let a = 0; a < n; a++) {
    for (let b = a + 1; b < n; b++) {
      if (adjacency[a][b] === 0 && rand() < extraEdgeProb) {
        adjacency[a][b] = adjacency[b][a] = 1;
      }
    }
  }

  // Layout: concentric rings sized to the node count, with nodes spread evenly
  // around each ring plus light jitter for an organic "network map" look.
  const positions: NodePos[] = [];
  const rings = n <= 9 ? 2 : n <= 20 ? 3 : 4;
  const perRing = Math.ceil(n / rings);
  for (let i = 0; i < n; i++) {
    const ring = Math.floor(i / perRing);
    const idxInRing = i % perRing;
    const countInRing = Math.min(perRing, n - ring * perRing);
    const radius = 0.14 + (rings === 1 ? 0 : (ring / (rings - 1)) * 0.34);
    const ringOffset = ring * 0.5; // stagger rings so nodes don't line up radially
    const angle = (idxInRing / countInRing) * Math.PI * 2 + ringOffset + (rand() - 0.5) * 0.25;
    positions.push({
      id: i,
      x: 0.5 + Math.cos(angle) * radius + (rand() - 0.5) * 0.04,
      y: 0.5 + Math.sin(angle) * radius + (rand() - 0.5) * 0.04,
    });
  }

  let start = Math.floor(rand() * n);
  let target = Math.floor(rand() * n);
  while (target === start) target = Math.floor(rand() * n);

  // Give the entry and the database enough connections that they can't be born in
  // a corner (and so a player can't cheaply wall/sever them off in one or two
  // clicks). The reachability guard in the UI is the hard rule; this just keeps
  // the board interesting by guaranteeing real routing choices around both hubs.
  const MIN_HUB_DEGREE = Math.min(3, n - 1);
  const degreeOf = (node: number) => adjacency[node].reduce((s, v) => s + v, 0);
  const boostDegree = (node: number) => {
    let guard = 0;
    while (degreeOf(node) < MIN_HUB_DEGREE && guard < n) {
      const candidates: number[] = [];
      for (let v = 0; v < n; v++) if (v !== node && adjacency[node][v] === 0) candidates.push(v);
      if (candidates.length === 0) break;
      const v = candidates[Math.floor(rand() * candidates.length)];
      adjacency[node][v] = adjacency[v][node] = 1;
      guard++;
    }
  };
  boostDegree(start);
  boostDegree(target);

  const nodeStates: NodeState[] = Array(n).fill(OPEN) as NodeState[];

  return { adjacency, positions, nodeStates, start, target };
}

/** Hop distance over passable nodes (firewalls block). Returns Infinity if unreachable. */
export function bfsDistance(
  adjacency: number[][],
  nodeStates: NodeState[],
  src: number,
  dst: number,
): number {
  if (src === dst) return 0;
  const n = adjacency.length;
  const visited = new Array(n).fill(false);
  visited[src] = true;
  let frontier = [src];
  let dist = 0;
  while (frontier.length) {
    dist++;
    const next: number[] = [];
    for (const u of frontier) {
      for (let v = 0; v < n; v++) {
        if (adjacency[u][v] === 1 && !visited[v]) {
          if (nodeStates[v] === FIREWALL && v !== dst) continue;
          if (v === dst) return dist;
          visited[v] = true;
          next.push(v);
        }
      }
    }
    frontier = next;
  }
  return Infinity;
}

export function neighbors(adjacency: number[][], node: number): number[] {
  const out: number[] = [];
  for (let v = 0; v < adjacency.length; v++) if (adjacency[node][v] === 1) out.push(v);
  return out;
}

/** Build the four observation tensors in the exact shape/order the ONNX model expects. */
export function buildObservation(net: NetworkState, current: number) {
  const n = NODE_COUNT;
  const node_states = Float32Array.from(net.nodeStates);
  const adjacency_matrix = new Float32Array(n * n);
  for (let i = 0; i < n; i++)
    for (let j = 0; j < n; j++) adjacency_matrix[i * n + j] = net.adjacency[i][j];
  const target_node = new Float32Array(n);
  target_node[net.target] = 1;
  const current_position = new Float32Array(n);
  current_position[current] = 1;
  return { node_states, adjacency_matrix, target_node, current_position };
}
