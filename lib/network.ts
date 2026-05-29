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
  foothold: number; // first objective the agent must reach (multi-stage attack)
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
export function generateNetwork(seed = Date.now(), extraEdgeProb = 0.18, ensureTwoPaths = false): NetworkState {
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

  // Layout: scatter nodes across the WHOLE canvas using a lightly-jittered grid
  // (landscape-biased), so the graph fills the rectangle — corners included —
  // instead of clustering in a centered blob. Each node lands in its own cell
  // with random jitter, keeping spacing even but organic.
  const positions: NodePos[] = [];
  const cols = Math.max(1, Math.round(Math.sqrt(n * 1.4)));
  const rows = Math.ceil(n / cols);
  const cells = Array.from({ length: cols * rows }, (_, k) => k);
  for (let k = cells.length - 1; k > 0; k--) {
    const j = Math.floor(rand() * (k + 1));
    [cells[k], cells[j]] = [cells[j], cells[k]];
  }
  for (let i = 0; i < n; i++) {
    const cell = cells[i];
    const cx = cell % cols;
    const cy = Math.floor(cell / cols);
    const x = (cx + 0.5) / cols + (rand() - 0.5) * (0.62 / cols);
    const y = (cy + 0.5) / rows + (rand() - 0.5) * (0.62 / rows);
    positions.push({
      id: i,
      x: Math.min(0.97, Math.max(0.03, x)),
      y: Math.min(0.97, Math.max(0.03, y)),
    });
  }

  let start = Math.floor(rand() * n);
  let target = Math.floor(rand() * n);
  while (target === start) target = Math.floor(rand() * n);
  // Multi-stage objective: the agent must reach a foothold before the database.
  let foothold = Math.floor(rand() * n);
  while (foothold === start || foothold === target) foothold = Math.floor(rand() * n);

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
  boostDegree(foothold);

  // On harder difficulties, guarantee two node-disjoint routes along each leg of the
  // attack (start->foothold, foothold->database) so there's no single natural
  // chokepoint — and so the sever rule that preserves this is actually satisfiable.
  if (ensureTwoPaths) {
    const reachSet = (src: number, removed: number): boolean[] => {
      const vis = new Array(n).fill(false);
      vis[src] = true;
      const q = [src];
      while (q.length) {
        const u = q.shift() as number;
        for (let w = 0; w < n; w++) {
          if (adjacency[u][w] === 1 && w !== removed && !vis[w]) {
            vis[w] = true;
            q.push(w);
          }
        }
      }
      return vis;
    };
    const ensure2 = (a: number, b: number) => {
      let guard = 0;
      while (guard++ < n) {
        let cut = -1;
        for (let v = 0; v < n; v++) {
          if (v === a || v === b) continue;
          if (!linkReachable(adjacency, a, b, v)) {
            cut = v;
            break;
          }
        }
        if (cut === -1) break; // already 2-connected on this leg
        // Bridge the two sides around the cut with one new edge (avoid the trivial
        // a-b direct link so the leg keeps some length).
        const side = reachSet(a, cut);
        const aSide: number[] = [];
        const bSide: number[] = [];
        for (let v = 0; v < n; v++) {
          if (v === cut) continue;
          (side[v] ? aSide : bSide).push(v);
        }
        const uA = aSide.find((x) => x !== a) ?? a;
        const uB = bSide.find((x) => x !== b) ?? bSide[0] ?? b;
        if (uA === uB) break;
        adjacency[uA][uB] = adjacency[uB][uA] = 1;
      }
    };
    ensure2(start, foothold);
    ensure2(foothold, target);
  }

  const nodeStates: NodeState[] = Array(n).fill(OPEN) as NodeState[];

  return { adjacency, positions, nodeStates, start, foothold, target };
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

/** Can `from` reach `to` over links only, optionally with one node removed?
 *  Node states never block here — firewalls are penetrable, everything else is
 *  passable — so reachability depends purely on severed links. */
export function linkReachable(
  adjacency: number[][],
  from: number,
  to: number,
  removed = -1,
): boolean {
  if (from === to) return true;
  const n = adjacency.length;
  const visited = new Array(n).fill(false);
  visited[from] = true;
  if (removed >= 0 && removed !== from && removed !== to) visited[removed] = true;
  const queue = [from];
  while (queue.length) {
    const u = queue.shift() as number;
    for (let v = 0; v < n; v++) {
      if (adjacency[u][v] === 1 && !visited[v]) {
        if (v === to) return true;
        visited[v] = true;
        queue.push(v);
      }
    }
  }
  return false;
}

/** True if there are >=2 node-disjoint paths between `from` and `to` — i.e. no
 *  single intermediate node whose removal disconnects them (Menger). Used to stop
 *  the defender funnelling the agent through one chokepoint. */
export function twoDisjointPaths(adjacency: number[][], from: number, to: number): boolean {
  if (from === to) return true;
  if (!linkReachable(adjacency, from, to)) return false;
  const n = adjacency.length;
  for (let v = 0; v < n; v++) {
    if (v === from || v === to) continue;
    if (!linkReachable(adjacency, from, to, v)) return false;
  }
  return true;
}

/** Build the four observation tensors in the exact shape/order the ONNX model expects.
 *  `goal` is the agent's CURRENT objective (foothold first, then the database). */
export function buildObservation(net: NetworkState, current: number, goal: number = net.target) {
  const n = NODE_COUNT;
  const node_states = Float32Array.from(net.nodeStates);
  const adjacency_matrix = new Float32Array(n * n);
  for (let i = 0; i < n; i++)
    for (let j = 0; j < n; j++) adjacency_matrix[i * n + j] = net.adjacency[i][j];
  const target_node = new Float32Array(n);
  target_node[goal] = 1;
  const current_position = new Float32Array(n);
  current_position[current] = 1;
  return { node_states, adjacency_matrix, target_node, current_position };
}
