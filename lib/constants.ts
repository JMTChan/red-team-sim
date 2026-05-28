// Keep NODE_COUNT identical to N_NODES in training/train.py and env.py.
// If you retrain with a different network size, change it in BOTH places.
// 24 nodes = a substantially larger maze than the original 16 (more routes,
// more defense decisions). Trainable on the daily CI runs; see README to push higher.
export const NODE_COUNT = 24;

// Node states (mirror env.py)
export const OPEN = 0;
export const FIREWALL = 1;
export const HONEYPOT = 2;
export const CORRUPTED = 3;

export type NodeState = 0 | 1 | 2 | 3;

export const TOOLS = {
  FIREWALL: "firewall",
  HONEYPOT: "honeypot",
  CORRUPT: "corrupt",
  SEVER: "sever",
} as const;

export type Tool = (typeof TOOLS)[keyof typeof TOOLS];
