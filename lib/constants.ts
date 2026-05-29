// Keep NODE_COUNT identical to N_NODES in training/train.py and env.py.
// If you retrain with a different network size, change it in BOTH places.
export const NODE_COUNT = 24;

// Node states (mirror env.py)
export const OPEN = 0;
export const FIREWALL = 1;
export const HONEYPOT = 2;
export const CORRUPTED = 3;
export const SLOW = 4; // tarpit: passable but the agent can stall on entry

export type NodeState = 0 | 1 | 2 | 3 | 4;

export const TOOLS = {
  FIREWALL: "firewall",
  HONEYPOT: "honeypot",
  CORRUPT: "corrupt",
  TARPIT: "tarpit",
  SEVER: "sever",
  MONITOR: "monitor",
} as const;

export type Tool = (typeof TOOLS)[keyof typeof TOOLS];
