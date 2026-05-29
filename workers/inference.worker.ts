/// <reference lib="webworker" />
//
// Runs ONNX inference for the Red Team agent on a background thread so the game
// loop and SVG rendering on the main thread never stall.
//
// Protocol (main thread -> worker):
//   { type: "init",  modelUrl }
//   { type: "infer", requestId, obs, validActions }
// Protocol (worker -> main thread):
//   { type: "ready" }
//   { type: "error",  message }
//   { type: "action", requestId, action }   // chosen node id, already masked

import * as ort from "onnxruntime-web";
import { NODE_COUNT } from "../lib/constants";

// Serve the WASM backend from a CDN so it works regardless of the GitHub Pages
// basePath. Keep the version aligned with package.json's onnxruntime-web.
ort.env.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.18.0/dist/";

let session: ort.InferenceSession | null = null;
// Node the agent occupied on the previous inference, so we can discourage it from
// immediately bouncing straight back (the classic A->B->A->B loop a memoryless,
// deterministic policy falls into once its productive paths are walled off).
let lastPos = -1;

interface Obs {
  node_states: Float32Array;
  adjacency_matrix: Float32Array;
  target_node: Float32Array;
  current_position: Float32Array;
}

function init(modelUrl: string): Promise<void> {
  return fetch(modelUrl).then(async (resp) => {
    if (!resp.ok) throw new Error(`Failed to fetch model (${resp.status}) at ${modelUrl}`);
    const bytes = new Uint8Array(await resp.arrayBuffer());
    session = await ort.InferenceSession.create(bytes, {
      executionProviders: ["wasm"],
      graphOptimizationLevel: "all",
    });
  });
}

function indexOfOne(arr: Float32Array): number {
  let idx = -1;
  let best = -Infinity;
  for (let i = 0; i < arr.length; i++) {
    if (arr[i] > best) {
      best = arr[i];
      idx = i;
    }
  }
  return idx;
}

function chooseAction(
  logits: Float32Array,
  validActions: number[],
  avoid: number,
): { action: number; dist: number[] } {
  // Greedy legal move.
  let best = -1;
  let bestVal = -Infinity;
  for (const a of validActions) {
    if (logits[a] > bestVal) {
      bestVal = logits[a];
      best = a;
    }
  }
  // Softmax over the legal logits -> a "where does it want to go" distribution
  // (kept as the true policy intent, so the heatmap reflects what it learned).
  const dist = new Array(NODE_COUNT).fill(0);
  let sum = 0;
  for (const a of validActions) {
    const e = Math.exp(logits[a] - bestVal);
    dist[a] = e;
    sum += e;
  }
  if (sum > 0) for (const a of validActions) dist[a] /= sum;

  // Anti-backtrack: if the greedy pick is to walk straight back where we came
  // from, take the next-best legal move instead — unless backtracking is the
  // ONLY option (a genuine dead end), in which case we allow it.
  let action = best;
  if (action === avoid && validActions.length > 1) {
    let alt = -1;
    let altVal = -Infinity;
    for (const a of validActions) {
      if (a === avoid) continue;
      if (logits[a] > altVal) {
        altVal = logits[a];
        alt = a;
      }
    }
    if (alt >= 0) action = alt;
  }
  return { action, dist };
}

async function infer(obs: Obs, validActions: number[]): Promise<{ action: number; dist: number[] }> {
  if (!session) throw new Error("Session not initialized");
  const feeds: Record<string, ort.Tensor> = {
    node_states: new ort.Tensor("float32", obs.node_states, [1, NODE_COUNT]),
    adjacency_matrix: new ort.Tensor("float32", obs.adjacency_matrix, [1, NODE_COUNT * NODE_COUNT]),
    target_node: new ort.Tensor("float32", obs.target_node, [1, NODE_COUNT]),
    current_position: new ort.Tensor("float32", obs.current_position, [1, NODE_COUNT]),
  };
  const out = await session.run(feeds);
  const logits = out.logits.data as Float32Array;
  const currentPos = indexOfOne(obs.current_position);
  const result = chooseAction(logits, validActions, lastPos);
  lastPos = currentPos; // next move, "backtracking" means returning here
  return result;
}

self.onmessage = async (e: MessageEvent) => {
  const msg = e.data;
  try {
    if (msg.type === "init") {
      await init(msg.modelUrl);
      (self as DedicatedWorkerGlobalScope).postMessage({ type: "ready" });
    } else if (msg.type === "reset") {
      lastPos = -1; // new round / new board: forget the previous trajectory
    } else if (msg.type === "infer") {
      const { action, dist } = await infer(msg.obs as Obs, msg.validActions as number[]);
      (self as DedicatedWorkerGlobalScope).postMessage({
        type: "action",
        requestId: msg.requestId,
        action,
        dist,
      });
    }
  } catch (err) {
    (self as DedicatedWorkerGlobalScope).postMessage({
      type: "error",
      message: err instanceof Error ? err.message : String(err),
    });
  }
};
