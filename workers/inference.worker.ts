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

interface Obs {
  node_states: Float32Array;
  adjacency_matrix: Float32Array;
  target_node: Float32Array;
  current_position: Float32Array;
}

async function init(modelUrl: string) {
  const resp = await fetch(modelUrl);
  if (!resp.ok) throw new Error(`Failed to fetch model (${resp.status}) at ${modelUrl}`);
  const bytes = new Uint8Array(await resp.arrayBuffer());
  session = await ort.InferenceSession.create(bytes, {
    executionProviders: ["wasm"],
    graphOptimizationLevel: "all",
  });
}

function argmaxMasked(logits: Float32Array, validActions: number[]): number {
  // Restrict the agent to legal moves; if none are legal it stays put.
  let best = -1;
  let bestVal = -Infinity;
  for (const a of validActions) {
    if (logits[a] > bestVal) {
      bestVal = logits[a];
      best = a;
    }
  }
  return best;
}

async function infer(obs: Obs, validActions: number[]): Promise<number> {
  if (!session) throw new Error("Session not initialized");
  const feeds: Record<string, ort.Tensor> = {
    node_states: new ort.Tensor("float32", obs.node_states, [1, NODE_COUNT]),
    adjacency_matrix: new ort.Tensor("float32", obs.adjacency_matrix, [1, NODE_COUNT * NODE_COUNT]),
    target_node: new ort.Tensor("float32", obs.target_node, [1, NODE_COUNT]),
    current_position: new ort.Tensor("float32", obs.current_position, [1, NODE_COUNT]),
  };
  const out = await session.run(feeds);
  const logits = out.logits.data as Float32Array;
  return argmaxMasked(logits, validActions);
}

self.onmessage = async (e: MessageEvent) => {
  const msg = e.data;
  try {
    if (msg.type === "init") {
      await init(msg.modelUrl);
      (self as DedicatedWorkerGlobalScope).postMessage({ type: "ready" });
    } else if (msg.type === "infer") {
      const action = await infer(msg.obs as Obs, msg.validActions as number[]);
      (self as DedicatedWorkerGlobalScope).postMessage({
        type: "action",
        requestId: msg.requestId,
        action,
      });
    }
  } catch (err) {
    (self as DedicatedWorkerGlobalScope).postMessage({
      type: "error",
      message: err instanceof Error ? err.message : String(err),
    });
  }
};
