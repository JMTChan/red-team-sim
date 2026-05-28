"""Optional: verify the exported ONNX policy loads and produces N logits.

Run after training:  python scripts/verify_onnx.py
"""

import os
import numpy as np
import onnxruntime as ort

N = 24
HERE = os.path.dirname(os.path.abspath(__file__))
MODEL = os.path.join(os.path.dirname(HERE), "public", "red_team_agent.onnx")


def main():
    sess = ort.InferenceSession(MODEL, providers=["CPUExecutionProvider"])
    feeds = {
        "node_states": np.zeros((1, N), dtype=np.float32),
        "adjacency_matrix": np.zeros((1, N * N), dtype=np.float32),
        "target_node": np.eye(N, dtype=np.float32)[None, 5],
        "current_position": np.eye(N, dtype=np.float32)[None, 0],
    }
    out = sess.run(None, feeds)[0]
    assert out.shape == (1, N), f"unexpected output shape {out.shape}"
    print("OK — logits shape:", out.shape, "| argmax:", int(out.argmax()))


if __name__ == "__main__":
    main()
