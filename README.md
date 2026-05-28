# Red Team Network Simulator

An interactive, browser-based reinforcement-learning game. You play **Blue Team** and
defend a procedurally generated network against a **Red Team** AI agent — a trained
RL policy that pathfinds in real time from an entry node to a moving "database" target.
Deploy firewalls, honeypots, corruptors, and link cuts on the live board to contain it.

The whole thing runs **client-side** as a static site. The AI runs in your browser via
ONNX, there's no backend, and the agent is **retrained automatically every day** by
GitHub Actions and committed back to the repo.

## How it works

Each round generates a fresh random topology (so the agent can't memorize a path) with
a random database target. The agent moves one step per tick toward the target; you spend
a regenerating pool of credits to slow or trap it:

- **Firewall** — makes a node impassable
- **Honeypot** — traps the agent and ends the run
- **Corrupt** — randomizes the agent's next move
- **Sever Link** — cuts an edge

Firewalls and severs are rejected if they would *completely* cut the agent off from the
database, so defenses funnel the agent rather than instantly winning. Contain the agent
to score; let it breach the database and you score nothing.

If no trained model is present yet, the agent falls back to a greedy heuristic — the
header shows `HEURISTIC AGENT` vs `ONNX POLICY` so you always know which is driving.

## Tech stack

- **Frontend:** Next.js (App Router) + Tailwind, exported as a fully static site
- **Inference:** `onnxruntime-web` running in a Web Worker
- **Training:** Python — Gymnasium (custom env) + `sb3-contrib` MaskablePPO
- **CI/CD:** GitHub Actions (daily training + Pages deploy)
- **Leaderboard (optional):** Cloudflare Worker + Workers KV (server-authoritative)

## Quick start

```bash
npm install
npm run dev        # http://localhost:3000
npm run build      # static export -> ./out
```

## How the agent learns

The agent is trained with **MaskablePPO**. The action space is "move to node N," but
only a few nodes are ever legal (adjacent, non-firewalled), so action masking removes
the illegal choices during training — this is what lets it learn on dense graphs. The
exported ONNX model outputs raw logits for all nodes; the browser masks illegal moves
before choosing, so the same masking logic applies at inference.

Training runs in a continuous loop on GitHub Actions: each ~5.5-hour session resumes
from the previously committed model, so the agent keeps improving around the clock.
`rollout/ep_rew_mean` (average score) and `rollout/ep_len_mean` (steps per episode) in
the logs show whether it's improving.

To train locally:
```bash
pip install -r training/requirements.txt
python training/train.py        # writes training/latest_model.zip + public/red_team_agent.onnx
```

## Deployment

The site deploys to GitHub Pages. To enable the full automatic chain:

1. **Settings → Actions → General → Workflow permissions → Read and write** (lets
   training push the new model).
2. **Settings → Pages → Source → GitHub Actions** (lets the deploy publish).
3. **Actions tab → Train Red Team Agent → Run workflow** once to seed the first model;
   the 6-hour cron handles every run after that.

Once set up: **train → commit model → deploy → the live header flips to `ONNX POLICY`**,
hands-off, every cycle.

## Optional: global leaderboard

A Cloudflare Worker + KV namespace (free tier) provides a shared leaderboard. The client
submits only the *ingredients* of a round (result, turns, credits left, difficulty); the
Worker recomputes the score server-side and bounds-checks it, so the board can't hold
impossible numbers. Setup is dashboard-only — see `leaderboard-worker/README.md`. Drop
the Worker URL into `lib/config.ts`. To reset the board, delete the `top` key in the KV
namespace.

## Tunables

| Setting | Where |
|---|---|
| Network size | `NODE_COUNT` in `lib/constants.ts` **and** `N_NODES` / `n_nodes` in `training/train.py` + `training/env.py` (must match; retrain from scratch) |
| Difficulty | `DIFFICULTY` table in `components/NetworkSimulator.tsx` |
| Scoring | `roundScore` in `components/NetworkSimulator.tsx` **and** the mirror in `leaderboard-worker/worker.js` |
| Training time / frequency | `TRAIN_MAX_SECONDS` and `cron` in `.github/workflows/train-agent.yml` |
| Canvas / layout | `VIEW_W`/`VIEW_H` + the layout block in `lib/network.ts` |
