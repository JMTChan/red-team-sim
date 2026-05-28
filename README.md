# Red Team Network Simulator

An interactive, browser-based reinforcement-learning game. You play **Blue Team**:
defend a procedurally generated network against a **Red Team** PPO agent that
pathfinds in real time to a dynamic database node. The agent runs **entirely in
your browser** via `onnxruntime-web` in a Web Worker — no backend, no server.

- **Frontend:** Next.js (App Router) + Tailwind, statically exported for GitHub Pages.
- **Agent inference:** ONNX model executed client-side in a Web Worker.
- **Training:** `gymnasium` + `stable-baselines3` (PPO), exported to `.onnx`.
- **CI/CD:** GitHub Actions retrains every 6 hours (4 sessions/day), checkpoints under 5.5h, and commits the
  new weights back to `main`.

```
red-team-sim/
├─ training/            env.py, train.py, requirements.txt   (Phase 1)
├─ .github/workflows/   train-agent.yml, deploy.yml          (Phase 2 + deploy)
├─ workers/             inference.worker.ts                  (Phase 3)
├─ lib/                 constants.ts, network.ts             (shared with the env)
├─ leaderboard-worker/  worker.js, wrangler.toml             (optional free leaderboard)
├─ components/          NetworkSimulator.tsx                 (Phase 4)
├─ app/                 layout.tsx, page.tsx, globals.css
└─ public/              red_team_agent.onnx  ← produced by training
```

> The app is fully playable **before** any model exists: if `red_team_agent.onnx`
> is missing, the agent automatically falls back to a greedy heuristic (you'll see
> `HEURISTIC AGENT` in the header). Once training commits a real model, it loads
> automatically and the header switches to `ONNX POLICY`.

---

## 1. Put it in your repo

1. Create a repo named **`red-team-sim`** (the name is referenced by `basePath` in
   `next.config.js` — if you use a different name, change `repo` there).
2. Copy everything in this folder into the repo and push to `main`.

---

## 2. Run the game locally

```bash
npm install
npm run dev          # http://localhost:3000
```

Build the static export (what GitHub Pages serves):

```bash
npm run build        # outputs ./out
```

---

## 3. ▶ Start training the agent on GitHub Actions  ⭐

This is the key step. The workflow at `.github/workflows/train-agent.yml` trains the
PPO agent, exports `public/red_team_agent.onnx`, and commits it back.

1. **Allow Actions to push commits.**
   Repo → **Settings → Actions → General → Workflow permissions** →
   select **“Read and write permissions”** → **Save**.
   *(The workflow also declares `permissions: contents: write`, but this repo
   setting must be enabled or the auto-commit step will fail.)*

2. **Kick off the first training run manually** (don't wait for the next scheduled run):
   Repo → **Actions** tab → **“Train Red Team Agent”** in the left sidebar →
   **“Run workflow”** button → **Run workflow**.

3. **Watch it train.** The job:
   - resumes from `training/latest_model.zip` if present, else starts fresh;
   - trains until **5.5 hours** elapse (`TRAIN_MAX_SECONDS=19800`), checkpointing
     every 50k steps so nothing is lost (hard job cap `timeout-minutes: 350`,
     safely under GitHub's 6-hour limit);
   - overwrites `training/latest_model.zip` and writes `public/red_team_agent.onnx`;
   - commits both back to `main` (`chore: auto-trained model …`).

4. **After that**, it runs **automatically every 6 hours** (00:00, 06:00, 12:00, 18:00 UTC) and keeps
   improving the same model: train ~5.5h, ~30-min idle gap, then the next run starts. Each run continues from the previous weights, so the
   agent gets progressively harder over time. Adjust the schedule by editing the
   `cron` line in `train-agent.yml`.

> Want a quick model without waiting 5.5h? Run the workflow, then cancel it after a
> few minutes — or set a smaller `TRAIN_MAX_SECONDS` — and the checkpoint logic
> still produces a usable `.onnx`. You can also train locally:
> ```bash
> pip install -r training/requirements.txt
> TRAIN_MAX_SECONDS=600 python training/train.py   # 10-minute local run
> python scripts/verify_onnx.py                      # optional sanity check
> ```

---

## 4. Publish the site (GitHub Pages)

`.github/workflows/deploy.yml` builds the static export and deploys it on every push
to `main` (including the model commits from training).

1. Repo → **Settings → Pages → Build and deployment → Source** → **GitHub Actions**.
2. Push to `main` (or re-run the **Deploy to GitHub Pages** workflow).
3. Your game goes live at `https://<your-username>.github.io/red-team-sim/`.

---

## How to play

- Pick a **countermeasure** (Firewall / Honeypot / Corrupt / Sever Link). Each costs
  credits, which slowly regenerate during a round.
- **Click a node** to deploy a node defense; with **Sever Link** selected, click a
  connection (or two linked nodes) to cut it.
- Hit **Launch Intrusion**. Each tick, the agent rebuilds its observation from your
  live defenses and recomputes its move:
  - **Firewall** — impassable, the agent must route around it.
  - **Honeypot** — if the agent steps in, it's trapped (you win the round).
  - **Corrupt** — the agent's *next* move is scrambled to a random neighbor.
  - **Sever Link** — removes an edge, reshaping the topology mid-run.
- You win by **containing** the agent (honeypot, dead-end, or timeout). The agent
  wins by reaching the **DATABASE** node.

---

## Global leaderboard (optional, free)

Local high scores work out of the box (saved per browser). For a shared online
board, deploy the included Cloudflare Worker — it's free and the browser never
holds a secret. You can do the whole thing from the **Cloudflare dashboard (web),
no command line required** — full click-by-click steps are in
`leaderboard-worker/README.md`. The essence: create a Worker, paste in
`leaderboard-worker/worker.js`, create a KV namespace, and bind it to the Worker
with the variable name `SCORES`.

Then put the Worker's URL in `lib/config.ts` (`LEADERBOARD_URL`) and redeploy the
site. A pure GitHub-hosted endpoint can't do this safely — anonymous score
submission needs a write credential, and anything in the static bundle is public.
The Worker keeps that credential server-side.

## Notes & knobs

- **Network size:** the board is **24 nodes**. This value lives in three places that
  MUST stay equal: `NODE_COUNT` in `lib/constants.ts`, and `N_NODES` in
  `training/train.py` (plus the `n_nodes` default in `training/env.py`). To make the
  maze bigger/harder, set all of them to the same new value (e.g. 32) **before** you
  start training, then train from scratch. Bigger N = richer maze but slower to reach
  a strong policy, so give it more daily runs to converge. Don't change it after a
  model exists — the ONNX input shape is fixed at training time.
- **ONNX runtime WASM:** loaded from a jsDelivr CDN in `workers/inference.worker.ts`.
  Keep that version aligned with `onnxruntime-web` in `package.json`.
- **Model in git:** `latest_model.zip` is committed so training can resume across
  runs. If history grows too large, consider Git LFS.
- **Pinned versions** in `package.json` / `training/requirements.txt` are known-good;
  bump them together if you upgrade.
