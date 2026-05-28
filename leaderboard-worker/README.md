# Global leaderboard (Cloudflare Workers + KV) — free, no CLI needed

The browser only ever sees the public Worker URL; no secret ships in client code.
Free tier easily covers a hobby game (100k requests/day, 1,000 KV writes/day — each
submitted score is one write).

You can set this up entirely from the **Cloudflare dashboard** (web) — no terminal.

## Dashboard setup (web only)

1. **Create the Worker.**
   Dashboard -> **Workers & Pages** (newer dashboards call this **Compute (Workers)**)
   -> **Create application** -> **Create Worker** -> **Start with "Hello World!"**.
   Name it `red-team-leaderboard` -> **Deploy**.

2. **Paste in the code.**
   On the Worker's page -> **Edit code**. Delete everything in the editor and paste
   the entire contents of `worker.js` (in this folder). The template is already a
   "module" Worker (it starts with `export default {`), which matches this code.
   Click **Deploy** (or **Save and deploy**).

3. **Create the KV namespace.**
   Dashboard -> **Storage & Databases -> Workers KV** (or the **KV** tab) ->
   **Create instance / Create** -> name it `red-team-scores` -> **Create**.

4. **Bind the namespace to the Worker — the variable name MUST be `SCORES`.**
   Back on your Worker -> **Settings -> Bindings -> Add -> KV namespace**.
   - **Variable name:** `SCORES`  (must be exactly this — the code reads `env.SCORES`)
   - **KV namespace:** select `red-team-scores`
   Save, then **Deploy** the Worker once more so the binding takes effect.

5. **Grab the URL + test it.**
   The Worker page shows a URL like
   `https://red-team-leaderboard.<your-subdomain>.workers.dev`.
   Open `https://red-team-leaderboard.<your-subdomain>.workers.dev/scores` in a new
   tab — you should see `{"scores":[]}`. That means it's live.

6. **Wire it into the game.**
   Edit `lib/config.ts` and paste the **base** URL (no `/scores` on the end):
   ```ts
   export const LEADERBOARD_URL = "https://red-team-leaderboard.<your-subdomain>.workers.dev";
   ```
   Commit/push. Your deploy Action rebuilds the site and the leaderboard goes live.

## Lock it down (optional, after it works)

On the Worker -> **Settings -> Variables and Secrets** (or **Settings -> Variables**)
-> add a plaintext variable:
- **Name:** `ALLOW_ORIGIN`
- **Value:** `https://YOUR_GITHUB_USERNAME.github.io`

Deploy again. This restricts reads/posts to your site's origin. You can also add a
Cloudflare **Rate limiting** rule or **Turnstile** from the dashboard for stronger
anti-abuse — the Worker code intentionally has no KV-based limiter (KV's minimum
expiration is 60s, so a fine-grained limiter doesn't fit there).

## Notes

- `wrangler.toml` in this folder is **only** for the CLI path; ignore it if you're
  using the dashboard. The KV binding you set in step 4 is what matters.
- Without a configured URL the game runs in **local-only** mode: per-browser high
  scores still work; only the shared online board is hidden.
- **Anti-cheat:** the client submits the round's *ingredients* (outcome, turns,
  credits left, difficulty), and the Worker **recomputes the score and rejects
  anything outside the legitimate per-difficulty range** — so no submission can ever
  exceed what a real top player earns. Because the game runs in the browser, a
  determined user can still submit a *plausible* round they didn't truly play; the
  only fully cheat-proof option is to re-simulate the round with the agent model on
  an authoritative server, which isn't practical on free Workers.

## CLI alternative (if you ever use a terminal)

```bash
npm install -g wrangler && wrangler login
cd leaderboard-worker
wrangler kv namespace create SCORES   # paste the id into wrangler.toml
wrangler deploy
```
