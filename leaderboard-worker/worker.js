// Red Team Simulator — global leaderboard Worker (Cloudflare Workers + KV).
//
// Endpoints (CORS-enabled):
//   GET  /scores  -> { scores: [...all], boards: { recruit:[], operator:[], elite:[] } }
//                    (top 50 PER difficulty; `scores` is the three boards concatenated)
//   POST /scores  -> body { name, result, turns, creditsLeft, difficulty, turnstileToken? }
//                    The server RECOMPUTES the score from these ingredients and
//                    bounds-checks every field, so the client can't assert a score.
//
// Anti-abuse layers (all optional / degrade gracefully):
//   * score recompute + per-difficulty bounds  -> no impossible numbers (always on)
//   * KV per-IP rate limit (60s window)         -> blocks floods (always on)
//   * Cloudflare Turnstile verification         -> blocks bots (on if TURNSTILE_SECRET set)
//
// Set these as Worker "Variables and Secrets" in the dashboard (all optional):
//   ALLOW_ORIGIN       e.g. https://you.github.io   (lock CORS to your site)
//   TURNSTILE_SECRET   your Turnstile secret key     (enables bot verification)
//   RATE_LIMIT         max submits per IP per minute (default 30)

const TOP_KEY = "top";
const MAX_PER_DIFF = 50; // top-N kept PER threat level
const DEFAULT_RATE_LIMIT = 30; // submits per IP per 60s
const DIFFS = ["recruit", "operator", "elite"];

// Must mirror DIFFICULTY + DIFF_MULT + roundScore() in components/NetworkSimulator.tsx.
const DIFF = {
  recruit: { maxTurns: 55, creditCap: 22, mult: 1 },
  operator: { maxTurns: 80, creditCap: 20, mult: 2 },
  elite: { maxTurns: 120, creditCap: 18, mult: 4 },
};
const VALID_RESULTS = ["honeypot", "timeout", "stalled"]; // containment only

function scoreFor(result, turns, creditsLeft, cfg) {
  const speed = Math.max(0, cfg.maxTurns - turns) * 8;
  const efficiency = Math.max(0, creditsLeft) * 10;
  const method = result === "honeypot" ? 200 : 0;
  return Math.round((300 + speed + efficiency + method) * cfg.mult);
}

// Boards are stored per difficulty: { recruit: [...], operator: [...], elite: [...] }.
// This keeps each threat level its own top-N so high-multiplier Elite scores can't
// evict Recruit/Operator entries from the table.
function emptyBoards() {
  return { recruit: [], operator: [], elite: [] };
}

function readBoards(raw) {
  if (!raw) return emptyBoards();
  let parsed;
  try {
    parsed = JSON.parse(raw);
  } catch {
    return emptyBoards();
  }
  const boards = emptyBoards();
  if (Array.isArray(parsed)) {
    // Migrate the old single combined list into per-difficulty buckets.
    for (const e of parsed) {
      const d = DIFFS.includes(e.difficulty) ? e.difficulty : "operator";
      boards[d].push(e);
    }
  } else if (parsed && typeof parsed === "object") {
    for (const d of DIFFS) if (Array.isArray(parsed[d])) boards[d] = parsed[d];
  }
  for (const d of DIFFS) {
    boards[d].sort((a, b) => b.score - a.score);
    boards[d] = boards[d].slice(0, MAX_PER_DIFF);
  }
  return boards;
}

function flatten(boards) {
  return [...boards.recruit, ...boards.operator, ...boards.elite];
}

function corsHeaders(origin) {
  return {
    "Access-Control-Allow-Origin": origin || "*",
    "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
    "Access-Control-Allow-Headers": "Content-Type",
    "Access-Control-Max-Age": "86400",
  };
}
function json(body, origin, status = 200) {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "Content-Type": "application/json", ...corsHeaders(origin) },
  });
}

// Loose KV-based per-IP limiter. KV's minimum TTL is 60s and it's eventually
// consistent, so this is a flood deterrent, not exact accounting — which is all a
// hobby leaderboard needs.
async function rateLimited(env, ip, limit) {
  if (!ip) return false;
  const key = `rl:${ip}`;
  const current = Number((await env.SCORES.get(key)) || 0);
  if (current >= limit) return true;
  await env.SCORES.put(key, String(current + 1), { expirationTtl: 60 });
  return false;
}

async function verifyTurnstile(secret, token, ip) {
  const form = new FormData();
  form.append("secret", secret);
  form.append("response", token || "");
  if (ip) form.append("remoteip", ip);
  const res = await fetch("https://challenges.cloudflare.com/turnstile/v0/siteverify", {
    method: "POST",
    body: form,
  });
  const data = await res.json();
  return data.success === true;
}

export default {
  async fetch(request, env) {
    const origin = (env && env.ALLOW_ORIGIN) || "*";
    const url = new URL(request.url);

    if (request.method === "OPTIONS") return new Response(null, { headers: corsHeaders(origin) });
    if (!url.pathname.endsWith("/scores")) return json({ error: "not found" }, origin, 404);

    if (request.method === "GET") {
      const boards = readBoards(await env.SCORES.get(TOP_KEY));
      return json({ scores: flatten(boards), boards }, origin);
    }

    if (request.method === "POST") {
      const ip = request.headers.get("CF-Connecting-IP");
      const limit = Number(env.RATE_LIMIT) || DEFAULT_RATE_LIMIT;
      if (await rateLimited(env, ip, limit)) {
        return json({ error: "rate limited — slow down" }, origin, 429);
      }

      let body;
      try {
        body = await request.json();
      } catch {
        return json({ error: "invalid json" }, origin, 400);
      }

      // Bot check (only enforced if you've set a Turnstile secret).
      if (env.TURNSTILE_SECRET) {
        const ok = await verifyTurnstile(env.TURNSTILE_SECRET, body.turnstileToken, ip);
        if (!ok) return json({ error: "failed bot check" }, origin, 403);
      }

      // Validate every field against the difficulty's hard limits.
      const difficulty = body.difficulty;
      const cfg = DIFF[difficulty];
      if (!cfg) return json({ error: "bad difficulty" }, origin, 400);

      const result = body.result;
      if (!VALID_RESULTS.includes(result)) return json({ error: "non-scoring result" }, origin, 400);

      const turns = Math.floor(Number(body.turns));
      const creditsLeft = Math.floor(Number(body.creditsLeft));
      if (!Number.isFinite(turns) || turns < 1 || turns > cfg.maxTurns)
        return json({ error: "turns out of range" }, origin, 400);
      if (!Number.isFinite(creditsLeft) || creditsLeft < 0 || creditsLeft > cfg.creditCap)
        return json({ error: "credits out of range" }, origin, 400);

      const name =
        String(body.name || "anon").slice(0, 16).replace(/[^\w \-]/g, "").trim() || "anon";

      // The SERVER computes the score; the client never gets to assert it.
      const score = scoreFor(result, turns, creditsLeft, cfg);
      if (score <= 0) return json({ error: "invalid score" }, origin, 400);

      const raw = await env.SCORES.get(TOP_KEY);
      const boards = readBoards(raw);
      boards[difficulty].push({ name, score, difficulty, ts: Date.now() });
      boards[difficulty].sort((a, b) => b.score - a.score);
      boards[difficulty] = boards[difficulty].slice(0, MAX_PER_DIFF);
      await env.SCORES.put(TOP_KEY, JSON.stringify(boards));
      return json({ scores: flatten(boards), boards }, origin);
    }

    return json({ error: "method not allowed" }, origin, 405);
  },
};
