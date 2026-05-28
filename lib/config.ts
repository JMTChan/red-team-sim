// Global leaderboard endpoint (Cloudflare Worker URL).
//
// Paste your deployed Worker URL here, e.g.
//   export const LEADERBOARD_URL = "https://red-team-leaderboard.you.workers.dev";
// Leave empty for LOCAL-ONLY mode (per-browser high scores still work).
export const LEADERBOARD_URL = "https://red-team-leaderboard.funtee123.workers.dev/";

// Optional bot protection. Paste your Cloudflare Turnstile *site key* here (the
// public one — safe to ship). Leave empty to disable the widget. If you set this,
// also set the matching TURNSTILE_SECRET variable on the Worker (dashboard).
export const TURNSTILE_SITE_KEY = "0x4AAAAAADYBupRkTUd1x4ao";
