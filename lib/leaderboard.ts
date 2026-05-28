import { LEADERBOARD_URL } from "./config";

// What the global board returns (score is computed by the server).
export interface ScoreEntry {
  name: string;
  score: number;
  difficulty: string;
  ts?: number;
}

// What the client submits: the ingredients of the round, NOT a score. The Worker
// recomputes and bounds-checks the score so a tampered client can't inflate it.
export interface ScoreSubmission {
  name: string;
  result: string; // "honeypot" | "timeout" | "stalled"
  turns: number;
  creditsLeft: number;
  difficulty: string;
  turnstileToken?: string; // present only when Turnstile is enabled
}

export const leaderboardEnabled = (): boolean => LEADERBOARD_URL.trim().length > 0;

const base = () => LEADERBOARD_URL.trim().replace(/\/$/, "");

export async function fetchScores(): Promise<ScoreEntry[]> {
  if (!leaderboardEnabled()) return [];
  const r = await fetch(`${base()}/scores`);
  if (!r.ok) throw new Error(`leaderboard fetch failed: ${r.status}`);
  const data = await r.json();
  return Array.isArray(data.scores) ? data.scores : [];
}

export async function submitScore(entry: ScoreSubmission): Promise<ScoreEntry[]> {
  if (!leaderboardEnabled()) return [];
  const r = await fetch(`${base()}/scores`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(entry),
  });
  if (!r.ok) throw new Error(`leaderboard submit failed: ${r.status}`);
  const data = await r.json();
  return Array.isArray(data.scores) ? data.scores : [];
}
