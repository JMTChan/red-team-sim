"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  NODE_COUNT,
  OPEN,
  FIREWALL,
  HONEYPOT,
  CORRUPTED,
  SLOW,
  TOOLS,
  type Tool,
  type NodeState,
} from "@/lib/constants";
import {
  generateNetwork,
  bfsDistance,
  neighbors,
  buildObservation,
  linkReachable,
  twoDisjointPaths,
  type NetworkState,
} from "@/lib/network";
import {
  leaderboardEnabled,
  fetchScores,
  submitScore,
  type ScoreEntry,
  type ScoreSubmission,
} from "@/lib/leaderboard";
import { TURNSTILE_SITE_KEY } from "@/lib/config";

type Status = "idle" | "running" | "breach" | "trapped" | "stalled" | "detected";

interface ToolDef {
  id: Tool;
  label: string;
  hint: string;
  cost: number;
  color: string;
}

const TOOL_DEFS: ToolDef[] = [
  { id: TOOLS.FIREWALL, label: "Firewall", hint: "Agent must breach it (~35% per try) — slows, doesn't block", cost: 2, color: "#f59e0b" },
  { id: TOOLS.HONEYPOT, label: "Honeypot", hint: "~70% chance to trap the agent — layer them", cost: 3, color: "#a78bfa" },
  { id: TOOLS.CORRUPT, label: "Corrupt", hint: "Randomizes the agent's next move", cost: 2, color: "#22d3ee" },
  { id: TOOLS.TARPIT, label: "Tarpit", hint: "Slows the agent — ~40% chance to stall each entry", cost: 1, color: "#14b8a6" },
  { id: TOOLS.SEVER, label: "Sever Link", hint: "Cut a connection between nodes", cost: 1, color: "#f43f5e" },
  { id: TOOLS.MONITOR, label: "Monitor", hint: "Raises detection near it — catch the agent before it breaches", cost: 2, color: "#38bdf8" },
];

const STATE_COLOR: Record<number, string> = {
  [OPEN]: "#1f3a44",
  [FIREWALL]: "#f59e0b",
  [HONEYPOT]: "#a78bfa",
  [CORRUPTED]: "#22d3ee",
  [SLOW]: "#14b8a6",
};

// Difficulty presets. NODE_COUNT stays 16 (the model's input is fixed at that
// size), so harder levels lean on a faster agent, a denser network (more escape
// routes the defender must funnel), a longer agent turn-budget, and a tighter —
// but never starved — credit economy.
type Difficulty = "recruit" | "operator" | "elite";

interface DifficultyConfig {
  label: string;
  blurb: string;
  tickMs: number; // agent move interval (lower = faster agent)
  startCredits: number;
  creditCap: number;
  regenMs: number; // ms per +1 credit during a live round
  maxTurns: number; // agent's move budget before it times out
  extraEdgeProb: number; // network density (more edges = more routes = harder)
}

const DIFFICULTY: Record<Difficulty, DifficultyConfig> = {
  recruit: {
    label: "Recruit",
    blurb: "Slow agent · sparse net · generous credits",
    tickMs: 1300,
    startCredits: 16,
    creditCap: 22,
    regenMs: 2500,
    maxTurns: 55,
    extraEdgeProb: 0.12,
  },
  operator: {
    label: "Operator",
    blurb: "Balanced speed, density & economy",
    tickMs: 850,
    startCredits: 14,
    creditCap: 20,
    regenMs: 3500,
    maxTurns: 80,
    extraEdgeProb: 0.2,
  },
  elite: {
    label: "Elite",
    blurb: "Fast agent · dense net · lean credits",
    tickMs: 500,
    startCredits: 12,
    creditCap: 18,
    regenMs: 5000,
    maxTurns: 120,
    extraEdgeProb: 0.3,
  },
};

const DEFAULT_DIFFICULTY: Difficulty = "operator";

const turnstileOn = () => leaderboardEnabled() && TURNSTILE_SITE_KEY.trim().length > 0;

// Load the Cloudflare Turnstile script once (explicit-render mode).
function loadTurnstileScript(): Promise<void> {
  return new Promise((resolve, reject) => {
    if (typeof window === "undefined") return resolve();
    if ((window as unknown as { turnstile?: unknown }).turnstile) return resolve();
    const existing = document.getElementById("cf-turnstile-script");
    if (existing) {
      existing.addEventListener("load", () => resolve());
      return;
    }
    const s = document.createElement("script");
    s.id = "cf-turnstile-script";
    s.src = "https://challenges.cloudflare.com/turnstile/v0/api.js?render=explicit";
    s.async = true;
    s.defer = true;
    s.onload = () => resolve();
    s.onerror = () => reject(new Error("turnstile failed to load"));
    document.head.appendChild(s);
  });
}

// Scoring: containment earns points; a breach scores nothing. Rewards fast
// containment, leftover credits (efficiency), actively trapping the agent in a
// honeypot, and — heavily — playing on harder difficulty.
const DIFF_MULT: Record<Difficulty, number> = { recruit: 1, operator: 2, elite: 4 };

function roundScore(
  result: string,
  turns: number,
  creditsLeft: number,
  maxTurns: number,
  diff: Difficulty,
): number {
  if (result === "breach") return 0;
  const speed = Math.max(0, maxTurns - turns) * 8;
  const efficiency = Math.max(0, creditsLeft) * 10;
  const method = result === "honeypot" ? 200 : 0; // active trap > passive timeout
  return Math.round((300 + speed + efficiency + method) * DIFF_MULT[diff]);
}

// Agent-vs-defense resolution odds. These MUST mirror training/env.py so the live
// agent behaves the way it was trained. If you retune one side, retune both.
const FIREWALL_BREACH = 0.35; // P(agent forces through a firewall on an attempt)
const HONEYPOT_TRAP = 0.7; // P(a honeypot actually catches the agent when entered)
const SLOW_PASS = 0.6; // P(agent crosses a tarpit on an attempt; else it stalls a tick)
const DETECTION_PER_STEP = 0.014; // base "noise" the agent accrues each move
const DETECTION_ON_FAIL = 0.05; // extra noise when a firewall breach attempt fails
// Monitors are a frontend-only accelerant (the env trains against base noise only):
// being on or beside a monitored node raises the detection rate so the defender can
// catch the agent before it breaches.
const MONITOR_RATE = 0.07;

// Layout mapping (normalized 0..1 -> SVG viewBox).
const VIEW_W = 1000;
const VIEW_H = 760;
const PAD = 60;
const px = (x: number) => PAD + x * (VIEW_W - 2 * PAD);
const py = (y: number) => PAD + y * (VIEW_H - 2 * PAD);

// Node glyph sizing scales down as the network grows, so a 24-node board stays legible.
const NODE_R = NODE_COUNT >= 28 ? 11 : NODE_COUNT >= 20 ? 13 : 16;
const NODE_FONT = NODE_R - 3;
const AGENT_R = Math.round(NODE_R * 0.62);

export default function NetworkSimulator() {
  const [difficulty, setDifficulty] = useState<Difficulty>(DEFAULT_DIFFICULTY);
  const initialCfg = DIFFICULTY[DEFAULT_DIFFICULTY];
  const [net, setNet] = useState<NetworkState>(() =>
    generateNetwork(Date.now(), initialCfg.extraEdgeProb, DEFAULT_DIFFICULTY !== "recruit"),
  );
  const [agentPos, setAgentPos] = useState<number>(() => net.start);
  const [status, setStatus] = useState<Status>("idle");
  const [tool, setTool] = useState<Tool>(TOOLS.FIREWALL);
  const [tickMs, setTickMs] = useState(initialCfg.tickMs);
  const [credits, setCredits] = useState(initialCfg.startCredits);
  const [turns, setTurns] = useState(0);
  const [trail, setTrail] = useState<number[]>([net.start]);
  const [log, setLog] = useState<string[]>(["// system online. awaiting trained policy…"]);
  const [score, setScore] = useState({ breaches: 0, contained: 0 });
  const [engine, setEngine] = useState<"model" | "heuristic" | "loading">("loading");
  const [severPick, setSeverPick] = useState<number | null>(null);

  // Scoring + leaderboard.
  const [lastScore, setLastScore] = useState<number | null>(null);
  const [bestScore, setBestScore] = useState(0);
  // Ingredients of the best-scoring round, so we submit those (not a raw number).
  const bestRecordRef = useRef<ScoreSubmission | null>(null);
  const [board, setBoard] = useState<ScoreEntry[]>([]);
  const [boardDiff, setBoardDiff] = useState<Difficulty>(DEFAULT_DIFFICULTY);
  const [playerName, setPlayerName] = useState("");
  const [submitting, setSubmitting] = useState(false);
  const [submittedBest, setSubmittedBest] = useState(0);
  const [turnstileToken, setTurnstileToken] = useState("");
  const turnstileDivRef = useRef<HTMLDivElement>(null);
  const turnstileWidgetId = useRef<string | null>(null);

  // Active difficulty config (and a ref for the async game loop).
  const cfg = DIFFICULTY[difficulty];
  const cfgRef = useRef(cfg);
  useEffect(() => void (cfgRef.current = cfg), [cfg]);

  // Pan/zoom of the network map.
  const [view, setView] = useState({ s: 1, tx: 0, ty: 0 });
  const svgRef = useRef<SVGSVGElement>(null);
  const panning = useRef(false);
  const panStart = useRef({ x: 0, y: 0, tx: 0, ty: 0 });
  const viewRef = useRef(view);
  const pointers = useRef<Map<number, { x: number; y: number }>>(new Map());
  const pinch = useRef<{ dist: number; s: number; cx: number; cy: number; tx: number; ty: number } | null>(null);

  // Feature state: policy heatmap, escalating waves, streaks, sound, tutorial, juice.
  const [heatOn, setHeatOn] = useState(false);
  const [heat, setHeat] = useState<number[]>([]);
  const [muted, setMuted] = useState(false);
  const mutedRef = useRef(false);
  const [wavesMode, setWavesMode] = useState(false);
  const wavesRef = useRef(false);
  const [wave, setWave] = useState(1);
  const waveRef = useRef(1);
  const [bestWave, setBestWave] = useState(0);
  const [streak, setStreak] = useState(0);
  const streakRef = useRef(0);
  const [bestStreak, setBestStreak] = useState(0);
  const [sessionPoints, setSessionPoints] = useState(0);
  const [flash, setFlash] = useState<null | "breach" | "contain">(null);
  const [showTutorial, setShowTutorial] = useState(false);
  // Multi-objective + detection (Phases 3–4).
  const [detection, setDetection] = useState(0); // 0..1 noise meter
  const detectionRef = useRef(0);
  const [reachedFoothold, setReachedFoothold] = useState(false);
  const reachedFootholdRef = useRef(false);
  const [monitors, setMonitors] = useState<Set<number>>(new Set());
  const monitorsRef = useRef<Set<number>>(new Set());
  const audioCtx = useRef<AudioContext | null>(null);
  const tickRef = useRef<() => void>(() => {});

  // Refs so the async game loop always reads live values (no stale closures).
  const netRef = useRef(net);
  // Pristine snapshot of the current network (full links, no defenses) so Reset
  // can fully restore it — including severed links, not just node defenses.
  const baselineRef = useRef<NetworkState>({
    ...net,
    adjacency: net.adjacency.map((r) => [...r]),
    nodeStates: [...net.nodeStates],
  });
  const logRef = useRef<HTMLDivElement>(null);
  const posRef = useRef(agentPos);
  const statusRef = useRef(status);
  const turnsRef = useRef(turns);
  const corruptNext = useRef(false);
  const modelReady = useRef(false);
  const workerRef = useRef<Worker | null>(null);
  const pending = useRef<Map<number, (r: { action: number; dist: number[] }) => void>>(new Map());
  const reqSeq = useRef(0);
  const creditsRef = useRef(credits);
  const diffRef = useRef(difficulty);
  useEffect(() => void (creditsRef.current = credits), [credits]);
  useEffect(() => void (diffRef.current = difficulty), [difficulty]);

  useEffect(() => void (netRef.current = net), [net]);
  useEffect(() => void (posRef.current = agentPos), [agentPos]);
  useEffect(() => void (statusRef.current = status), [status]);
  useEffect(() => void (turnsRef.current = turns), [turns]);
  useEffect(() => void (viewRef.current = view), [view]);
  useEffect(() => void (mutedRef.current = muted), [muted]);
  useEffect(() => void (wavesRef.current = wavesMode), [wavesMode]);

  const pushLog = useCallback((line: string) => {
    setLog((l) => [...l.slice(-80), line]);
  }, []);

  // Keep the event log pinned to the newest entry.
  useEffect(() => {
    const el = logRef.current;
    if (el) el.scrollTop = el.scrollHeight;
  }, [log]);

  // Restore local high score + name, and load the global board (if configured).
  useEffect(() => {
    try {
      const b = Number(localStorage.getItem("rts-best") || 0);
      if (b > 0) setBestScore(b);
      const rec = localStorage.getItem("rts-best-record");
      if (rec) bestRecordRef.current = JSON.parse(rec);
      const n = localStorage.getItem("rts-name") || "";
      if (n) setPlayerName(n);
      setBestStreak(Number(localStorage.getItem("rts-best-streak") || 0));
      setBestWave(Number(localStorage.getItem("rts-best-wave") || 0));
      if (!localStorage.getItem("rts-tutorial-seen")) setShowTutorial(true);
    } catch {
      /* localStorage unavailable */
    }
    if (leaderboardEnabled()) fetchScores().then(setBoard).catch(() => {});
  }, []);

  // Render the Turnstile bot-check widget (only if a site key is configured).
  useEffect(() => {
    if (!turnstileOn()) return;
    let cancelled = false;
    loadTurnstileScript()
      .then(() => {
        if (cancelled) return;
        const ts = (window as unknown as { turnstile?: any }).turnstile;
        if (ts && turnstileDivRef.current && turnstileWidgetId.current === null) {
          turnstileWidgetId.current = ts.render(turnstileDivRef.current, {
            sitekey: TURNSTILE_SITE_KEY.trim(),
            theme: "dark",
            size: "flexible",
            callback: (token: string) => setTurnstileToken(token),
            "expired-callback": () => setTurnstileToken(""),
            "error-callback": () => setTurnstileToken(""),
          });
        }
      })
      .catch(() => {});
    return () => {
      cancelled = true;
    };
  }, []);

  // ----------------------------------------------------------------- sound
  const sfx = useCallback((name: "place" | "remove" | "contain" | "honeypot" | "breach" | "wave") => {
    if (mutedRef.current || typeof window === "undefined") return;
    if (!audioCtx.current) {
      try {
        const Ctor = window.AudioContext || (window as unknown as { webkitAudioContext: typeof AudioContext }).webkitAudioContext;
        audioCtx.current = new Ctor();
      } catch {
        return;
      }
    }
    const ctx = audioCtx.current;
    if (!ctx) return;
    if (ctx.state === "suspended") ctx.resume().catch(() => {});
    const beep = (freq: number, dur: number, type: OscillatorType, gain: number, delay = 0) => {
      const t0 = ctx.currentTime + delay;
      const osc = ctx.createOscillator();
      const g = ctx.createGain();
      osc.type = type;
      osc.frequency.setValueAtTime(freq, t0);
      g.gain.setValueAtTime(gain, t0);
      g.gain.exponentialRampToValueAtTime(0.0001, t0 + dur);
      osc.connect(g);
      g.connect(ctx.destination);
      osc.start(t0);
      osc.stop(t0 + dur);
    };
    switch (name) {
      case "place": beep(440, 0.08, "square", 0.04); break;
      case "remove": beep(280, 0.07, "square", 0.03); break;
      case "contain": beep(523, 0.12, "triangle", 0.05); beep(784, 0.16, "triangle", 0.05, 0.09); break;
      case "honeypot": beep(659, 0.1, "triangle", 0.05); beep(988, 0.18, "triangle", 0.05, 0.09); break;
      case "breach": beep(200, 0.3, "sawtooth", 0.06); beep(110, 0.4, "sawtooth", 0.06, 0.12); break;
      case "wave": beep(587, 0.1, "square", 0.04); beep(880, 0.14, "square", 0.04, 0.1); break;
    }
  }, []);

  // Award points for the round outcome (containment scores; a breach scores 0).
  // Returns the single-round base score (used for streak/session bonuses).
  const award = useCallback(
    (result: string): number => {
      const turns = turnsRef.current;
      const creditsLeft = creditsRef.current;
      const diff = diffRef.current;
      const s = roundScore(result, turns, creditsLeft, cfgRef.current.maxTurns, diff);
      setLastScore(s);
      if (s > 0) {
        setBestScore((b) => {
          if (s <= b) return b;
          const record: ScoreSubmission = {
            name: "",
            result,
            turns,
            creditsLeft,
            difficulty: diff,
          };
          bestRecordRef.current = record;
          try {
            localStorage.setItem("rts-best", String(s));
            localStorage.setItem("rts-best-record", JSON.stringify(record));
          } catch {
            /* ignore */
          }
          return s;
        });
        pushLog(`★ round score +${s}`);
      }
      return s;
    },
    [pushLog],
  );

  // Reset the per-round dynamic state (detection meter + foothold progress, and
  // optionally the placed monitors when the board itself changes).
  const resetDynamics = useCallback((clearMonitors: boolean) => {
    detectionRef.current = 0;
    setDetection(0);
    reachedFootholdRef.current = false;
    setReachedFoothold(false);
    if (clearMonitors) {
      monitorsRef.current = new Set();
      setMonitors(new Set());
    }
  }, []);

  // Extra detection accrued per step from monitors on or beside the agent.
  const monitorBoost = useCallback((node: number): number => {
    const mon = monitorsRef.current;
    if (mon.size === 0) return 0;
    if (mon.has(node)) return MONITOR_RATE;
    for (const v of neighbors(netRef.current.adjacency, node)) if (mon.has(v)) return MONITOR_RATE * 0.5;
    return 0;
  }, []);

  // Escalating waves: regenerate a harder board (faster agent + denser links) and
  // auto-relaunch. Uses tickRef so it doesn't have to depend on the game loop.
  const advanceWave = useCallback(
    (nextWave: number) => {
      const base = cfgRef.current;
      const edgeProb = Math.min(0.5, base.extraEdgeProb + (nextWave - 1) * 0.03);
      const fresh = generateNetwork(Date.now(), edgeProb, diffRef.current !== "recruit");
      setNet(fresh);
      netRef.current = fresh;
      baselineRef.current = {
        ...fresh,
        adjacency: fresh.adjacency.map((r) => [...r]),
        nodeStates: [...fresh.nodeStates],
      };
      setAgentPos(fresh.start);
      posRef.current = fresh.start;
      setTurns(0);
      turnsRef.current = 0;
      corruptNext.current = false;
      setTrail([fresh.start]);
      setCredits(base.startCredits);
      creditsRef.current = base.startCredits;
      setTickMs(Math.max(200, base.tickMs - (nextWave - 1) * 60));
      setSeverPick(null);
      setHeat([]);
      resetDynamics(true);
      workerRef.current?.postMessage({ type: "reset" });
      // Stay IDLE so the defender can fortify the new board, then launch the wave.
      setStatus("idle");
      statusRef.current = "idle";
      sfx("wave");
      pushLog(`▲ WAVE ${nextWave} ready — denser network, faster agent. Deploy defenses, then launch.`);
    },
    [sfx, pushLog, resetDynamics],
  );

  // Centralized round-end: streak + session points + waves + sound + screen flash.
  const concludeRound = useCallback(
    (contained: boolean, result: string) => {
      const base = award(result);
      if (contained) {
        sfx(result === "honeypot" ? "honeypot" : "contain");
        setFlash("contain");
        streakRef.current += 1;
        const st = streakRef.current;
        setStreak(st);
        setBestStreak((b) => {
          const nb = Math.max(b, st);
          try {
            localStorage.setItem("rts-best-streak", String(nb));
          } catch {
            /* ignore */
          }
          return nb;
        });
        const factor = 1 + Math.min(st - 1, 5) * 0.1; // up to +50% at a 6-streak
        setSessionPoints((p) => p + Math.round(base * factor));
        if (st >= 2) pushLog(`✦ streak x${st} (+${Math.round((factor - 1) * 100)}% combo)`);
      } else {
        sfx("breach");
        setFlash("breach");
        streakRef.current = 0;
        setStreak(0);
      }
      setTimeout(() => setFlash(null), 480);

      if (wavesRef.current) {
        if (contained) {
          const next = waveRef.current + 1;
          waveRef.current = next;
          setWave(next);
          setBestWave((b) => {
            const nb = Math.max(b, next);
            try {
              localStorage.setItem("rts-best-wave", String(nb));
            } catch {
              /* ignore */
            }
            return nb;
          });
          advanceWave(next);
        } else {
          pushLog(`✖ wave run ended at wave ${waveRef.current}.`);
          waveRef.current = 1;
          setWave(1);
          setTickMs(cfgRef.current.tickMs); // back to base speed for the next run
        }
      }
    },
    [award, sfx, pushLog, advanceWave],
  );

  // ---------------------------------------------------------------- worker
  useEffect(() => {
    const worker = new Worker(new URL("../workers/inference.worker.ts", import.meta.url));
    workerRef.current = worker;

    worker.onmessage = (e: MessageEvent) => {
      const msg = e.data;
      if (msg.type === "ready") {
        modelReady.current = true;
        setEngine("model");
        pushLog("// onnx policy loaded — adversary online.");
      } else if (msg.type === "error") {
        modelReady.current = false;
        setEngine("heuristic");
        pushLog("// no trained model found — running heuristic adversary.");
      } else if (msg.type === "action") {
        const resolve = pending.current.get(msg.requestId);
        if (resolve) {
          pending.current.delete(msg.requestId);
          resolve({ action: msg.action, dist: msg.dist ?? [] });
        }
      }
    };

    const basePath = process.env.NEXT_PUBLIC_BASE_PATH || "";
    worker.postMessage({ type: "init", modelUrl: `${basePath}/red_team_agent.onnx` });
    return () => worker.terminate();
  }, [pushLog]);

  // Credits regenerate while a round is running (rate & cap depend on difficulty).
  useEffect(() => {
    if (status !== "running") return;
    const id = setInterval(
      () => setCredits((c) => Math.min(cfg.creditCap, c + 1)),
      cfg.regenMs,
    );
    return () => clearInterval(id);
  }, [status, cfg.creditCap, cfg.regenMs]);

  // -------------------------------------------------------------- policies
  // Every adjacent node is a legal move now — firewalls are attackable rather than
  // impassable, so the agent can choose to attempt a breach (resolved in applyMove).
  // This mirrors env.py's action_masks(), which also exposes all neighbors.
  const legalMoves = useCallback((n: NetworkState, pos: number): number[] => {
    return neighbors(n.adjacency, pos);
  }, []);

  // Greedy fallback used until the .onnx policy is available. Also produces an
  // intent distribution (favoring distance-reducing moves) so the heatmap works
  // even before a trained model is loaded.
  const heuristicMove = useCallback(
    (n: NetworkState, pos: number, moves: number[]): { action: number; dist: number[] } => {
      const goal = reachedFootholdRef.current ? n.target : n.foothold;
      let best = moves[0];
      let bestD = Infinity;
      const dists = moves.map((m) => bfsDistance(n.adjacency, n.nodeStates, m, goal));
      moves.forEach((m, i) => {
        if (dists[i] < bestD) {
          bestD = dists[i];
          best = m;
        }
      });
      // weight ~ closeness to target (softmin over distances)
      const dist = new Array(NODE_COUNT).fill(0);
      let sum = 0;
      moves.forEach((m, i) => {
        const w = Math.exp(-(dists[i] - bestD));
        dist[m] = w;
        sum += w;
      });
      if (sum > 0) moves.forEach((m) => (dist[m] /= sum));
      return { action: best, dist };
    },
    [],
  );

  const inferViaWorker = useCallback(
    (n: NetworkState, pos: number, moves: number[]): Promise<{ action: number; dist: number[] }> => {
      return new Promise((resolve) => {
        const id = ++reqSeq.current;
        pending.current.set(id, resolve);
        workerRef.current?.postMessage({
          type: "infer",
          requestId: id,
          obs: buildObservation(n, pos, reachedFootholdRef.current ? n.target : n.foothold),
          validActions: moves,
        });
      });
    },
    [],
  );

  // ------------------------------------------------------------- game loop
  const applyMove = useCallback(
    (action: number) => {
      const n = netRef.current;
      const state = n.nodeStates[action];

      // Penetrable firewall: an attempt forces through only sometimes; a failed
      // attempt rebuffs the agent (it stays put, turn still spent) and is noisy.
      if (state === FIREWALL) {
        if (Math.random() >= FIREWALL_BREACH) {
          detectionRef.current = Math.min(1, detectionRef.current + DETECTION_ON_FAIL);
          setHeat([]);
          pushLog(`!! firewall held at node ${action} — breach failed (detection +).`);
          return;
        }
        pushLog(`>> agent forced the firewall at node ${action}.`);
      } else if (state === SLOW) {
        // Tarpit: sometimes the agent bogs down and loses the tick.
        if (Math.random() >= SLOW_PASS) {
          setHeat([]);
          pushLog(`~~ agent bogged down in the tarpit at node ${action}.`);
          return;
        }
      }

      posRef.current = action;
      setAgentPos(action);
      setTrail((t) => (t[t.length - 1] === action ? t : [...t, action]));

      // Multi-stage objective: secure the foothold first, then go for the database.
      if (!reachedFootholdRef.current && action === n.foothold) {
        reachedFootholdRef.current = true;
        setReachedFoothold(true);
        setHeat([]);
        pushLog(`** foothold seized at node ${action} — agent now advancing on the database.`);
        return;
      }

      if (reachedFootholdRef.current && action === n.target) {
        statusRef.current = "breach";
        setStatus("breach");
        setScore((s) => ({ ...s, breaches: s.breaches + 1 }));
        setHeat([]);
        pushLog(`>> BREACH — database node ${action} compromised.`);
        concludeRound(false, "breach");
      } else if (state === HONEYPOT) {
        // Leaky honeypot: only sometimes catches the agent; otherwise it slips by.
        if (Math.random() < HONEYPOT_TRAP) {
          statusRef.current = "trapped";
          setStatus("trapped");
          setScore((s) => ({ ...s, contained: s.contained + 1 }));
          setHeat([]);
          pushLog(`<< TRAPPED — agent ensnared in honeypot at node ${action}.`);
          concludeRound(true, "honeypot");
        } else {
          pushLog(`.. agent slipped past the honeypot at node ${action}.`);
        }
      } else if (state === CORRUPTED) {
        corruptNext.current = true;
        pushLog(`~~ agent corrupted at node ${action} — next move scrambled.`);
      }
    },
    [pushLog, concludeRound],
  );

  const tick = useCallback(async () => {
    if (statusRef.current !== "running") return;
    const n = netRef.current;
    const pos = posRef.current;

    if (turnsRef.current >= cfgRef.current.maxTurns) {
      statusRef.current = "stalled";
      setStatus("stalled");
      setScore((s) => ({ ...s, contained: s.contained + 1 }));
      setHeat([]);
      pushLog("== agent ran out of time — network held.");
      concludeRound(true, "timeout");
      return;
    }

    const moves = legalMoves(n, pos);
    if (moves.length === 0) {
      statusRef.current = "stalled";
      setStatus("stalled");
      setScore((s) => ({ ...s, contained: s.contained + 1 }));
      setHeat([]);
      pushLog("== agent contained — no legal route remains.");
      concludeRound(true, "stalled");
      return;
    }

    let action: number;
    if (corruptNext.current) {
      corruptNext.current = false;
      setHeat([]);
      const all = neighbors(n.adjacency, pos);
      action = all[Math.floor(Math.random() * all.length)];
    } else if (modelReady.current) {
      const res = await inferViaWorker(n, pos, moves);
      action = res.action;
      setHeat(res.dist);
    } else {
      const res = heuristicMove(n, pos, moves);
      action = res.action;
      setHeat(res.dist);
    }

    if (statusRef.current !== "running") return;
    if (action == null || action < 0) {
      setTimeout(tick, tickMs);
      return;
    }

    applyMove(action);
    setTurns((t) => t + 1);

    // Detection rises each move (more near monitors); if it caps, the intrusion is
    // traced and evicted — a containment win for the defender. Only meaningful once
    // the agent is still live after this move.
    if (statusRef.current === "running") {
      detectionRef.current = Math.min(1, detectionRef.current + DETECTION_PER_STEP + monitorBoost(posRef.current));
      setDetection(detectionRef.current);
      if (detectionRef.current >= 1) {
        statusRef.current = "detected";
        setStatus("detected");
        setScore((s) => ({ ...s, contained: s.contained + 1 }));
        setHeat([]);
        pushLog(">> DETECTED — intrusion traced and the agent evicted.");
        concludeRound(true, "detected");
        return;
      }
    }

    if (statusRef.current === "running") setTimeout(tick, tickMs);
  }, [applyMove, heuristicMove, inferViaWorker, legalMoves, pushLog, tickMs, concludeRound, monitorBoost]);

  // Keep a live ref to tick so wave auto-relaunch can call the latest version.
  useEffect(() => void (tickRef.current = tick), [tick]);

  const start = () => {
    if (status === "running") return;
    if (status !== "idle") resetRound(false);
    setHeat([]);
    sfx("place"); // also unlocks the AudioContext on this user gesture
    resetDynamics(false);
    workerRef.current?.postMessage({ type: "reset" });
    setStatus("running");
    statusRef.current = "running";
    pushLog(
      wavesRef.current
        ? `// wave ${waveRef.current} — intrusion live. defend the database.`
        : "// intrusion initiated. defend the database.",
    );
    setTimeout(tick, 300);
  };

  const pause = () => {
    setStatus("idle");
    statusRef.current = "idle";
    pushLog("// paused.");
  };

  const resetRound = (newLog = true) => {
    // Restore the network to its pristine state: all links back, all defenses cleared.
    const base = baselineRef.current;
    const restored: NetworkState = {
      ...net,
      adjacency: base.adjacency.map((r) => [...r]),
      nodeStates: Array(NODE_COUNT).fill(OPEN) as NodeState[],
    };
    statusRef.current = "idle";
    setStatus("idle");
    setNet(restored);
    netRef.current = restored;
    setAgentPos(net.start);
    posRef.current = net.start;
    setTurns(0);
    turnsRef.current = 0;
    corruptNext.current = false;
    setTrail([net.start]);
    setCredits(cfg.startCredits);
    setSeverPick(null);
    resetDynamics(true);
    if (newLog) pushLog("// round reset — defenses cleared, links restored, agent back at entry.");
  };

  const newNetwork = () => {
    const fresh = generateNetwork(Date.now(), cfg.extraEdgeProb, difficulty !== "recruit");
    setNet(fresh);
    netRef.current = fresh;
    baselineRef.current = {
      ...fresh,
      adjacency: fresh.adjacency.map((r) => [...r]),
      nodeStates: [...fresh.nodeStates],
    };
    setAgentPos(fresh.start);
    posRef.current = fresh.start;
    setStatus("idle");
    statusRef.current = "idle";
    setTurns(0);
    turnsRef.current = 0;
    corruptNext.current = false;
    setTrail([fresh.start]);
    setCredits(cfg.startCredits);
    setSeverPick(null);
    resetDynamics(true);
    pushLog("// new topology generated — domain randomized.");
  };

  // Switch difficulty: retune the economy/agent and generate a matching board.
  const applyDifficulty = (d: Difficulty) => {
    if (d === difficulty) return;
    const c = DIFFICULTY[d];
    setDifficulty(d);
    setBoardDiff(d);
    cfgRef.current = c;
    setTickMs(c.tickMs);
    const fresh = generateNetwork(Date.now(), c.extraEdgeProb, d !== "recruit");
    setNet(fresh);
    netRef.current = fresh;
    baselineRef.current = {
      ...fresh,
      adjacency: fresh.adjacency.map((r) => [...r]),
      nodeStates: [...fresh.nodeStates],
    };
    setAgentPos(fresh.start);
    posRef.current = fresh.start;
    setStatus("idle");
    statusRef.current = "idle";
    setTurns(0);
    turnsRef.current = 0;
    corruptNext.current = false;
    setTrail([fresh.start]);
    setCredits(c.startCredits);
    setSeverPick(null);
    resetDynamics(true);
    pushLog(`// difficulty set to ${c.label.toUpperCase()} — new network generated.`);
  };

  // ----------------------------------------------------------- leaderboard
  const submitBest = async () => {
    const record = bestRecordRef.current;
    if (!leaderboardEnabled() || !record || bestScore <= 0 || submitting) return;
    if (turnstileOn() && !turnstileToken) {
      pushLog("xx complete the bot check before submitting.");
      return;
    }
    const name = (playerName || "anon").slice(0, 16).replace(/[^\w \-]/g, "").trim() || "anon";
    try {
      localStorage.setItem("rts-name", name);
    } catch {
      /* ignore */
    }
    setSubmitting(true);
    try {
      // Send the round's ingredients; the Worker recomputes + bounds-checks the score.
      const updated = await submitScore({
        ...record,
        name,
        turnstileToken: turnstileOn() ? turnstileToken : undefined,
      });
      setBoard(updated);
      setSubmittedBest(bestScore);
      pushLog(`☁ submitted best round as "${name}".`);
      // Turnstile tokens are single-use — get a fresh one for next time.
      const ts = (window as unknown as { turnstile?: any }).turnstile;
      if (turnstileOn() && ts && turnstileWidgetId.current !== null) {
        ts.reset(turnstileWidgetId.current);
        setTurnstileToken("");
      }
    } catch {
      pushLog("xx leaderboard submit failed — check the endpoint URL.");
    } finally {
      setSubmitting(false);
    }
  };

  // ----------------------------------------------------------- defenses
  const currentToolDef = TOOL_DEFS.find((t) => t.id === tool)!;
  const CREDIT_CAP = cfg.creditCap;

  // Which node-state each node tool produces.
  const DEFENSE_STATE: Partial<Record<Tool, NodeState>> = {
    [TOOLS.FIREWALL]: FIREWALL,
    [TOOLS.HONEYPOT]: HONEYPOT,
    [TOOLS.CORRUPT]: CORRUPTED,
    [TOOLS.TARPIT]: SLOW,
  };
  // Cost to refund when an existing defense is removed/replaced (mirrors TOOL_DEFS costs).
  const COST_BY_STATE: Record<number, number> = {
    [FIREWALL]: 2,
    [HONEYPOT]: 3,
    [CORRUPTED]: 2,
    [SLOW]: 1,
  };

  const setNodeState = (id: number, state: NodeState) => {
    setNet((n) => {
      const states = [...n.nodeStates];
      states[id] = state;
      const updated = { ...n, nodeStates: states };
      netRef.current = updated;
      return updated;
    });
  };

  // A complete route start -> foothold -> database must always survive a sever. On
  // Operator/Elite each leg must survive as TWO node-disjoint paths, so severing can
  // never funnel the agent through a single chokepoint; Recruit stays lenient (one
  // path) as the beginner mode. Firewalls are penetrable, so node states don't affect
  // reachability — only severed links do, which is why this checks adjacency alone.
  const routeSurvives = (adjacency: number[][]): boolean => {
    const n = netRef.current;
    const strict = diffRef.current !== "recruit";
    const legOk = (a: number, b: number) =>
      strict ? twoDisjointPaths(adjacency, a, b) : linkReachable(adjacency, a, b);
    return legOk(n.start, n.foothold) && legOk(n.foothold, n.target);
  };

  const deployOnNode = (id: number) => {
    if (id === posRef.current || id === net.target || id === net.start || id === net.foothold) {
      pushLog("xx cannot place a defense on that node.");
      return;
    }
    if (tool === TOOLS.SEVER) return; // sever works on edges/pairs, handled separately

    // Monitor: a placed sensor (not a node-state the agent observes) that raises
    // detection while the agent is on or beside it. Toggles on repeat click.
    if (tool === TOOLS.MONITOR) {
      const mon = monitorsRef.current;
      if (mon.has(id)) {
        const next = new Set(mon);
        next.delete(id);
        monitorsRef.current = next;
        setMonitors(next);
        setCredits((c) => Math.min(CREDIT_CAP, c + currentToolDef.cost));
        sfx("remove");
        pushLog(`-- monitor removed from node ${id} (+${currentToolDef.cost}cr).`);
        return;
      }
      if (currentToolDef.cost > credits) {
        pushLog("xx insufficient credits.");
        return;
      }
      const next = new Set(mon);
      next.add(id);
      monitorsRef.current = next;
      setMonitors(next);
      setCredits((c) => Math.min(CREDIT_CAP, c - currentToolDef.cost));
      sfx("place");
      pushLog(`++ monitor deployed on node ${id}.`);
      return;
    }

    const targetState = DEFENSE_STATE[tool]!;
    const current = netRef.current.nodeStates[id];

    // Clicking the same defense again REMOVES it and refunds its cost.
    // (This is what previously let repeated clicks drain credits — every click
    //  was charged, even when it was just toggling the defense back off.)
    if (current === targetState) {
      setNodeState(id, OPEN as NodeState);
      setCredits((c) => Math.min(CREDIT_CAP, c + currentToolDef.cost));
      sfx("remove");
      pushLog(`-- ${currentToolDef.label.toLowerCase()} removed from node ${id} (+${currentToolDef.cost}cr).`);
      return;
    }

    // Placing fresh, or replacing a different defense (refund the old one).
    const refund = current === OPEN ? 0 : COST_BY_STATE[current] ?? 0;
    const netCost = currentToolDef.cost - refund;
    if (netCost > credits) {
      pushLog("xx insufficient credits.");
      return;
    }
    // Firewalls are penetrable now, so placing one never seals the agent off — no
    // reachability check needed. (Only Sever can change connectivity; see severLink.)
    setNodeState(id, targetState);
    setCredits((c) => Math.min(CREDIT_CAP, c - netCost));
    sfx("place");
    pushLog(`++ ${currentToolDef.label.toLowerCase()} deployed on node ${id}.`);
  };

  const severLink = (a: number, b: number) => {
    if (credits < 1) {
      pushLog("xx insufficient credits.");
      return;
    }
    // Don't allow severing that funnels the agent through a single chokepoint.
    const probe = netRef.current.adjacency.map((row) => [...row]);
    probe[a][b] = probe[b][a] = 0;
    if (!routeSurvives(probe)) {
      pushLog(
        diffRef.current === "recruit"
          ? "xx cannot sever — that would cut the agent off from an objective."
          : "xx cannot sever — the agent must keep two separate routes (no single chokepoint).",
      );
      return;
    }
    setNet((n) => {
      const adj = n.adjacency.map((row) => [...row]);
      adj[a][b] = adj[b][a] = 0;
      const updated = { ...n, adjacency: adj };
      netRef.current = updated;
      return updated;
    });
    setCredits((c) => c - 1);
    sfx("remove");
    pushLog(`-- link severed: ${a} <-> ${b}.`);
  };

  const onNodeClick = (id: number) => {
    if (tool === TOOLS.SEVER) {
      if (severPick === null) {
        setSeverPick(id);
        pushLog(`.. select a neighbor of node ${id} to cut the link.`);
      } else if (severPick === id) {
        setSeverPick(null);
      } else if (net.adjacency[severPick][id] === 1) {
        severLink(severPick, id);
        setSeverPick(null);
      } else {
        pushLog("xx those nodes are not directly linked.");
        setSeverPick(id);
      }
      return;
    }
    deployOnNode(id);
  };

  // ----------------------------------------------------------- pan / zoom
  const SCALE_MIN = 0.6;
  const SCALE_MAX = 5;

  const zoomAt = (factor: number, fx: number, fy: number) => {
    setView((v) => {
      const s = Math.min(SCALE_MAX, Math.max(SCALE_MIN, v.s * factor));
      const ratio = s / v.s;
      return { s, tx: fx - (fx - v.tx) * ratio, ty: fy - (fy - v.ty) * ratio };
    });
  };
  const zoomButton = (factor: number) => zoomAt(factor, VIEW_W / 2, VIEW_H / 2);
  const resetView = () => setView({ s: 1, tx: 0, ty: 0 });

  const clampScale = (s: number) => Math.min(SCALE_MAX, Math.max(SCALE_MIN, s));

  const onPointerDownSvg = (e: React.PointerEvent) => {
    pointers.current.set(e.pointerId, { x: e.clientX, y: e.clientY });
    if (pointers.current.size >= 2) {
      // begin pinch: cancel any pan, capture initial finger spread + midpoint
      panning.current = false;
      const pts = [...pointers.current.values()];
      const dist = Math.hypot(pts[0].x - pts[1].x, pts[0].y - pts[1].y) || 1;
      const rect = svgRef.current?.getBoundingClientRect();
      const midX = (pts[0].x + pts[1].x) / 2;
      const midY = (pts[0].y + pts[1].y) / 2;
      const cx = rect ? ((midX - rect.left) / rect.width) * VIEW_W : VIEW_W / 2;
      const cy = rect ? ((midY - rect.top) / rect.height) * VIEW_H : VIEW_H / 2;
      const v = viewRef.current;
      pinch.current = { dist, s: v.s, cx, cy, tx: v.tx, ty: v.ty };
    } else {
      const v = viewRef.current;
      panning.current = true;
      panStart.current = { x: e.clientX, y: e.clientY, tx: v.tx, ty: v.ty };
    }
  };

  const onPointerMoveSvg = (e: React.PointerEvent) => {
    if (pointers.current.has(e.pointerId)) {
      pointers.current.set(e.pointerId, { x: e.clientX, y: e.clientY });
    }
    if (pinch.current && pointers.current.size >= 2) {
      const pts = [...pointers.current.values()];
      const dist = Math.hypot(pts[0].x - pts[1].x, pts[0].y - pts[1].y) || 1;
      const s = clampScale(pinch.current.s * (dist / pinch.current.dist));
      const k = s / pinch.current.s;
      setView({
        s,
        tx: pinch.current.cx - (pinch.current.cx - pinch.current.tx) * k,
        ty: pinch.current.cy - (pinch.current.cy - pinch.current.ty) * k,
      });
      return;
    }
    if (!panning.current) return;
    const rect = svgRef.current?.getBoundingClientRect();
    if (!rect) return;
    const dx = ((e.clientX - panStart.current.x) / rect.width) * VIEW_W;
    const dy = ((e.clientY - panStart.current.y) / rect.height) * VIEW_H;
    setView((v) => ({ ...v, tx: panStart.current.tx + dx, ty: panStart.current.ty + dy }));
  };

  const onPointerUpSvg = (e: React.PointerEvent) => {
    pointers.current.delete(e.pointerId);
    if (pointers.current.size < 2) pinch.current = null;
    if (pointers.current.size === 0) {
      panning.current = false;
    } else if (pointers.current.size === 1) {
      // one finger remains after a pinch -> resume panning from it
      const [pt] = [...pointers.current.values()];
      const v = viewRef.current;
      panning.current = true;
      panStart.current = { x: pt.x, y: pt.y, tx: v.tx, ty: v.ty };
    }
  };

  // Mouse-wheel zoom toward the cursor (native non-passive listener so we can
  // preventDefault the page scroll while hovering the map).
  useEffect(() => {
    const el = svgRef.current;
    if (!el) return;
    const onWheel = (e: WheelEvent) => {
      e.preventDefault();
      const rect = el.getBoundingClientRect();
      const fx = ((e.clientX - rect.left) / rect.width) * VIEW_W;
      const fy = ((e.clientY - rect.top) / rect.height) * VIEW_H;
      const factor = e.deltaY < 0 ? 1.12 : 1 / 1.12;
      setView((v) => {
        const s = Math.min(SCALE_MAX, Math.max(SCALE_MIN, v.s * factor));
        const ratio = s / v.s;
        return { s, tx: fx - (fx - v.tx) * ratio, ty: fy - (fy - v.ty) * ratio };
      });
    };
    el.addEventListener("wheel", onWheel, { passive: false });
    return () => el.removeEventListener("wheel", onWheel);
  }, []);

  // --------------------------------------------------------------- edges
  const edges = useMemo(() => {
    const out: { a: number; b: number }[] = [];
    for (let i = 0; i < NODE_COUNT; i++)
      for (let j = i + 1; j < NODE_COUNT; j++) if (net.adjacency[i][j] === 1) out.push({ a: i, b: j });
    return out;
  }, [net.adjacency]);

  const statusMeta: Record<Status, { text: string; color: string }> = {
    idle: { text: "STANDBY", color: "#94a3b8" },
    running: { text: "INTRUSION ACTIVE", color: "#f43f5e" },
    breach: { text: "DATABASE BREACHED", color: "#f43f5e" },
    trapped: { text: "AGENT CONTAINED", color: "#34d399" },
    stalled: { text: "NETWORK HELD", color: "#34d399" },
    detected: { text: "AGENT DETECTED", color: "#38bdf8" },
  };

  return (
    <div className="font-mono">
      {/* Header */}
      <header className="mb-6 flex flex-wrap items-end justify-between gap-4 border-b border-edge pb-4">
        <div>
          <h1 className="font-display text-2xl font-bold tracking-wider text-cyan md:text-3xl">
            RED&nbsp;TEAM<span className="text-slate-500"> // </span>NETWORK&nbsp;SIMULATOR
          </h1>
          <p className="mt-1 text-xs text-slate-500">
            Goal-conditioned RL adversary · client-side ONNX inference · domain-randomized topology
          </p>
        </div>
        <div className="flex items-center gap-2 text-xs">
          <button
            onClick={() => setHeatOn((v) => !v)}
            title="Toggle policy intent heatmap"
            className={`rounded border px-2 py-1 ${
              heatOn ? "border-blood text-blood" : "border-edge text-slate-400 hover:border-slate-600"
            }`}
          >
            ◎ Intent
          </button>
          <button
            onClick={() => setMuted((v) => !v)}
            title={muted ? "Unmute" : "Mute"}
            className="rounded border border-edge px-2 py-1 text-slate-400 hover:border-slate-600"
          >
            {muted ? "🔇" : "🔊"}
          </button>
          <button
            onClick={() => setShowTutorial(true)}
            title="How to play"
            className="rounded border border-edge px-2 py-1 text-slate-400 hover:border-slate-600"
          >
            ?
          </button>
          <span
            className={`ml-1 inline-block h-2 w-2 rounded-full ${engine === "model" ? "bg-viper" : "bg-amber"} ${
              engine === "loading" ? "blink" : ""
            }`}
          />
          <span className="text-slate-400">
            {engine === "model"
              ? "ONNX POLICY"
              : engine === "heuristic"
                ? "HEURISTIC AGENT"
                : "LOADING…"}
          </span>
        </div>
      </header>

      <div className="grid grid-cols-1 gap-5 lg:grid-cols-[1fr_320px] lg:items-start">
        {/* Graph panel */}
        <section className="relative overflow-hidden rounded-lg border border-edge bg-panel shadow-glow">
          <div className="flex items-center justify-between border-b border-edge px-4 py-2 text-xs">
            <span className="font-display tracking-widest" style={{ color: statusMeta[status].color }}>
              ● {statusMeta[status].text}
            </span>
            <span className="text-slate-500">
              <span className={reachedFoothold ? "text-viper" : "text-amber"}>
                {reachedFoothold ? "→ DATABASE" : "→ FOOTHOLD"}
              </span>{" "}
              · turn {turns}/{cfg.maxTurns}
            </span>
          </div>

          {(status === "running" || detection > 0) && (
            <div className="flex items-center gap-2 border-b border-edge px-4 py-1 text-[10px]">
              <span className="tracking-widest text-sky-400">DETECTION</span>
              <div className="h-1.5 flex-1 overflow-hidden rounded bg-edge">
                <div
                  className="h-full transition-all"
                  style={{
                    width: `${Math.round(detection * 100)}%`,
                    background: detection > 0.66 ? "#f43f5e" : detection > 0.33 ? "#f59e0b" : "#38bdf8",
                  }}
                />
              </div>
              <span className="text-slate-500">{Math.round(detection * 100)}%</span>
            </div>
          )}

          <svg
            ref={svgRef}
            viewBox={`0 0 ${VIEW_W} ${VIEW_H}`}
            className="block w-full select-none"
            style={{ touchAction: "none" }}
            onPointerDown={onPointerDownSvg}
            onPointerMove={onPointerMoveSvg}
            onPointerUp={onPointerUpSvg}
            onPointerCancel={onPointerUpSvg}
            onPointerLeave={onPointerUpSvg}
          >
            {/* transparent backdrop so presses on empty space register (and bubble) */}
            <rect
              x={0}
              y={0}
              width={VIEW_W}
              height={VIEW_H}
              fill="transparent"
              className="cursor-grab active:cursor-grabbing"
            />
            <g transform={`translate(${view.tx} ${view.ty}) scale(${view.s})`}>
            {/* edges */}
            {edges.map(({ a, b }, i) => {
              const pa = net.positions[a];
              const pb = net.positions[b];
              const severable = tool === TOOLS.SEVER;
              const hot =
                severable && severPick !== null && (severPick === a || severPick === b);
              return (
                <line
                  key={i}
                  x1={px(pa.x)}
                  y1={py(pa.y)}
                  x2={px(pb.x)}
                  y2={py(pb.y)}
                  stroke={hot ? "#f43f5e" : "#1b2330"}
                  strokeWidth={severable ? 5 : 1.5}
                  strokeOpacity={severable ? 0.9 : 1}
                  className={severable ? "cursor-pointer hover:stroke-blood" : ""}
                  onClick={severable ? () => severLink(a, b) : undefined}
                />
              );
            })}

            {/* travel trail */}
            {trail.length > 1 &&
              trail.slice(1).map((node, i) => {
                const from = net.positions[trail[i]];
                const to = net.positions[node];
                return (
                  <line
                    key={`tr-${i}`}
                    x1={px(from.x)}
                    y1={py(from.y)}
                    x2={px(to.x)}
                    y2={py(to.y)}
                    stroke="#f43f5e"
                    strokeOpacity={0.25}
                    strokeWidth={2}
                    strokeDasharray="4 4"
                  />
                );
              })}

            {/* nodes */}
            {net.positions.map((p) => {
              const isTarget = p.id === net.target;
              const isAgent = p.id === agentPos;
              const isStart = p.id === net.start;
              const isFoothold = p.id === net.foothold;
              const isMonitored = monitors.has(p.id);
              const st = net.nodeStates[p.id];
              const fill = isTarget ? "#0b3d2e" : STATE_COLOR[st];
              const stroke = isTarget
                ? "#34d399"
                : st === OPEN
                  ? "#2c5562"
                  : STATE_COLOR[st];
              const picked = severPick === p.id;
              return (
                <g
                  key={p.id}
                  className="cursor-pointer"
                  onClick={() => onNodeClick(p.id)}
                  transform={`translate(${px(p.x)},${py(p.y)})`}
                >
                  {heatOn && status === "running" && (heat[p.id] ?? 0) > 0.02 && (
                    <circle
                      r={NODE_R + 4 + (heat[p.id] ?? 0) * 10}
                      fill="none"
                      stroke="#f43f5e"
                      strokeOpacity={Math.min(0.85, 0.2 + (heat[p.id] ?? 0))}
                      strokeWidth={1.5 + (heat[p.id] ?? 0) * 4}
                    />
                  )}
                  {isTarget && (
                    <circle r={NODE_R + 6} fill="none" stroke="#34d399" strokeWidth={1.5} className="pulse-ring" />
                  )}
                  {isFoothold && !isTarget && (
                    <circle
                      r={NODE_R + 5}
                      fill="none"
                      stroke="#fbbf24"
                      strokeWidth={1.5}
                      strokeDasharray="3 3"
                      strokeOpacity={reachedFoothold ? 0.25 : 0.9}
                    />
                  )}
                  {isMonitored && (
                    <circle r={NODE_R + 8} fill="none" stroke="#38bdf8" strokeWidth={1.5} strokeDasharray="2 3" strokeOpacity={0.8} />
                  )}
                  <circle
                    r={picked ? NODE_R + 3 : NODE_R}
                    fill={fill}
                    stroke={picked ? "#f43f5e" : stroke}
                    strokeWidth={picked ? 3 : 2}
                  />
                  <text
                    textAnchor="middle"
                    dy="4"
                    fontSize={NODE_FONT}
                    fill={isTarget ? "#a7f3d0" : "#7d93a5"}
                    fontFamily="var(--font-mono)"
                  >
                    {p.id}
                  </text>
                  {isStart && (
                    <text textAnchor="middle" dy={-(NODE_R + 6)} fontSize="9" fill="#64748b">
                      ENTRY
                    </text>
                  )}
                  {isTarget && (
                    <text textAnchor="middle" dy={-(NODE_R + 10)} fontSize="9" fill="#34d399">
                      DATABASE
                    </text>
                  )}
                  {isFoothold && !isTarget && (
                    <text
                      textAnchor="middle"
                      dy={-(NODE_R + 10)}
                      fontSize="9"
                      fill="#fbbf24"
                      opacity={reachedFoothold ? 0.4 : 1}
                    >
                      {reachedFoothold ? "FOOTHOLD ✓" : "FOOTHOLD"}
                    </text>
                  )}
                </g>
              );
            })}

            {/* agent marker */}
            <g
              transform={`translate(${px(net.positions[agentPos].x)},${py(net.positions[agentPos].y)})`}
              style={{ transition: "transform 0.4s ease" }}
            >
              <circle r={AGENT_R} fill="#f43f5e">
                <animate
                  attributeName="r"
                  values={`${AGENT_R};${AGENT_R + 3};${AGENT_R}`}
                  dur="1s"
                  repeatCount="indefinite"
                />
              </circle>
              <circle r={AGENT_R} fill="none" stroke="#fda4af" strokeWidth={1.5} />
            </g>
            </g>
          </svg>

          {/* outcome flash */}
          {flash && (
            <div
              className={`flash-fx pointer-events-none absolute inset-0 ${
                flash === "breach" ? "bg-blood/25" : "bg-viper/20"
              }`}
            />
          )}

          {/* zoom controls */}
          <div className="absolute bottom-3 right-3 flex flex-col gap-1">
            <button
              onClick={() => zoomButton(1.25)}
              title="Zoom in"
              className="h-8 w-8 rounded border border-edge bg-void/80 text-lg leading-none text-slate-300 backdrop-blur hover:border-cyan hover:text-cyan"
            >
              +
            </button>
            <button
              onClick={() => zoomButton(0.8)}
              title="Zoom out"
              className="h-8 w-8 rounded border border-edge bg-void/80 text-lg leading-none text-slate-300 backdrop-blur hover:border-cyan hover:text-cyan"
            >
              −
            </button>
            <button
              onClick={resetView}
              title="Reset view"
              className="h-8 w-8 rounded border border-edge bg-void/80 text-xs leading-none text-slate-300 backdrop-blur hover:border-cyan hover:text-cyan"
            >
              ⤢
            </button>
          </div>
        </section>

        {/* Control panel */}
        <aside className="flex flex-col gap-4">
          {/* difficulty */}
          <div className="rounded-lg border border-edge bg-panel p-3">
            <h2 className="mb-2 font-display text-xs tracking-widest text-slate-400">THREAT LEVEL</h2>
            <div className="grid grid-cols-3 gap-2">
              {(Object.keys(DIFFICULTY) as Difficulty[]).map((d) => {
                const active = difficulty === d;
                return (
                  <button
                    key={d}
                    onClick={() => applyDifficulty(d)}
                    className={`rounded border px-2 py-2 text-center text-xs font-semibold transition ${
                      active
                        ? "border-cyan bg-cyan/10 text-cyan"
                        : "border-edge text-slate-300 hover:border-slate-600"
                    }`}
                  >
                    {DIFFICULTY[d].label}
                  </button>
                );
              })}
            </div>
            <p className="mt-2 text-[10px] leading-tight text-slate-500">{cfg.blurb}</p>
            <div className="mt-1 flex justify-between text-[10px] text-slate-600">
              <span>agent {cfg.tickMs}ms</span>
              <span>{cfg.startCredits}→{cfg.creditCap}cr</span>
              <span>{cfg.maxTurns} turns</span>
            </div>
          </div>

          {/* tools */}
          <div className="rounded-lg border border-edge bg-panel p-3">
            <div className="mb-2 flex items-center justify-between">
              <h2 className="font-display text-xs tracking-widest text-slate-400">COUNTERMEASURES</h2>
              <span className="text-xs text-amber">◆ {credits} cr</span>
            </div>
            <div className="grid grid-cols-2 gap-2">
              {TOOL_DEFS.map((t) => {
                const active = tool === t.id;
                return (
                  <button
                    key={t.id}
                    onClick={() => {
                      setTool(t.id);
                      setSeverPick(null);
                    }}
                    className={`rounded border px-2 py-2 text-left text-xs transition ${
                      active
                        ? "border-cyan bg-cyan/10 text-cyan"
                        : "border-edge text-slate-300 hover:border-slate-600"
                    }`}
                    style={active ? { borderColor: t.color, color: t.color } : undefined}
                  >
                    <span className="flex items-center justify-between font-semibold">
                      {t.label}
                      <span className="text-[10px] opacity-70">{t.cost}cr</span>
                    </span>
                    <span className="mt-0.5 block text-[10px] leading-tight text-slate-500">{t.hint}</span>
                  </button>
                );
              })}
            </div>
          </div>

          {/* run controls */}
          <div className="rounded-lg border border-edge bg-panel p-3">
            <h2 className="mb-2 font-display text-xs tracking-widest text-slate-400">SESSION</h2>
            <div className="grid grid-cols-2 gap-2">
              {status === "running" ? (
                <button
                  onClick={pause}
                  className="col-span-2 rounded border border-amber bg-amber/10 py-2 text-xs font-semibold text-amber hover:bg-amber/20"
                >
                  ❚❚ PAUSE
                </button>
              ) : (
                <button
                  onClick={start}
                  className="col-span-2 rounded border border-blood bg-blood/10 py-2 text-xs font-semibold text-blood hover:bg-blood/20"
                >
                  ▶ {wavesMode ? `LAUNCH WAVE ${wave}` : "LAUNCH INTRUSION"}
                </button>
              )}
              <button
                onClick={() => resetRound()}
                className="rounded border border-edge py-2 text-xs text-slate-300 hover:border-slate-600"
              >
                ↺ Reset
              </button>
              <button
                onClick={newNetwork}
                className="rounded border border-edge py-2 text-xs text-slate-300 hover:border-slate-600"
              >
                ⟳ New Net
              </button>
            </div>

            <label className="mt-3 block text-[10px] text-slate-500">
              TICK RATE · {tickMs}ms
              <input
                type="range"
                min={250}
                max={1800}
                step={50}
                value={tickMs}
                onChange={(e) => setTickMs(Number(e.target.value))}
                className="mt-1 w-full accent-cyan"
              />
            </label>

            <button
              onClick={() => {
                setWavesMode((v) => !v);
                waveRef.current = 1;
                setWave(1);
                streakRef.current = 0;
                setStreak(0);
                setTickMs(cfgRef.current.tickMs);
                if (status === "running") {
                  setStatus("idle");
                  statusRef.current = "idle";
                }
              }}
              className={`mt-3 w-full rounded border py-2 text-xs font-semibold transition ${
                wavesMode ? "border-blood bg-blood/10 text-blood" : "border-edge text-slate-300 hover:border-slate-600"
              }`}
            >
              {wavesMode ? "▲ WAVES MODE: ON" : "△ Waves Mode: Off"}
            </button>
            {wavesMode && (
              <p className="mt-1 text-[10px] leading-tight text-slate-500">
                Each containment escalates: faster agent, denser network. A breach ends the run.
              </p>
            )}
          </div>

          {/* waves / streak status (waves mode) */}
          {wavesMode && (
            <div className="grid grid-cols-2 gap-2">
              <div className="rounded-lg border border-edge bg-panel p-3 text-center">
                <div className="text-2xl font-bold text-blood">{wave}</div>
                <div className="text-[10px] tracking-widest text-slate-500">WAVE · best {bestWave}</div>
              </div>
              <div className="rounded-lg border border-edge bg-panel p-3 text-center">
                <div className="text-2xl font-bold text-amber">×{streak}</div>
                <div className="text-[10px] tracking-widest text-slate-500">STREAK · best {bestStreak}</div>
              </div>
            </div>
          )}

          {/* score */}
          <div className="grid grid-cols-2 gap-2">
            <div className="rounded-lg border border-edge bg-panel p-3 text-center">
              <div className="text-2xl font-bold text-viper">{score.contained}</div>
              <div className="text-[10px] tracking-widest text-slate-500">CONTAINED</div>
            </div>
            <div className="rounded-lg border border-edge bg-panel p-3 text-center">
              <div className="text-2xl font-bold text-blood">{score.breaches}</div>
              <div className="text-[10px] tracking-widest text-slate-500">BREACHES</div>
            </div>
          </div>

          {/* points */}
          <div className="rounded-lg border border-edge bg-panel p-3">
            <div className="flex items-end justify-between">
              <div>
                <div className="text-[10px] tracking-widest text-slate-500">LAST ROUND</div>
                <div className="text-xl font-bold text-cyan">
                  {lastScore === null ? "—" : `+${lastScore}`}
                </div>
              </div>
              <div className="text-right">
                <div className="text-[10px] tracking-widest text-slate-500">BEST</div>
                <div className="text-xl font-bold text-amber">{bestScore}</div>
              </div>
            </div>
            <div className="mt-2 flex items-center justify-between border-t border-edge pt-2 text-[10px] text-slate-500">
              <span>
                SESSION <span className="font-bold text-slate-300">{sessionPoints}</span>
              </span>
              <span>
                STREAK <span className={`font-bold ${streak > 1 ? "text-amber" : "text-slate-300"}`}>×{streak}</span>
                {streak >= 2 && <span className="text-amber"> (+{Math.min(streak - 1, 5) * 10}%)</span>}
              </span>
            </div>
          </div>

          {/* leaderboard */}
          <div className="rounded-lg border border-edge bg-panel p-3">
            <h2 className="mb-2 font-display text-xs tracking-widest text-slate-400">LEADERBOARD</h2>
            {leaderboardEnabled() ? (
              <>
                <div className="mb-2 grid grid-cols-3 gap-1">
                  {(Object.keys(DIFFICULTY) as Difficulty[]).map((d) => (
                    <button
                      key={d}
                      onClick={() => setBoardDiff(d)}
                      className={`rounded border px-1 py-1 text-[10px] font-semibold transition ${
                        boardDiff === d
                          ? "border-cyan bg-cyan/10 text-cyan"
                          : "border-edge text-slate-400 hover:border-slate-600"
                      }`}
                    >
                      {DIFFICULTY[d].label}
                    </button>
                  ))}
                </div>
                <div className="mb-2 flex gap-2">
                  <input
                    value={playerName}
                    onChange={(e) => setPlayerName(e.target.value.slice(0, 16))}
                    placeholder="callsign"
                    maxLength={16}
                    className="min-w-0 flex-1 rounded border border-edge bg-void px-2 py-1 text-xs text-slate-200 outline-none focus:border-cyan"
                  />
                  <button
                    onClick={submitBest}
                    disabled={
                      submitting ||
                      bestScore <= 0 ||
                      bestScore === submittedBest ||
                      (turnstileOn() && !turnstileToken)
                    }
                    className="shrink-0 rounded border border-cyan bg-cyan/10 px-2 py-1 text-xs font-semibold text-cyan disabled:cursor-not-allowed disabled:opacity-40"
                  >
                    {submitting ? "…" : bestScore === submittedBest && bestScore > 0 ? "Sent" : "Submit"}
                  </button>
                </div>
                {turnstileOn() && <div ref={turnstileDivRef} className="mb-2" />}
                <ol className="log-scroll max-h-44 space-y-1 overflow-y-auto text-[11px]">
                  {board.filter((e) => e.difficulty === boardDiff).length === 0 && (
                    <li className="text-slate-600">No {DIFFICULTY[boardDiff].label} scores yet — be the first.</li>
                  )}
                  {board
                    .filter((e) => e.difficulty === boardDiff)
                    .slice(0, 15)
                    .map((e, i) => (
                      <li key={i} className="flex items-center justify-between gap-2">
                        <span className="flex min-w-0 items-center gap-2">
                          <span className="w-4 shrink-0 text-right text-slate-600">{i + 1}</span>
                          <span className="truncate text-slate-300">{e.name}</span>
                        </span>
                        <span className="shrink-0 font-bold text-amber">{e.score}</span>
                      </li>
                    ))}
                </ol>
              </>
            ) : (
              <p className="text-[11px] leading-relaxed text-slate-500">
                Local high scores are active. To enable a shared online board, deploy the
                Worker in <span className="text-slate-300">leaderboard-worker/</span> and set
                its URL in <span className="text-slate-300">lib/config.ts</span>.
              </p>
            )}
          </div>

          {/* log */}
          <div className="rounded-lg border border-edge bg-panel p-3">
            <h2 className="mb-1 font-display text-xs tracking-widest text-slate-400">EVENT LOG</h2>
            <div
              ref={logRef}
              className="log-scroll h-40 overflow-y-auto text-[11px] leading-relaxed text-slate-400"
            >
              {log.map((line, i) => (
                <div
                  key={i}
                  className={
                    line.startsWith(">>")
                      ? "text-blood"
                      : line.startsWith("<<") || line.startsWith("==")
                        ? "text-viper"
                        : line.startsWith("~~")
                          ? "text-cyan"
                          : line.startsWith("xx")
                            ? "text-amber"
                            : ""
                  }
                >
                  {line}
                </div>
              ))}
            </div>
          </div>
        </aside>
      </div>

      <p className="mt-4 text-center text-[10px] text-slate-600">
        Select a countermeasure, click a node to deploy it, then launch the intrusion. The agent
        recomputes its path every tick against your live defenses.
      </p>

      {/* tutorial / how-to-play overlay */}
      {showTutorial && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-void/80 p-4 backdrop-blur-sm">
          <div className="max-h-[85vh] w-full max-w-lg overflow-y-auto rounded-lg border border-edge bg-panel p-5 shadow-glow">
            <h2 className="font-display text-lg font-bold tracking-wider text-cyan">HOW TO PLAY</h2>
            <p className="mt-2 text-xs leading-relaxed text-slate-300">
              A Red Team AI (the pulsing <span className="text-blood">red dot</span>) is trying to
              reach the <span className="text-viper">DATABASE</span>. You&apos;re Blue Team — stop it
              by spending credits to fortify the network before and during the intrusion.
            </p>
            <div className="mt-3 space-y-1.5 text-xs text-slate-300">
              {TOOL_DEFS.map((t) => (
                <div key={t.id} className="flex items-start gap-2">
                  <span className="mt-0.5 shrink-0 font-semibold" style={{ color: t.color }}>
                    {t.label}
                  </span>
                  <span className="text-slate-500">— {t.hint} ({t.cost}cr)</span>
                </div>
              ))}
            </div>
            <ul className="mt-3 space-y-1 text-[11px] leading-relaxed text-slate-400">
              <li>• Credits regenerate while a round runs. Click a placed defense again to remove it and refund the cost.</li>
              <li>• Contain the agent (trap, dead-end, or run out its clock) to score; a breach scores nothing.</li>
              <li>• <span className="text-blood">◎ Intent</span> shows where the AI wants to move next. Try <span className="text-blood">Waves</span> mode for an escalating gauntlet.</li>
              <li>• Pan by dragging the map; zoom with the wheel, the +/− buttons, or pinch on touch.</li>
            </ul>
            <button
              onClick={() => {
                setShowTutorial(false);
                try {
                  localStorage.setItem("rts-tutorial-seen", "1");
                } catch {
                  /* ignore */
                }
              }}
              className="mt-4 w-full rounded border border-cyan bg-cyan/10 py-2 text-xs font-semibold text-cyan hover:bg-cyan/20"
            >
              Got it — let me in
            </button>
          </div>
        </div>
      )}
    </div>
  );
}
