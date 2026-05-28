"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  NODE_COUNT,
  OPEN,
  FIREWALL,
  HONEYPOT,
  CORRUPTED,
  TOOLS,
  type Tool,
  type NodeState,
} from "@/lib/constants";
import {
  generateNetwork,
  bfsDistance,
  neighbors,
  buildObservation,
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

type Status = "idle" | "running" | "breach" | "trapped" | "stalled";

interface ToolDef {
  id: Tool;
  label: string;
  hint: string;
  cost: number;
  color: string;
}

const TOOL_DEFS: ToolDef[] = [
  { id: TOOLS.FIREWALL, label: "Firewall", hint: "Blocks the node entirely", cost: 2, color: "#f59e0b" },
  { id: TOOLS.HONEYPOT, label: "Honeypot", hint: "Traps & terminates the agent", cost: 3, color: "#a78bfa" },
  { id: TOOLS.CORRUPT, label: "Corrupt", hint: "Randomizes the agent's next move", cost: 2, color: "#22d3ee" },
  { id: TOOLS.SEVER, label: "Sever Link", hint: "Cut a connection between nodes", cost: 1, color: "#f43f5e" },
];

const STATE_COLOR: Record<number, string> = {
  [OPEN]: "#1f3a44",
  [FIREWALL]: "#f59e0b",
  [HONEYPOT]: "#a78bfa",
  [CORRUPTED]: "#22d3ee",
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

// Layout mapping (normalized 0..1 -> SVG viewBox).
const VIEW_W = 1000;
const VIEW_H = 640;
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
    generateNetwork(Date.now(), initialCfg.extraEdgeProb),
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
  const pending = useRef<Map<number, (a: number) => void>>(new Map());
  const reqSeq = useRef(0);
  const creditsRef = useRef(credits);
  const diffRef = useRef(difficulty);
  useEffect(() => void (creditsRef.current = credits), [credits]);
  useEffect(() => void (diffRef.current = difficulty), [difficulty]);

  useEffect(() => void (netRef.current = net), [net]);
  useEffect(() => void (posRef.current = agentPos), [agentPos]);
  useEffect(() => void (statusRef.current = status), [status]);
  useEffect(() => void (turnsRef.current = turns), [turns]);

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

  // Award points for the round outcome (containment scores; a breach scores 0).
  const award = useCallback(
    (result: string) => {
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
    },
    [pushLog],
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
          resolve(msg.action);
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
  const legalMoves = useCallback((n: NetworkState, pos: number): number[] => {
    return neighbors(n.adjacency, pos).filter((v) => n.nodeStates[v] !== FIREWALL);
  }, []);

  // Greedy fallback used until the .onnx policy is available.
  const heuristicMove = useCallback(
    (n: NetworkState, pos: number, moves: number[]): number => {
      let best = moves[0];
      let bestD = Infinity;
      for (const m of moves) {
        const d = bfsDistance(n.adjacency, n.nodeStates, m, n.target);
        if (d < bestD) {
          bestD = d;
          best = m;
        }
      }
      return best;
    },
    [],
  );

  const inferViaWorker = useCallback(
    (n: NetworkState, pos: number, moves: number[]): Promise<number> => {
      return new Promise((resolve) => {
        const id = ++reqSeq.current;
        pending.current.set(id, resolve);
        workerRef.current?.postMessage({
          type: "infer",
          requestId: id,
          obs: buildObservation(n, pos),
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
      posRef.current = action;
      setAgentPos(action);
      setTrail((t) => (t[t.length - 1] === action ? t : [...t, action]));

      if (action === n.target) {
        statusRef.current = "breach";
        setStatus("breach");
        setScore((s) => ({ ...s, breaches: s.breaches + 1 }));
        pushLog(`>> BREACH — database node ${action} compromised.`);
        award("breach");
      } else if (state === HONEYPOT) {
        statusRef.current = "trapped";
        setStatus("trapped");
        setScore((s) => ({ ...s, contained: s.contained + 1 }));
        pushLog(`<< TRAPPED — agent ensnared in honeypot at node ${action}.`);
        award("honeypot");
      } else if (state === CORRUPTED) {
        corruptNext.current = true;
        pushLog(`~~ agent corrupted at node ${action} — next move scrambled.`);
      }
    },
    [pushLog, award],
  );

  const tick = useCallback(async () => {
    if (statusRef.current !== "running") return;
    const n = netRef.current;
    const pos = posRef.current;

    if (turnsRef.current >= cfgRef.current.maxTurns) {
      statusRef.current = "stalled";
      setStatus("stalled");
      setScore((s) => ({ ...s, contained: s.contained + 1 }));
      pushLog("== agent ran out of time — network held.");
      award("timeout");
      return;
    }

    const moves = legalMoves(n, pos);
    if (moves.length === 0) {
      statusRef.current = "stalled";
      setStatus("stalled");
      setScore((s) => ({ ...s, contained: s.contained + 1 }));
      pushLog("== agent contained — no legal route remains.");
      award("stalled");
      return;
    }

    let action: number;
    if (corruptNext.current) {
      corruptNext.current = false;
      const all = neighbors(n.adjacency, pos);
      action = all[Math.floor(Math.random() * all.length)];
      if (n.nodeStates[action] === FIREWALL) {
        pushLog("~~ scrambled move slammed into a firewall.");
        setTurns((t) => t + 1);
        if (statusRef.current === "running") setTimeout(tick, tickMs);
        return;
      }
    } else if (modelReady.current) {
      action = await inferViaWorker(n, pos, moves);
    } else {
      action = heuristicMove(n, pos, moves);
    }

    if (statusRef.current !== "running") return;
    if (action == null || action < 0) {
      setTimeout(tick, tickMs);
      return;
    }

    applyMove(action);
    setTurns((t) => t + 1);
    if (statusRef.current === "running") setTimeout(tick, tickMs);
  }, [applyMove, heuristicMove, inferViaWorker, legalMoves, pushLog, tickMs, award]);

  const start = () => {
    if (status === "running") return;
    if (status !== "idle") resetRound(false);
    setStatus("running");
    statusRef.current = "running";
    pushLog("// intrusion initiated. defend the database.");
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
    if (newLog) pushLog("// round reset — defenses cleared, links restored, agent back at entry.");
  };

  const newNetwork = () => {
    const fresh = generateNetwork(Date.now(), cfg.extraEdgeProb);
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
    pushLog("// new topology generated — domain randomized.");
  };

  // Switch difficulty: retune the economy/agent and generate a matching board.
  const applyDifficulty = (d: Difficulty) => {
    if (d === difficulty) return;
    const c = DIFFICULTY[d];
    setDifficulty(d);
    cfgRef.current = c;
    setTickMs(c.tickMs);
    const fresh = generateNetwork(Date.now(), c.extraEdgeProb);
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

  const CREDIT_CAP = cfg.creditCap;

  // Which node-state each node tool produces.
  const DEFENSE_STATE: Partial<Record<Tool, NodeState>> = {
    [TOOLS.FIREWALL]: FIREWALL,
    [TOOLS.HONEYPOT]: HONEYPOT,
    [TOOLS.CORRUPT]: CORRUPTED,
  };
  // Cost to refund when an existing defense is removed/replaced (mirrors TOOL_DEFS costs).
  const COST_BY_STATE: Record<number, number> = {
    [FIREWALL]: 2,
    [HONEYPOT]: 3,
    [CORRUPTED]: 2,
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

  // The core invariant: a passable route from the agent's current position to the
  // database must always exist. This blocks the "instant enclosure" exploits —
  // walling the entry so the agent can't move, or sealing the target so it can
  // never be reached. Defenses can reshape and lengthen the path, never erase it.
  // (Only firewalls and severed links reduce reachability; honeypots/corrupts are
  // passable, so they never need this check.)
  const routeSurvives = (adjacency: number[][], nodeStates: NodeState[]): boolean => {
    const from = posRef.current;
    const to = netRef.current.target;
    return Number.isFinite(bfsDistance(adjacency, nodeStates, from, to));
  };

  const deployOnNode = (id: number) => {
    if (id === posRef.current || id === net.target || id === net.start) {
      pushLog("xx cannot place a defense on that node.");
      return;
    }
    if (tool === TOOLS.SEVER) return; // sever works on edges/pairs, handled separately

    const targetState = DEFENSE_STATE[tool]!;
    const current = netRef.current.nodeStates[id];

    // Clicking the same defense again REMOVES it and refunds its cost.
    // (This is what previously let repeated clicks drain credits — every click
    //  was charged, even when it was just toggling the defense back off.)
    if (current === targetState) {
      setNodeState(id, OPEN as NodeState);
      setCredits((c) => Math.min(CREDIT_CAP, c + currentToolDef.cost));
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
    // A firewall removes a node from play — make sure a route to the DB survives.
    if (targetState === FIREWALL) {
      const probe = [...netRef.current.nodeStates];
      probe[id] = FIREWALL as NodeState;
      if (!routeSurvives(netRef.current.adjacency, probe)) {
        pushLog("xx firewall blocked — that would seal the agent off from the database.");
        return;
      }
    }
    setNodeState(id, targetState);
    setCredits((c) => Math.min(CREDIT_CAP, c - netCost));
    pushLog(`++ ${currentToolDef.label.toLowerCase()} deployed on node ${id}.`);
  };

  const severLink = (a: number, b: number) => {
    if (credits < 1) {
      pushLog("xx insufficient credits.");
      return;
    }
    // Don't allow severing the agent's last route to the database.
    const probe = netRef.current.adjacency.map((row) => [...row]);
    probe[a][b] = probe[b][a] = 0;
    if (!routeSurvives(probe, netRef.current.nodeStates)) {
      pushLog("xx cannot sever — that would cut the agent off from the database entirely.");
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
          <span
            className={`inline-block h-2 w-2 rounded-full ${engine === "model" ? "bg-viper" : "bg-amber"} ${
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

      <div className="grid grid-cols-1 gap-5 lg:grid-cols-[1fr_320px]">
        {/* Graph panel */}
        <section className="relative overflow-hidden rounded-lg border border-edge bg-panel shadow-glow">
          <div className="flex items-center justify-between border-b border-edge px-4 py-2 text-xs">
            <span className="font-display tracking-widest" style={{ color: statusMeta[status].color }}>
              ● {statusMeta[status].text}
            </span>
            <span className="text-slate-500">
              turn {turns}/{cfg.maxTurns}
            </span>
          </div>

          <svg viewBox={`0 0 ${VIEW_W} ${VIEW_H}`} className="block w-full">
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
                  {isTarget && (
                    <circle r={NODE_R + 6} fill="none" stroke="#34d399" strokeWidth={1.5} className="pulse-ring" />
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
          </svg>
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
                  ▶ LAUNCH INTRUSION
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
          </div>

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
          </div>

          {/* leaderboard */}
          <div className="rounded-lg border border-edge bg-panel p-3">
            <h2 className="mb-2 font-display text-xs tracking-widest text-slate-400">LEADERBOARD</h2>
            {leaderboardEnabled() ? (
              <>
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
                  {board.length === 0 && <li className="text-slate-600">No scores yet — be the first.</li>}
                  {board.slice(0, 15).map((e, i) => (
                    <li key={i} className="flex items-center justify-between gap-2">
                      <span className="flex min-w-0 items-center gap-2">
                        <span className="w-4 shrink-0 text-right text-slate-600">{i + 1}</span>
                        <span className="truncate text-slate-300">{e.name}</span>
                        <span className="shrink-0 text-[9px] uppercase text-slate-600">{e.difficulty}</span>
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
    </div>
  );
}
