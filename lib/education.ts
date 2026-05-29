// Educational / framing content for the simulator. None of this affects the agent
// or the model — it's the layer that maps the game's mechanics onto the real-world
// concepts they represent (asset types, ATT&CK-style phases, defensive techniques).

import { TOOLS } from "./constants";

// ---------------------------------------------------------------- asset types
export type AssetRole =
  | "workstation"
  | "webserver"
  | "appserver"
  | "domaincontroller"
  | "database";

export interface AssetDef {
  label: string;
  code: string; // short tag shown on the node
  color: string;
  blurb: string;
}

export const ASSETS: Record<AssetRole, AssetDef> = {
  workstation: {
    label: "Workstation",
    code: "WS",
    color: "#7d93a5",
    blurb: "An employee endpoint — the usual initial-access landing spot via phishing or a malicious attachment.",
  },
  webserver: {
    label: "Web Server",
    code: "WEB",
    color: "#0ea5e9",
    blurb: "An internet-facing service — a common entry and pivot point into the internal network.",
  },
  appserver: {
    label: "App / File Server",
    code: "APP",
    color: "#8b5cf6",
    blurb: "An internal application or file server holding business data and cached credentials.",
  },
  domaincontroller: {
    label: "Domain Controller",
    code: "DC",
    color: "#fbbf24",
    blurb: "The identity backbone of the network — owning it hands an attacker the keys to the kingdom.",
  },
  database: {
    label: "Database",
    code: "DB",
    color: "#34d399",
    blurb: "The crown jewels — the high-value data the attacker is ultimately after.",
  },
};

// --------------------------------------------------------- kill-chain phases
export interface PhaseInfo {
  label: string;
  tactic: string; // MITRE ATT&CK tactic id
  desc: string;
}

// The full chain, for the field guide.
export const KILL_CHAIN: PhaseInfo[] = [
  { label: "Initial Access", tactic: "TA0001", desc: "The attacker gains a foothold on an exposed host — here, the entry workstation." },
  { label: "Lateral Movement", tactic: "TA0008", desc: "Pivoting host-to-host through the network toward higher-value systems." },
  { label: "Privilege Escalation", tactic: "TA0004", desc: "Seizing the domain controller — the foothold objective that unlocks the rest." },
  { label: "Collection & Exfiltration", tactic: "TA0010", desc: "Reaching and extracting the database — the crown jewels." },
];

// The phase the agent is currently in, given whether it holds the foothold yet.
export function currentPhase(reachedFoothold: boolean): PhaseInfo {
  return reachedFoothold ? KILL_CHAIN[3] : KILL_CHAIN[1];
}

// --------------------------------------------------- defense concept cards
export interface ConceptCard {
  tool: string;
  title: string;
  realWorld: string;
  tip: string;
}

export const DEFENSE_CONCEPTS: ConceptCard[] = [
  {
    tool: TOOLS.FIREWALL,
    title: "Firewall → Network Segmentation",
    realWorld: "Splitting the network into zones with access controls so a compromised host can't freely reach everything.",
    tip: "Penetrable — it slows the agent rather than stopping it. Use it to funnel and buy time.",
  },
  {
    tool: TOOLS.HONEYPOT,
    title: "Honeypot → Deception & Canaries",
    realWorld: "Decoy systems that look valuable; an intruder touching one reveals itself and gets trapped.",
    tip: "Probabilistic (~70%) — layer several across likely paths to raise the odds.",
  },
  {
    tool: TOOLS.TARPIT,
    title: "Tarpit → Rate Limiting & Throttling",
    realWorld: "Deliberately slowing connections to bog down automated movement through the network.",
    tip: "Cheap friction; stacks with monitors to keep the agent in view longer.",
  },
  {
    tool: TOOLS.MONITOR,
    title: "Monitor → Intrusion Detection (IDS / EDR / SIEM)",
    realWorld: "Sensors that raise alerts as an intruder moves; enough signal triggers an eviction response.",
    tip: "Place along chokepoints and pair with friction so the agent lingers where you can see it.",
  },
  {
    tool: TOOLS.SEVER,
    title: "Sever → Isolation & Containment",
    realWorld: "Cutting a link to contain movement — like isolating a segment during incident response.",
    tip: "Can't funnel the agent to one chokepoint: two separate routes always remain on harder levels.",
  },
  {
    tool: TOOLS.CORRUPT,
    title: "Corrupt → Active Defense / Disruption",
    realWorld: "Disrupting an attacker's tooling or session so their next action misfires.",
    tip: "Scrambles the agent's next move — a one-off disruption rather than a wall.",
  },
];

// ------------------------------------------------- incident report narrative
export interface RoundReport {
  result: string;
  contained: boolean;
  turns: number;
  creditsLeft: number;
  detection: number; // 0-100
  reachedFoothold: boolean;
  fwHold: number;
  fwBreach: number;
  hpTrap: number;
  hpEvade: number;
  tarpit: number;
  corrupt: number;
}

export function incidentNarrative(r: RoundReport): {
  headline: string;
  outcome: "breach" | "contained";
  techniques: string[];
  takeaway: string;
} {
  const techniques: string[] = [];
  techniques.push(r.reachedFoothold ? "Established a foothold on the domain controller" : "Failed to establish a foothold");
  if (r.fwBreach > 0) techniques.push(`Bypassed network segmentation ${r.fwBreach}×`);
  if (r.fwHold > 0) techniques.push(`Rebuffed by firewalls ${r.fwHold}×`);
  if (r.hpEvade > 0) techniques.push(`Slipped past deception ${r.hpEvade}×`);
  if (r.tarpit > 0) techniques.push(`Throttled in tarpits ${r.tarpit}×`);
  if (r.corrupt > 0) techniques.push(`Disrupted at corrupted nodes ${r.corrupt}×`);

  let headline: string;
  let takeaway: string;
  switch (r.result) {
    case "breach":
      headline = "BREACH — data exfiltrated";
      takeaway = r.reachedFoothold
        ? "The agent pivoted through the DC and reached the database. Spread deception and monitoring across both routes — don't rely on a single chokepoint."
        : "The agent reached the database fast. Tighten the approaches to the crown jewels.";
      break;
    case "honeypot":
      headline = "CONTAINED — agent ensnared in a honeypot";
      takeaway = "Deception paid off. Honeypots are probabilistic, so layering them across likely paths raises your hit rate.";
      break;
    case "detected":
      headline = "CONTAINED — intrusion detected & evicted";
      takeaway = "Monitoring traced the agent before it reached the data. Friction (firewalls, tarpits) keeps it on the network long enough to be caught.";
      break;
    case "timeout":
    case "stalled":
      headline = "CONTAINED — network held";
      takeaway = "You ran the attacker out of time. Solid, but active defenses (honeypots, monitors) end intrusions faster and score higher.";
      break;
    default:
      headline = "Round complete";
      takeaway = "";
  }
  return { headline, outcome: r.result === "breach" ? "breach" : "contained", techniques, takeaway };
}
