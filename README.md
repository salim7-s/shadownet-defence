---
title: ShadowNet
emoji: 🛡️
colorFrom: gray
colorTo: red
sdk: docker
pinned: false
---

# ShadowNet

**Deceptive containment for cyber defense agents.**

In cybersecurity, speed matters — but timing matters more. A defender that reacts too early destroys evidence, alerts the attacker, and loses the chance to understand what is actually happening. Human analysts know this. They watch quietly, gather clues, redirect the attacker into controlled systems, and only *then* contain the incident.

ShadowNet asks whether a language model can learn to do the same thing.

---

**Quick links:** [Blog Post](BLOG_POST.md) · [Training Guide](TRAINING.md) · [Colab Notebook](notebooks/ShadowNet_SFT_Colab.ipynb) · [License](LICENSE)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/salim7-s/ShadowNet-When-Defense-Thinks-Like-the-Attacker/blob/main/notebooks/ShadowNet_SFT_Colab.ipynb)

---

## What this repo includes

- An OpenEnv-compatible environment with a Hugging Face Space deployment path
- A working Colab training notebook
- A trained adapter artifact in `artifacts/shadownet-sft-adapter`
- Real training evidence in `training/sft_loss_curve.png`
- A trained-vs-baseline comparison in `training/trained_vs_baseline_heatmap_better.png`

---

## The problem

The defender works under partial information. It can see anomalous nodes, attacker behavior tier, SIEM-style alerts, available forensic artifacts, and valid actions. It cannot directly see `detection_risk` — the hidden variable representing how close the attacker is to realising the defender is onto them.

That missing variable changes the whole problem. The agent has to infer when patience is safe and when delay becomes dangerous. A policy that blocks everything immediately destroys the investigation. A policy that waits too long loses the assets. The task is finding the space between.

---

## Episode structure

Each episode moves through three phases in sequence:

```
Track  →  Contain  →  Evidence
```

**Track:** Observe anomalous nodes. Build a behavioral model of the attacker without triggering suspicion.

**Contain:** Mirror traffic, introduce honeypot infrastructure, redirect flows. Keep detection risk low.

**Evidence:** Lock forensic artifacts before they decay. Preserve the investigation before the window closes.

A bad move in the tracking phase can ruin the evidence phase much later. This is what makes the task long-horizon rather than reactive.

---

## Control loop

The environment follows a simple but meaningful decision cycle:

```
Observe partial state
        ↓
Infer attacker behavior + hidden suspicion
        ↓
Choose one action
        ↓
Environment updates (attacker moves, artifacts decay, risk shifts)
        ↓
Reward + phase progress
        ↓
↻ back to observe
```

---

## Environment layer diagram

The environment is structured as four nested layers. The agent has access to the outer two only. The inner two are computed by the environment and inform the reward but are never directly observable.

```
┌─────────────────────────────────────────────────────────┐
│  Agent interface                              [visible]  │
│  alerts · anomalous nodes · valid actions · artifacts    │
│                                                          │
│  ┌───────────────────────────────────────────────────┐  │
│  │  Environment state                      [visible]  │  │
│  │  network graph · artifact decay · phase tracker    │  │
│  │                                                    │  │
│  │  ┌─────────────────────────────────────────────┐  │  │
│  │  │  Attacker model                   [visible]  │  │  │
│  │  │  behavior tier · movement · suspicion path   │  │  │
│  │  │                                              │  │  │
│  │  │  ┌────────────────────────────────────────┐ │  │  │
│  │  │  │  Hidden variables              [hidden] │ │  │  │
│  │  │  │  detection_risk · attacker suspicion    │ │  │  │
│  │  │  └────────────────────────────────────────┘ │  │  │
│  │  └─────────────────────────────────────────────┘  │  │
│  └───────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────┘
```

---

## Action space

The agent has eight available actions. These are not cosmetic choices — they create real tradeoffs between stealth, speed, deception quality, and evidence preservation.

| Action | Tier | Effect |
|---|---|---|
| `observe` | passive | Gather state without acting. Zero exposure, zero progress. |
| `wait_and_track` | passive | Extend observation window. Builds behavioral profile. |
| `mirror_traffic` | covert | Duplicate flows to out-of-band capture. Invisible to attacker. |
| `redirect` | covert | Route attacker into controlled honeypot infrastructure. |
| `lock_artifact` | forensic | Preserve forensic artifacts before decay window closes. |
| `partial_covert` | covert | Partial containment. Lower noise, moderate effect. |
| `loud_contain` | hard | Hard containment. Effective but spikes detection risk. |
| `emergency_expel` | hard | Last resort eviction. Ends investigation, protects assets. |

---

## Reward design

The final score is a weighted combination of six signals:

```
reward = 0.25 × asset_safety
       + 0.25 × forensic_value
       + 0.20 × stealth_score
       + 0.15 × honeypot_quality
       + 0.10 × phase_completion
       + 0.05 × efficiency
```

A policy that stays stealthy but protects nothing should not win. A policy that blocks everything immediately but destroys the investigation should not win either. The multi-objective reward prevents either collapse.

---

## Training

**Base model:** `Qwen/Qwen2.5-1.5B-Instruct`  
**Method:** LoRA adapters via TRL SFTTrainer  
**Notebook:** [notebooks/ShadowNet_SFT_Colab.ipynb](notebooks/ShadowNet_SFT_Colab.ipynb)  
**Adapter:** [artifacts/shadownet-sft-adapter](artifacts/shadownet-sft-adapter)

### Training loss

![SFT Training Loss](training/sft_loss_curve.png)

The loss curve shows the model learning a stable mapping from environment observations to structured actions. Training achieved 100% valid output generation (zero parse failures).

### Trained vs baseline comparison

![Trained vs Baseline Heatmap](training/trained_vs_baseline_heatmap_better.png)

**SFT results (learning from baseline demonstrations):**
- Easy + Stealthy: 0.627 → 0.662 (+5.6%)
- Medium + Stealthy: 0.567 → 0.594 (+4.8%)
- Medium + Adaptive: 0.502 → 0.546 (+8.8%)

The model learned action formatting and defensive sequences. Hard scenarios suggest RL training would be needed to surpass baseline on complex cases.

### Baseline reference

| Task | Random | Baseline | SFT Result |
|---|---|---|---|
| `shadow-easy` | ~0.36 | ~0.52–0.59 | Matches or slightly exceeds |
| `shadow-medium` | ~0.35 | ~0.47–0.50 | Mixed, profile-dependent |
| `shadow-hard` | ~0.35 | ~0.45–0.47 | Below baseline |

ShadowNet is not trivial—scores have headroom for improvement via RL training.

Detailed data: [artifacts/eval_summary.md](artifacts/eval_summary.md) · [training/eval_baseline.json](training/eval_baseline.json)

---

## OpenEnv compliance

ShadowNet is fundamentally stateful — the next observation depends on what the defender just did, what the attacker inferred, and which evidence is still available. This is not a one-shot QA benchmark or a stateless tool-call task. It is a persistent environment where actions shape future context.

- `openenv.yaml` manifest included
- Standard `reset()` / `step()` interface
- Client/server separation maintained
- Deployable Hugging Face Space target
- Re-runnable Colab training notebook

---

## API endpoints

| Endpoint | Method | Purpose |
|---|---|---|
| `/health` | GET | Liveness check |
| `/tasks` | GET | Task metadata |
| `/reset` | POST | Start an episode |
| `/step` | POST | Apply one action |
| `/state` | GET | Debug state |
| `/grader` | GET | Grading breakdown |
| `/baseline` | GET | Baseline performance |
| `/network-state` | GET | Graph state for dashboard |
| `/siem-alerts` | GET | Live alert feed |
| `/reasoning-log` | GET | Structured action reasoning |
| `/demo/compare` | GET | Good vs bad policy comparison |

---

## Repo layout

```
.
├── openenv.yaml
├── Dockerfile
├── environment.py
├── grader.py
├── data.py
├── agent.py
├── inference.py
├── client.py
├── eval_harness.py
├── server/
├── training/
│   ├── sft_loss_curve.png
│   ├── trained_vs_baseline_heatmap_better.png
│   ├── eval_baseline.json
│   └── eval_baseline_table.md
├── artifacts/
│   ├── shadownet-sft-adapter/
│   └── eval_summary.md
├── notebooks/
│   └── ShadowNet_SFT_Colab.ipynb
├── TRAINING.md
└── BLOG_POST.md
```

---

## Deployment checklist

- [ ] Push repo to GitHub
- [ ] Deploy Docker Space on Hugging Face
- [ ] Verify `/health`, `/tasks`, `/baseline`, `/demo/compare`
- [ ] Keep the Colab notebook public
- [ ] Keep the Space public
- [ ] Add public Space URL below
- [ ] Add public W&B run URL below
- [ ] Add demo video link below

---

## Links

- **GitHub:** (https://github.com/salim7-s/shadownet-defence.git)
- **Hugging Face Space:** [zizoha/shadownet-Cops](https://huggingface.co/spaces/zizoha/shadownet-Cops)
