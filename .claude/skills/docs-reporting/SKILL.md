---
name: docs-reporting
description: >
  Documentation and reporting discipline for the conflict-damage-monitor project. Use
  this skill at the end of every project phase, when the user asks to "write up" results,
  "update the docs", "document what we built", "write a report", "add to the portfolio",
  or "summarize the findings". Also trigger when a training run completes and metrics are
  available, when a new pipeline stage is finished, or when the user mentions sharing
  work externally. The report structure (motivation → data → model → results →
  limitations → ethics) is fixed — every report or phase doc follows it. The ethical
  disclaimer about macro-scale analysis is non-negotiable and must appear in any document
  intended for external audiences. When in doubt, write the docs — undocumented work
  cannot be shared, cited, or built upon.
---

# Docs & Reporting — conflict-damage-monitor

Good documentation serves two audiences: future-you picking up the project after a break,
and external readers (advisors, collaborators, portfolio reviewers). The structures below
are designed to satisfy both with one writing pass.

---

## When to update docs/

Update documentation at the end of every project phase, not continuously. The phases
from `CLAUDE.md` map directly to doc files:

| Phase completed | Document to update/create |
|---|---|
| Data ingestion + preprocessing | `docs/01-data.md` |
| Baseline EfficientNet-B4 classifier | `docs/02-baseline-model.md` |
| UNet++ segmentation model | `docs/03-segmentation-model.md` |
| Change detection + evaluation | `docs/04-evaluation.md` |
| FastAPI inference server | `docs/05-inference-api.md` |
| React dashboard | `docs/06-dashboard.md` |
| Any training run with final metrics | `docs/results/YYYY-MM-DD-<run-name>.md` |

Also maintain:
- `docs/README.md` — project overview and navigation index
- `docs/ethics.md` — standalone ethics statement (see below)
- `docs/debug-log.md` — running fix log (maintained by self-healing skill)

---

## Report structure — fixed for all phase docs

Every phase document and technical report follows this six-section structure. Write
sections in order; don't skip sections even if they're brief.

### Template

```markdown
# [Phase/Report Title]
*Last updated: YYYY-MM-DD*

## Motivation
Why this phase matters for the overall project goal. What question does it answer?
What would break downstream if this phase were missing or wrong?

## Data
- Datasets used: names, versions, sizes
- Preprocessing steps applied (tiling, normalization, augmentation)
- Class distribution (always include destroyed-class %)
- Train/val/test split sizes and how splits were created

## Model / Approach
- Architecture choices and why (not just what)
- Key hyperparameters from configs/train.yaml
- Loss function and class imbalance strategy
- Training infrastructure (2×A100, mixed precision, etc.)

## Results
- Primary metrics: mIoU, macro F1, per-class F1 (especially destroyed)
- Comparison to baseline or prior phase if applicable
- wandb run link or experiment ID for reproducibility
- At least one figure (confusion matrix, learning curves, or sample predictions)

## Limitations
Honest assessment of what doesn't work well and why. Be specific:
- Which damage class has lowest F1 and likely why
- Geographic or event-type generalization gaps
- Data quality issues encountered

## Ethics & Responsible Use
[Always include — see ethics section below]
```

---

## Ethical disclaimer — required in all external documents

This section is mandatory in any document intended for external audiences: reports,
portfolio entries, presentations, README files, or publications.

```markdown
## Ethics & Responsible Use

This system is designed for **macro-scale humanitarian damage assessment** — estimating
the extent and distribution of building damage across a disaster-affected region to
support relief coordination and resource allocation.

**This system must not be used for:**
- Individual targeting or surveillance of specific people or households
- Military targeting or strike planning of any kind
- Real-time tactical operations
- Any application where automated damage assessment is used as the sole basis for
  decisions affecting individuals without human review

**Limitations that users must understand:**
- Model predictions are probabilistic and carry uncertainty — always report confidence
  scores alongside damage classifications
- The model was trained primarily on xBD and BRIGHT datasets; performance may degrade
  on disaster types, geographic regions, or sensor configurations not represented in
  training data
- Satellite imagery resolution and cloud cover affect prediction quality
- Ground truth validation is required before using outputs in formal humanitarian reports

This project is conducted for research and educational purposes. Any operational
deployment must involve qualified humanitarian professionals and comply with applicable
international humanitarian law and data protection regulations.
```

Include this verbatim (or a condensed version for brevity) — don't paraphrase it in
ways that soften the restrictions.

---

## Portfolio section format

When documenting this project for a portfolio (README, personal site, application
materials), use this four-part structure. Keep it factual and specific — metrics over
adjectives.

```markdown
## Conflict Damage Monitor

**Problem:** Rapid post-disaster building damage assessment from satellite imagery is
critical for directing humanitarian aid, but manual inspection of large conflict zones
is too slow and dangerous. Existing automated tools lack coverage of conflict-specific
damage patterns.

**Approach:** Fine-tuned EfficientNet-B4 classifier and UNet++ segmentation model on
850k+ labeled buildings from the xBD dataset and 350k from BRIGHT, covering earthquakes,
hurricanes, floods, and conflict damage. Trained on UF HiperGator HPC (2×A100 GPUs)
with PyTorch Lightning. Built a FastAPI inference server and React + Leaflet.js dashboard
for interactive damage visualization.

**Results:**
- Macro F1: [X.XX] across 4 damage classes (no-damage / minor / major / destroyed)
- mIoU: [X.XX] on xBD test set
- Destroyed-class F1: [X.XX] (the hardest and most critical class)
- Inference latency: [X]ms per 512×512 tile on A100

**Screenshots:** [link to dashboard screenshot] | [link to sample predictions]

**Code:** github.com/Ruthik27/conflict-damage-monitor

> ⚠️ For macro-scale humanitarian analysis only. Not for targeting or individual surveillance.
```

Fill in bracketed metrics from actual wandb runs before publishing. Never publish
placeholder values.

---

## Writing style rules for docs/

- **Passive voice for methods, active for findings**: "Images were tiled at 512×512 pixels" / "The model achieves F1=0.71 on the destroyed class"
- **Metrics always include context**: Don't write "F1=0.71" — write "F1=0.71 on the destroyed class (xBD test set, n=8,420 buildings)"
- **Limitations before discussion**: State what doesn't work before explaining why — it signals intellectual honesty
- **No marketing language**: "state-of-the-art", "powerful", "robust" — replace with actual numbers
- **Date every document**: Add `*Last updated: YYYY-MM-DD*` at the top — undated docs rot silently

---

## docs/ file naming conventions

```
docs/
├── README.md                          # navigation index
├── ethics.md                          # standalone ethics statement
├── debug-log.md                       # running error/fix log
├── 01-data.md                         # phase 1 write-up
├── 02-baseline-model.md               # phase 2 write-up
├── 03-segmentation-model.md           # phase 3 write-up
├── 04-evaluation.md                   # phase 4 write-up
├── 05-inference-api.md                # phase 5 write-up
├── 06-dashboard.md                    # phase 6 write-up
└── results/
    ├── 2025-01-15-efficientnet-b4-baseline.md
    └── 2025-02-03-unetpp-xbd-full.md
```
