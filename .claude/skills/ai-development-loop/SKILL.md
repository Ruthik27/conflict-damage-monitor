---
name: ai-development-loop
description: >
  Disciplined AI development loop for the conflict-damage-monitor project. Use this skill
  whenever the user asks Claude to implement a feature, write new code, add a module, fix
  a bug, or start any non-trivial coding task. Trigger on phrases like "implement X",
  "add Y to the codebase", "write the training script", "build the data pipeline", "create
  the API endpoint", or any request that will touch more than one file. This skill enforces
  a plan-first, one-file-at-a-time, sanity-checked workflow that keeps sessions coherent
  and prevents runaway half-finished implementations. When in doubt, use this skill — it
  costs nothing to plan and prevents a lot of rework.
---

# AI Development Loop — conflict-damage-monitor

The core idea behind this workflow is that the biggest source of wasted effort in AI-assisted
coding is starting implementation before fully understanding the context. Reading first and
planning before coding makes every subsequent step faster, catches conflicts early, and
keeps the user informed and in control.

---

## Phase 1: Orient before you touch anything

Before writing a single line of code, read the project context files:

1. **Read `CLAUDE.md`** (both root and `.claude/CLAUDE.md`) — understand the stack, paths, constraints, and coding rules.
2. **Read `PROJECT_SPEC.md`** if it exists — understand the current phase, what's already built, and what's planned.
3. **Skim relevant existing source files** — find where the task fits and what it might interact with.

This takes two minutes and saves an hour of backtracking. Don't skip it even for "small" tasks.

---

## Phase 2: Write the plan, wait for approval

Present a numbered implementation checklist *before* writing any code. Each item should be
one concrete, completable action (create file X, add function Y, update config Z). The plan
should also call out:

- Files to be created or modified
- Any new dependencies to add to `environment.yml`
- Any config keys needed in `configs/train.yaml`
- Potential risks or open questions

**Wait for the user to say "looks good" or request changes before proceeding.** This is the
most important gate in the workflow — a plan review is far cheaper than undoing half-written code.

Example plan format:
```
Implementation plan: [task name]

1. Create `src/data/xbd_dataset.py` — PyTorch Dataset for xBD tiles
2. Add `XBDDataModule` to `src/data/datamodule.py` — wraps dataset with train/val splits
3. Update `configs/train.yaml` — add `dataset.name: xbd` and `dataset.tile_size: 512`
4. Add `rasterio>=1.3` to `environment.yml` if not already present
5. Smoke-test with: `python -c "from src.data.xbd_dataset import XBDDataset; print('ok')"`

Open questions:
- Should the dataset support both xBD and BRIGHT in one class, or separate classes?

Ready to proceed once you confirm.
```

---

## Phase 3: Implement one file at a time

Work through the plan sequentially, one checklist item at a time:

1. **Implement the file** — write clean, minimal code that does exactly what the plan says.
   - No hardcoded paths — use `configs/train.yaml` via argparse or OmegaConf.
   - All outputs (checkpoints, logs, results) go to `/blue/smin.fgcu/rkale.fgcu/cdm/`.
   - Log to wandb (project: `conflict-damage-monitor`).
2. **Run a sanity check** immediately after writing:
   - Syntax check: `python -m py_compile src/path/to/file.py`
   - Import test: `python -c "from src.module.file import ClassName; print('ok')"`
   - Quick unit test if one exists: `pytest tests/test_file.py -x -q`
3. **Report the result** — show the user the check output and confirm it passed before moving to the next item.

If a sanity check fails, fix it before moving on. Don't stack up multiple broken files.

---

## Phase 4: End-of-session summary

Before closing out a coding session, give the user a brief structured summary:

```
## Session summary

### What changed
- Created `src/data/xbd_dataset.py` — XBDDataset class, returns (pre_img, post_img, label) tensors
- Updated `configs/train.yaml` — added dataset section
- `environment.yml` unchanged (rasterio was already present)

### Sanity checks
- All imports pass ✓
- `pytest tests/test_xbd_dataset.py` — 3 passed ✓

### What's next
- Step 4 of plan: implement XBDDataModule
- Open question still unresolved: unified vs. separate dataset classes
```

This makes it easy for the user to pick up the session later, or hand it off to another agent.

---

## Quick rules

- **Never write to `/home`** for data/models/outputs — only code lives there.
- **Never hardcode `/blue/...` paths** in source files — use `configs/train.yaml`.
- **Never skip the plan approval gate** — even if the task seems obvious.
- **One file per step** — resist the urge to write three files at once.
- **Sanity check before moving on** — a broken import caught early is trivial; caught late it becomes a debugging session.
