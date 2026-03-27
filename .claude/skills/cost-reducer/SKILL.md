---
name: cost-reducer
description: >
  Token and cost efficiency for the conflict-damage-monitor project. Use this skill
  whenever the context window is getting long, a task feels repetitive, the user asks
  about cost, or you're about to paste a large file into the conversation. Trigger on:
  "we're running low on context", "this is getting expensive", "compress the context",
  "summarize what we've done", "which model should I use", or any time you're about to
  repeat a large code block that already exists in a file. Also self-trigger when the
  conversation has exceeded roughly 60% of the context window — proactively suggest
  compression before it becomes a problem. The goal is to do the same quality of work
  with fewer tokens, not to cut corners.
---

# Cost Reducer — conflict-damage-monitor

Token costs compound quickly on long ML sessions: reading large files repeatedly,
restating context, using a powerful model for trivial tasks. The strategies below
keep sessions efficient without sacrificing quality.

---

## Model selection: match model to task complexity

| Task type | Model | Examples |
|---|---|---|
| **Simple / mechanical** | Haiku | Renaming variables, adding docstrings, formatting YAML, writing a one-liner, adding argparse flags, updating requirements |
| **Moderate** | Sonnet (default) | Writing a new function, debugging a specific error, adding a test, explaining code |
| **Complex / architectural** | Sonnet with full context | Designing a new module, multi-file refactors, diagnosing subtle training bugs, planning a new pipeline stage |

When a task is clearly in the "simple" bucket, explicitly note: *"This is a Haiku-level
task — using Haiku to save budget."* Don't use Sonnet for tasks that are essentially
search-and-replace or boilerplate generation.

---

## Reference files, don't repeat them

When discussing code that already exists in a file, reference the path rather than
pasting the full content into the conversation. Large pastes consume tokens on every
subsequent message in the context window.

```
# Instead of pasting 200 lines of DamageClassifier:
"See src/models/classifier.py:45 — the _shared_step method"

# Instead of showing the full train.yaml:
"The relevant section is configs/train.yaml under training.lr"
```

When you do need to show code, show only the relevant function or block — not the
entire file. Use line number references (`file.py:23-45`) to point the user to context
without reproducing it.

---

## Context compression: when the window is ~60% full

When the conversation is getting long, compress before it becomes a problem. Signs to
watch for:
- Responses starting to feel repetitive or re-explaining things
- The session has covered 2+ distinct phases (e.g., wrote dataset → wrote model → now debugging)
- You're about to start a new major task

**Compression steps:**

1. Write a phase summary (see format below)
2. Tell the user: *"The context window is getting full. I've summarized what we've done above. Want to start a fresh session with that summary as context?"*
3. Don't discard anything the user might still need — ask first

---

## Phase summary format

Before compressing or ending a long session, produce a summary that can be pasted into
the next session's opening message:

```
## Session summary — [date] [phase name]

### Completed
- Created `src/data/xbd_dataset.py` — XBDDataset, returns (pre, post, label) tensors
- Created `tests/data/test_xbd_dataset.py` — shape, label range, split overlap tests
- Updated `configs/train.yaml` — added data.tile_size: 512, data.num_classes: 4
- All tests passing: `pytest tests/data/ -x -q` → 6 passed

### Current state
- Next step: implement XBDDataModule in `src/data/datamodule.py`
- Open question: unified vs. separate dataset class for xBD + BRIGHT

### Key decisions made
- Using WeightedRandomSampler (not focal loss) for class imbalance — revisit if destroyed F1 < 0.4
- Checkpoints every 5 epochs, monitored on val/f1_macro

### Files modified
- src/data/xbd_dataset.py (new)
- tests/data/test_xbd_dataset.py (new)
- configs/train.yaml (modified: data section)
```

---

## Budget awareness: flag expensive operations

Warn the user before starting tasks that are likely to be token-heavy:

- Reading + summarizing multiple large files in one session
- Running evals or test suites that will produce long output
- Writing a whole new module from scratch (suggest breaking into 2 sessions)
- Debugging a complex multi-file issue where many files need to be read

Format: *"This task will likely use significant context — it involves reading 4 files
totalling ~800 lines. Want me to proceed, or would you prefer to scope it down?"*

The 15% weekly Pro budget threshold: if a single task appears likely to consume more
than ~15% of a typical weekly Pro plan (rough guide: sessions longer than 2 hours of
complex coding, or reading+writing 10+ large files), flag it upfront so the user can
decide whether to split the work across sessions.

---

## Quick wins checklist

Before starting any new task in an existing session:

- [ ] Is this a Haiku-level task? Say so and use Haiku.
- [ ] Does the code I need to reference already exist in a file? Use path + line numbers.
- [ ] Is the context window >60% full? Offer a phase summary first.
- [ ] Will this task involve reading many large files? Warn the user upfront.
- [ ] Am I about to re-explain something that was already established this session? Skip it.
