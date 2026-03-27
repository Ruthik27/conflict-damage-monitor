---
name: self-healing
description: >
  Error diagnosis and fix workflow for the conflict-damage-monitor project. Use this
  skill whenever a script crashes, a SLURM job fails, a training run dies, or a pipeline
  produces unexpected output. Trigger on: pasting a Python traceback, sharing a SLURM
  .err file, "my job failed", "this script is crashing", "I'm getting an error",
  "the training stopped", "CUDA error", "OOM", "ImportError", "FileNotFoundError",
  "job timed out", or any error output from the pipeline. The workflow is always:
  read the error → diagnose in 2-3 sentences → propose one fix → wait for confirmation
  → apply → verify. Never apply fixes without user confirmation and never skip the
  verification re-run. Log every fix to docs/debug-log.md.
---

# Self-Healing — conflict-damage-monitor

When something breaks in an overnight HPC job, the worst outcome is spending an hour
debugging only to realize the fix was wrong and you need to requeue. The discipline here
is: diagnose carefully, propose precisely, confirm before touching anything, then verify
the fix actually worked before declaring success.

---

## Step 1: Read the full error

Before diagnosing, collect:
- The full traceback or SLURM `.err` content (not just the last line)
- The command or script that was running
- Relevant config values from `configs/train.yaml`

SLURM logs are at `/blue/smin.fgcu/rkale.fgcu/cdm/logs/<job-id>.err`. If the user
pastes a truncated error, ask for the full log:
```
tail -200 /blue/smin.fgcu/rkale.fgcu/cdm/logs/<job-id>.err
```

---

## Step 2: Diagnose in 2–3 sentences

State:
1. What went wrong (the proximate cause)
2. Why it went wrong (the root cause)
3. What the fix targets

Keep it short — the user needs to understand and confirm the diagnosis, not read an essay.

**Format:**
```
Diagnosis: The job ran out of GPU memory during the forward pass of epoch 3.
Root cause: batch_size=32 with EfficientNet-B4 + 512×512 tiles exceeds 2×A100 VRAM
(~40GB needed, ~80GB available — but DDP splits across 2 GPUs, so ~20GB per card, which
is below the 40GB A100 limit — the actual issue is gradient accumulation is off).
Fix target: Disable gradient checkpointing override in train.yaml that was accidentally
set to False, re-enabling it restores VRAM headroom.
```

---

## Step 3: Propose one specific fix

One fix only — not a list of options. The most likely root cause, stated precisely.
Include the exact change (diff, config line, command).

**Format:**
```
Proposed fix: In configs/train.yaml, change:
  training.gradient_checkpointing: false
to:
  training.gradient_checkpointing: true

This re-enables gradient checkpointing, reducing peak VRAM from ~40GB to ~22GB per card.

Shall I apply this? (yes / no / suggest alternative)
```

Wait for explicit confirmation before making any change.

---

## Step 4: Apply the fix

After confirmation, make the minimal change. Do not refactor surrounding code, rename
variables, or "clean up while I'm in there." Fix exactly what was diagnosed.

```python
# Good: targeted single-line fix
# Bad: rewriting the whole training config while fixing one field
```

---

## Step 5: Verify

Run the appropriate verification immediately after applying the fix:

| Fix type | Verification |
|---|---|
| Python/import error | `python -m py_compile src/path/to/file.py && python -c "from src... import ...; print('ok')"` |
| Config/path error | `python src/script.py --config configs/train.yaml --dry-run` (if supported) |
| CUDA/memory error | `pytest tests/models/test_classifier.py::test_training_smoke -x` (CPU smoke test) |
| SLURM job failure | Re-submit with shortened `--time` for a 2-epoch validation run |
| Data/shape error | `pytest tests/data/ -x -q` |

Report the verification output to the user before closing.

---

## Step 6: Log to docs/debug-log.md

Every diagnosed+fixed error gets a log entry. This builds a searchable record that
prevents the same issue from taking an hour to debug the second time.

Append to `docs/debug-log.md` (create it if absent):

```markdown
## YYYY-MM-DD — <one-line error description>

**Error:** <error type and message, 1-2 lines>
**Context:** <script/job that failed, relevant config>
**Root cause:** <2-3 sentence diagnosis>
**Fix applied:** <exact change made>
**Verification:** <command run + result>
**Status:** Resolved ✓
```

---

## Common error patterns

| Error | Likely cause | Quick fix |
|---|---|---|
| `CUDA out of memory` | Batch size too large or checkpointing off | Reduce `batch_size` or enable `gradient_checkpointing` |
| `DUE TO TIME LIMIT` in .err | Walltime exceeded | Add `--time` buffer or checkpoint more frequently |
| `conda: command not found` | Missing `module load conda` in .sbatch | Add `module purge && module load conda` before activate |
| `ModuleNotFoundError` | Wrong conda env active | Verify `conda activate cdm` ran; check `which python` |
| `FileNotFoundError` on `/blue/...` | Path doesn't exist yet | `mkdir -p` the directory or check `DATA_ROOT` env var |
| `AssertionError: No CRS` | GeoDataFrame loaded without CRS | `gdf.set_crs("EPSG:4326")` before any spatial op |
| `KeyError: 'subtype'` | xBD JSON field name changed | Check actual field names with `gdf.columns.tolist()` |
| `wandb: ERROR` | Not logged in or no network on compute node | Set `WANDB_MODE=offline` in .sbatch, sync after |
| NaN loss at epoch 1 | LR too high or bad weight init | Lower `lr` by 10×, check `torch.isfinite(logits)` |
