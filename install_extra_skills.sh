#!/bin/bash
# Run from: /home/rkale.fgcu/local_project/conflict-damage-monitor/

echo "Installing 5 additional skills..."
mkdir -p .claude/skills

# ─────────────────────────────────────────────
# 1. claude-mem (persistent memory across sessions)
# Source: github.com/thedotmack/claude-mem
# ─────────────────────────────────────────────
git clone --depth 1 https://github.com/thedotmack/claude-mem.git /tmp/claude-mem
# claude-mem is a plugin (hooks + skill) — copy the whole thing
cp -r /tmp/claude-mem .claude/skills/claude-mem
rm -rf /tmp/claude-mem
echo "✓ claude-mem installed"

# ─────────────────────────────────────────────
# 2. ralph-loop (autonomous iterative task loop)
# Source: github.com/openclaw/skills (jordyvandomselaar's version)
# ─────────────────────────────────────────────
git clone --depth 1 https://github.com/openclaw/skills.git /tmp/openclaw-skills
mkdir -p .claude/skills/ralph-loop
cp /tmp/openclaw-skills/skills/jordyvandomselaar/ralph-loop/SKILL.md .claude/skills/ralph-loop/
rm -rf /tmp/openclaw-skills
echo "✓ ralph-loop installed"

# ─────────────────────────────────────────────
# 3. self-healing (error detection + auto-fix loop)
# Source: generate via skill-creator (no single canonical repo)
# Scaffold placeholder — fill in first Claude session
# ─────────────────────────────────────────────
mkdir -p .claude/skills/self-healing
cat > .claude/skills/self-healing/SKILL.md << 'EOF'
---
name: self-healing
description: Use when a script, training job, or pipeline errors — diagnose root cause, propose fix, verify with user, apply and re-run
version: 0.1.0
---

# Self-Healing Skill

TODO: In first Claude session run:
"Use skill-creator to build a self-healing skill that:
- Detects errors in scripts, SLURM jobs, and pipeline outputs
- Diagnoses root cause from logs/tracebacks
- Proposes a specific fix with explanation
- Asks for confirmation before applying
- Re-runs verification after fix
- Logs the fix to docs/debug-log.md for future reference"
EOF
echo "✓ self-healing scaffolded (fill via skill-creator)"

# ─────────────────────────────────────────────
# 4. cost-reducer (token optimization)
# Source: alirezarezvani/claude-skills (engineering-team folder)
# ─────────────────────────────────────────────
git clone --depth 1 https://github.com/alirezarezvani/claude-skills.git /tmp/alireza-skills

# Check for cost/token related skill
if [ -d "/tmp/alireza-skills/engineering-team" ]; then
  # Look for cost or token optimizer skill
  COST_SKILL=$(find /tmp/alireza-skills -type d -name "*cost*" -o -type d -name "*token*" | head -1)
  if [ -n "$COST_SKILL" ]; then
    cp -r "$COST_SKILL" .claude/skills/cost-reducer
    echo "✓ cost-reducer installed from alirezarezvani/claude-skills"
  else
    # Scaffold if not found
    mkdir -p .claude/skills/cost-reducer
    cat > .claude/skills/cost-reducer/SKILL.md << 'EOF'
---
name: cost-reducer
description: Use to reduce token usage — compress context, use concise outputs, cache repeated prompts, prefer Haiku for simple tasks
version: 0.1.0
---

# Cost Reducer Skill

TODO: In first Claude session run:
"Use skill-creator to build a cost-reducer skill that:
- Uses Haiku model for simple/repetitive tasks, Sonnet for complex ones
- Compresses context when window exceeds 50% full
- Avoids repeating large code blocks — reference file paths instead
- Summarizes completed work before clearing context
- Warns when a task is likely to exceed 20% of weekly token budget"
EOF
    echo "✓ cost-reducer scaffolded (fill via skill-creator)"
  fi
else
  mkdir -p .claude/skills/cost-reducer
  cat > .claude/skills/cost-reducer/SKILL.md << 'EOF'
---
name: cost-reducer
description: Use to reduce token usage and extend Pro weekly limits
version: 0.1.0
---
TODO: Generate via skill-creator in first Claude session.
EOF
  echo "✓ cost-reducer scaffolded"
fi
rm -rf /tmp/alireza-skills

# ─────────────────────────────────────────────
# 5. context7 (live up-to-date library documentation)
# Source: github.com/upstash/context7 (official CLI install)
# ─────────────────────────────────────────────
# Option A: Official CLI install (recommended if npx available)
if command -v npx &> /dev/null; then
  npx context7 install --claude
  echo "✓ context7 installed via official CLI"
else
  # Option B: Manual SKILL.md from netresearch
  git clone --depth 1 https://github.com/netresearch/context7-skill.git /tmp/context7-skill
  if [ -f "/tmp/context7-skill/SKILL.md" ]; then
    mkdir -p .claude/skills/context7
    cp /tmp/context7-skill/SKILL.md .claude/skills/context7/
    echo "✓ context7 installed from netresearch/context7-skill"
  else
    mkdir -p .claude/skills/context7
    cat > .claude/skills/context7/SKILL.md << 'EOF'
---
name: context7
description: Use when working with any library (rasterio, GeoPandas, PyTorch Lightning, Sentinel API) to fetch current up-to-date documentation
version: 0.1.0
---

# Context7 Integration

Fetch live library documentation before writing any API calls.
Use context7 MCP tool: resolve-library-id then get-library-docs.
Priority libraries: rasterio, geopandas, pytorch-lightning, sentinelsat, shapely, xarray.
EOF
    echo "✓ context7 scaffolded"
  fi
  rm -rf /tmp/context7-skill
fi

echo ""
echo "✅ All 5 skills done. Summary:"
ls .claude/skills/ | grep -E "claude-mem|ralph-loop|self-healing|cost-reducer|context7"
