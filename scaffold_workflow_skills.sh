#!/bin/bash
WORKFLOW_SKILLS=(
  "ai-development-loop"
  "senior-data-engineer"
  "data-scientist"
  "hpc-slurm"
  "testing-quality"
  "geopandas-spatial"
  "geospatial-visualization"
  "docs-reporting"
)

for skill in "${WORKFLOW_SKILLS[@]}"; do
  mkdir -p .claude/skills/$skill
  cat > .claude/skills/$skill/SKILL.md << EOF
---
name: $skill
description: PLACEHOLDER - generate via skill-creator in first Claude session
version: 0.1.0
---
TODO: Run skill-creator to generate this properly.
EOF
  echo "✓ Scaffolded: $skill"
done
echo "✅ Done! Fill via skill-creator in first Claude session."
