#!/bin/bash
mkdir -p .claude/skills

git clone --depth 1 https://github.com/K-Dense-AI/claude-scientific-skills.git /tmp/claude-scientific-skills

SKILLS=(
  "geopandas"
  "pytorch-lightning"
  "exploratory-data-analysis"
  "scientific-visualization"
  "dask"
  "plotly"
  "statistical-analysis"
  "scientific-writing"
  "transformers"
  "timesfm-forecasting"
)

for skill in "${SKILLS[@]}"; do
  cp -r /tmp/claude-scientific-skills/scientific-skills/$skill .claude/skills/
  echo "✓ Copied: $skill"
done

git clone --depth 1 https://github.com/anthropics/skills.git /tmp/anthropic-skills
cp -r /tmp/anthropic-skills/skills/skill-creator .claude/skills/
cp -r /tmp/anthropic-skills/skills/webapp-testing .claude/skills/

rm -rf /tmp/claude-scientific-skills /tmp/anthropic-skills
echo "✅ Done! Layer 2 installed."
ls .claude/skills/
