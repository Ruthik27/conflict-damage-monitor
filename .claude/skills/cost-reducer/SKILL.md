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
