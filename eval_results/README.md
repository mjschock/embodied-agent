# Evaluation Results

This directory is reserved for selected, reviewable benchmark records produced by live model runs.

Live physics runs can write a self-describing JSON record with:

```bash
embodied-agent-eval-openai-physics \
  --model "YOUR_MODEL_ID" \
  --xlerobot-runtime-root /path/to/XLeRobot \
  --humanoid-runtime-root /path/to/lerobot-humanoid-runtime \
  --output eval_results/three-robot-physics-comparison/YOUR_MODEL_ID/run.json
```

Each record contains:

- a result schema version;
- benchmark, provider, and model identity;
- UTC creation time and maximum agent-step budget;
- repository revision and detected upstream simulator/model revisions where available;
- the complete deterministic baseline result;
- the complete live AgentModel result;
- direct model-vs-deterministic score deltas.

The manual `openai-physics-benchmark` GitHub Actions workflow uploads the same JSON record as an artifact for every completed benchmark, including runs where the model does not achieve a perfect score. A non-perfect model score is data, not an infrastructure failure.

Commit only intentional comparison records here. Do not commit API keys, provider credentials, raw private prompts, or unrelated logs. The current benchmark record contains only the fixed repository task instructions, semantic actions/metrics, and reproducibility metadata.
