Round 12 Final Response Blend Check
===================================

Saved answers:

- `tab1_response_blender_decision.txt`: reject deterministic response blenders. Keep only `judge_clean_mix` as the primary and `lexplus_softened` as the response-only backup/challenger unless a manual Gemini-style audit shows a large naturalness advantage for a blend.

Local probes:

- `scripts/blend_response_predictions.py` was added to create response-only blends while refusing to run if ranking hashes differ.
- Dev lexical scores with the frozen weighted-RRF ranking:
  - primary `judge_clean_mix`: `0.19958`
  - `lexplus_softened`: `0.20531`
  - ratio 25% softened: `0.20089`
  - ratio 50% softened: `0.20247`
  - ratio 75% softened: `0.20374`
  - repeated-opening replacement: `0.19279`
  - hybrid: `0.20420`
- Because every blend is below `lexplus_softened`, no blend is promoted.

Final practical stop condition:

- Ranking is frozen.
- Response variants are frozen to primary plus `lexplus_softened` backup.
- Remaining work is only package validation, hash recording, and documentation.
