# Phase2 Session Ledger

## Goal

Implement Candidate A from [Phase2_improvement.md](Phase2_improvement.md): stabilize the `vggt_omega_alpha` projector before the user runs server-side VSIBench/CVBench validation.

## Active Constraints

- Preserve Phase2 as an independent `vggt_omega_alpha` input-side wrapper.
- Do not introduce Candidate B/C attention adapters in this step.
- Do not change token counts, packing semantics, `image_grid_thw`, or M-RoPE expansion.
- Do not modify Phase1 `vggt` / `vggt_omega` fusion paths.
- Keep Omega encoder frozen.

## Architecture Rules

- Each frame still inserts exactly `17` alpha tokens before the corresponding Qwen merged visual span.
- Qwen preprocessing and `image_grid_thw` remain the source of truth.
- Expanded placeholder count, visual embed count, and position-id length must stay identical.
- The alpha projector remains the only Phase2 trainable alpha-side module in this Candidate A implementation.

## Decisions

- Added `LayerNorm(input_dim)` before the progressive projector MLP.
- Added learnable scalar `alpha_gate` after the projector output.
- Initialized `alpha_gate` to `1e-2`, not `0`, to reduce initial perturbation while preserving early gradient flow through the projector.
- Left optimizer parameter grouping unchanged; `alpha_projector` continues to follow existing trainable-parameter handling.

## Progress

- Updated `SpatialStack/src/qwen_vl/model/vggt_omega_alpha_projector.py`.
- Updated `Phase2_plan.md` to reflect the stabilized projector.
- Updated `Phase2_improvement.md` to mark Candidate A as implemented.
- Added this session ledger and [Phase2_knowledge_graph.md](Phase2_knowledge_graph.md).

## Open Problems

- Server-side benchmark validation is still pending.
- It is unknown whether Candidate A alone recovers the VSIBench/CVBench gap.
- If Candidate A is insufficient, Candidate B/C attention adapter ablations remain the next planned path.

## Next Actions

- Run training/eval on the server with the same Phase2 setup.
- Compare against the recorded Phase2 alpha scores: VSIBench `64.3450`, CVBench `82.3312`.
- If results remain below baseline, inspect Candidate B/C implementation scope before modifying architecture further.

