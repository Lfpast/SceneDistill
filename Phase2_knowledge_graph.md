# Phase2 Knowledge Graph

## Scope

- Workspace: `/home/jackson/python/SpatialStack-omega`
- Branch focus: Qwen3.5 `geometry_encoder_type=vggt_omega_alpha`
- Current method family: Phase2 alpha input-side wrapper

## Nodes

| Node | Type | Current state |
|---|---|---|
| `Phase2 / vggt_omega_alpha` | method | Independent input-side wrapper; no Phase1 deepstack / feature_fusion / geometry_merger path |
| `VGGT-Omega alpha encoder` | module | Frozen encoder exposing per-frame `1 camera + 16 register(scene)` tokens |
| `VGGTOmegaAlphaProjector` | module | Trainable stabilized progressive projector: `LayerNorm(2048) -> Linear -> GELU -> Linear -> alpha_gate` |
| `alpha_gate` | parameter | Learnable scalar initialized to `1e-2` to reduce initial LLM input perturbation while preserving gradient flow |
| `image_grid_thw` | invariant | Source of truth for geometry input sizing and per-frame Qwen visual token counts |
| `frame_center` M-RoPE | invariant | Current inserted-token position strategy |
| `Phase2 Candidate A` | experiment | Projector stabilization implemented; awaiting server benchmark validation |
| VSIBench/CVBench | datasets | User-observed Phase2 alpha underperforms SpatialStack paper baseline before Candidate A |

## Relationships

- `Phase2 / vggt_omega_alpha` -> `VGGT-Omega alpha encoder`: consumes frozen Omega special tokens instead of patch-token features.
- `VGGT-Omega alpha encoder` -> `VGGTOmegaAlphaProjector`: outputs `(T, 17, 2048)` tokens projected to Qwen text hidden size.
- `VGGTOmegaAlphaProjector` -> `alpha_gate`: scales projected alpha embeddings before they are prepended to Qwen visual spans.
- `image_grid_thw` -> `Phase2 / vggt_omega_alpha`: determines Omega-side resize and Qwen visual span lengths; no fixed `196` or `224x224`.
- `Phase2 Candidate A` -> VSIBench/CVBench: next validation target is whether projector stabilization recovers any of the observed gap.

## Decisions

- Candidate A is implemented with a small nonzero gate (`1e-2`) rather than zero gate. A zero multiplicative gate would minimize initial perturbation but can starve projector weights of early gradient.
- Candidate A does not add self-attention, cross-attention, new batch metadata, or LLM-internal fusion.
- Phase1 `vggt` / `vggt_omega` paths are intentionally untouched.

## Open Questions

- Does the stabilized projector improve VSIBench/CVBench relative to the current Phase2 alpha scores?
- If the score remains below baseline, the next planned experiment is Candidate B/C attention adapter ablation from [Phase2_improvement.md](Phase2_improvement.md).

