# Phase2 Improvement Proposal

## 0. Scope

本文档只讨论 `geometry_encoder_type=vggt_omega_alpha` 的 Phase2 改进方案，不直接修改代码。

Source of truth:

- 用户实验观察：Phase2 已完成训练和评测，但 VSIBench/CVBench 均弱于 SpatialStack 原论文 baseline。
- [Phase2_plan.md](Phase2_plan.md)：Phase2 是独立外挂分支，不走 Phase1 的 `deepstack / feature_fusion / geometry_merger` 路径。
- 当前代码：
  - [modeling_qwen3_5_vggt_omega_alpha.py](/home/jackson/python/SpatialStack-omega/SpatialStack/src/qwen_vl/model/modeling_qwen3_5_vggt_omega_alpha.py:209)：在线抽取 Omega special tokens，经过 projector 后插入 Qwen visual span。
  - [vggt_omega_alpha_projector.py](/home/jackson/python/SpatialStack-omega/SpatialStack/src/qwen_vl/model/vggt_omega_alpha_projector.py:25)：当前 projector 是 `Linear + GELU + Linear`。
  - [vggt_omega_alpha_packing.py](/home/jackson/python/SpatialStack-omega/SpatialStack/src/qwen_vl/model/vggt_omega_alpha_packing.py:150)：当前插入策略是每帧 `alpha_embeds + image_embeds[frame]`。
  - [vggt_omega_alpha_encoder.py](/home/jackson/python/SpatialStack-omega/SpatialStack/src/qwen_vl/model/geometry_encoders/vggt_omega_alpha_encoder.py:79)：Omega alpha encoder 返回每帧前 `17` 个 special tokens。
- 可参考实现：
  - [SpatialStack_temp/src/qwen_vl/model/cam_distill.py](/home/jackson/python/SpatialStack-omega/SpatialStack_temp/src/qwen_vl/model/cam_distill.py:41)：`FrameCrossAttentionLayer`。
  - [SpatialStack_temp/src/qwen_vl/model/cam_distill.py](/home/jackson/python/SpatialStack-omega/SpatialStack_temp/src/qwen_vl/model/cam_distill.py:113)：`GlobalCamSceneSelfAttentionLayer`。
  - [SpatialStack_temp/src/qwen_vl/model/feature_fusion.py](/home/jackson/python/SpatialStack-omega/SpatialStack_temp/src/qwen_vl/model/feature_fusion.py:32)：Phase1 风格 `CrossAttentionBlock`。

## 1. Problem Definition

实验观察：

| Model | VSIBench | CVBench |
|---|---:|---:|
| SpatialStack paper baseline | 67.25698 | 85.4491 |
| Current Phase2 alpha | 64.3450 | 82.3312 |
| Delta | -2.91198 | -3.1179 |
| Relative delta | -4.33% | -3.65% |

结论必须保守表述：

- 事实：当前 Phase2 alpha 在两个 benchmark 上均弱于论文 baseline。
- 假设：Omega camera/register tokens 含有可用几何信息。
- 假说：当前 Phase2 的主要瓶颈不是 token 来源，而是 `project -> prepend -> LLM self-attention` 的交互方式太弱，无法替代论文 baseline 的 layered fusion / injection。
- 需要验证：引入 self-attention 和 cross-attention 是否能恢复或超过论文 baseline。

## 2. Current Phase2 Failure Hypothesis

当前 Phase2 的主路径可以抽象为：

```text
Qwen image encoder -> image_embeds
VGGT-Omega frozen encoder -> (T, 17, 2048) special tokens
special tokens -> MLP projector -> (T, 17, text_hidden)
per frame: [17 alpha tokens, Qwen patch tokens]
LLM consumes expanded sequence
```

这个路径的优点是接口干净，能保持 Qwen 原始 visual span、`image_grid_thw`、M-RoPE 扩张逻辑稳定。

但它有三个可能不足：

1. **没有专门的几何 token 适配层**  
   `17` 个 Omega special tokens 是 VGGT-Omega 内部几何空间的 token。直接线性投影到 Qwen text hidden 后，它们不一定已经是 LLM 可读的语义 token。

2. **没有和本帧 Qwen patch tokens 做显式对齐**  
   当前 prepend 后，alpha tokens 与 patch tokens 的第一次交互发生在 LLM decoder 的 causal self-attention 中。对于 visual span 内的 token，LLM 是否能稳定学到“这些 17 个 token 是本帧几何摘要”并不保证。

3. **没有跨帧 special-token 建模**  
   VSIBench 和 CVBench 都会受多视角、空间关系和全局场景一致性影响。当前每帧 special tokens 在进入 LLM 前没有跨帧 self-attention。

这和 SpatialStack 论文主张存在结构差异：SpatialStack 论文强调 layered geometry-language fusion，认为只做深层或晚期融合会丢失 hierarchical geometry signals。Phase2 当前做法比 late fusion 更极端，因为它只是把 special tokens 插到输入侧，没有多层同步融合。

## 3. Literature Review

### 3.1 SpatialStack

SpatialStack 论文指出，VLM 的 3D spatial reasoning 瓶颈来自缺少 fine-grained 3D geometry 和 spatial relationships；它提出 layered geometry-language fusion，逐层对齐 vision、geometry、language 表示，而不是只做 conventional late-stage fusion。

对 Phase2 的启发：

- 当前 alpha 方案没有 layered fusion。
- 如果当前结果弱于 baseline，不能简单归因于 Omega tokens 无效；更可能是融合方式太弱。
- 改进应优先补足 interaction，而不是先更换 token 来源或改变输入尺寸。

Reference: https://arxiv.org/abs/2603.27437

### 3.2 VGGT

VGGT 直接从单张、多张或大量 views 预测 camera parameters、point maps、depth maps 和 tracks，并且作者报告 pretrained VGGT 作为 feature backbone 能增强下游任务。

对 Phase2 的启发：

- Omega special tokens 很可能承载全局几何/相机/scene-level 信息。
- 但 VGGT/VGGT-Omega 的 token 是为几何任务训练出的内部表示，不等价于 Qwen LLM 已对齐的视觉语言 token。
- 需要一个 adapter/refiner 把 geometry-special-token 表示转成 Qwen 可消费的表示。

Reference: https://arxiv.org/abs/2503.11651

### 3.3 BLIP-2

BLIP-2 用轻量 Q-Former 连接 frozen image encoder 和 frozen LLM，并通过两阶段训练弥合 modality gap。

对 Phase2 的启发：

- 当两个 backbone 都很强但表示空间不同，单层 projector 往往不是唯一选择。
- 用 query/cross-attention bridge 是成熟路线。
- Phase2 可以把 Omega special tokens 当作 geometry latent，把 Qwen patch tokens 当作 visual context，通过小型 transformer adapter 做对齐。

Reference: https://arxiv.org/abs/2301.12597

### 3.4 Flamingo

Flamingo 引入 Perceiver-style visual resampler 和 gated cross-attention blocks 来连接 frozen vision-only 和 language-only models，支持 interleaved images/text 和 video。

对 Phase2 的启发：

- 对 LLM 注入外部视觉/几何信息时，cross-attention 和 gating 是稳定训练的重要设计。
- Phase2 当前没有 gate，alpha tokens 初始就直接进入 LLM 序列，可能造成语言侧分布漂移。
- 可以先加 zero-init gate 或 LayerScale，控制新增几何分支初始扰动。

Reference: https://arxiv.org/abs/2204.14198

### 3.5 Perceiver-VL

Perceiver-VL 通过 iterative latent cross-attention 处理高维视频/图文输入，复杂度相对 self-attention 更可控。

对 Phase2 的启发：

- `17` 个 alpha tokens 本身就是小 latent set，适合做 Q/K/V adapter。
- 让小 token 集合 cross-attend 大 patch 集合，比让所有 patch tokens 互相 full attention 更省。
- 这支持一个轻量方案：只更新 alpha tokens，不改 Qwen patch tokens。

Reference: https://arxiv.org/abs/2211.11701

### 3.6 Vision Transformers Need Registers

Register tokens 被用于承载全局计算/信息，能改善 ViT feature maps 和 attention maps。

对 Phase2 的启发：

- Omega 的 register/scene tokens 不应被当作普通 patch token。
- 这些 tokens 更像 compact global state，需要专门位置、mask、adapter 和稳定化训练。
- 直接 prepend 可能让 LLM 把它们视为普通 visual placeholders，无法充分解码 register 语义。

Reference: https://arxiv.org/abs/2309.16588

### 3.7 Qwen2-VL / Qwen3.5-VL Design Constraint

Qwen-VL 系列强调 dynamic resolution 和 M-RoPE，以不同数量的 visual tokens 保持空间/时间信息。

对 Phase2 的约束：

- 继续以 `image_grid_thw` 为 source of truth。
- 不恢复固定 `196` 或固定 `224x224`。
- 不对 expanded visual sequence 重新跑原始 rope scan。
- 所有新 attention/refiner 只能消费已对齐的 frame patch spans，不能破坏 placeholder/image_embed/position_id 三者一致性。

Reference: https://arxiv.org/abs/2409.12191

## 4. Knowledge Graph Snapshot

| Node | Type | Current relationship |
|---|---|---|
| SpatialStack paper baseline | method | 强调 layered geometry-language fusion，是当前对照目标 |
| Phase2 `vggt_omega_alpha` | method | 输入侧 wrapper，每帧插入 `1 camera + 16 register(scene)` |
| VGGT-Omega special tokens | representation | 几何/相机/scene latent，当前只经过 MLP projector |
| Qwen patch tokens | representation | 由 Qwen visual encoder 产生，按 `image_grid_thw` 动态长度展开 |
| `image_grid_thw` | invariant | Phase2 sizing 和 per-frame token count 的 source of truth |
| M-RoPE `frame_center` | invariant | 当前 alpha 插入 token 的 position strategy |
| CamDistill temp module | prior implementation | 有 frame cross-attn + global self-attn，但原始目标是蒸馏学生 tokens |
| Current failure | experiment | VSIBench/CVBench 全面低于论文 baseline |
| AlphaTokenInteractionAdapter | proposed method | 在 LLM 前显式建模 alpha-alpha 和 alpha-patch 交互 |

## 5. Candidate Improvements

### Candidate A: Projector Stabilization

Implementation status: implemented in [vggt_omega_alpha_projector.py](/home/jackson/python/SpatialStack-omega/SpatialStack/src/qwen_vl/model/vggt_omega_alpha_projector.py:25) as `LayerNorm(input_dim) + Linear + GELU + Linear + alpha_gate`, with `alpha_gate` initialized to `1e-2`.

目标：最小改动，先确认当前差距是否来自新增 token 对 LLM 的初始扰动。

结构：

```text
alpha_raw: (T, 17, 2048)
alpha_norm = LayerNorm(alpha_raw)
delta = MLP(alpha_norm)
alpha_embeds = gate * delta
gate initialized to 0 or very small scalar
```

建议细节：

- 在当前 `VGGTOmegaAlphaProjector` 前加 `LayerNorm(2048)`。
- 输出乘一个 `alpha_gate`，当前实现初始化为 `1e-2`，避免 gate=0 时 projector 权重初期梯度过弱。
- 保持 `2048 -> midpoint_hidden -> text_hidden` 的 progressive rule。

优点：

- 代码风险低。
- 不改变 token 数、不改变 packing、不改变 M-RoPE。
- 能测试“当前 Phase2 是否因为随机 projector 输出扰动 LLM”。

缺点：

- 仍然没有 alpha-patch interaction。
- 如果核心问题是缺少 fusion，它最多改善稳定性，不太可能完全追上 SpatialStack baseline。

实验优先级：P0。

### Candidate B: Alpha Self-Attention Refiner

目标：只让 `17` 个 special tokens 内部、跨帧交互，不触碰 patch tokens。

结构：

```text
alpha_raw: (T, 17, 2048)
alpha = LayerNorm + Linear(2048 -> text_hidden)
for block in depth:
    alpha = per-frame self-attn over 17 tokens
    alpha = per-sample global self-attn over T * 17 tokens
    alpha = FFN
alpha = gated residual output
```

和 temp 的关系：

- 参考 `GlobalCamSceneSelfAttentionLayer` 的跨帧隔离思想。
- 不照搬 `CamSceneTokenModule` 的 learned query 初始化，因为 Phase2 已经有真实 Omega special tokens，不需要再从 Qwen ViT 学生 tokens 起步。
- 不引入 distill cache 和 `CamSceneDistillLoss`。

优点：

- 增强 camera/scene tokens 内部建模。
- 计算量小：每帧 17 tokens，跨帧也只是 `T * 17`。
- 不改 Qwen patch embeddings。

缺点：

- 仍然没有显式读取 Qwen patch tokens。
- 如果 alpha tokens 与 Qwen visual span 的局部对应关系是瓶颈，收益有限。

实验优先级：P1。

### Candidate C: Alpha-Patch Cross-Attention Adapter

目标：让 Omega special tokens 在进入 LLM 前读取本帧 Qwen patch tokens，从而把几何 global tokens 和 Qwen visual semantics 对齐。

这是本文档推荐的主方案。

结构：

```text
Qwen image_embeds split by image_grid_thw:
    patches_f: (N_f, text_hidden)
Omega alpha raw:
    alpha_f: (17, 2048)

alpha_f = AlphaProjector(alpha_f)  # (17, text_hidden)

for layer in adapter_depth:
    alpha_f = CrossAttn(Q=alpha_f, K=patches_f, V=patches_f)
    alpha_all_sample = GlobalSelfAttn(alpha_all_sample)
    alpha_f = FFN(alpha_f)

expanded image embeds per frame:
    [alpha_f_refined, patches_f]
```

关键约束：

- 只更新 alpha tokens，不更新 Qwen patch tokens。
- 输入和输出 token 数不变，仍然每帧插入 `17` 个 tokens。
- `patches_per_frame` 继续由 `image_grid_thw` 推导。
- `position_ids` 仍由 unexpanded Qwen sequence 先算，再 expand。
- 不调用 Phase1 `feature_fusion` / `geometry_merger`。

为什么它比当前 concat/prepend 更合理：

- 当前 concat 只把信息放进序列，交互交给 LLM 自注意力自己学。
- Cross-attention adapter 显式指定：camera/scene tokens 应该从本帧 Qwen patch tokens 读取视觉语义。
- Global self-attention 显式指定：同一样本不同帧的 camera/scene tokens 应该互相建模。
- 这仍保留 Phase2 的核心设定，即“input-side special-token injection”，只是把 token 插入前的表示做可训练适配。

建议初始化：

- `LayerNorm` before attention。
- attention/FFN residual 使用 LayerScale，初始化 `1e-2`。
- adapter final output 使用 gate，初始化小值。
- 若担心 gate=0 导致 alpha tokens 初始全零，可使用 `alpha_out = base_projected + gate * refined_delta`，其中 `gate=0`，初始等价于当前 projector baseline。

推荐默认配置：

| Hyperparameter | Initial value |
|---|---:|
| adapter_hidden_dim | `text_hidden` |
| adapter_depth | 2 |
| num_heads | 8 or `text_hidden // 128` |
| ffn_ratio | 4 |
| dropout | 0.0 or 0.1 |
| layerscale_init | 1e-2 |
| output_gate_init | 0 |

实验优先级：P1，主推荐。

### Candidate D: Patch-Update Cross-Attention

目标：不仅让 alpha tokens 读取 patch tokens，也让 Qwen patch tokens 读取 alpha tokens。

结构：

```text
patches_f = CrossAttn(Q=patches_f, K=alpha_f, V=alpha_f)
expanded frame = [alpha_f, patches_f_refined]
```

优点：

- 更接近 Phase1 fusion/injection 的效果，因为 patch tokens 也被几何增强。
- 如果 baseline 优势来自“几何影响视觉 token 表示”，这个方案可能更强。

缺点：

- 行为更接近 Phase1 feature fusion，Phase2 独立性变弱。
- 改动 patch tokens 可能破坏 Qwen visual encoder 的原始语义分布。
- 需要严格作为单独 ablation，不能和 Candidate C 混在一起同时改。

实验优先级：P2。

### Candidate E: LLM Internal Gated Cross-Attention

目标：在 LLM decoder 若干层内部，让 text/visual hidden states cross-attend alpha memory。

结构接近：

```text
hidden_states = decoder_layer(hidden_states)
hidden_states = hidden_states + gate_l * CrossAttn(Q=hidden_states, K=alpha_memory, V=alpha_memory)
```

参考：

- Flamingo gated cross-attention。
- SpatialStack_temp 中 `deepstack_language_cross_attn` 的思想。
- SpatialStack paper 的 layered geometry-language fusion。

优点：

- 最接近论文 baseline 的 layered injection。
- 可能恢复 SpatialStack baseline 的主要优势。

缺点：

- 架构侵入大。
- 会从“input-side wrapper”转向“LLM internal fusion”，与 Phase2 原始边界冲突。
- 需要用户明确批准后才能实现。

实验优先级：P3，上界实验。

### Candidate F: Hybrid Alpha + Phase1 Deepstack

目标：同时保留 Phase2 alpha tokens 和 Phase1 `deepstack_language_add/cross_attn`。

用途：

- 不是推荐最终架构。
- 作为诊断上界：如果 hybrid 明显恢复 baseline，而 alpha-only 不行，说明 alpha tokens 不能独立替代 layered fusion。

风险：

- 变量过多，不能用于回答“camera/register tokens 本身是否有效”。
- 训练成本更高。

实验优先级：P3，诊断用。

## 6. Recommended Experimental Plan

### Stage 0: Reconfirm Evaluation Invariants

目的：排除评测和输入路径问题。

检查项：

- VSIBench/CVBench 的 eval 路径确实加载 `Qwen3_5ForConditionalGenerationWithVGGTOmegaAlpha`。
- `GEOMETRY_ENCODER_TYPE=vggt_omega_alpha`。
- `geometry_encoder_inputs` 非空，帧数与 `image_grid_thw` 条数一致。
- expanded placeholder count = expanded image embed count = expanded position id length。
- eval 没有走 native `pixel_values_videos`。

必要输出：

- 每个 benchmark 抽 3 个样本打印：`image_grid_thw`, `patches_per_frame`, `alpha_frames`, `expanded_visual_tokens`。

### Stage 1: P0 Stabilization Ablation

Run:

- `A0`: current Phase2 alpha, fixed seed rerun small eval subset。
- `A1`: projector `LayerNorm + output gate`。
- `A2`: projector `LayerNorm + zero-init final linear + residual/gate`。

判定：

- 如果 A1/A2 明显恢复，说明当前主要问题可能是新增 token 初始化扰动。
- 如果几乎不变，进入 attention adapter。

### Stage 2: P1 Attention Adapter Ablation

Run:

- `B1`: alpha self-attn only，depth=2。
- `C1`: alpha-patch cross-attn only，depth=1。
- `C2`: alpha-patch cross-attn + global self-attn，depth=2。
- `C3`: C2 + gate init 0，初始等价于 current projector baseline。

判定：

- 如果 C 系列显著优于 B 系列，说明 alpha-patch 对齐是关键。
- 如果 B 系列已显著提升，说明 special tokens 内部/跨帧建模是主要缺口。

### Stage 3: P2 Patch-Update Ablation

Run:

- `D1`: alpha -> patch cross-attn，仅更新 patch tokens。
- `D2`: C2 + D1，双向但分层实现。

判定：

- 如果 D 系列提升明显，说明仅更新 alpha tokens 不足，需要让几何影响 Qwen visual tokens。
- 如果 D 系列不稳定或掉点，保留 Candidate C 作为主线。

### Stage 4: P3 Upper Bound

Run:

- `E1`: LLM internal gated cross-attn，选少量 decoder layers。
- `F1`: alpha tokens + Phase1 deepstack_language_cross_attn。

判定：

- 如果 E/F 才能追上 baseline，Phase2 alpha-only 的“纯输入侧 token injection”可能不是充分架构。
- 如果 C2/C3 已接近 baseline，优先保留轻量 adapter，不进入 internal fusion。

## 7. Implementation Notes For Candidate C

建议新增模块，不直接改已有 projector 语义：

```text
SpatialStack/src/qwen_vl/model/vggt_omega_alpha_adapter.py
```

建议类：

```python
class AlphaPatchCrossAttentionBlock(nn.Module):
    # Q = alpha tokens, K/V = Qwen patch tokens of the same frame

class AlphaGlobalSelfAttentionBlock(nn.Module):
    # self-attn over T * 17 alpha tokens, isolated by sample

class VGGTOmegaAlphaInteractionAdapter(nn.Module):
    # projector + cross-attn + global self-attn + output gate
```

Forward integration point:

- Current line: [modeling_qwen3_5_vggt_omega_alpha.py](/home/jackson/python/SpatialStack-omega/SpatialStack/src/qwen_vl/model/modeling_qwen3_5_vggt_omega_alpha.py:214)
- Replace:

```python
alpha_embeds = self.alpha_projector(alpha_features)
```

- With conceptually:

```python
alpha_embeds = self.alpha_adapter(
    alpha_features=alpha_features,
    image_embeds=image_embeds,
    patches_per_frame=patches_per_frame,
    frames_per_sample=frames_per_sample,
)
```

Important: current code does not carry `frames_per_sample`. For training/eval batches where each row may contain different image counts, global self-attn needs sample boundaries. There are two choices:

1. Derive sample boundaries from `input_ids` visual runs per row inside the wrapper. This avoids adding a new batch field.
2. Add an explicit `frames_per_sample` field in data collation. This is clearer but changes the batch interface.

Given Phase2_plan originally avoided extra batch metadata, option 1 is more consistent. Candidate B/C can compute runs per row from `input_ids == image_token_id`, matching existing packing logic.

## 8. Validation Checklist

Shape tests:

- single image: `T=1`, variable `N_patch`。
- multi-image: `T>1` within one sample。
- batched samples with different numbers of images。
- non-square grid, for example `grid_thw=[1, 24, 32]`。

Invariant tests:

- no fixed `196` assumption。
- no fixed `224x224` assumption。
- adapter output shape is `(sum_frames, 17, text_hidden)`。
- expanded image embeds count equals original image embeds count + `17 * sum_frames`。
- labels for inserted tokens are `-100`。
- inserted tokens use `frame_center` position strategy unless ablated。

Training tests:

- only adapter/projector trainable if base LLM freezing policy requires it。
- Omega encoder remains frozen。
- no new `pixel_values_videos` path。
- no change to Phase1 `vggt` / `vggt_omega` behavior。

Evaluation tests:

- minimal VSIBench subset。
- minimal CVBench subset。
- full benchmark only after subset passes。

## 9. Error Analysis Plan

If Phase2 remains worse after Candidate C:

1. Split benchmark errors by question type: relative position, depth/order, counting, object relation, camera/viewpoint relation。
2. Compare failures where answer requires multi-view geometry vs single-image semantics。
3. Log attention/gate magnitudes:
   - adapter output norm vs Qwen patch norm；
   - alpha gate value；
   - cross-attn entropy per frame；
   - self-attn entropy across frames。
4. Check whether camera token and scene/register tokens behave differently:
   - camera-only ablation；
   - scene/register-only ablation；
   - `1 + 16` full tokens。
5. Compare layer source:
   - Omega layer 23 only；
   - layers 4/11/17/23 pooled or selected；
   - if multi-layer Omega special tokens help, the missing piece may be hierarchical geometry, matching SpatialStack paper.

## 10. Recommendation

Recommended next implementation order:

1. **P0: projector stabilization**  
   Add LayerNorm and a small/zero output gate. This is the cheapest check for distribution drift.

2. **P1: Alpha-Patch Cross-Attention Adapter**  
   Implement Candidate C as the main Phase2 improvement. It keeps the Phase2 input-side wrapper, preserves token counts and M-RoPE invariants, and directly addresses the likely concat/prepend bottleneck.

3. **P1 ablations: self-attn vs cross-attn**  
   Run B1, C1, C2/C3 separately so the causal factor is identifiable.

4. **P2/P3 only if needed**  
   Patch-update or LLM internal gated cross-attn should be treated as stronger architecture changes, not silent Phase2 modifications.

Most defensible hypothesis:

> Phase2 underperforms because current Omega special tokens are geometrically informative but insufficiently language-aligned. A small gated attention adapter before insertion should be tested before abandoning the alpha-token direction.

This is consistent with:

- SpatialStack: layered interaction matters.
- BLIP-2 / Perceiver-VL: query/cross-attention is a standard modality bridge.
- Flamingo: gated cross-attention stabilizes frozen-backbone multimodal fusion.
- Qwen-VL constraints: dynamic token counts and M-RoPE invariants must remain intact.
