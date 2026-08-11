# SceneDistill Architecture

本文介绍当前工作区中 **Stage 2 SceneDistill** 的完整架构。Stage 2 在 Stage 1 的 Pre-LLM SceneDistill 路径上增加了 Post-LLM SceneDistill 路径，形成双端蒸馏：

```text
Qwen Vision
  → 4-stage Pre-GCTE
  → 17 student tokens / frame
  → prepend 到 Qwen visual span
  → Qwen LLM
  → 6-stage Post-GCTE（仅训练期辅助分支）
```

训练时，Pre-GCTE 和 Post-GCTE 分别对齐同一个冻结 VGGT-Omega teacher。评估时不加载 teacher，也不执行 Post-GCTE；真正进入生成路径的是 Pre-GCTE 产生的 17 个 student tokens。

核心实现：

- [scene_distill_module.py](../SpatialStack/src/qwen_vl/model/scene_distill_module.py)：Pre/Post GCTE、projector 与 distillation loss。
- [modeling_qwen3_5_scene_distill.py](../SpatialStack/src/qwen_vl/model/modeling_qwen3_5_scene_distill.py)：完整模型 dataflow、teacher 调用、token packing 与双端 loss。
- [modeling_qwen3_5.py](../SpatialStack/src/qwen_vl/model/modeling_qwen3_5.py)：Qwen LLM 层执行和选择性 hidden-state capture。
- [vggt_omega_direct_packing.py](../SpatialStack/src/qwen_vl/model/vggt_omega_direct_packing.py)：17-token placeholder、embedding、label 与 MRoPE 扩展。

## 1. 架构常量与符号

SceneDistill 使用固定结构：

```text
camera tokens              = 1
scene/register tokens      = 16
special tokens per frame   = 17
special stream dimension   = 1024
teacher/student feature    = 2048
attention heads            = 16
Pre Vision hidden_states   = [1, 5, 9, 13]
Pre Vision block outputs   = [0, 4, 8, 12]
Post LLM layers            = [4, 8, 12, 16, 20, 24]
Pre depth                  = 4
Post depth                 = 6
```

这些常量定义在 [scene_distill_module.py:13](../SpatialStack/src/qwen_vl/model/scene_distill_module.py#L13)。Transformers 会把首个 Vision block 的输入放在 `hidden_states[0]`，随后依次记录各 block 输出；因此 `[1,5,9,13]` 精确对应 zero-based block `[0,4,8,12]` 的输出，也就是人类计数的第 1、5、9、13 个 Vision block。LLM 层号仍是 zero-based block index。

下文使用以下符号：

| 符号 | 含义 |
|---|---|
| `B` | batch 中的视频/样本数 |
| `T_b` | 第 `b` 个视频的帧数 |
| `T=Σ_b T_b` | batch 中总帧数 |
| `P_f` | 第 `f` 帧 Qwen Vision merger 前的 raw visual token 数 |
| `M_f` | 第 `f` 帧 merger 后的 visual token 数 |
| `D_v` | Qwen Vision hidden dimension |
| `D_s=1024` | SceneDistill special-token stream dimension |
| `D_l` | Qwen LLM hidden dimension |
| `D_y=2048` | VGGT teacher 与 SceneDistill distillation feature dimension |

对 `image_grid_thw[f]=(t_f,h_f,w_f)`：

$$
P_f=t_fh_fw_f,
\qquad
M_f=\frac{t_fh_fw_f}{s^2},
$$

其中 `s` 是 Qwen Vision 的 `spatial_merge_size`。SceneDistill 要求每帧作为独立 image 输入，因此 `t_f=1`。

## 2. 整体 Dataflow

### 2.1 训练路径

```text
输入文本 + 按顺序排列的多帧图像
  │
  ├─ Qwen image processor
  │    ├─ pixel_values
  │    └─ image_grid_thw
  │
  ├─ geometry image preprocessing
  │    └─ geometry_encoder_inputs: List[(T_b, 3, H_b, W_b)]
  │
  ├─ Qwen Vision
  │    ├─ raw hidden states at blocks [0,4,8,12]
  │    └─ final merged visual embeddings
  │
  ├─ SceneDistillPreModule
  │    ├─ pre_features:       (T,17,2048) ────────┐
  │    ├─ pre_global_tokens:  (T,17,1024) ─────┐ │
  │    └─ pre_embeds:         (T,17,D_l)        │ │
  │                                             │ │
  ├─ VGGT-Omega teacher（冻结、只运行一次）       │ │
  │    └─ teacher_features: (T,17,2048) ────────┼─┼─ Pre loss
  │                                             │ │
  ├─ [17 pre_embeds + M_f visual embeddings]    │ │
  │    └─ expanded multimodal sequence          │ │
  │                                             │ │
  ├─ Qwen LLM                                   │ │
  │    ├─ final hidden states → lm_head → SFT    │ │
  │    └─ blocks [4,8,12,16,20,24] image spans ─┘ │
  │                                               │
  └─ SceneDistillPostModule                       │
       └─ post_features: (T,17,2048) ─────────────┘─ Post loss
```

总损失：

$$
L_{\mathrm{total}}
=
L_{\mathrm{SFT}}
+\lambda_{\mathrm{pre}}L_{\mathrm{pre}}
+\lambda_{\mathrm{post}}L_{\mathrm{post}}.
$$

### 2.2 Evaluation / Generation 路径

```text
输入文本 + 多帧图像
  → Qwen Vision blocks [0,4,8,12]
  → SceneDistillPreModule
  → 每帧 17 个 pre_embeds
  → prepend 到对应 visual span
  → Qwen LLM prefill
  → lm_head
  → autoregressive decoding with KV cache
```

Evaluation 中：

- 不构造、不加载 VGGT-Omega teacher。
- 不计算 Pre/Post distillation loss。
- 不捕获六层 LLM image-span hidden states。
- 不执行 `SceneDistillPostModule.forward()`。
- Post-GCTE 的输出从不参与 logits 或 generation cache。

student-only evaluation 由 [qwen3_5_scene_distill.py:22](../SpatialStack/src/lmms_eval/models/qwen3_5_scene_distill.py#L22) 配置。

## 3. 输入组织与帧边界

SceneDistill 不使用 native `pixel_values_videos` 路径。一个视频先被展开为连续的多张 image frames：

```text
video
  → frame_1, frame_2, ..., frame_T
  → <image><image>...<image> placeholders
```

模型接收三组与视觉相关的输入：

| 输入 | 内容 | 用途 |
|---|---|---|
| `pixel_values` | 所有帧的 Qwen image tensors | Qwen Vision |
| `image_grid_thw` | 每帧 Qwen patch grid | raw/merged token 数、MRoPE、frame split |
| `geometry_encoder_inputs` | 按样本分组的 RGB tensors | 视频边界；训练时也作为 VGGT teacher 输入 |

`geometry_encoder_inputs` 保留 batch 中每个视频的独立边界：

```text
geometry_encoder_inputs = [video_1_tensor, ..., video_B_tensor]
video_b_tensor.shape     = (T_b, 3, H_b, W_b)
video_sizes              = [T_1, ..., T_B]
```

Pre/Post global attention 均按照 `video_sizes` 切分，因此不同视频之间不会相互 attention。batch 内总帧顺序必须与 `image_grid_thw` 的行顺序完全一致：[modeling_qwen3_5_scene_distill.py:164](../SpatialStack/src/qwen_vl/model/modeling_qwen3_5_scene_distill.py#L164)。

## 4. Online VGGT-Omega Teacher

Teacher 使用冻结的 `VGGTOmegaDirectEncoder`，配置固定为：

```text
encoder_type      = vggt_omega_direct
direct_token_mode = special17
reference_frame   = first
freeze_encoder    = true
```

每个视频独立执行：

```text
RGB frames: (T_b,3,H,W)
  → VGGT-Omega patch embed
  → 24 × (frame block → inter-frame block)
  → final cached layer
  → 取每帧前 17 个 special tokens
```

VGGT-Omega 每个 cached token 由当前 frame state 与 inter-frame state 拼接：

$$
Y_{f,i}
=
\operatorname{Concat}
\left(Y^{\mathrm{frame}}_{f,i},Y^{\mathrm{global}}_{f,i}\right)
\in\mathbb{R}^{2048}.
$$

teacher 输出：

```text
index 0       = camera token
indices 1:17  = 16 register/scene tokens
shape         = (T_b,17,2048)
```

batch 内各视频输出沿 frame 维拼接为 `(T,17,2048)`。Teacher 始终运行在 `eval()` 和 `torch.no_grad()` 下：[vggt_omega_direct_encoder.py:53](../SpatialStack/src/qwen_vl/model/geometry_encoders/vggt_omega_direct_encoder.py#L53)。

Teacher 仅在训练且至少一个 distillation loss 被启用时运行；Pre/Post loss 同时启用时，每个 batch 也只提取一次 teacher features。

## 5. GCTE 的两个基础 Attention 模块

Pre-GCTE 与 Post-GCTE 共享同一种计算结构，但各自拥有独立参数。

### 5.1 Frame Cross-Attention

Frame Cross-Attention 让每帧的 17 个 special tokens 读取该帧的视觉特征。

输入：

```text
special_tokens:  (T,17,D_s)
visual_features: (Σ_f N_f,D_kv)
frame_sizes:     [N_1,...,N_T]
```

第 `f` 帧内部：

$$
Q_f=W_Q\operatorname{LN}(Z_f),
\qquad
K_f=W_K\operatorname{LN}(X_f),
\qquad
V_f=W_V\operatorname{LN}(X_f),
$$

$$
A_f=\operatorname{SDPA}
\left(\operatorname{QKNorm}(Q_f),
      \operatorname{QKNorm}(K_f),
      V_f\right).
$$

Attention 和 FFN 均使用可学习 LayerScale residual，初值为 `0.01`：

$$
Z'_f=Z_f+\gamma_{\mathrm{attn}}W_OA_f,
$$

$$
\widetilde Z_f
=
Z'_f+\gamma_{\mathrm{ffn}}
\operatorname{FFN}(\operatorname{LN}(Z'_f)).
$$

其中 FFN 为 `Linear(D_s,4D_s) → GELU → Linear(4D_s,D_s)`。实现使用 16 个 attention heads、QK-Norm、无 dropout、非 causal SDPA：[scene_distill_module.py:54](../SpatialStack/src/qwen_vl/model/scene_distill_module.py#L54)。

为支持不同图像分辨率，代码先按 `N_f` 对帧分桶；只有 token 数相同的帧才 stack 后并行计算。不同帧的 K/V 不会混合。

### 5.2 Global Camera-Scene Self-Attention

Global attention 让同一视频中的 camera/scene tokens 跨帧交互。

对第 `b` 个视频：

$$
S_b
=
\operatorname{reshape}
(\widetilde Z_b,T_b\times17,D_s).
$$

随后执行标准 multi-head self-attention：

$$
Q_b,K_b,V_b
=
\operatorname{split}
\left(W_{QKV}\operatorname{LN}(S_b)\right),
$$

$$
G_b
=
S_b
+\gamma_{\mathrm{attn}}W_O
\operatorname{SDPA}(Q_b,K_b,V_b),
$$

$$
S'_b
=
G_b
+\gamma_{\mathrm{ffn}}
\operatorname{FFN}(\operatorname{LN}(G_b)).
$$

最后 reshape 回 `(T_b,17,D_s)`。每个视频单独执行 self-attention，再沿 frame 维拼接，因此不存在跨视频信息泄漏：[scene_distill_module.py:152](../SpatialStack/src/qwen_vl/model/scene_distill_module.py#L152)。

## 6. SceneDistill Pre Module

`SceneDistillPreModule` 是推理主路径的一部分，在 training 和 evaluation 都会执行。

### 6.1 Qwen Vision 多层输入

Qwen Vision 一次 forward 同时返回：

- 最终 `pooler_output`：merger 后 visual embeddings，稍后直接送入 LLM。
- `hidden_states`：merger 前的 raw Vision block outputs。

Pre-GCTE 固定选择：

```text
V^(1)  = hidden_states[0]
V^(2)  = hidden_states[4]
V^(3)  = hidden_states[8]
V^(4)  = hidden_states[12]
shape  = (Σ_f P_f,D_v)
```

这些 raw visual features 在进入 Pre-GCTE 前执行 `detach()`。因此 Pre/Post distillation loss 不通过该支路更新 Qwen Vision：[scene_distill_module.py:291](../SpatialStack/src/qwen_vl/model/scene_distill_module.py#L291)。

### 6.2 Camera/Scene Token 初始化

Pre module 维护两组可学习初始化参数：

```text
pre_camera_token: (1,2,1,1024)
pre_scene_token:  (1,2,16,1024)
```

第二维表示 reference-frame variant：

- 每个视频第一帧使用 variant 0。
- 同一视频的其余帧共享 variant 1。

拼接顺序固定为：

```text
[camera, scene_1, ..., scene_16]
```

得到初始状态：

$$
Z^{(0)}\in\mathbb{R}^{T\times17\times1024}.
$$

初始化使用 `normal_(std=10^{-3})`：[scene_distill_module.py:255](../SpatialStack/src/qwen_vl/model/scene_distill_module.py#L255)。

### 6.3 四组 Pre-GCTE

第 `m` 组依次执行：

$$
\widetilde Z^{(m)}
=
\operatorname{PreFrameCrossAttn}^{(m)}
(Z^{(m-1)},V^{(m)}),
$$

$$
Z^{(m)}
=
\operatorname{PreGlobalSelfAttn}^{(m)}
(\widetilde Z^{(m)}).
$$

四组参数彼此独立。每组先汇聚当前帧 Qwen visual information，再让 special tokens 在同一视频内跨帧交互。

### 6.4 Pre Distillation Feature

第 4 组结束时保留 frame/global 两个状态：

```text
pre_after_frame  = Z_tilde^(4): (T,17,1024)
pre_after_global = Z^(4):       (T,17,1024)
```

二者拼接为：

$$
F_{\mathrm{pre}}
=
\operatorname{Concat}
(\widetilde Z^{(4)},Z^{(4)})
\in\mathbb{R}^{T\times17\times2048}.
$$

`F_pre` 用于 Pre distillation loss；`Z^(4)` 同时作为 Post-GCTE 的初始 query state。

### 6.5 Pre Projector

Pre projector 把 2048-D student feature 映射到 Qwen LLM hidden space：

$$
E_{\mathrm{pre}}
=
W_2\operatorname{GELU}
\left(W_1\operatorname{LN}(F_{\mathrm{pre}})\right),
$$

```text
LayerNorm(2048)
→ Linear(2048,2048)
→ GELU
→ Linear(2048,D_l)
```

输出：

$$
E_{\mathrm{pre}}\in\mathbb{R}^{T\times17\times D_l}.
$$

这些 embeddings 是 SceneDistill 唯一直接写入 LLM input sequence 的新增信息。

## 7. Visual Span 扩展与 Token Packing

Qwen Vision merger 后，第 `f` 帧原有：

$$
E^{\mathrm{vis}}_f\in\mathbb{R}^{M_f\times D_l}.
$$

SceneDistill 固定在其前方插入 17 个 Pre embeddings：

$$
E^{\mathrm{image}}_f
=
\left[
E^{\mathrm{pre}}_{f,0:17};
E^{\mathrm{vis}}_{f,0:M_f}
\right]
\in\mathbb{R}^{(17+M_f)\times D_l}.
$$

每帧在 LLM 中的 image span 顺序为：

```text
[CAM, SCENE_1, ..., SCENE_16, VIS_1, ..., VIS_M]
```

Packing 同步扩展四类张量：

| 张量 | 17 个新增位置的值 |
|---|---|
| `input_ids` | `image_token_id` placeholder |
| `inputs_embeds` | `E_pre` |
| `labels` | `-100`，不计算 token-level SFT loss |
| `attention_mask` | `1` |
| `position_ids` | 复制当前 visual span 的帧中心 MRoPE 坐标，即 `(temporal, min_h+H//2, min_w+W//2)` |

插入后重新计算 `rope_deltas` 和需要时的 `cache_position`。`build_direct_only_mask` 标记每帧 image span 开头的 17 个位置；Post-GCTE 的 K/V 使用整个 image span，而不是只使用 direct positions：[vggt_omega_direct_packing.py:198](../SpatialStack/src/qwen_vl/model/vggt_omega_direct_packing.py#L198)。

SceneDistill 只允许 `geometry_token_insert_position="front"`。

## 8. Qwen LLM 与选择性 Hidden-State Capture

扩展后的 multimodal embeddings 进入正常 Qwen LLM：

$$
H^{(\ell+1)}
=
\operatorname{DecoderBlock}^{(\ell)}(H^{(\ell)}).
$$

最终 hidden states 经 `lm_head` 得到 logits：

$$
\operatorname{logits}=W_{\mathrm{lm\_head}}H^{\mathrm{final}}.
$$

当且仅当训练时启用 Post loss，LLM 在 block `[4,8,12,16,20,24]` 执行完成后，用 `image_mask` 立即截取 image-span hidden states：

```text
H_image^(l_m): (Σ_f (17+M_f),D_l)
m = 1,...,6
```

捕获发生在目标 decoder block 之后、最终 RMSNorm 之前。实现只保存六个 masked tensors，不保存全部层的完整文本序列，也不使用长期 forward hooks：[modeling_qwen3_5.py:286](../SpatialStack/src/qwen_vl/model/modeling_qwen3_5.py#L286)。

捕获的 hidden states 不执行 `detach()`，因此 Post loss 可以向对应 LLM blocks 反向传播。正常 evaluation 不传入 capture 参数，`outputs.hidden_states` 也不用于 Post-GCTE。

## 9. SceneDistill Post Module

`SceneDistillPostModule` 是 Stage 2 新增的训练辅助分支。它拥有六组全新、彼此独立的 frame/global attention 参数，但不拥有新的 camera/scene 初始化 token，也不拥有 projector。

### 9.1 输入

初始 special state 直接复用 Pre 第 4 组的 global 输出：

$$
U^{(0)}=Z^{(4)}
\in\mathbb{R}^{T\times17\times1024}.
$$

第 `m` 组 K/V 来自对应 LLM 层的完整 image span：

```text
Q   = 当前帧 17 个 Post special tokens: (17,1024)
K/V = 当前帧 [17 specials + M_f visuals]: (17+M_f,D_l)
```

文本 tokens、padding positions 和其他帧不会进入该帧 Frame Cross-Attention。

### 9.2 六组 Post-GCTE

第 `m` 组：

$$
\widetilde U^{(m)}
=
\operatorname{PostFrameCrossAttn}^{(m)}
\left(U^{(m-1)},H_{\mathrm{image}}^{(\ell_m)}\right),
$$

$$
U^{(m)}
=
\operatorname{PostGlobalSelfAttn}^{(m)}
(\widetilde U^{(m)}),
$$

其中：

```text
(l_1,...,l_6) = (4,8,12,16,20,24)
```

Frame layer 负责从对应深度的 LLM image representation 更新每帧 special tokens；Global layer 负责同一视频内部的跨帧 camera/scene interaction。

### 9.3 Post Distillation Feature

第 6 组的 frame/global 输出拼接为：

$$
F_{\mathrm{post}}
=
\operatorname{Concat}
(\widetilde U^{(6)},U^{(6)})
\in\mathbb{R}^{T\times17\times2048}.
$$

`F_post` 只用于 Post distillation loss。它不会：

- 投影为新的 LLM input embeddings；
- 覆盖 `outputs.last_hidden_state`；
- 修改 logits；
- 写入 generation KV cache；
- 在 evaluation/generation 中执行。

Post module 实现见 [scene_distill_module.py:328](../SpatialStack/src/qwen_vl/model/scene_distill_module.py#L328)。

## 10. 双端 Distillation Loss

Pre/Post 使用同一个 index-aligned cosine loss：

$$
\mathcal{D}(F,Y)
=
\frac{1}{T}
\sum_{f=1}^{T}
\sum_{i=1}^{17}
\left(1-\cos(F_{f,i},Y_{f,i})\right).
$$

因此：

$$
L_{\mathrm{pre}}=\mathcal{D}(F_{\mathrm{pre}},Y),
\qquad
L_{\mathrm{post}}=\mathcal{D}(F_{\mathrm{post}},Y).
$$

Loss 的具体语义：

- camera 对齐 camera：`index 0 ↔ index 0`。
- 16 个 scene tokens 逐 index 对齐：`index i ↔ index i`。
- 不执行 pooling、matching、permutation 或 temporal interpolation。
- cosine 在 float32 中计算。
- teacher features 被 `detach()`。
- 17 个 token 先求和，再对所有有效帧求平均。
- student 或 teacher 出现非有限值时立即报错。

总损失：

$$
L_{\mathrm{total}}
=
L_{\mathrm{SFT}}
+\lambda_{\mathrm{pre}}L_{\mathrm{pre}}
+\lambda_{\mathrm{post}}L_{\mathrm{post}}.
$$

当前 Stage 2 训练母脚本默认：

```text
PRE_DISTILL_WEIGHT  = 0.2
POST_DISTILL_WEIGHT = 0.05
```

共享脚本和 dataclass 的 fallback 均为 `0.05/0.05`；实际 SceneDistill 训练以母脚本值为准：[train_scene_distill.sh:35](../SpatialStack/scripts/train/train_scene_distill.sh#L35)。

## 11. 参数与梯度流

当前 Stage 2 母脚本的训练状态：

| 模块 | 是否可训练 | 接收的梯度 |
|---|---:|---|
| Qwen Vision Encoder | 否 | 无 |
| Qwen Vision merger/MM MLP | 否 | 无 |
| VGGT-Omega teacher | 否 | 无 |
| Pre camera/scene parameters | 是 | Pre loss、Post loss、SFT |
| 4 组 Pre frame/global layers | 是 | Pre loss、Post loss、SFT |
| Pre projector | 是 | SFT、Post loss；不接收 Pre loss |
| Qwen LLM | 是 | SFT；启用 Post 时也接收 Post loss |
| lm_head | 是 | SFT |
| 6 组 Post frame/global layers | 是 | Post loss |

梯度路径可概括为：

```text
L_pre  → pre_features → Pre-GCTE

L_post → Post-GCTE
       ├─→ captured LLM image states → Qwen LLM
       │                              → Pre projector → Pre-GCTE
       └─→ pre_global_tokens → Pre-GCTE

L_SFT  → lm_head → Qwen LLM
       → pre_embeds → Pre projector → Pre-GCTE
```

参数冻结逻辑位于 [train_qwen.py:91](../SpatialStack/src/qwen_vl/train/train_qwen.py#L91)。

## 12. 模块启用阶段

| 模块/操作 | Training | Evaluation prefill | Autoregressive decode |
|---|:---:|:---:|:---:|
| 输入帧组织与 `video_sizes` | ✓ | ✓ | — |
| Qwen Vision forward | ✓ | ✓ | — |
| Vision blocks `[0,4,8,12]` capture | ✓ | ✓ | — |
| Pre camera/scene token 初始化 | ✓ | ✓ | — |
| 4-stage Pre-GCTE | ✓ | ✓ | — |
| Pre projector | ✓ | ✓ | — |
| 17-token front packing | ✓ | ✓ | — |
| Qwen LLM | ✓ | ✓ | ✓，复用 KV cache |
| `lm_head` | ✓ | ✓ | ✓ |
| VGGT-Omega teacher | 仅任一 distill weight `>0` | — | — |
| Pre distillation loss | 仅 `pre_distill_weight>0` | — | — |
| LLM 六层 image-span capture | 仅 `post_distill_weight>0` | — | — |
| 6-stage Post-GCTE | 仅 `post_distill_weight>0` | — | — |
| Post distillation loss | 仅 `post_distill_weight>0` | — | — |
| SFT loss | 有 labels 时 | — | — |

SceneDistill 的视觉主路径只在首个 visual/prefill step 激活：

```text
pixel_values is not None
and image_grid_thw is not None
and (cache_position is None or cache_position[0] == 0)
```

后续 decode step 使用已经建立的 KV cache，不重复运行 Qwen Vision、Pre-GCTE、packing、teacher 或 Post-GCTE：[modeling_qwen3_5_scene_distill.py:253](../SpatialStack/src/qwen_vl/model/modeling_qwen3_5_scene_distill.py#L253)。

## 13. Training Forward 的执行顺序

一次带 labels 的 Stage 2 forward 按以下顺序执行：

1. 判断 SceneDistill 是否处于首个 visual step。
2. 为每帧增加 17 个 label placeholders，新增 labels 设为 `-100`。
3. 根据 `training`、labels 和两个权重决定是否计算 Pre/Post loss。
4. Qwen Vision 返回 final embeddings 和指定中间层。
5. Pre-GCTE 返回 `pre_embeds`、`F_pre`、`pre_global_tokens`。
6. 若任一 distillation loss 启用，在线 teacher 只运行一次并生成 `Y`。
7. 若 Pre loss 启用，计算 `D(F_pre,Y)`。
8. 将 `pre_embeds` 插入每帧 visual span 前方。
9. Qwen LLM 正常前向；若 Post loss 启用，同时捕获六层 image spans。
10. 若 Post loss 启用，Post-GCTE 生成 `F_post` 并计算 `D(F_post,Y)`。
11. `lm_head` 计算 logits，expanded labels 计算 SFT loss。
12. 加权合并三个 loss。
13. 清空 `_last_pre_distill_loss` 与 `_last_post_distill_loss`，避免跨 batch 残留计算图。

外层 loss 组合见 [modeling_qwen3_5_scene_distill.py:486](../SpatialStack/src/qwen_vl/model/modeling_qwen3_5_scene_distill.py#L486)。

## 14. Evaluation 与 Generation 细节

Stage 2 checkpoint 的正常 evaluation 应使用：

```text
model_impl = qwen3_5_scene_distill
```

adapter 会固定：

```text
use_geometry_encoder            = true
geometry_encoder_type           = scene_distill
geometry_encoder_freeze         = true
geometry_direct_token_mode      = special17
geometry_token_insert_position  = front
reference_frame                 = first
geometry_encoder_path           = None
```

因此 evaluation model 是纯 student：

```text
Qwen Vision + Pre-GCTE + Pre projector + Qwen LLM + lm_head
```

首个 prefill step 完成 17-token 注入并建立 KV cache；后续每生成一个文本 token，只执行必要的 LLM decode。

当前 [eval_qwen35_scene_distill.sh](../SpatialStack/scripts/evaluation/eval_qwen35_scene_distill.sh) 直接加载 Stage 2 checkpoint，不再提供 Stage 1 checkpoint key mapping。

## 15. Checkpoint 契约

Stage 2 checkpoint 保存：

- Pre camera/scene initialization parameters。
- 4 组 Pre frame/global layers。
- Pre projector。
- 6 组 Post frame/global layers。
- Qwen LLM、lm_head 及其他正常训练权重。
- `pre_distill_weight`、`post_distill_weight` 和 SceneDistill config。

Stage 2 checkpoint 不保存冻结的 VGGT-Omega teacher。`state_dict()` 会过滤：

```text
geometry_encoder.*
model.geometry_encoder.*
```

Teacher 在训练开始时由 `geometry_encoder_path` 单独加载；evaluation adapter 清空该路径，因此不会构造 teacher：[scene_distill_module.py:26](../SpatialStack/src/qwen_vl/model/scene_distill_module.py#L26)。

当前 checkpoint discovery 会识别：

```text
scene_distill_pre_module
scene_distill_post_module
```

## 16. 结构约束与运行时检查

SceneDistill 会显式检查以下契约：

- `geometry_encoder_type == "scene_distill"`。
- token mode 必须为 `special17`。
- insert position 必须为 `front`。
- reference frame 必须为 `first`。
- teacher 必须冻结。
- Pre/Post distillation weight 不得为负。
- LLM 层数必须大于 24。
- 只支持 ordered multi-image frames，不支持 native `pixel_values_videos`。
- 每个 `image_grid_thw` 的 `t` 必须等于 1。
- `sum(video_sizes)` 必须等于 frame 数。
- 每帧 image span 必须以恰好 17 个 direct positions 开始。
- Pre/Post feature 必须为 `(T,17,2048)`。
- teacher/student frame 数、token 数和维度必须完全一致。
- teacher/student features 必须全部 finite。

这些检查保证数据从 frame grouping、GCTE、packing、LLM capture 到 loss 的 shape 和顺序一致。

## 17. 模块级 Input/Output 总表

| 模块 | 输入 | 输出 | 阶段 |
|---|---|---|---|
| Qwen Vision | `pixel_values`, `image_grid_thw` | 4 层 raw features；final merged embeddings | Training + Evaluation prefill |
| VGGT-Omega Teacher | `(T_b,3,H,W)` | `(T_b,17,2048)` | Training only，按 loss gating |
| Pre token initializer | `video_sizes` | `(T,17,1024)` | Training + Evaluation prefill |
| Pre Frame Cross-Attn ×4 | special tokens + raw Vision layer | `(T,17,1024)` | Training + Evaluation prefill |
| Pre Global Self-Attn ×4 | `(T,17,1024)` + `video_sizes` | `(T,17,1024)` | Training + Evaluation prefill |
| Pre feature concat | final frame/global states | `(T,17,2048)` | Training + Evaluation prefill |
| Pre projector | `(T,17,2048)` | `(T,17,D_l)` | Training + Evaluation prefill |
| Packing | pre embeddings + merged visual embeddings | 每帧 `(17+M_f,D_l)` | Training + Evaluation prefill |
| Qwen LLM | expanded multimodal sequence | final hidden states | Training + Evaluation |
| LLM layer capture | blocks `[4,8,12,16,20,24]` + image mask | 6 × `(Σ_f(17+M_f),D_l)` | Training only，`post_weight>0` |
| Post Frame Cross-Attn ×6 | Post tokens + LLM image spans | `(T,17,1024)` | Training only，`post_weight>0` |
| Post Global Self-Attn ×6 | `(T,17,1024)` + `video_sizes` | `(T,17,1024)` | Training only，`post_weight>0` |
| Post feature concat | final frame/global states | `(T,17,2048)` | Training only，`post_weight>0` |
| Distillation loss | student `(T,17,2048)` + teacher | scalar | Training only |
| lm_head | final LLM hidden states | vocabulary logits | Training + Evaluation |

## 18. 一句话总结

SceneDistill Stage 2 用 **4-stage Pre-GCTE** 从冻结 Qwen Vision 多层特征中生成每帧 17 个 camera/scene embeddings，并将其前置注入 Qwen LLM；训练时再用 **6-stage Post-GCTE** 读取多层 LLM image representations，通过同一个冻结 VGGT-Omega teacher 对 Pre 和 Post 两端施加 index-aligned cosine supervision，而 evaluation 只保留 Pre-GCTE 主路径进行 student-only generation。
