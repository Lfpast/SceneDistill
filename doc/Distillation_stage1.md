# 将 CamDistill 的 Camera Token 蒸馏迁移为 SceneDistill 的 17-token GCTE

## 1. 目标架构与已锁定决策

新增独立的 `scene_distill` 模型路径，保留现有 SpatialStack 和 `vggt_omega_direct` 基线不变。每个视频帧生成：

- 1 个 camera token。
- 16 个 scene token。
- 共 17 个 special tokens，顺序固定为 `[camera, scene_1, ..., scene_16]`。

整体数据流为：

$$
\text{Qwen Vision blocks }[1,5,9,13]
\rightarrow 4\times(\text{Frame Cross-Attn}\rightarrow\text{Global Camera-Scene Self-Attn})
$$

$$
Z=\operatorname{Concat}(Z_{\text{frame}}^{(4)},Z_{\text{global}}^{(4)})
\in \mathbb{R}^{T\times17\times2048}
$$

$$
L_{\text{total}}
=
L_{\text{SFT}}
+
0.05\cdot
\frac{1}{T}
\sum_{f=1}^{T}
\sum_{i=1}^{17}
\left(1-\cos(Z_{f,i},Y_{f,i})\right)
$$

其中 `Y` 是在线、冻结的 VGGT-Omega 最后一层 17 个 special tokens。

已锁定的实现选择：

- 用户所说的第 `1/5/9/13` 层按人类 1-based block 编号解释；由于 `hidden_states[0]` 是首层输入，代码读取 tuple 索引 `[1,5,9,13]`。
- 教师特征在线计算，复用 SceneDistill 当前的冻结 `VGGTOmegaDirectEncoder`，不引入离线缓存。
- 17 个 token 逐 index 对齐；先在每帧内求和，再对 batch 中所有有效帧求平均，最后乘 `0.05`。
- 最终 17 个 student tokens 始终放在每帧 Qwen visual tokens 之前。
- 不实现 `空间智能方案.md` 中额外的 LLM-layer GCTE、输出端第二次蒸馏或 gated residual；这些属于另一套更大的架构提案，[空间智能方案.md:132–231](/home/jackson/python/SceneDistill/空间智能方案.md:132) 与本次明确需求冲突时，以本次需求为准。

## 2. 三个代码库的实现依据

| 目标行为 | 复用依据 |
|---|---|
| Cross-attention 中 special tokens 为 Q，当前帧 visual tokens 为 K/V | CamDistill 已将 Q/K/V 分开投影，并支持 `cam_dim != vis_dim`，[camdistill_model.py:18–64](/home/jackson/python/CamDistill/camera_movement_sft/plugins/camdistill_model.py:18)；其按帧切分、同尺寸分桶和 SDPA 实现位于 [camdistill_model.py:66–137](/home/jackson/python/CamDistill/camera_movement_sft/plugins/camdistill_model.py:66)。 |
| 跨帧 global attention 不包含 visual tokens | CamDistill 已按视频隔离 camera self-attention，[camdistill_model.py:140–221](/home/jackson/python/CamDistill/camera_movement_sft/plugins/camdistill_model.py:140)；VGGT-Omega 的 register 模式只取头部 special tokens、展平为 `frames × 17` 后做 inter-frame attention，[aggregator.py:170–217](/home/jackson/python/SceneDistill/vggt-omega/vggt_omega/models/aggregator.py:170)。 |
| 第一帧为参考系，其他帧共享另一组初始化 token | CamDistill 的 camera 参数是 `(1,2,1,D)`，第 0 组给首帧、第 1 组给其余帧，[camdistill_model.py:294–358](/home/jackson/python/CamDistill/camera_movement_sft/plugins/camdistill_model.py:294)。VGGT-Omega 同时维护 `(1,2,1,D)` camera 和 `(1,2,16,D)` register 参数，[aggregator.py:81–98](/home/jackson/python/SceneDistill/vggt-omega/vggt_omega/models/aggregator.py:81)，并通过相同的首帧/其他帧切片规则展开，[aggregator.py:246–250](/home/jackson/python/SceneDistill/vggt-omega/vggt_omega/models/aggregator.py:246)。 |
| 教师 token 顺序与特征维度 | VGGT-Omega 先拼接 camera、16 register、patch tokens，[aggregator.py:111–150](/home/jackson/python/SceneDistill/vggt-omega/vggt_omega/models/aggregator.py:111)，最后缓存 `cat([frame_tokens,tokens], dim=-1)`，因此每个 special token 是 2048 维。模型输出直接截取 `:patch_token_start`，[vggt_omega.py:41–49](/home/jackson/python/SceneDistill/vggt-omega/vggt_omega/models/vggt_omega.py:41)。 |
| 在线提取教师 17 tokens | SceneDistill 已冻结、以 `torch.no_grad()` 运行 VGGT-Omega，[vggt_omega_direct_encoder.py:53–72](/home/jackson/python/SceneDistill/SpatialStack/src/qwen_vl/model/geometry_encoders/vggt_omega_direct_encoder.py:53)，`special17` 模式取 `tokens[:, :patch_token_start]` 并检查数量，[vggt_omega_direct_encoder.py:85–109](/home/jackson/python/SceneDistill/SpatialStack/src/qwen_vl/model/geometry_encoders/vggt_omega_direct_encoder.py:85)，输出维度明确为 2048，[vggt_omega_direct_encoder.py:120–124](/home/jackson/python/SceneDistill/SpatialStack/src/qwen_vl/model/geometry_encoders/vggt_omega_direct_encoder.py:120)。 |
| GCTE 最终双分支特征 | CamDistill 保留最后一层 frame 输出和 global 输出，拼成 `2×stream_dim` 后再进入 projector，[camdistill_model.py:393–420](/home/jackson/python/CamDistill/camera_movement_sft/plugins/camdistill_model.py:393)。这与 VGGT-Omega 的双分支 2048 维教师特征严格对应。 |
| 17 tokens 放在 visual tokens 前 | SceneDistill 已有按帧切分 final merged visual embeds、再 prepend direct tokens 的函数，[vggt_omega_direct_packing.py:292–320](/home/jackson/python/SceneDistill/SpatialStack/src/qwen_vl/model/vggt_omega_direct_packing.py:292)。新增 placeholder 的 label 为 `-100`、attention 为 `1`，MRoPE 使用对应帧的空间中心坐标，[vggt_omega_direct_packing.py:88–195](/home/jackson/python/SceneDistill/SpatialStack/src/qwen_vl/model/vggt_omega_direct_packing.py:88)。 |
| 视频帧顺序和跨视频边界 | 数据集将视频转换为多张连续 image frames，[data_qwen.py:463–481](/home/jackson/python/SceneDistill/SpatialStack/src/qwen_vl/data/data_qwen.py:463)；collator 以相同顺序拼接 Qwen frames，同时保留每个样本独立的 `geometry_encoder_inputs`，[data_qwen.py:666–725](/home/jackson/python/SceneDistill/SpatialStack/src/qwen_vl/data/data_qwen.py:666)。这可以直接得到 `video_sizes` 并防止不同视频之间发生 global attention。 |

## 3. 核心实现

### 3.1 新增独立 SceneDistill 模块

新增：

- `SpatialStack/src/qwen_vl/model/scene_distill_module.py`
- `SpatialStack/src/qwen_vl/model/modeling_qwen3_5_scene_distill.py`

不直接修改或替换 `modeling_qwen3_5_vggt_omega_direct.py`，以免改变现有 baseline 的 checkpoint 和推理行为。新 wrapper 以现有 direct wrapper 的初始化、packing 和外层 loss 结构为模板；现有入口及形状处理可参考 [modeling_qwen3_5_vggt_omega_direct.py:53–163](/home/jackson/python/SceneDistill/SpatialStack/src/qwen_vl/model/modeling_qwen3_5_vggt_omega_direct.py:53) 和 [modeling_qwen3_5_vggt_omega_direct.py:321–417](/home/jackson/python/SceneDistill/SpatialStack/src/qwen_vl/model/modeling_qwen3_5_vggt_omega_direct.py:321)。

`SceneDistillPreModule` 固定以下结构：

- `NUM_SCENE_TOKENS = 16`
- `NUM_SPECIAL_TOKENS = 17`
- `PRE_VISION_BLOCK_INDICES = (1,5,9,13)`；这里是 Transformers `hidden_states` tuple 索引，对应 zero-based Vision block `0,4,8,12` 的输出
- `STREAM_DIM = 1024`
- `FEATURE_DIM = 2048`
- `NUM_HEADS = 16`
- `PRE_DISTILL_DEPTH = 4`
- `PRE_DISTILL_WEIGHT = 0.05`

初始化两个参数：

```text
pre_camera_token: (1, 2, 1, 1024)
pre_scene_token:  (1, 2, 16, 1024)
```

二者均使用 `normal_(std=1e-3)`。每个视频第一帧使用 variant 0，其余帧使用 variant 1；拼接后得到：

```text
special_tokens: (T_total, 17, 1024)
```

依据是 VGGT-Omega 的 camera/register 初始化和拼接顺序，[aggregator.py:81–118](/home/jackson/python/SceneDistill/vggt-omega/vggt_omega/models/aggregator.py:81)。

### 3.2 四组 GCTE

从 CamDistill 小修改得到公用 `FrameCrossAttentionLayer`：

- 将输入从 `(T,1,D)` 泛化为 `(T,17,D)`。
- Q reshape 为 `(group, heads, 17, head_dim)`。
- K/V 仍只来自对应帧的 Qwen visual tokens。
- 保留 Pre-Norm、QK-Norm、SDPA、FFN、LayerScale 和按 `frame_size` 分桶。
- 不把 special tokens 加入 K/V。

CamDistill 已提供所需的不同 Q/KV 维度投影和残差结构，[camdistill_model.py:29–64](/home/jackson/python/CamDistill/camera_movement_sft/plugins/camdistill_model.py:29)；这里只修改 query length，不重写注意力机制。

从 CamDistill 小修改复制 global layer：

1. 按 `video_sizes` 将 `(T_total,17,1024)` 切成不同视频。
2. 每个视频 reshape 为 `(T_video×17,1024)`。
3. 对这 `T_video×17` 个 special tokens 做双向 self-attention。
4. 恢复为 `(T_video,17,1024)`。
5. 不同视频绝不互相 attention，visual tokens 也不进入该层。

这同时遵循 CamDistill 的视频隔离逻辑和 VGGT-Omega 的 special-only register attention，[aggregator.py:190–217](/home/jackson/python/SceneDistill/vggt-omega/vggt_omega/models/aggregator.py:190)。

四个 stage 严格按下列顺序执行：

```text
stage 1: Qwen block 1  visual tokens -> frame cross-attn -> global special self-attn
stage 2: Qwen block 5  visual tokens -> frame cross-attn -> global special self-attn
stage 3: Qwen block 9  visual tokens -> frame cross-attn -> global special self-attn
stage 4: Qwen block 13 visual tokens -> frame cross-attn -> global special self-attn
```

最后保留 stage 4 的 `post-frame` 和 `post-global`：

```text
student_features = concat(post_frame, post_global, dim=-1)
                 = (T_total, 17, 2048)
```

不添加额外的 2048→2048 teacher-alignment projector，因为 CamDistill 和 VGGT-Omega 已通过双分支拼接在维度及语义上对齐，[camdistill_model.py:250–268](/home/jackson/python/CamDistill/camera_movement_sft/plugins/camdistill_model.py:250)。

### 3.3 提取 Qwen3.5 四层 visual tokens

调用 Qwen3.5 现有：

```python
get_image_features(..., return_dict=True, output_hidden_states=True)
```

然后精确选择 `hidden_states[0]`、`[4]`、`[8]`、`[12]`。SceneDistill 已有调用和读取 `hidden_states` 的路径，[modeling_qwen3_5.py:641–684](/home/jackson/python/SceneDistill/SpatialStack/src/qwen_vl/model/modeling_qwen3_5.py:641)；依赖版本固定为 Transformers 5.3.0，[setup.py:10](/home/jackson/python/SceneDistill/SpatialStack/setup.py:10)，其捕获机制以 [Transformers v5.3.0 官方 Qwen3.5 实现](https://github.com/huggingface/transformers/blob/v5.3.0/src/transformers/models/qwen3_5/modeling_qwen3_5.py) 为准。

具体约束：

- 取 merger 之前的 raw visual hidden states，而不是 `pooler_output`。
- 使用 `image_grid_thw.prod(-1)` 计算每帧的 raw token 数；当前代码已有相同的 split 逻辑，[modeling_qwen3_5.py:627–637](/home/jackson/python/SceneDistill/SpatialStack/src/qwen_vl/model/modeling_qwen3_5.py:627)。
- 当前数据将视频拆成多图，故 SceneDistill 路径要求每个 `image_grid_thw` 行的 `t == 1`；遇到 native `pixel_values_videos` 时显式报错，保持与 direct wrapper 的输入边界一致，[modeling_qwen3_5_vggt_omega_direct.py:181–213](/home/jackson/python/SceneDistill/SpatialStack/src/qwen_vl/model/modeling_qwen3_5_vggt_omega_direct.py:181)。
- visual features 在进入 GCTE 前 `detach()`；训练母脚本同时冻结 Vision Encoder。CamDistill 也在每层 block 输出后立即 detach，[modeling_qwen3_vl_camdistill.py:1052–1070](/home/jackson/python/CamDistill/camera_movement_sft/plugins/modeling_qwen3_vl_camdistill.py:1052)。
- `vis_dim` 从 `config.vision_config.hidden_size` 读取，special stream 固定为 1024；Cross-Attention 的 K/V projection 负责映射维度。当前目标 Qwen3.5-4B 的预期 visual hidden size 为 1024，但仍在初始化时检查配置，禁止静默使用错误维度。

### 3.4 在线 VGGT-Omega 教师和蒸馏 loss

在 `scene_distill` wrapper 中单独初始化教师：

```text
encoder_type = vggt_omega_direct
direct_token_mode = special17
reference_frame = first
freeze_encoder = true
```

直接复用 `VGGTOmegaDirectEncoder`，不改 `SceneDistill/vggt-omega` 源码，也不复用 CamDistill 的离线 `.safetensors` cache。教师仍执行现有 first-reference transform，[vggt_omega_direct_encoder.py:66–72](/home/jackson/python/SceneDistill/SpatialStack/src/qwen_vl/model/geometry_encoders/vggt_omega_direct_encoder.py:66)。

教师输出与 student 输出都必须通过以下检查：

```text
shape == (T_total, 17, 2048)
frame count == sum(video_sizes)
token index 0 == camera
token indices 1:17 == scene/register tokens
all values finite
```

loss 使用 float32 计算：

```python
per_token = 1 - cosine_similarity(
    student_features.float(),
    teacher_features.float(),
    dim=-1,
)                                   # (T_total, 17)

distill_loss = per_token.sum(dim=-1).mean()
total_loss = sft_loss + 0.05 * distill_loss
```

严格逐 index 计算，不进行 Hungarian matching、平均池化、scene-token permutation 或 temporal interpolation。CamDistill 在 teacher/student 维度一致后使用 float32 cosine loss，[camdistill_loss.py:219–258](/home/jackson/python/CamDistill/camera_movement_sft/plugins/camdistill_loss.py:219)，并在总 loss 中相加，[camdistill_loss.py:261–262](/home/jackson/python/CamDistill/camera_movement_sft/plugins/camdistill_loss.py:261)；本实现只按本次要求将 reduction 改为 17-token index sum，并固定系数为 `0.05`。

不继承 CamDistill 当前的 `0.3`、200-step warmup、两个 1024 维 half 分别求 loss或离线 cache。在线教师仅在 `self.training and labels is not None` 时运行；生成和 benchmark evaluation 只执行 student GCTE。

为避免跨 batch 残留：

- inner forward 开始时将 `_last_pre_distill_loss` 清为 `None`。
- 同一 forward 内计算标量并由外层 causal-LM wrapper立即读取。
- 外层加入 SFT loss 后再次清空引用。
- 如果训练 batch 缺少 teacher inputs、帧数不一致或出现非有限值，直接报错，不把 distillation loss 静默置零。

### 3.5 Projector 与 17-token 前置拼接

对 `student_features: (T,17,2048)` 使用 CamDistill projector 的原结构：

```text
LayerNorm(2048)
Linear(2048, 2048)
GELU
Linear(2048, text_hidden_size)
```

依据是 [camdistill_model.py:224–247](/home/jackson/python/CamDistill/camera_movement_sft/plugins/camdistill_model.py:224) 和其 2048→LLM 初始化位置 [camdistill_model.py:311–316](/home/jackson/python/CamDistill/camera_movement_sft/plugins/camdistill_model.py:311)。不复用当前 progressive direct projector，因为它会根据输入/输出宽度调整 hidden width，不是 CamDistill 的原 projector 结构。

投影后：

```text
student_embeds: (T_total, 17, text_hidden_size)
```

逐帧与 Qwen merger 后的 visual embeddings 拼接：

```text
[CAM, SCENE_1, ..., SCENE_16, VISUAL_1, ..., VISUAL_N]
```

直接调用：

- `expand_image_embeds_with_direct_tokens(..., insert_position="front")`
- `expand_visual_placeholders(..., num_extra_per_frame=17, insert_position="front")`；新增位置固定使用帧中心 MRoPE

现有 wrapper 已展示完整的 placeholder 扩展、masked scatter 和 language-model 输入流程，[modeling_qwen3_5_vggt_omega_direct.py:221–318](/home/jackson/python/SceneDistill/SpatialStack/src/qwen_vl/model/modeling_qwen3_5_vggt_omega_direct.py:221)。SceneDistill 路径不提供 `back` 模式，配置不是 `front` 时直接拒绝。

## 4. 训练、保存与评测接线

### 4.1 公共接口

新增模型配置值：

```text
geometry_encoder_type = "scene_distill"
```

继续复用：

```text
geometry_encoder_path
geometry_encoder_freeze = true
reference_frame = first
```

`scene_distill` 内部固定为 `special17`，不允许通过 `geometry_direct_token_mode` 改成 `camera` 或 `scene16`，从而保证 teacher/student 契约不会被配置破坏。

需要更新：

- 训练参数和模型 dispatcher；当前 Qwen3.5 分派位置为 [train_qwen.py:232–279](/home/jackson/python/SceneDistill/SpatialStack/src/qwen_vl/train/train_qwen.py:232)。
- geometry patch-size 映射，为 `scene_distill` 使用 VGGT-Omega 的 patch size 16；当前映射入口为 [data/utils.py:14–18](/home/jackson/python/SceneDistill/SpatialStack/src/qwen_vl/data/utils.py:14)，实际 resize 继续由 `image_grid_thw` 推导，不另写尺寸常量。
- 推理模型分派；当前入口为 [infer.py:154–193](/home/jackson/python/SceneDistill/SpatialStack/scripts/inference/infer.py:154)。
- LMMS-Eval 模型分派及注册；分别参考 [qwen3_5.py:161–205](/home/jackson/python/SceneDistill/SpatialStack/src/lmms_eval/models/qwen3_5.py:161) 和 [models/__init__.py:47–50](/home/jackson/python/SceneDistill/SpatialStack/src/lmms_eval/models/__init__.py:47)。

新增 LMMS-Eval 名称 `qwen3_5_scene_distill`，兼容适配器仿照现有 direct adapter，不改变原模型名。

### 4.2 训练母脚本

新增 `SpatialStack/scripts/train/train_scene_distill.sh`，复用共享 `train.sh` 的启动、DeepSpeed 和数据参数传递方式，但由母脚本固定：

```text
GEOMETRY_ENCODER_TYPE=scene_distill
GEOMETRY_ENCODER_PATH=<VGGT-Omega checkpoint>
REFERENCE_FRAME=first
GEOMETRY_ENCODER_FREEZE=true
TUNE_VISION=false
TUNE_LLM=true
special tokens=17
vision layers=1,5,9,13
distill weight=0.05
insert position=front
```

现有 SceneDistill 名称下仍指向旧 SpatialStack fusion 的脚本不覆盖；在训练文档中把两条路径明确区分。共享脚本当前负责参数默认值与 CLI 拼装，[train.sh:22–38](/home/jackson/python/SceneDistill/SpatialStack/scripts/train/train.sh:22) 和 [train.sh:110–169](/home/jackson/python/SceneDistill/SpatialStack/scripts/train/train.sh:110)。

训练初始化后打印并断言参数组：

- 冻结：Qwen Vision Encoder、VGGT-Omega teacher。
- 可训练：4 组 GCTE、camera/scene token 参数、projector、LLM。
- 任何 teacher 参数出现梯度立即作为测试失败。

现有冻结逻辑位于 [train_qwen.py:73–116](/home/jackson/python/SceneDistill/SpatialStack/src/qwen_vl/train/train_qwen.py:73)，新增路径在这里加入精确检查，不改变其他模型族的默认行为。

### 4.3 Checkpoint 恢复

将 `scene_distill_pre_module` 加入 Qwen3.5 自定义子模块 checkpoint 发现和恢复范围。当前 shard 筛选只识别 geometry/projector/fusion 关键词，[modeling_qwen3_5.py:44–50](/home/jackson/python/SceneDistill/SpatialStack/src/qwen_vl/model/modeling_qwen3_5.py:44)，加载逻辑位于 [modeling_qwen3_5.py:237–268](/home/jackson/python/SceneDistill/SpatialStack/src/qwen_vl/model/modeling_qwen3_5.py:237)。

保存/恢复契约：

- 保存 student GCTE、camera/scene 参数和 projector。
- 不把冻结的外部 VGGT-Omega 权重重复写入训练 checkpoint。
- `config.json` 保存 `geometry_encoder_type=scene_distill` 和 teacher checkpoint 路径。
- reload 后所有 student 参数必须与保存前逐项一致；缺少 student key 时明确报错，不能随机初始化后继续评测。

## 5. 测试与验收

新增独立测试文件 `SpatialStack/tests/test_scene_distill.py`，覆盖：

1. **初始化与参考帧**

   - 两个视频分别含 2 帧和 3 帧。
   - 检查输出为 `(5,17,1024)`。
   - 每个视频第一帧使用 variant 0，其余帧使用 variant 1。
   - 检查 index 0 为 camera、indices 1–16 为 scene。

2. **Frame-wise Cross-Attention**

   - 使用不同长度的 frame visual tokens。
   - 检查输出为 `(T,17,1024)`。
   - 改变某一帧 K/V 只影响该帧对应的 17 个 queries。
   - 验证 K/V token 数总和与 raw `image_grid_thw.prod(-1)` 一致。对应的 CamDistill 防错检查依据为 [camdistill_model.py:84–103](/home/jackson/python/CamDistill/camera_movement_sft/plugins/camdistill_model.py:84)。

3. **Global Camera-Scene Self-Attention**

   - 同一视频的 `T×17` tokens 可以相互影响。
   - 不同视频之间完全隔离。
   - visual tensor 不出现在 global layer 接口中。
   - 恢复后顺序仍为 frame-major、special-index-minor。

4. **四层提取**

   - mock Qwen Vision 输出 24 层可区分 tensor。
   - 只检查并消费 `hidden_states` tuple 索引 `[1,5,9,13]`，对应 block `[0,4,8,12]` 的输出。
   - 输入 GCTE 的是 pre-merger raw tokens，最终拼接使用的是 `pooler_output` merged tokens。
   - 层数量、token 总数或 hidden width 不符合契约时 fail fast。

5. **双分支输出与 projector**

   - 最后 post-frame/post-global 均为 `(T,17,1024)`。
   - 拼接严格为 `(T,17,2048)`。
   - projector 输出 `(T,17,text_hidden_size)`。

6. **逐 index cosine loss**

   - student 与 teacher 完全相同时 loss 为 0。
   - 单个相反方向 token 只给对应 index 贡献 2。
   - 交换两个 scene token 后 loss 增大，证明没有集合匹配或 permutation。
   - 用手算 tensor 验证“每帧 sum 17，再对有效帧 mean”。
   - 验证总 loss 精确满足 `SFT + 0.05 × distill`，没有 warmup 或额外平均。

7. **Teacher 契约**

   - mock `VGGTOmegaDirectEncoder` 返回 `(T,17,2048)`。
   - 验证 teacher 在训练且有 labels 时调用一次。
   - `eval()`、generation subsequent decoding step、缺少 labels 时不调用 teacher。
   - teacher tensor 为 `requires_grad=False`。

8. **前置拼接与标签**

   - 每帧顺序严格为 `[17 specials, visual patches]`。
   - 新增 17 个位置的 labels 全为 `-100`、attention mask 为 1。
   - `input_ids`、labels、attention mask、position IDs 和 embeds 长度完全一致。
   - batch 内不同 visual patch 数仍能正确拆分；依据是现有 packing 的 frame count 和 token count 检查，[vggt_omega_direct_packing.py:300–319](/home/jackson/python/SceneDistill/SpatialStack/src/qwen_vl/model/vggt_omega_direct_packing.py:300)。

9. **Checkpoint round trip**

   - 保存一个已知 student 参数状态。
   - 从 checkpoint 重载。
   - 检查 GCTE、camera token、scene token、projector 参数逐项一致。
   - 验证现有 `vggt_omega_direct` checkpoint 加载行为未改变。

10. **真实模型 smoke test**

    在项目 GPU 环境运行单 batch Qwen3.5-4B + VGGT-Omega：

    - 四层 visual features、student 2048 特征、teacher 2048 特征和总 loss 全部 finite。
    - teacher/vision 无梯度；GCTE、projector、LLM 获得梯度。
    - 完成一次 optimizer step、保存、重载和一次 generation。
    - 运行 `py_compile`、新增 pytest、`git diff --check`，再进行短训练和 LMMS-Eval smoke test。

验收时将纯 Qwen SFT、保留的 `vggt_omega_direct special17` 和新 `scene_distill` 作为三个独立路径。`0.05` 只记录为用户提供的 ablation 结论；在没有完整 benchmark 结果前，不新增性能提升声明。

## 6. 边界与失败策略

- 不修改 VGGT-Omega aggregator、camera head 或预训练权重；教师使用 aggregator 最后一层 special tokens，而不是 camera head 的后处理输出。camera head 会继续混合这些 tokens，[camera_head.py:63–72](/home/jackson/python/SceneDistill/vggt-omega/vggt_omega/heads/camera_head.py:63)，因此不作为本次 target。
- 不把 VGGT-Omega 的 16 个 register tokens 做池化；在 student 侧称为 scene tokens，但按原 index `1–16` 蒸馏。
- 不把 visual tokens 放入 global camera-scene self-attention。
- 不支持 teacher/student 帧数自动插值、截断或重复；数据顺序或形状不一致直接报错。
- 不覆盖现有 SpatialStack、`vggt_omega_direct`、camera-only 或 scene16 baseline。
- 不引入离线 teacher cache、额外 loss warmup、第二个 distillation head或新的 LLM-layer 注入。
