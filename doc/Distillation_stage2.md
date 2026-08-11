# SceneDistill Stage 2：Qwen3.5 LLM Internal Distillation 实施方案

目标文档：`/home/jackson/python/SceneDistill/doc/Distillation_stage2.md`

## 1. 目标与架构边界

Stage 2 继续使用现有 `geometry_encoder_type="scene_distill"`，不新增或修改 geometry 注册，不改 VGGT-Omega 源码，不替换 Stage 1 路径。

保留 Stage 1 全部组件：

- Vision Encoder zero-based block `[0,4,8,12]` 的输出；在 Transformers `hidden_states` tuple 中对应 `[1,5,9,13]`。
- 四组 pre-LLM GCTE。
- 1 camera + 16 scene tokens。
- Stage 1 VGGT-Omega cosine distillation。
- 17 tokens 前置拼接和原始 SFT loss。
- 现有推理与 LMMS-Eval 注册。

新增彼此不共享参数的 post-LLM 模块：

- LLM 层索引固定为 `[4,8,12,16,20,24]`。
- 六组 `frame-wise cross-attention → global camera-scene self-attention`。
- 独立 post distillation loss 和权重。
- 只读 LLM intermediate hidden states，不把 post-GCTE 输出写回 LLM。
- pre/post 使用同一次联合 forward 正常反传，但不共享 attention 参数或 loss 字段。

旧设计文档曾提出 gated residual 回写，[空间智能方案.md:157–177](/home/jackson/python/SceneDistill/doc/空间智能方案.md:157)；本次已明确选择“只读蒸馏头”，因此不实现 gated residual、逐层 special-token 替换或额外注入路径。

## 2. 完整数据流与张量契约

设：

- 总帧数为 `T`。
- special token 数为 `S=17`。
- Stage 1 stream dimension 为 `D_s=1024`。
- VGGT-Omega feature dimension 为 `D_t=2048`。
- Qwen3.5 text hidden dimension 为 `D_l=config.text_config.hidden_size`。
- 第 `f` 帧 merger 后 visual token 数为 `P_f`。

### 2.1 Stage 1 保持不变

当前 Stage 1 按四个 Vision 层执行 alternating attention，并保留最后一次 frame/global 输出，[scene_distill_module.py:261–290](/home/jackson/python/SceneDistill/SpatialStack/src/qwen_vl/model/scene_distill_module.py:261)：

$$
\widetilde Z_{\mathrm{pre}}\in\mathbb{R}^{T\times17\times1024},
\qquad
Z_{\mathrm{pre}}\in\mathbb{R}^{T\times17\times1024}
$$

$$
F_{\mathrm{pre}}
=
\operatorname{Concat}
(\widetilde Z_{\mathrm{pre}},Z_{\mathrm{pre}})
\in\mathbb{R}^{T\times17\times2048}
$$

其中：

- `F_pre` 继续用于 Stage 1 distillation 和 LLM projector。
- `Z_pre`，即最后一次 global attention 后的 `17×1024` 状态，作为 Stage 2 初始 Q。
- `SceneDistillPreModule.forward` 的内部返回值由两个扩展为三个：

```python
pre_embeds, pre_features, pre_global_tokens
```

这是内部接口变更；现有 `scene_distill_module.py` 源码文件名、模型类型和 geometry 注册保持不变。

### 2.2 LLM hidden-state 提取

17 个 Stage 1 embeddings 已按帧放在 visual tokens 前方，[modeling_qwen3_5_scene_distill.py:247–264](/home/jackson/python/SceneDistill/SpatialStack/src/qwen_vl/model/modeling_qwen3_5_scene_distill.py:247)，placeholder 和 mask 同步扩展于 [modeling_qwen3_5_scene_distill.py:277–309](/home/jackson/python/SceneDistill/SpatialStack/src/qwen_vl/model/modeling_qwen3_5_scene_distill.py:277)。

因此，每帧在 LLM 内部的目标 span 固定为：

```text
[camera, scene_1, ..., scene_16, visual_1, ..., visual_Pf]
```

每个选中层仅提取 image span，不包含：

- 文本 tokens。
- padding。
- 其他帧的 tokens。

第 `f` 帧 K/V 长度为：

$$
L_f=17+P_f
$$

六层特征契约为：

```text
llm_layer_features[m]:
    flattened shape = (sum_f(17 + P_f), D_l)
    per-frame shape = (17 + P_f, D_l)
```

捕获点位于 decoder layer 完成之后。当前自定义 LLM 循环在 [modeling_qwen3_5.py:366–398](/home/jackson/python/SceneDistill/SpatialStack/src/qwen_vl/model/modeling_qwen3_5.py:366) 执行 decoder layer 和可选 fusion；Stage 2 将在同一循环中仅捕获指定的六层，避免启用全部层的 `output_hidden_states`。

依赖版本继续以仓库固定的 Transformers 5.3.0 为准，[setup.py:9–13](/home/jackson/python/SceneDistill/SpatialStack/setup.py:9)，并以 [Transformers v5.3.0 Qwen3.5 官方实现](https://github.com/huggingface/transformers/blob/v5.3.0/src/transformers/models/qwen3_5/modeling_qwen3_5.py) 校准输出语义。

### 2.3 六组 post-LLM GCTE

新增独立的 `SceneDistillPostModule`，但继续放在现有 `scene_distill_module.py` 中，不创建新源码模块。

常量固定为：

```python
LLM_BLOCK_INDICES = (4, 8, 12, 16, 20, 24)
POST_DISTILL_DEPTH = 6
POST_DISTILL_WEIGHT = 0.05
```

初始状态：

```text
post_tokens = pre_global_tokens
shape = (T, 17, 1024)
```

第 `m` 组执行：

$$
\widetilde Z^{(m)}
=
\operatorname{FrameCrossAttn}^{(m)}
\left(
Z^{(m-1)}, H_{\mathrm{image}}^{(\ell_m)}
\right)
$$

其中每帧：

- Q：该帧的 17 个 `1024-D` camera/scene tokens。
- K/V：该帧当前 LLM 层的完整 image span，即 `17 special + P_f visual`，维度为 `D_l`。
- 不读取文本或其他帧。
- 公用 `FrameCrossAttentionLayer` 的不同 Q/KV 维度投影同时服务 Pre/Post；两条路径使用不同实例、互不共享参数，[scene_distill_module.py:51](/home/jackson/python/SceneDistill/SpatialStack/src/qwen_vl/model/scene_distill_module.py:51)。
- 其机制源自 CamDistill 的 Q/K/V projection、QK norm、SDPA、FFN 和 LayerScale，[camdistill_model.py:18–137](/home/jackson/python/CamDistill/camera_movement_sft/plugins/camdistill_model.py:18)。

随后：

$$
Z^{(m)}
=
\operatorname{GlobalSelfAttn}^{(m)}
(\widetilde Z^{(m)})
$$

- 每个视频独立 reshape 为 `(T_video×17,1024)`。
- 只让 camera/scene tokens 双向交互。
- visual 和文本 tokens 不进入 global attention。
- 不同视频之间严格隔离。

该部分直接复用 Stage 1 的视频隔离实现，[scene_distill_module.py:136–193](/home/jackson/python/SceneDistill/SpatialStack/src/qwen_vl/model/scene_distill_module.py:136)，其来源分别是 CamDistill 的跨帧 self-attention [camdistill_model.py:140–221](/home/jackson/python/CamDistill/camera_movement_sft/plugins/camdistill_model.py:140) 和 VGGT-Omega 的 special-only register attention [aggregator.py:190–217](/home/jackson/python/SceneDistill/vggt-omega/vggt_omega/models/aggregator.py:190)。

第六组结束时保留：

```text
post_after_frame:  (T, 17, 1024)
post_after_global: (T, 17, 1024)
```

最终：

$$
F_{\mathrm{post}}
=
\operatorname{Concat}
(\widetilde Z^{(6)},Z^{(6)})
\in\mathbb{R}^{T\times17\times2048}
$$

该双分支拼接严格复用 CamDistill 的 final frame/global 组合，[camdistill_model.py:393–420](/home/jackson/python/CamDistill/camera_movement_sft/plugins/camdistill_model.py:393)，并对应 VGGT-Omega 缓存的 `cat([frame_tokens, tokens], dim=-1)`，[aggregator.py:129–154](/home/jackson/python/SceneDistill/vggt-omega/vggt_omega/models/aggregator.py:129)。

不新增 1024→2048 或 text-hidden→2048 projector。

## 3. Teacher 与双端 loss

### 3.1 Teacher 只运行一次

训练 batch 中，VGGT-Omega 最后一层 teacher features 只提取一次，然后同时供 pre/post loss 使用。

现有 teacher：

- 在线运行且被冻结，[vggt_omega_direct_encoder.py:53–72](/home/jackson/python/SceneDistill/SpatialStack/src/qwen_vl/model/geometry_encoders/vggt_omega_direct_encoder.py:53)。
- `special17` 精确取 camera 和 16 register tokens，[vggt_omega_direct_encoder.py:92–109](/home/jackson/python/SceneDistill/SpatialStack/src/qwen_vl/model/geometry_encoders/vggt_omega_direct_encoder.py:92)。
- 输出维度固定为 2048，[vggt_omega_direct_encoder.py:120–121](/home/jackson/python/SceneDistill/SpatialStack/src/qwen_vl/model/geometry_encoders/vggt_omega_direct_encoder.py:120)。
- 来源是 VGGT-Omega aggregator 最后一层，而不是 camera head，[vggt_omega.py:41–49](/home/jackson/python/SceneDistill/vggt-omega/vggt_omega/models/vggt_omega.py:41)。

Teacher 契约：

```text
teacher_features.shape == (T, 17, 2048)
index 0      == camera
indices 1:17 == scene/register
requires_grad == False
all values finite
```

### 3.2 Pre loss

现有 `scene_distillation_loss` 原样复用：

$$
L_{\mathrm{pre}}
=
\frac{1}{T}
\sum_{f=1}^{T}
\sum_{i=1}^{17}
\left(
1-\cos(F_{\mathrm{pre},f,i},Y_{f,i})
\right)
$$

当前逐 index、帧内求和、跨帧平均的实现位于 [scene_distill_module.py:293–315](/home/jackson/python/SceneDistill/SpatialStack/src/qwen_vl/model/scene_distill_module.py:293)。

### 3.3 Post loss

使用相同纯 loss 函数，但传入完全独立的 `F_post`：

$$
L_{\mathrm{post}}
=
\frac{1}{T}
\sum_{f=1}^{T}
\sum_{i=1}^{17}
\left(
1-\cos(F_{\mathrm{post},f,i},Y_{f,i})
\right)
$$

不使用：

- scene-token matching。
- permutation。
- pooling。
- temporal interpolation。
- warmup 权重。
- layer-wise auxiliary loss。
- VGGT-Omega 最后一层之外的 teacher features。

CamDistill 在 2048 维空间用 float32 cosine 对齐并加回 SFT loss，[camdistill_loss.py:248–262](/home/jackson/python/CamDistill/camera_movement_sft/plugins/camdistill_loss.py:248)；Stage 2 沿用该数值策略。

总损失固定为：

$$
L_{\mathrm{total}}
=
L_{\mathrm{SFT}}
+
\lambda_{\mathrm{pre}}L_{\mathrm{pre}}
+
\lambda_{\mathrm{post}}L_{\mathrm{post}}
$$

默认 SceneDistill 母脚本：

```text
PRE_DISTILL_WEIGHT  = 0.2   # 保留当前 Stage 1 母脚本默认值
POST_DISTILL_WEIGHT = 0.05
```

共享训练脚本和 dataclass 的兜底默认值：

```text
pre_distill_weight  = 0.05
post_distill_weight = 0.05
```

## 4. 代码实施

### 4.1 扩展 `scene_distill_module.py`

依据现有 attention、projector、loss 集中在同一文件的结构，[scene_distill_module.py:13–315](/home/jackson/python/SceneDistill/SpatialStack/src/qwen_vl/model/scene_distill_module.py:13)，在原文件完成：

1. 将 `DISTILL_WEIGHT` 硬重命名为 `PRE_DISTILL_WEIGHT`。
2. 新增 `POST_DISTILL_WEIGHT`。
3. 明确区分：

```python
PRE_DISTILL_DEPTH = len(PRE_VISION_BLOCK_INDICES)
POST_DISTILL_DEPTH = len(LLM_BLOCK_INDICES)
```

4. 扩展 `SceneDistillPreModule.forward`，额外返回最终 pre-global `17×1024` 状态。
5. 新增 `SceneDistillPostModule`：
   - 6 个全新公用 `FrameCrossAttentionLayer` 实例。
   - 6 个全新公用 `GlobalSelfAttentionLayer` 实例。
   - 不与 Stage 1 `ModuleList` 共享对象或参数。
   - 不拥有新的 camera/scene initialization parameters。
   - 不拥有 projector。
6. 增加形状检查：
   - 精确收到六层 LLM features。
   - 每层 hidden width 等于 `text_hidden_dim`。
   - 每层 token 总数等于 `sum(17+P_f)`。
   - `pre_global_tokens.shape == (T,17,1024)`。
   - 最终 `post_features.shape == (T,17,2048)`。
7. 继续用同一个 `scene_distillation_loss` 计算两个独立标量。
8. 将 Stage 1 的 `SceneDistillModule` 重命名为 `SceneDistillPreModule`；Pre/Post 保留不同的 `ModuleList` 属性名，但复用同一组公用 attention 类。

### 4.2 为自定义 Qwen3.5 LLM 增加选择性捕获

当前 `Qwen3_5TextModelWithGeometry.forward` 自己执行全部 decoder layers，但只返回最终 hidden state，[modeling_qwen3_5.py:272–403](/home/jackson/python/SceneDistill/SpatialStack/src/qwen_vl/model/modeling_qwen3_5.py:272)。

增加两个仅供内部 wrapper 使用的可选参数：

```python
capture_hidden_state_layers: Optional[Sequence[int]] = None
capture_hidden_state_mask: Optional[torch.Tensor] = None
```

行为：

- 参数为空时，现有所有模型路径完全不变。
- Stage 2 传入固定层 `[4,8,12,16,20,24]`。
- 每个目标 decoder layer 完成后，用 image-span mask 立即截取对应 hidden states。
- 只保留六个 masked tensors，不保存全部 LLM 层和全部文本序列。
- 按层号升序返回 tuple，wrapper 再逐项验证层数与形状。
- 不 `detach()` LLM features，使 post loss 能监督 LLM。
- 捕获发生在 decoder layer 输出后、最终 RMSNorm 前。
- 不注册长期 forward hooks，避免梯度 checkpoint 重算时覆盖缓存或残留计算图。

Stage 1 已使用 `image_mask[...,0]` 定位 visual positions，[modeling_qwen3_5.py:380–396](/home/jackson/python/SceneDistill/SpatialStack/src/qwen_vl/model/modeling_qwen3_5.py:380)。Stage 2 复用该 mask 约定。

### 4.3 扩展 SceneDistill wrapper

在 [modeling_qwen3_5_scene_distill.py](/home/jackson/python/SceneDistill/SpatialStack/src/qwen_vl/model/modeling_qwen3_5_scene_distill.py:59) 原文件内：

1. 将 pre 模块属性硬重命名为 `scene_distill_pre_module`。
2. 新增独立属性：

```python
self.scene_distill_pre_module
self.scene_distill_post_module
self._last_pre_distill_loss
self._last_post_distill_loss
```

3. 初始化 post 模块时使用：

```text
special_dim = 1024
llm_hidden_dim = config.text_config.hidden_size
num_heads = 16
depth = 6
```

4. `align_geometry_modules` 同时移动 pre/post 模块，但 teacher 仍只按 geometry encoder 的 dtype/device 规则运行。当前 pre 模块对齐入口为 [modeling_qwen3_5_scene_distill.py:113–122](/home/jackson/python/SceneDistill/SpatialStack/src/qwen_vl/model/modeling_qwen3_5_scene_distill.py:113)。
5. 验证：
   - `pre_distill_weight >= 0`。
   - `post_distill_weight >= 0`。
   - `text_config.num_hidden_layers > 24`。
6. 运行 Stage 1 后，同时取得：
   - `student_embeds`。
   - `pre_features`。
   - `pre_global_tokens`。
7. 构建 expanded placeholder 和 image mask 后，检查：
   - 每帧开头正好 17 个 direct positions。
   - `direct_only_mask.sum() == T×17`。
   - 每帧 LLM K/V split size 为 `17+merged_frame_size`。

Packing 的每帧前置拼接依据是 [vggt_omega_direct_packing.py:292–320](/home/jackson/python/SceneDistill/SpatialStack/src/qwen_vl/model/vggt_omega_direct_packing.py:292)，direct-only mask 的 run 检测依据是 [vggt_omega_direct_packing.py:323–352](/home/jackson/python/SceneDistill/SpatialStack/src/qwen_vl/model/vggt_omega_direct_packing.py:323)。

8. 仅在以下条件成立时计算 loss：

```text
self.training
and labels is not None
and first visual decoding step
and corresponding weight > 0
```

9. 当 pre/post 任一 loss 启用时，只调用一次 `_collect_teacher_features`；当前 teacher 收集入口位于 [modeling_qwen3_5_scene_distill.py:151–159](/home/jackson/python/SceneDistill/SpatialStack/src/qwen_vl/model/modeling_qwen3_5_scene_distill.py:151)。
10. 将当前单 loss 相加逻辑 [modeling_qwen3_5_scene_distill.py:427–433](/home/jackson/python/SceneDistill/SpatialStack/src/qwen_vl/model/modeling_qwen3_5_scene_distill.py:427) 改为两个显式分支：
    - 权重大于零时，对应 loss 缺失立即报错。
    - 权重为零时，不要求对应 loss，也不运行其额外计算。
    - forward 结束立即清空两个 transient loss 引用，防止跨 batch 残留。
11. 不把 post tokens 写入 `outputs.last_hidden_state`，不改变 logits 或 generation cache。

### 4.4 Checkpoint 与训练参数

在 checkpoint 发现列表 [modeling_qwen3_5.py:44–51](/home/jackson/python/SceneDistill/SpatialStack/src/qwen_vl/model/modeling_qwen3_5.py:44) 中显式加入：

```python
"scene_distill_pre_module"
"scene_distill_post_module"
```

保存契约：

- 保存 Stage 1 GCTE、tokens、projector。
- 保存 Stage 2 六组 attention。
- 不保存冻结 VGGT-Omega teacher；当前 teacher filter 位于 [scene_distill_module.py:23–32](/home/jackson/python/SceneDistill/SpatialStack/src/qwen_vl/model/scene_distill_module.py:23)。
- `config.json` 保存 `pre_distill_weight` 和 `post_distill_weight`。
- 不再保存或读取 `distill_weight`。
- reload 后 pre/post 参数都必须完成 state-dict round trip。

训练参数在 [argument.py:13–29](/home/jackson/python/SceneDistill/SpatialStack/src/qwen_vl/train/argument.py:13) 中硬迁移为：

```python
pre_distill_weight: float = 0.05
post_distill_weight: float = 0.05
```

彻底删除：

```python
distill_weight
```

训练配置写入位置 [train_qwen.py:274–304](/home/jackson/python/SceneDistill/SpatialStack/src/qwen_vl/train/train_qwen.py:274) 同步改为两个新字段；若加载的旧 config 仍含 `distill_weight`，删除该旧属性，不提供回退或 CLI alias。

训练参数冻结逻辑 [train_qwen.py:120–132](/home/jackson/python/SceneDistill/SpatialStack/src/qwen_vl/train/train_qwen.py:120) 调整为：

- VGGT-Omega teacher：全部冻结。
- Qwen Vision Encoder：继续冻结。
- Stage 1 module：可训练。
- Stage 2 module：可训练。
- LLM/lm_head：按现有 `TUNE_MM_LLM=True` 训练。
- 分别打印 pre/post trainable parameter 数量；现有统计入口是 [train_qwen.py:352–367](/home/jackson/python/SceneDistill/SpatialStack/src/qwen_vl/train/train_qwen.py:352)。

### 4.5 四个训练脚本

1. [train_scene_distill.sh:35–52](/home/jackson/python/SceneDistill/SpatialStack/scripts/train/train_scene_distill.sh:35)

```bash
export PRE_DISTILL_WEIGHT="${PRE_DISTILL_WEIGHT:-0.2}"
export POST_DISTILL_WEIGHT="${POST_DISTILL_WEIGHT:-0.05}"
```

删除 `DISTILL_WEIGHT`。

2. [train_scene_distill_multinode.sh:37–55](/home/jackson/python/SceneDistill/SpatialStack/scripts/train/train_scene_distill_multinode.sh:37)

使用完全相同的两个变量和默认值；其 Slurm `3 nodes × 2 GPUs` 资源申请、master 地址及 launcher 逻辑不改。

3. [train.sh:22–40](/home/jackson/python/SceneDistill/SpatialStack/scripts/train/train.sh:22)

新增共享默认值：

```bash
set_default_env PRE_DISTILL_WEIGHT "0.05"
set_default_env POST_DISTILL_WEIGHT "0.05"
```

CLI 拼装位置 [train.sh:136–146](/home/jackson/python/SceneDistill/SpatialStack/scripts/train/train.sh:136) 改为：

```bash
--pre_distill_weight "${PRE_DISTILL_WEIGHT}"
--post_distill_weight "${POST_DISTILL_WEIGHT}"
```

启动日志 [train.sh:175–184](/home/jackson/python/SceneDistill/SpatialStack/scripts/train/train.sh:175) 分别打印两个权重。

4. [train_multinode.sh:18–59](/home/jackson/python/SceneDistill/SpatialStack/scripts/train/train_multinode.sh:18)

保留 `SLURM_EXPORT_ENV=ALL` 和 GPU allocation 推导；在 srun 前及各节点启动日志中打印 pre/post 权重，验证环境变量没有在 Slurm job step 中丢失。

同步更新 [train.md:75](/home/jackson/python/SceneDistill/SpatialStack/scripts/train/train.md:75) 的命令示例，不再出现旧字段。

新调用方式：

```bash
PRE_DISTILL_WEIGHT=0.2 \
POST_DISTILL_WEIGHT=0.05 \
bash scripts/train/train_scene_distill.sh
```

多节点：

```bash
sbatch \
  --export=ALL,PRE_DISTILL_WEIGHT=0.2,POST_DISTILL_WEIGHT=0.05 \
  scripts/train/train_scene_distill_multinode.sh
```

### 4.6 评估脚本
`eval_qwen35_scene_distill.sh` 直接评估 Stage 2 checkpoint，不再保留旧 Stage 1 checkpoint 的 key mapping。

## 5. 明确不改的部分

- 不新增 geometry encoder type。
- 不修改 geometry factory。
- 不修改 VGGT-Omega aggregator、camera head 或 checkpoint。
- 不修改现有前置 17-token packing 规则。
- 不新增 LMMS-Eval 模型名。
- 不修改 inference dispatcher。
- 不让 post-GCTE 在 generation 时运行。
- 不让 post-GCTE 输出回写 LLM。
- 不新增 gated residual、额外 projector、逐层 loss 或 teacher cache。
- Pre 读取 Transformers Vision `hidden_states` 的 `[1,5,9,13]`，即 zero-based block `[0,4,8,12]` 的输出。
- 不让 pre/post attention 层共享参数。
- SFT labels、attention mask 与 cache-position 扩展语义保持不变；SceneDistill MRoPE anchor 改为对应帧的空间中心。

## 6. 测试与验收

当前基线已验证：

- `SpatialStack/tests/test_scene_distill.py`：12 tests passed。
- 四个目标 shell 脚本均通过 `bash -n`。
- 当前工作树干净。

实施后扩展同一个 [test_scene_distill.py](/home/jackson/python/SceneDistill/SpatialStack/tests/test_scene_distill.py:1)，覆盖：

1. **Stage 1 回归**
   - 原有 12 项测试继续通过。
   - 新第三返回值为 `(T,17,1024)`。
   - `pre_features[...,1024:]` 与返回的 post-global state 一致。
   - `POST_DISTILL_WEIGHT=0` 时不捕获 LLM layers、不运行 post 模块，总 loss 等价于旧 Stage 1 公式。

2. **固定 LLM 层**
   - 常量严格等于 `(4,8,12,16,20,24)`。
   - 缺少任一层、层数小于 25、顺序不符时 fail fast。
   - 捕获的是 decoder layer 输出，不是 embedding 输入或最终 norm 输出。

3. **LLM image-span K/V**
   - 每帧 K/V 顺序严格为 `[17 specials, visual]`。
   - K/V 长度严格为 `17+P_f`。
   - 文本、padding 和其他帧不进入该帧 cross-attention。
   - 修改某帧 hidden states 只影响该帧 frame-attention 输出。

4. **Post global attention**
   - 同一视频的 `T×17` tokens 可以相互作用。
   - 不同视频完全隔离。
   - visual hidden states 不直接进入 global layer。

5. **六组模块独立**
   - pre/post `ModuleList` 参数对象无交集。
   - 六组 post frame/global layers 均拥有独立参数。
   - post 参数获得梯度。
   - LLM hidden K/V 获得 post-loss 梯度。
   - teacher 始终无梯度。

6. **Post feature 与 loss**
   - final frame/global 都是 `(T,17,1024)`。
   - concat 后是 `(T,17,2048)`。
   - 相同 teacher 时 loss 为零。
   - 交换两个 scene indices 后 loss 增大。
   - 单个反向 token 只贡献对应 index 的 cosine loss。
   - 非有限 student/teacher 立即报错。

7. **独立权重**
   - 验证：

```text
SFT + pre_weight × pre_loss + post_weight × post_loss
```

   - `pre=0, post>0`、`pre>0, post=0`、两者都启用分别正确。
   - 负权重被拒绝。

8. **Teacher 复用**
   - 两个 loss 都启用时 teacher encoder 每个 batch 只调用一次。
   - post-only 时仍调用一次。
   - eval、generation 和无 labels 时不调用。

9. **Checkpoint**
   - pre/post 参数保存、重载后逐项一致。
   - teacher 权重不进入 checkpoint。
   - 保存后的 config 只含新字段。
   - 仓库源码、脚本和文档不再存在独立旧字段 `DISTILL_WEIGHT/distill_weight`。

10. **GPU smoke test**
    - 单个 Qwen3.5-4B + VGGT-Omega batch。
    - 六层 capture、pre/post features、两个 loss 和总 loss 全部 finite。
    - 完成 backward 和一个 optimizer step。
    - Vision/teacher 无梯度。
    - pre module、post module、LLM 获得梯度。
    - 保存后重载，并完成一次 generation。
    - generation 不触发 post module 或 teacher。

验证命令：

```bash
cd /home/jackson/python/SceneDistill

env PYTHONPATH="$PWD/SpatialStack/src" \
  conda run -n OKT \
  pytest SpatialStack/tests/test_scene_distill.py -q

bash -n SpatialStack/scripts/train/train_scene_distill.sh
bash -n SpatialStack/scripts/train/train_scene_distill_multinode.sh
bash -n SpatialStack/scripts/train/train.sh
bash -n SpatialStack/scripts/train/train_multinode.sh

git diff --check
```

项目 `spatialstack` 环境额外执行完整 `py_compile` 和真实 GPU smoke test。

## 7. 实验与消融计划

最小对照矩阵：

| 条件 | PRE | POST | 目的 |
|---|---:|---:|---|
| Stage 1 回归 | 0.2 | 0 | 验证 Stage 2 关闭时完全保留 Stage 1 |
| Post-only loss | 0 | 0.05 | 测量输出端蒸馏的独立贡献 |
| Stage 1 + Stage 2 | 0.2 | 0.05 | 完整方法 |
| 两个 loss 关闭 | 0 | 0 | 区分蒸馏监督与 special-token/SFT 路径 |

所有对照固定：

- 相同初始化 checkpoint。
- 相同数据版本、样本顺序与 seed。
- 相同 world size、per-device batch 和实际 global batch。
- 相同 optimizer steps、warmup steps 和 learning rate。
- 相同 GPU 型号、软件版本和 git commit。
- 多节点与单节点不混入同一严格消融，或将拓扑作为实验元数据记录。

先运行 100–500 optimizer steps 的 paired pilot，检查：

- total loss 是否 finite。
- pre/post loss 是否稳定。
- gradient norm 是否异常。
- GPU memory 和 step time。
- Stage 1-only 与完整 Stage 2 是否从相同 SFT 初值起步。

只有在同一正式 benchmark、相同预算和多 seed 条件下报告 downstream 指标。至少报告均值与标准差；在完成实验前，不声明 Stage 2 带来性能提升。

## 8. 已锁定的假设

- Stage 2 Q 使用 Stage 1 最终 post-global 的 `17×1024` 状态。
- LLM K/V 是当前帧完整 image span，包含 17 个 special tokens 和 visual tokens，但不包含文本。
- Post-GCTE 不回写 LLM。
- Pre/post 参数和权重独立，但使用一次联合 forward 的正常梯度图。
- LLM 层固定为 0-based `[4,8,12,16,20,24]`，不增加运行时可配置层列表。
- Post loss 默认权重为 `0.05`。
- SceneDistill 母脚本保留当前 pre 默认 `0.2`。
- `DISTILL_WEIGHT/distill_weight` 执行硬重命名，不提供旧 alias 或 checkpoint 字段回退。
- 四个同步脚本包括 `train_scene_distill_multinode.sh`；请求中重复出现的第二个 `train_multinode.sh` 按已确认的母脚本笔误处理。
- 最终 Markdown 文件名为 `doc/Distillation_stage2.md`。
