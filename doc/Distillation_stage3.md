# SceneDistill Stage 3：Camera/Scene Internal Injection 实施方案

## 1. 目标与依据

最终文档创建为 `doc/Distillation_stage3.md`。实施基于当前干净工作区：

- SceneDistill：commit `738185a2154a8d0ecad7d6c1b00867d08bf051cb`
- 本地 SpatioLM：commit `9cdbee057fcb11e22ec7ba778d05b58dda470904`
- 论文：[SpatioLM，§3.2、Figure 3、Eq. (5)–(6)](https://arxiv.org/pdf/2608.01899#page=4)

事实优先级锁定为：用户本次要求 > 当前代码 > Stage 1/2 文档 > SpatioLM 可迁移原则。具体边界如下：

- 保留 `geometry_encoder_type="scene_distill"`、Stage 1 Pre-GCTE、17-token packing、VGGT-Omega teacher、Pre/Post cosine loss和现有训练参数。
- Stage 2 已固定 Post 层为 `(4,8,12,16,20,24)`，[scene_distill_module.py:19](/home/jackson/python/SceneDistill/SpatialStack/src/qwen_vl/model/scene_distill_module.py:19)，每层已有独立 Frame/Global attention，[scene_distill_module.py:334](/home/jackson/python/SceneDistill/SpatialStack/src/qwen_vl/model/scene_distill_module.py:334)。
- Stage 2 当前在 decoder block 前捕获 hidden state，[modeling_qwen3_5.py:405](/home/jackson/python/SceneDistill/SpatialStack/src/qwen_vl/model/modeling_qwen3_5.py:405)，并在完整 LLM forward 后离线执行 Post-GCTE，[modeling_qwen3_5_scene_distill.py:395](/home/jackson/python/SceneDistill/SpatialStack/src/qwen_vl/model/modeling_qwen3_5_scene_distill.py:395)。Stage 3 将其改为目标 block 输出后的在线交互与回写。
- SpatioLM 同样在 LM block 后执行 zero-projected residual，[qwen3.py:332](/home/jackson/python/spatio-lm/src/spatiolm/models/modules/qwen3.py:332)、[qwen3.py:343](/home/jackson/python/spatio-lm/src/spatiolm/models/modules/qwen3.py:343)，并显式零初始化相关投影，[qwen3.py:382](/home/jackson/python/spatio-lm/src/spatiolm/models/modules/qwen3.py:382)。
- 不照搬 SpatioLM 的 `in_zero_proj`、全视觉 token 注入、冻结整个 VLM或其默认层号。SceneDistill 只采用“一层一组 side block、输出端零投影、残差注入”的原则；17-token Q、完整 image-span K/V和当前训练策略仍以 Stage 2 为准。

当前 `SceneDistillPreModule`、`SceneDistillPostModule` 命名已经系统化，且名称参与 checkpoint 发现，[modeling_qwen3_5.py:44](/home/jackson/python/SceneDistill/SpatialStack/src/qwen_vl/model/modeling_qwen3_5.py:44)。因此不重命名模块、不新增源码模块，只删除被在线 injection 取代的离线 capture 代码。

## 2. Stage 3 数据流与数学契约

设总帧数为 $T$，每帧 special token 数为 $S=17$，旁侧维度为 $D_s=1024$，LLM hidden size 为 $D_l$，目标层为：

$$
\ell_m\in(4,8,12,16,20,24),\qquad m=0,\ldots,5.
$$

Stage 1 保持原样：

$$
Z^{(-1)}=Z_{\mathrm{pre}}\in\mathbb{R}^{T\times17\times1024}.
$$

在目标 decoder block $\ell_m$ 完成计算后、最终 RMSNorm 前，取得该层输出中的完整 image spans：

$$
H_{\mathrm{img}}^{(\ell_m)}
=
[\text{camera},\text{scene}_{1:16},\text{visual}_{1:P_f}]
\in\mathbb{R}^{\sum_f(17+P_f)\times D_l}.
$$

该顺序由前置 packing [vggt_omega_direct_packing.py:292](/home/jackson/python/SceneDistill/SpatialStack/src/qwen_vl/model/vggt_omega_direct_packing.py:292) 和 direct-only mask [vggt_omega_direct_packing.py:323](/home/jackson/python/SceneDistill/SpatialStack/src/qwen_vl/model/vggt_omega_direct_packing.py:323) 保证。

每个 Post stage 依次执行：

$$
\widetilde Z^{(m)}
=
\operatorname{FrameCrossAttn}^{(m)}
\left(
Z^{(m-1)},H_{\mathrm{img}}^{(\ell_m)}
\right),
$$

$$
Z^{(m)}
=
\operatorname{GlobalSelfAttn}^{(m)}
\left(\widetilde Z^{(m)}\right).
$$

其中：

- Q 是对应帧的 `17×1024` 旁侧 token。
- K/V 是同一帧的完整 `[17 special + visual]` block 输出。
- Frame attention 的分帧、变长 token 和 LayerScale 结构复用现有实现，[scene_distill_module.py:89](/home/jackson/python/SceneDistill/SpatialStack/src/qwen_vl/model/scene_distill_module.py:89)。
- Global attention 继续仅处理各视频内部的 `T_video×17` special tokens，[scene_distill_module.py:172](/home/jackson/python/SceneDistill/SpatialStack/src/qwen_vl/model/scene_distill_module.py:172)。

每个 stage 新增独立投影：

$$
P_m:\mathbb{R}^{1024}\rightarrow\mathbb{R}^{D_l},
\qquad W_{P_m}=0,
$$

并仅写回该层的 17 个对应位置：

$$
H^{(\ell_m)}[I_{\mathrm{special}}]
\leftarrow
H^{(\ell_m)}[I_{\mathrm{special}}]
+
P_m(Z^{(m)}).
$$

实现约束：

- `I_special` 必须使用 `_direct_only_mask`，不能使用覆盖 visual patches 的 `image_mask_2d`。
- `(T,17,D_l)` 按 batch、frame、special-index 顺序 flatten，与 boolean mask 的顺序逐项对应。
- 先 `hidden_states.clone()` 再赋值，避免在计算图上进行危险的原位修改；现有 geometry fusion 已采用同一模式，[modeling_qwen3_5.py:434](/home/jackson/python/SceneDistill/SpatialStack/src/qwen_vl/model/modeling_qwen3_5.py:434)。
- 注入源固定为每级 Global attention 后的 $Z^{(m)}$，不是 `post_after_frame`，也不是二者拼接后的 2048-D feature。
- 不增加独立标量 $\gamma_m$。零初始化的 $P_m$ 本身承担 gated residual 的初始关闭作用，符合[空间智能方案.md:163](/home/jackson/python/SceneDistill/doc/空间智能方案.md:163)和用户指定的零投影机制。
- 更新后的 LLM special tokens继续进入更深 decoder；更新后的旁侧 $Z^{(m)}$ 同时作为下一 Post stage 的 Q，从而形成闭环渐进演化。

最终 Post loss仍使用：

$$
F_{\mathrm{post}}
=
\operatorname{Concat}
\left(\widetilde Z^{(5)},Z^{(5)}\right)
\in\mathbb{R}^{T\times17\times2048},
$$

并复用当前逐 index cosine loss，[scene_distill_module.py:408](/home/jackson/python/SceneDistill/SpatialStack/src/qwen_vl/model/scene_distill_module.py:408)。不增加逐层 loss或第三个蒸馏项。

## 3. 代码实施

### 3.1 `scene_distill_module.py`

扩展现有 `SceneDistillPostModule`，不新增类：

- 新增 `post_injection_projections`，包含 6 个独立的 `nn.Linear(1024, D_l, bias=False)`，顺序严格对应 `(4,8,12,16,20,24)`。
- 保留 `post_frame_layers`、`post_global_layers` 和全部 Stage 2 attention 参数。
- 调整初始化顺序：现有 attention/projector Linear 仍按 Xavier 初始化，[scene_distill_module.py:344](/home/jackson/python/SceneDistill/SpatialStack/src/qwen_vl/model/scene_distill_module.py:344)；之后单独对六个 injection projection 执行 `nn.init.zeros_`，防止通用初始化覆盖零门。
- 将当前“一次接收六层 features”的 `forward` 改为在线单-stage接口：

```python
forward(
    stage_index,
    post_tokens,
    llm_layer_features,
    frame_sizes,
    video_sizes,
) -> (
    post_after_frame,
    post_after_global,
    injection_delta,
)
```

- `stage_index` 必须处于 `[0,5]`；`llm_layer_features` 必须是当前目标 block 输出的完整 flattened image span。
- 返回的 `injection_delta` 为对应 `post_injection_projections[stage_index](post_after_global)`。
- 删除当前离线六层循环 [scene_distill_module.py:392](/home/jackson/python/SceneDistill/SpatialStack/src/qwen_vl/model/scene_distill_module.py:392)，因为它无法让第 $m$ 次 injection 影响更深 LLM输出。

### 3.2 `modeling_qwen3_5.py`

把 Stage 2 的选择性 capture 改为可选在线 Post/injection 接口。删除：

```python
capture_hidden_state_layers
capture_hidden_state_mask
```

新增仅供 SceneDistill wrapper 使用的内部参数：

```python
scene_distill_post_module=None
scene_distill_post_tokens=None
scene_distill_image_mask=None
scene_distill_special_mask=None
scene_distill_frame_sizes=None
scene_distill_video_sizes=None
return_scene_distill_post_features=False
```

执行方式：

1. 参数要么全部关闭，要么完整提供；其他 geometry 类型继续走原路径。
2. decoder block先按现有逻辑运行，[modeling_qwen3_5.py:408](/home/jackson/python/SceneDistill/SpatialStack/src/qwen_vl/model/modeling_qwen3_5.py:408)。
3. 当 `layer_idx` 匹配 Post 模块的固定层号时：

   - 用 `scene_distill_image_mask`提取完整 image span。
   - 调用对应 Post stage。
   - 用 `scene_distill_special_mask`只更新 camera/scene位置。
   - 将更新后的 hidden states送入下一 decoder block。

4. 第六级结束后，若 `return_scene_distill_post_features=True`，通过内部 `outputs.hidden_states` 返回 `(final_post_after_frame, final_post_after_global)`；外层 wrapper立即消费，不将它们暴露到最终生成输出。
5. 最终 RMSNorm仍位于全部六次注入之后，[modeling_qwen3_5.py:441](/home/jackson/python/SceneDistill/SpatialStack/src/qwen_vl/model/modeling_qwen3_5.py:441)。
6. 非 SceneDistill 调用不提供这些参数，执行路径和输出保持不变。

### 3.3 `modeling_qwen3_5_scene_distill.py`

调整 wrapper 调度，不改输入、config或 geometry 注册：

- Pre-GCTE、teacher收集、packing、MRoPE、label扩展均保持不变。
- 继续构建：

  - `image_mask_2d`：完整 `[17 special + visual]` span；
  - `_direct_only_mask`：每帧开头精确17个 special位置；
  - `llm_frame_sizes = 17 + merged_frame_size`。

  当前验证入口见 [modeling_qwen3_5_scene_distill.py:360](/home/jackson/python/SceneDistill/SpatialStack/src/qwen_vl/model/modeling_qwen3_5_scene_distill.py:360)和[modeling_qwen3_5_scene_distill.py:373](/home/jackson/python/SceneDistill/SpatialStack/src/qwen_vl/model/modeling_qwen3_5_scene_distill.py:373)。

- 只要处于首个 multimodal prefill，即始终把 Post 模块、`pre_global_tokens`、两个 masks和frame/video sizes传给 language model，不再以 `compute_post_distill_loss` 控制 Post 是否运行。
- `compute_post_distill_loss` 只控制：

  - 是否要求返回最终 Post features；
  - 是否运行 teacher；
  - 是否计算并加上 $L_{\mathrm{post}}$。

- 训练时 teacher仍只在 Pre/Post 任一 loss启用时调用一次，[modeling_qwen3_5_scene_distill.py:299](/home/jackson/python/SceneDistill/SpatialStack/src/qwen_vl/model/modeling_qwen3_5_scene_distill.py:299)。
- Evaluation/generation首个视觉 prefill运行 Post/injection，但不构建 VGGT-Omega teacher；现有 LMMS-Eval adapter已经强制 student-only，[qwen3_5_scene_distill.py:46](/home/jackson/python/SceneDistill/SpatialStack/src/lmms_eval/models/qwen3_5_scene_distill.py:46)。
- 后续 autoregressive decode没有 `pixel_values/image_grid_thw`，不重复运行 Post；首轮注入产生的状态已经进入更深层 hidden states和KV cache。这与 SpatioLM只在尚无目标层 cache 时运行 condition的约束一致，[qwen3.py:314](/home/jackson/python/spatio-lm/src/spatiolm/models/modules/qwen3.py:314)。
- 总 loss保持当前公式和权重分支，[modeling_qwen3_5_scene_distill.py:529](/home/jackson/python/SceneDistill/SpatialStack/src/qwen_vl/model/modeling_qwen3_5_scene_distill.py:529)。

### 3.4 Checkpoint与训练兼容

不修改 config、训练脚本或评估脚本：

- 新增 state-dict keys：

```text
model.scene_distill_post_module.post_injection_projections.{0..5}.weight
```

- 这些 keys 已天然被 `"scene_distill_post_module"` checkpoint过滤规则覆盖，[modeling_qwen3_5.py:47](/home/jackson/python/SceneDistill/SpatialStack/src/qwen_vl/model/modeling_qwen3_5.py:47)，无需修改 checkpoint发现代码。
- Stage 2 checkpoint缺少新 keys时按现有 `strict=False` 子模块加载机制载入，[modeling_qwen3_5.py:267](/home/jackson/python/SceneDistill/SpatialStack/src/qwen_vl/model/modeling_qwen3_5.py:267)；六个缺失投影保持显式零初始化，因此初始 LLM输出等价于 Stage 2。
- Stage 3 checkpoint必须完整保存并恢复六个投影；冻结 teacher仍由 `remove_teacher_weights` 排除。
- 当前训练逻辑会把整个 Post module设为可训练，[train_qwen.py:125](/home/jackson/python/SceneDistill/SpatialStack/src/qwen_vl/train/train_qwen.py:125)、[train_qwen.py:133](/home/jackson/python/SceneDistill/SpatialStack/src/qwen_vl/train/train_qwen.py:133)，因此新投影自动进入训练参数，无需脚本接线。
- LLM是否训练继续由现有 `tune_mm_llm` 决定，[train_qwen.py:109](/home/jackson/python/SceneDistill/SpatialStack/src/qwen_vl/train/train_qwen.py:109)；不照搬 SpatioLM冻结整个VLM的策略。

## 4. 测试与验收

扩展现有 `SpatialStack/tests/test_scene_distill.py`，不新建测试框架。

### 结构与初始化

- 六个 projection实例互不共享参数，shape均为 `(D_l,1024)`，无bias。
- 构造、外层 `post_init/reset_parameters`、Stage 2 checkpoint加载后三个时点，权重都必须精确为零。
- 新 Stage 3 checkpoint round trip后，投影权重逐项一致且不再为零。
- Pre/Post attention仍参数独立；不新增camera/scene初始化参数。

### 时序与索引

- Dummy decoder验证 Post读取的是 0-based目标 block执行后的输出，而不是该层输入或最终RMSNorm输出。
- 六次调用顺序严格为 `(4,8,12,16,20,24)`，每层恰好调用一次。
- 第 $m$ 层 injection必须影响第 $m+1$ 个目标层看到的LLM features，证明是在线闭环而非离线旁路。
- 每帧 injection严格映射camera index `0`、scene indices `1:17`；跨batch、不同视频长度和不同patch数量时顺序不漂移。
- 修改一个projection或一帧旁侧tokens，只影响对应special位置；同层visual和text位置不直接被写入。

### 零初始化等价性

- 使用相同Stage 2参数和输入，六个projection全零时：

  - 注入前后 target-layer hidden states相同；
  - 最终 hidden state、logits和首轮KV cache相同；
  - Post side path仍可计算，但对主干为零扰动。

- 该测试是“初始保持预训练/Stage 2行为”的核心验收，不以训练loss下降替代。

### 训练、梯度与loss

- 初始零投影时，SFT梯度能够到达projection权重；由于零门，SFT对更早Post attention的梯度可暂为零。
- Post loss启用时，Frame/Global attention即使在零门初始状态也必须获得梯度。
- 将projection设为非零后，SFT梯度必须进一步到达Post attention和`pre_global_tokens`。
- Vision Encoder和VGGT-Omega teacher仍无梯度。
- Post最终 feature仍为最后一级 frame/global拼接的 `(T,17,2048)`，总loss仍精确满足：

$$
L_{\mathrm{SFT}}
+\lambda_{\mathrm{pre}}L_{\mathrm{pre}}
+\lambda_{\mathrm{post}}L_{\mathrm{post}}.
$$

### Train/Eval/Generation phase gate

- `train + labels`：Post/injection始终运行；teacher是否运行只取决于两个distillation weights。
- `eval`或`labels=None`：Post/injection仍运行，teacher和两项distillation loss不运行。
- 后续cached decode step：Post/injection调用次数为0，不重复写入。
- `POST_DISTILL_WEIGHT=0`不得关闭injection，只关闭Post teacher loss。
- 非`scene_distill` geometry路径完全不进入新增逻辑。

### 验证命令

当前基线已实测为 `22 passed, 7 skipped`；7项跳过原因是 `OKT` 环境缺少Qwen3.5 Transformers runtime。实施后依次执行：

```bash
cd /home/jackson/python/SceneDistill

env PYTHONPATH="$PWD/SpatialStack/src" \
  conda run -n OKT \
  pytest SpatialStack/tests/test_scene_distill.py -q -rs

bash -n SpatialStack/scripts/train/train_scene_distill.sh
bash -n SpatialStack/scripts/train/train_scene_distill_multinode.sh
bash -n SpatialStack/scripts/train/train.sh
bash -n SpatialStack/scripts/train/train_multinode.sh

git diff --check
```

随后在包含Qwen3.5的 `spatialstack` 环境执行：

- 完整测试且不得因Qwen3.5 runtime缺失而skip。
- 单GPU Qwen3.5-4B + VGGT-Omega训练batch：forward、backward、optimizer step、两个loss和六次injection全部finite。
- 保存Stage 3 checkpoint后，在不加载teacher的条件下完成一次generation和LMMS-Eval smoke test。
- 记录Stage 2与Stage 3的峰值显存、prefill时间和generation token latency；Post现在会在评估首轮运行，因此必须量化新增开销。

## 5. 实验与消融边界

不为消融增加生产配置项，使用测试注入或固定checkpoint副本完成以下对照：

| 条件 | Injection projection | Post loss | 目的 |
|---|---|---:|---|
| Stage 2基线 | 不执行 | 现有值 | 原始参照 |
| 零门等价对照 | 固定为0 | 现有值 | 验证计算链存在但主干初始行为不变 |
| Stage 3完整方法 | 可训练 | 0.05 | 测量internal injection贡献 |
| 推理关闭注入 | 训练后临时置0 | 与推理无关 | 验证收益是否确由回写产生 |
| 无Post监督 | 可训练 | 0 | 区分SFT驱动的注入与Post蒸馏监督 |

所有对照固定初始化Stage 2 checkpoint、数据顺序、seed、world size、实际global batch、optimizer steps、learning rate、GPU型号、软件版本和评估协议。报告均值、标准差、显存及吞吐；没有完成同预算多seed评估前，不声明Stage 3带来性能提升。

## 6. 明确不改与验收结论

- 不修改geometry注册、factory、VGGT-Omega、packing规则、训练脚本、评估脚本、inference dispatcher或LMMS-Eval模型名。
- 不新增projection配置、layer配置、scalar gate、逐层loss、teacher cache或新源码模块。
- 不改变17-token顺序、Pre/Post loss权重、teacher抽取、labels、attention mask、MRoPE或SFT公式。
- 不重命名Pre/Post模块；仅删除不再正确的离线capture接口及对应测试。
- `doc/Distillation_stage3.md` 必须记录上述来源快照、公式、接口、phase gate、checkpoint兼容、测试和消融边界，并通过Markdown链接检查与 `git diff --check`。
