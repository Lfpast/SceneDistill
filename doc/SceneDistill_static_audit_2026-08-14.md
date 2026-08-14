# SceneDistill 静态证据审计（2026-08-14）

> 审计范围：当前工作树 `8946b82`、`doc/` 的全部阶段文档、训练/评估脚本、`performance.xlsx`。本机没有可用的模型运行环境；以下把**代码已证实**、**实验表观察**与**待运行假设**分开，不把静态检查写成运行结论。

## 结论先行

1. **没有发现一个能静态证明“Pre/Post 张量接错或 Stage3 写错位置”的致命实现错误。**当前实现对 17-token 数量、每 logical video 的 teacher 对齐、视频 placeholder 长度、special mask 与注入维度均有 fail-fast 校验；Stage3 确实在 decoder block 输出后，只回写 special positions。【[C1](/home/jackson/python/SceneDistill/SpatialStack/src/qwen_vl/model/modeling_qwen3_5_scene_distill.py:178)】【[C2](/home/jackson/python/SceneDistill/SpatialStack/src/qwen_vl/model/modeling_qwen3_5.py:480)】
2. **当前最先要排除的是实验契约错误，不是再改网络。**默认 `VGGT-direct` 训练为 `back`，默认评估为 `front`；这会改变每一帧 visual span 内 17 个 token 的相对位置。任何沿用脚本默认值的 direct checkpoint 都可能被错误位置评估。【[C3](/home/jackson/python/SceneDistill/SpatialStack/scripts/train/train_vggt_direct.sh:61)】【[C4](/home/jackson/python/SceneDistill/SpatialStack/scripts/evaluation/eval_qwen35_vggt_direct.sh:11)】
3. **原生视频重构同时改动了采样、Qwen temporal grouping/MRoPE、teacher 时间对齐、上下文长度和评估帧数。**因此“new dataflow 的 direct 更差”尚不能归因于 dataflow 或任何一个蒸馏模块；它是一个尚未做单变量回归的系统变化。【[C5](/home/jackson/python/SceneDistill/doc/Dataflow_refactor.md:25)】
4. `performance.xlsx` 支持“direct 有潜力、Stage2/ver2 有局部恢复、后续模型跨 benchmark 不稳定”的叙述，但**不支持模块因果结论**：每行缺少 commit、数据快照、seed、有效 global batch、训练步数、checkpoint、token 位置和 eval 参数，且没有多 seed 方差。

因此建议立刻冻结架构；先完成后文的 **Gate 0–2**。在这三关前，不应继续训练 Stage3 或重设计 Pre/Post。

---

## 1. 开发历程：代码与结果能支持的版本

1. 项目以 [SpatialStack](https://github.com/jzh15/SpatialStack) 的 geometry injection 为出发点：选择几何视觉层，把几何信息写入 LLM visual-token 通路。
2. [VGGT-Omega](https://github.com/facebookresearch/vggt-omega) 提供 camera/scene token；项目先把它们作为 17 个 `special17` token 与 Qwen visual token 拼接，验证教师几何信息本身是否有效。
3. 表中 `VGGT-direct` 相比 `SpatialStack` 在 CVBench、SparBench、VideoMME、MMSI、Blink-Spatial 更高（85.71/69.46/65.11/.294/.572 对 85.45/68.73/64.00/.282/.516），但 VSI 较低（65.70 对 67.26）。这证明“该 direct 配置可以有效”，不证明所有位置/数据流都有效。
4. **Stage1 / Pre-distillation**：取视觉编码器第 1/5/9/13 层，special token 对本帧 visual token cross-attention、再按 logical video self-attention；输出 17 token 拼到 LLM 头部，并以 teacher 特征计算 pre cosine loss。视觉特征会 `detach()`，因而该 loss 不会训练 Qwen vision encoder。【[C6](/home/jackson/python/SceneDistill/SpatialStack/src/qwen_vl/model/scene_distill_module.py:290)】
5. **Stage2 / Post-distillation（原设计）**：从 LLM 第 4/8/12/16/20/24 层的输入读取 `(17 + visual)` token，special 与它们交互并计算 post cosine loss；文档定义它为只读辅助监督，不回写 LLM。【[C7](/home/jackson/python/SceneDistill/doc/Distillation_stage2.md:99)】
6. **Stage3 / online injection（当前代码）**：Post 每个阶段把 special token 通过可训练 `1024 -> D_llm` projector 投影；projector 零初始化，decoder 每个指定 block 后将 delta 加回 special positions。它不再是 Stage2 的只读辅助模块，而是直接改变下游 LLM 隐状态的反馈系统。【[C8](/home/jackson/python/SceneDistill/SpatialStack/src/qwen_vl/model/scene_distill_module.py:336)】【[C2](/home/jackson/python/SceneDistill/SpatialStack/src/qwen_vl/model/modeling_qwen3_5.py:480)】
7. 随后项目把 SpatialStack 的 image-list 流改为 Qwen3.5 原生 `pixel_values_videos`、`video_grid_thw`、`mm_token_type_ids`，以保留视频 MRoPE；SceneDistill 明确拒绝 image fields。【[C9](/home/jackson/python/SceneDistill/SpatialStack/src/qwen_vl/model/modeling_qwen3_5_scene_distill.py:283)】
8. 当前训练目标是 `SFT + λ_pre L_pre + λ_post L_post`；teacher 只在训练且需要 distill loss 时载入，SceneDistill 的评估路径可以 student-only。【[C10](/home/jackson/python/SceneDistill/SpatialStack/src/qwen_vl/model/modeling_qwen3_5_scene_distill.py:337)】

## 2. 表中实际信号，而非结论

下表只取最能表示用户描述的单个行；六项指标量纲不同，不能平均。`Δ` 相对 `VGGT-direct`。

| 运行标签 | CV | VSI | Spar | VideoMME | MMSI | Blink | 静态解读 |
|---|---:|---:|---:|---:|---:|---:|---|
| SpatialStack | 85.45 | 67.26 | 68.73 | 64.00 | .282 | .516 | 参照 |
| VGGT-direct | 85.71 | 65.70 | 69.46 | 65.11 | .294 | .572 | direct 强基线 |
| stage1-005 | 85.37 | 63.41 | 68.83 | 64.96 | .269 | .579 | 除 Blink 外多项回落 |
| ver2-02 | **85.96** | **65.94** | 69.99 | 64.19 | **.310** | .528 | 部分指标超过 direct，非全指标胜出 |
| stage3-01 | 82.53 | 62.51 | 65.42 | 62.63 | .271 | **.701** | 主指标大幅回落、Blink 异常抬升 |
| SceneDistill-01 | 84.53 | 63.08 | 68.34 | 62.81 | .284 | .653 | Blink 高，其他基准低于 direct |

表中没有能唯一标识“native/new-dataflow direct”的 commit 或配置行。因此它不能量化第五点（重构后的 direct 变差）；需要从实际 checkpoint/run log 补录，而不是把任意 `VGGT-direct` 行当作新数据流结果。

---

## 3. 逐模块静态筛查

| 模块 | 已验证的正确约束 | 风险与判定 |
|---|---|---|
| 数据集/processor | 每个 logical video 保留为一个 video；collator 拼接 native patches/grid，并检查 grid 数与 teacher 输入数相等。【[C11](/home/jackson/python/SceneDistill/SpatialStack/src/qwen_vl/data/data_qwen.py:536)】 | processor 按 conversation message 分别调用后再拼接；真实 `AutoProcessor` 对多 message、多视频、奇数帧的 patch 顺序尚无集成测试。**待证伪**。 |
| 采样与 MRoPE | 训练默认 8–16 帧、8192 token；原生 video 字段进入 Qwen 的 video feature/MRoPE 路径。【[C12](/home/jackson/python/SceneDistill/SpatialStack/scripts/train/train.sh:43)】 | 评估默认 32 帧、12800 token。原生 temporal grouping 下，这是实质性 train/eval shift。**代码已证实，效果待证伪**。【[C20](/home/jackson/python/SceneDistill/SpatialStack/scripts/evaluation/eval_qwen35_vggt_direct.sh:14)】 |
| VGGT-Omega teacher | 按每个 video 独立对齐：1:1、`2T -> T` 相邻均值、`S>T` adaptive average pool，`S<T` 直接报错；没有跨视频 pooling。【[C13](/home/jackson/python/SceneDistill/SpatialStack/src/qwen_vl/model/modeling_qwen3_5_scene_distill.py:215)】 | Qwen 内部 temporal grouping 与 teacher 采样是否真对应同一时间窗，只能以固定样本的 indices/grid runtime dump 验证。**待证伪**。 |
| Pre | 4 个视觉层、frame-local cross-attention、video-local self-attention、17-token shape check；视觉层被 detach，训练边界清楚。【[C6](/home/jackson/python/SceneDistill/SpatialStack/src/qwen_vl/model/scene_distill_module.py:290)】 | 没有显见 shape/order bug；但若 teacher 时间窗不对，pre loss 会稳定地蒸馏错误配对。**依赖上项验证**。 |
| 拼接/position | expand 后会验证每个 temporal group 恰有 17 special positions，再重新计算 MRoPE delta。【[C14](/home/jackson/python/SceneDistill/SpatialStack/src/qwen_vl/model/modeling_qwen3_5_scene_distill.py:379)】 | 验证的是模型看到的 `video_mask`，不能证明 processor 构造的真实 runs 与“每组一帧”假设一致。**待证伪**。 |
| Post + loss | Post 接收 masked expanded video tokens，最后特征再与同一 teacher target 算 loss；Pre/Post 权重独立。 | Stage2 文档仍描述只读，而当前主干始终具备 injection 模块；分析 checkpoint 时必须记录是否加载/启用 injection，不能只写“stage2”。 |
| Stage3 injection | 注入 projector 零初始化，且只将 delta 写回 special mask；写入点在 `decoder_layer` 调用之后，机制与 Stage3 文档要求一致。【[C8](/home/jackson/python/SceneDistill/SpatialStack/src/qwen_vl/model/scene_distill_module.py:348)】【[C2](/home/jackson/python/SceneDistill/SpatialStack/src/qwen_vl/model/modeling_qwen3_5.py:490)】 | 六次在线残差会改变更深层 token；这是新因果路径，不是“小修 Stage2”。它是最强的性能下降候选之一，但**不是静态 bug 证据**。 |
| checkpoint/eval | Scene eval 可不加载 teacher；训练需要 loss 才取 teacher。【[C10](/home/jackson/python/SceneDistill/SpatialStack/src/qwen_vl/model/modeling_qwen3_5_scene_distill.py:337)】 | 必须确认 Stage2 checkpoint 缺失 inject 权重时保持全零，及 eval 是否加载了预期 config。当前无真实 checkpoint load smoke test。 |
| 测试 | 纯 Python 语法、六个相关 shell script 的 `bash -n` 通过；工作树除用户的 `performance.xlsx` 外干净。 | 关键 Qwen 集成测试已与 native migration 脱节：测试调用已不存在的 `_validate_expanded_image_spans`，而现代码只有 `_validate_expanded_video_spans`。【[C15](/home/jackson/python/SceneDistill/SpatialStack/tests/test_scene_distill.py:1012)】【[C1](/home/jackson/python/SceneDistill/SpatialStack/src/qwen_vl/model/modeling_qwen3_5_scene_distill.py:178)】默认 Python 没有 `pytest`，Conda 也没有可写 env，故本机未能运行 pytest；不能把历史 unit-test 绿灯当作当前 native 路径证明。 |

## 4. 优先级排序：先查什么，为什么

| 优先级 | 已知事实 / 假设 | 为什么足以阻断当前结论 | 最小决定性检查 |
|---|---|---|---|
| P0 | **已证实：direct 默认 train=`back`，eval=`front`** | 同一 checkpoint 的 token 相对位置被改变；直接污染 direct 基线及重构前后比较。文档也明示必须覆盖。【[C16](/home/jackson/python/SceneDistill/doc/recover_plan.md:146)】 | 对**同一 checkpoint**仅切换 `GEOMETRY_TOKEN_INSERT_POSITION=back/front` 重跑六 benchmark；先确认训练实际配置。 |
| P0 | **已证实：train 16 帧/8k，eval 32 帧/12.8k** | 这不是小参数变化：token 数、temporal group 数、位置范围、帧采样都变了。 | 固定 checkpoint、token position、任务和解码参数，只比较 eval 16/8192 与 32/12800。若差异大，先统一协议再谈模型。 |
| P0 | **已证实：实验身份缺失且配置漂移** | 文档的 Scene 默认是 `.2/.05`，单机脚本 `.05/.05`，多机脚本 `.1/.1`；同名 Stage 不代表同一训练条件。【[C17](/home/jackson/python/SceneDistill/doc/Distillation_stage2.md:257)】【[C18](/home/jackson/python/SceneDistill/SpatialStack/scripts/train/train_scene_distill.sh:43)】【[C19](/home/jackson/python/SceneDistill/SpatialStack/scripts/train/train_scene_distill_multinode.sh:45)】 | 为每个已有 checkpoint 补一个不可变 manifest：commit、config、数据 manifest hash、seed、world size、per-device batch、accumulation、optimizer steps、训练/评估帧数与位置。无法补齐的行降级为探索性结果。 |
| P1 | **待证伪：Stage3 online feedback 是退化来源** | Stage2 的 read-only loss 与 Stage3 六次写回不是可直接比较的消融。零初始化只保证**起点**近似等价，不保证训练后稳定。 | 同一 Stage2 checkpoint：`inject=0` 输出必须与 Stage2 参考完全一致（logits tolerance）；再仅打开 injection 评估。之后才训练 full Stage3。 |
| P1 | **待证伪：teacher/Qwen 时间窗错配** | 对齐代码的形状合法不等于语义帧一致；一旦错配，两个 distill loss 都会被系统性误导。 | 一个奇数帧、一个多 logical-video fixture：记录原始 indices、processor `video_grid_thw`、每 group 对应 teacher frame/index、17-token span。逐项人工核对。 |
| P1 | **已证实：关键集成测试陈旧** | 当前测试无法保护 native video、processor、MRoPE 和 eval 配置；任何“模块没问题”的判断都缺运行证据。 | 在有依赖的环境更新/运行真实 processor forward、1-step loss/backward、generate/cache、checkpoint reload 四类测试。 |

## 5. 不重训前的 Gate 路线图

### Gate 0：恢复可比较的 direct 基线（最高优先）

1. 选一个确定由 `train_vggt_direct.sh` 训练的 checkpoint，读取其启动命令/W&B config，确认 token 位置。
2. 以完全相同的 checkpoint、任务、few-shot、生成参数，分别评估 `back` 与 `front`；训练默认产物必须先用 `back`。不修改权重、不重训。
3. 固定 token position，再测 `16 frames + 8192` 与 `32 frames + 12800`。若这两项任一改变显著影响结果，历史 direct/SceneDistill 比较先按统一协议重算。

**通过条件：** 选定一个可复现的 direct 主基线，并把完整 model args 写入结果表。**失败含义：** 先修评估协议/记录，不碰 Pre/Post/Stage3。

### Gate 1：原生视频契约 fixture

为以下三种不训练样本保存一份 JSON/pt trace：单 16 帧、单奇数帧、多 logical video。

- 原始采样 indices 与 `VideoMetadata`；
- `pixel_values_videos.shape`、`video_grid_thw`、`mm_token_type_ids`；
- 每 video 的 Qwen temporal groups、teacher source/target frames；
- expand 前后 input length、每 group visual span 长度、17-token special mask；
- MRoPE position-id 的 temporal 行和 `rope_deltas`。

**通过条件：** 每个 logical video 独立、无跨视频 token/teacher pooling；每 group 的 special span 恰为 17；训练与评估的相同 fixture trace 一致（除明确指定的帧数差异）。

### Gate 2：先证明 Stage2 的 null model

以已知 Stage2 checkpoint 在当前代码加载：强制全部 `inject` 权重为零，检查 logits、loss、生成首 token 与对应只读参考的一致性。随后在同一 checkpoint 上只切换 injection 开/关评估，得到 **inference-only delta**。这一步将“Stage3 是训练失败”与“Stage3 是结构性推理干扰”分离。

### Gate 3：最小、可解释的训练矩阵

仅在 Gate 0–2 全部通过后，固定 data snapshot、commit、init checkpoint、effective global batch、optimizer steps、LR schedule、帧数、token position、eval protocol，至少三 seed：

| 组 | Pre | Post loss | Injection | 回答的问题 |
|---|---:|---:|---:|---|
| D0 | 0 | 0 | 0 | 当前 native SFT/direct 对照 |
| D1 | 1 | 0 | 0 | Pre 是否单独有益 |
| D2 | 1 | 1 | 0 | read-only Post 是否单独有益 |
| D3 | 1 | 1 | 1 | online injection 的净因果效应 |

先报告每个 benchmark、平均/标准差、训练曲线、pre/post cosine、梯度范数和有效 token/frame 分布；不要用 Blink 的单项高分替代跨 benchmark 结论。

## 6. 现在不建议做的事

- 不要立刻把 Stage3 改成另一个 gate/projector，或继续增加视觉层/损失；当前证据不足以指向这些模块。
- 不要把 xlsx 中不同命名、不同权重、不同数据流的单行结果排序成“方法进化曲线”。
- 不要把静态 shape test 或旧 image-path unit test 当成 native Qwen3.5 video runtime 证明。

## 7. 可保留的核心判断

你的原始判断并没有被结果否定：direct 在大多数主指标上超过 SpatialStack，说明 VGGT-Omega 的 camera/scene 信号对该模型族有价值；Stage2/ver2 的局部恢复也说明“把几何信息压进 LLM”并非没有希望。现在真正缺的不是第三个大模块，而是一条**位置、帧数、数据版本、checkpoint 和评估完全闭合的因果证据链**。先把 direct 基线和 native-video 契约钉死，才有资格审判蒸馏与 injection。

## 证据索引

- C1: [native span 校验](/home/jackson/python/SceneDistill/SpatialStack/src/qwen_vl/model/modeling_qwen3_5_scene_distill.py:178)
- C2: [decoder 后 Stage3 回写](/home/jackson/python/SceneDistill/SpatialStack/src/qwen_vl/model/modeling_qwen3_5.py:480)
- C3: [direct train 默认 back](/home/jackson/python/SceneDistill/SpatialStack/scripts/train/train_vggt_direct.sh:61)
- C4: [direct eval 默认 front](/home/jackson/python/SceneDistill/SpatialStack/scripts/evaluation/eval_qwen35_vggt_direct.sh:11)
- C5: [原生 video 重构范围](/home/jackson/python/SceneDistill/doc/Dataflow_refactor.md:25)
- C6: [Pre 的逐层处理与 visual detach](/home/jackson/python/SceneDistill/SpatialStack/src/qwen_vl/model/scene_distill_module.py:290)
- C7: [Stage2 的只读设计](/home/jackson/python/SceneDistill/doc/Distillation_stage2.md:99)
- C8: [inject 零初始化](/home/jackson/python/SceneDistill/SpatialStack/src/qwen_vl/model/scene_distill_module.py:336)
- C9: [SceneDistill 只接受 native video](/home/jackson/python/SceneDistill/SpatialStack/src/qwen_vl/model/modeling_qwen3_5_scene_distill.py:283)
- C10: [teacher/loss 边界](/home/jackson/python/SceneDistill/SpatialStack/src/qwen_vl/model/modeling_qwen3_5_scene_distill.py:337)
- C11: [collator 的 video/teacher 数量校验](/home/jackson/python/SceneDistill/SpatialStack/src/qwen_vl/data/data_qwen.py:536)
- C12: [训练 video 帧数默认](/home/jackson/python/SceneDistill/SpatialStack/scripts/train/train.sh:43)
- C13: [teacher 时序对齐](/home/jackson/python/SceneDistill/SpatialStack/src/qwen_vl/model/modeling_qwen3_5_scene_distill.py:215)
- C14: [拼接与 MRoPE 重建](/home/jackson/python/SceneDistill/SpatialStack/src/qwen_vl/model/modeling_qwen3_5_scene_distill.py:379)
- C15: [过期 image-span 测试](/home/jackson/python/SceneDistill/SpatialStack/tests/test_scene_distill.py:1012)
- C16: [脚本位置差异的既有说明](/home/jackson/python/SceneDistill/doc/recover_plan.md:146)
- C17: [文档中的 `.2/.05` 默认](/home/jackson/python/SceneDistill/doc/Distillation_stage2.md:257)
- C18: [单机脚本 `.05/.05`](/home/jackson/python/SceneDistill/SpatialStack/scripts/train/train_scene_distill.sh:43)
- C19: [多机脚本 `.1/.1`](/home/jackson/python/SceneDistill/SpatialStack/scripts/train/train_scene_distill_multinode.sh:45)
- C20: [direct eval 的 32 frames、12800](/home/jackson/python/SceneDistill/SpatialStack/scripts/evaluation/eval_qwen35_vggt_direct.sh:14)
