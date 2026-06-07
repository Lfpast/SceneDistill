## Phase 2 方案: `geometry_encoder_type=vggt_omega_alpha` 的正式集成计划

### Summary
- 这次 Phase2 的 source of truth 不再是我之前那版 `224x224 + h=w=0` 的 alpha 方案，而是你这次转述的学长要求，加上 `SpatialStack_temp/` 里已经验证过功能链路的实现思路。
- 目标架构仍然是一个**独立外挂分支**：
  - `USE_GEOMETRY_ENCODER` 保持布尔；
  - 新增 `geometry_encoder_type=vggt_omega_alpha`；
  - `vggt` / `vggt_omega` 现有 Phase1 路径不改语义；
  - `vggt_omega_alpha` 不走当前 deepstack / feature_fusion / geometry_merger 注入链路。
- 这次的核心不是“让 Qwen 自己适配额外 token”，而是：
  1. 保持 Qwen3.5 原始 visual span 语义和 eval key 解析方式不被破坏；
  2. 用 Qwen 真实生成的 `image_grid_thw` 作为 token grid 的唯一 source of truth；
  3. 令送入 VGGT-Omega 的 geometry-side 分辨率显式满足 token 对齐；
  4. 再把每帧的 `1 camera + 16 scene/register` token 插入到 Qwen 的视觉 span 起始处。

### 核心结论
- 学长截图里的“Qwen 看 `28*H, 28*W`，Omega 看 `32*H, 32*W`”在当前主仓里其实已经有一半基础设施：
  - 当前 [SpatialStack/src/qwen_vl/data/utils.py](/home/jackson/python/SpatialStack-omega/SpatialStack/src/qwen_vl/data/utils.py:196) 的 `build_qwen3_5_geometry_inputs(...)` 本质上就是按 `target_height = grid_h * patch_size`、`target_width = grid_w * patch_size` 构造 geometry 输入。
  - 对 Qwen3.5 来说，`image_grid_thw` 里的 `grid_h/grid_w` 是 **Qwen merger 之前** 的 patch grid；当 `geometry_encoder_type=vggt_omega` 且 `patch_size=16` 时，这个公式自然对应学长说的 `32*H, 32*W`。
- 因此，这次 Phase2 **不需要**恢复我之前那版“强制 224x224”的输入策略。
- `SpatialStack_temp/` 真正应该借鉴的是两类机制，而不是整套 CamDistill：
  - 可复用思想 1：用独立 Qwen3.5 wrapper 做输入侧视觉 span 扩展，而不是把大量 alpha 条件判断塞回 [modeling_qwen3_5.py](/home/jackson/python/SpatialStack-omega/SpatialStack/src/qwen_vl/model/modeling_qwen3_5.py:394)。
  - 可复用思想 2：把“插 token”拆成小工具函数，分别处理 placeholder 扩张、视觉 embedding 扩张、视觉 mask 修正。
- `SpatialStack_temp/` 里**不应直接迁移**的部分：
  - `CamSceneTokenModule` 的交替注意力学生模块；
  - `CamSceneDistillLoss` 的离线 cosine distill；
  - 离线 cache 抽取脚本和 `use_camdistill` 系列 flags；
  - temp 里自带的一整份 `src/qwen_vl/model/vggt_omega/` vendored 源码。

### Source Mapping
- 学长关于 token 对齐的文字说明：
  - [SpatialStack_temp/SpatialStack代码详解.md](/home/jackson/python/SpatialStack-omega/SpatialStack_temp/SpatialStack代码详解.md:1119)
  - [SpatialStack_temp/SpatialStack代码详解.md](/home/jackson/python/SpatialStack-omega/SpatialStack_temp/SpatialStack代码详解.md:1142)
- 学长实现里值得复用的“序列扩张”机制：
  - [SpatialStack_temp/src/qwen_vl/model/cam_distill.py](/home/jackson/python/SpatialStack-omega/SpatialStack_temp/src/qwen_vl/model/cam_distill.py:382)
  - [SpatialStack_temp/src/qwen_vl/model/cam_distill.py](/home/jackson/python/SpatialStack-omega/SpatialStack_temp/src/qwen_vl/model/cam_distill.py:572)
  - [SpatialStack_temp/src/qwen_vl/model/cam_distill.py](/home/jackson/python/SpatialStack-omega/SpatialStack_temp/src/qwen_vl/model/cam_distill.py:610)
- 学长实现里值得复用的“独立 wrapper”形态：
  - [SpatialStack_temp/src/qwen_vl/model/modeling_qwen3_5_camdistill.py](/home/jackson/python/SpatialStack-omega/SpatialStack_temp/src/qwen_vl/model/modeling_qwen3_5_camdistill.py:68)
  - [SpatialStack_temp/src/qwen_vl/model/modeling_qwen3_5_camdistill.py](/home/jackson/python/SpatialStack-omega/SpatialStack_temp/src/qwen_vl/model/modeling_qwen3_5_camdistill.py:506)
- 学长实现里确认 Omega special token 切片方式的证据：
  - [SpatialStack_temp/scripts/extract/extract_vggt_omega_camscene.py](/home/jackson/python/SpatialStack-omega/SpatialStack_temp/scripts/extract/extract_vggt_omega_camscene.py:229)
- 当前主仓 Phase1 的对接基座：
  - Omega frozen encoder wrapper: [SpatialStack/src/qwen_vl/model/geometry_encoders/vggt_omega_encoder.py](/home/jackson/python/SpatialStack-omega/SpatialStack/src/qwen_vl/model/geometry_encoders/vggt_omega_encoder.py:96)
  - Qwen3.5 geometry forward: [SpatialStack/src/qwen_vl/model/modeling_qwen3_5.py](/home/jackson/python/SpatialStack-omega/SpatialStack/src/qwen_vl/model/modeling_qwen3_5.py:394)
  - geometry input resize base: [SpatialStack/src/qwen_vl/data/utils.py](/home/jackson/python/SpatialStack-omega/SpatialStack/src/qwen_vl/data/utils.py:196)
  - inference geometry path: [SpatialStack/scripts/inference/infer.py](/home/jackson/python/SpatialStack-omega/SpatialStack/scripts/inference/infer.py:303)
  - eval geometry path: [SpatialStack/src/lmms_eval/models/qwen3_5.py](/home/jackson/python/SpatialStack-omega/SpatialStack/src/lmms_eval/models/qwen3_5.py:369)

### 与上一版撤回方案的差异
- 不再强制 `224x224`。
- 不再把 `rope2d.py` 里的 alpha 逻辑做成一套独立硬编码分支。
- 不再让 Qwen 通过“自适应裁切后碰运气对齐”来吸收 geometry 侧差异。
- 不再把 alpha 条件判断直接铺进现有 [modeling_qwen3_5.py](/home/jackson/python/SpatialStack-omega/SpatialStack/src/qwen_vl/model/modeling_qwen3_5.py) 主分支。
- 改为：
  - geometry-side 分辨率由 `image_grid_thw` 显式推导；
  - `17` 个特殊 token 的插入在独立 wrapper 内完成；
  - 原始 Phase1 `vggt` / `vggt_omega` 路径保持不动。

### Public / Interface Changes
- [SpatialStack/src/qwen_vl/train/argument.py](/home/jackson/python/SpatialStack-omega/SpatialStack/src/qwen_vl/train/argument.py:14)
  - `USE_GEOMETRY_ENCODER` 保持布尔，不改接口。
  - `geometry_encoder_type` 新增合法值：`vggt_omega_alpha`。
  - 不引入 temp 里的 `use_camdistill`、`camdistill_*` 训练 flags。
- 训练命令保持主仓风格：
  - 仍通过 `USE_GEOMETRY_ENCODER=True`
  - 再配 `GEOMETRY_ENCODER_TYPE=vggt_omega_alpha`
  - `GEOMETRY_ENCODER_PATH` 指向 Omega checkpoint
- batch 内部新增的字段尽量少：
  - `geometry_encoder_inputs` 继续复用，不新增 `alpha_geometry_inputs`
  - 最终实现不新增 `frames_per_sample`
  - 不新增 `sample_ids`、离线 cache 路径、teacher label 等 distill 专用字段

### 设计原则
- 原始 Qwen3.5 visual token 序列仍然是“每帧一段连续 patch tokens”；我们只是在每段前面插入 17 个 token。
- 视频继续沿用当前主仓的 multi-image 路径，而不是走 Qwen 原生 `pixel_values_videos` 路径。
  - 这是和学长文档 [SpatialStack_temp/SpatialStack代码详解.md](/home/jackson/python/SpatialStack-omega/SpatialStack_temp/SpatialStack代码详解.md:1142) 一致的。
- Phase2 不引入任何 deepstack 内部交互：
  - 不用 `language_feature_fusion`
  - 不用 `feature_fusion`
  - 不用 `geometry_merger`
  - 不用 temp 里的 distill branch

### 代码集成方案

#### A. 底层可复用部分

- [SpatialStack/src/qwen_vl/data/utils.py](/home/jackson/python/SpatialStack-omega/SpatialStack/src/qwen_vl/data/utils.py:196)
  - 继续复用 `build_qwen3_5_geometry_inputs(...)` 的主逻辑。
  - 只需要把 `GEOMETRY_ENCODER_PATCH_SIZES` 扩展为：
    - `vggt -> 14`
    - `vggt_omega -> 16`
    - `vggt_omega_alpha -> 16`
  - 同时补注释，明确这里的 `grid_h/grid_w` 是 Qwen merger 之前的 grid，因此对 Omega 来说会自然得到 `32*H, 32*W`。

- [SpatialStack/src/qwen_vl/model/geometry_encoders/vggt_omega_encoder.py](/home/jackson/python/SpatialStack-omega/SpatialStack/src/qwen_vl/model/geometry_encoders/vggt_omega_encoder.py:96)
  - 继续复用：
    - sibling `vggt-omega/` 仓库导入逻辑
    - Omega checkpoint 解析逻辑
    - frozen 权重加载逻辑
  - 不直接搬 `SpatialStack_temp/src/qwen_vl/model/vggt_omega/`。

- [SpatialStack/scripts/inference/infer.py](/home/jackson/python/SpatialStack-omega/SpatialStack/scripts/inference/infer.py:303)
- [SpatialStack/src/lmms_eval/models/qwen3_5.py](/home/jackson/python/SpatialStack-omega/SpatialStack/src/lmms_eval/models/qwen3_5.py:369)
  - 继续复用当前“先 processor 出 `image_grid_thw`，再构造 `geometry_encoder_inputs`”的主流程。
  - 只需要把 `geometry_encoder_type=vggt_omega_alpha` 纳入路由，不另起一套输入协议。

#### B. 需要新增但属于“外挂模块”的部分

- 新建 [SpatialStack/src/qwen_vl/model/geometry_encoders/vggt_omega_alpha_encoder.py](/home/jackson/python/SpatialStack-omega/SpatialStack/src/qwen_vl/model/geometry_encoders)
  - 职责：
    - 复用 Omega 主干加载；
    - 冻结全部参数；
    - 暴露一个只返回每帧 `camera + 16 registers` 的接口；
    - 输出 shape 约定为 `(num_frames, 17, 2048)`。
  - 与现有 `vggt_omega_encoder.py` 的差别：
    - `vggt_omega_encoder.py` 返回 patch token features；
    - `vggt_omega_alpha_encoder.py` 返回 special token features。
  - token 切片规则以 [extract_vggt_omega_camscene.py](/home/jackson/python/SpatialStack-omega/SpatialStack_temp/scripts/extract/extract_vggt_omega_camscene.py:229) 为准。

- 新建 [SpatialStack/src/qwen_vl/model/vggt_omega_alpha_projector.py](/home/jackson/python/SpatialStack-omega/SpatialStack/src/qwen_vl/model)
  - 职责：
    - 渐进投影：`2048 -> midpoint_hidden -> text_hidden`
    - 其中 `midpoint_hidden` 必须同时不同于 `input_dim` 和 `output_dim`
    - 结构保持简单，`Linear + GELU + Linear`
  - 这会是 Phase2 唯一新增的 trainable alpha-side 模块。
  - 不迁移 temp `CamSceneTokenProjector` 里的 LayerNorm / hidden_mult 设计，除非后续你明确要保留。

- 新建 [SpatialStack/src/qwen_vl/model/vggt_omega_alpha_packing.py](/home/jackson/python/SpatialStack-omega/SpatialStack/src/qwen_vl/model)
  - 这不是 temp `cam_distill.py` 的直接复制版，而是它的**瘦身重写版**。
  - 只保留三类工具：
    - `expand_visual_placeholders(...)`
    - `expand_image_embeds_with_alpha_tokens(...)`
    - `build_alpha_only_mask(...)`
  - 明确不包含：
    - `FrameCrossAttentionLayer`
    - `GlobalCamSceneSelfAttentionLayer`
    - `CamSceneTokenModule`
    - distill 相关逻辑

- 新建 [SpatialStack/src/qwen_vl/model/modeling_qwen3_5_vggt_omega_alpha.py](/home/jackson/python/SpatialStack-omega/SpatialStack/src/qwen_vl/model)
  - 这是本次 Phase2 的核心外挂文件。
  - 形态上参考 temp 的 [modeling_qwen3_5_camdistill.py](/home/jackson/python/SpatialStack-omega/SpatialStack_temp/src/qwen_vl/model/modeling_qwen3_5_camdistill.py:68)，但内部逻辑改成：
    1. 调用 Qwen3.5 visual encoder 得到原始 `image_embeds`
    2. 调用 frozen `vggt_omega_alpha_encoder` 得到 `(T, 17, 2048)` special tokens
    3. 用 projector 投到 LLM hidden
    4. 按帧把 `17` 个 token prepend 到每帧 merged visual span
    5. 同步扩张 `input_ids / labels / attention_mask / position_ids`
    6. 再喂给 `language_model`
  - 这条路径不调用当前 [modeling_qwen3_5.py](/home/jackson/python/SpatialStack-omega/SpatialStack/src/qwen_vl/model/modeling_qwen3_5.py:577) 里的 geometry fusion 逻辑。

#### C. 需要修改的现有文件

- [SpatialStack/src/qwen_vl/train/argument.py](/home/jackson/python/SpatialStack-omega/SpatialStack/src/qwen_vl/train/argument.py:14)
  - 新增 `vggt_omega_alpha` 到 `geometry_encoder_type` 注释/合法值说明。
  - 不额外暴露任何 alpha 专用 CLI flag。

- [SpatialStack/src/qwen_vl/model/geometry_encoders/factory.py](/home/jackson/python/SpatialStack-omega/SpatialStack/src/qwen_vl/model/geometry_encoders/factory.py)
- [SpatialStack/src/qwen_vl/model/geometry_encoders/__init__.py](/home/jackson/python/SpatialStack-omega/SpatialStack/src/qwen_vl/model/geometry_encoders/__init__.py)
  - 注册 `vggt_omega_alpha` encoder。

- [SpatialStack/src/qwen_vl/train/train_qwen.py](/home/jackson/python/SpatialStack-omega/SpatialStack/src/qwen_vl/train/train_qwen.py)
  - Qwen3.5 路由新增：
    - `geometry_encoder_type != vggt_omega_alpha` -> 继续走 `Qwen3_5ForConditionalGenerationWithGeometry`
    - `geometry_encoder_type == vggt_omega_alpha` -> 走新的 `Qwen3_5ForConditionalGenerationWithVGGTOmegaAlpha`
  - 不引入 temp 里的 `use_camdistill` 总开关。
  - projector 参数需要默认 `requires_grad=True`，Omega encoder 默认 frozen。

- [SpatialStack/src/qwen_vl/data/utils.py](/home/jackson/python/SpatialStack-omega/SpatialStack/src/qwen_vl/data/utils.py:14)
  - 扩展 patch-size map。
  - `prepare_image_inputs(...)` 继续复用当前逻辑，不再引入 `224x224` 分支。

- [SpatialStack/src/qwen_vl/data/data_qwen.py](/home/jackson/python/SpatialStack-omega/SpatialStack/src/qwen_vl/data/data_qwen.py:199)
  - 训练数据仍然走现有 image / multi-image / video->multi-image 流程。
  - 只需要补两个东西：
    - `geometry_encoder_type=vggt_omega_alpha` 时也正常构造 `geometry_encoder_inputs`
    - wrapper 直接以 `image_grid_thw` 的帧展开顺序作为对齐依据，不额外携带样本级 frame 计数
  - 不恢复旧 alpha 的 `alpha_geometry_inputs` / `alpha_visual_token_layout` 字段。

- [SpatialStack/scripts/inference/infer.py](/home/jackson/python/SpatialStack-omega/SpatialStack/scripts/inference/infer.py:154)
  - `resolve_model_class(...)` 新增 alpha wrapper 路由。
  - Qwen3.5 输入准备仍然先走 processor，再复用 `geometry_encoder_inputs`。
  - 不引入额外 alpha 专用 batch 元数据。

- [SpatialStack/src/lmms_eval/models/qwen3_5.py](/home/jackson/python/SpatialStack-omega/SpatialStack/src/lmms_eval/models/qwen3_5.py:376)
  - 与 inference 同步增加 alpha wrapper 路由。
  - 保证 eval 仍然走和训练一致的多图展开路径，不能走原生 video token 合并路径。

- [SpatialStack/TRAINING.md](/home/jackson/python/SpatialStack-omega/SpatialStack/TRAINING.md)
  - 补一节新的 Phase2 recipe，明确：
    - `USE_GEOMETRY_ENCODER=True`
    - `GEOMETRY_ENCODER_TYPE=vggt_omega_alpha`
    - Qwen 图像流保持原生 processor；
    - Omega geometry 流通过 `image_grid_thw` 推导到 `32*H, 32*W`；
    - 每帧插入 `17` 个 special tokens。

### Qwen3.5 Forward 具体改造思路

#### 1. 仍然先让 Qwen 正常生成 `image_embeds`
- 当前主仓的 Qwen3.5 geometry forward 在 [modeling_qwen3_5.py](/home/jackson/python/SpatialStack-omega/SpatialStack/src/qwen_vl/model/modeling_qwen3_5.py:663) 先拿 `image_outputs.pooler_output`，然后直接 `masked_scatter` 回 `inputs_embeds`。
- Phase2 alpha wrapper 不能改这条主线的语义，只能在 `image_embeds` 已经生成之后，扩张它对应的视觉 span。

#### 2. 不改 Qwen 原始 `compute_3d_position_ids`，而是在它之后扩张 `position_ids`
- 这次不建议重建旧版 `rope2d.py` alpha 路线。
- 更干净的方案是：
  - 先按原始 `input_ids + image_grid_thw` 生成标准 Qwen3.5 `position_ids`
  - 再用 alpha packing helper 把新增 17 个 token 的位置一起插进去
- 这正是 temp `expand_visual_placeholders(...)` 最值得复用的地方。
- 这里要加一个**强约束**：
  - **禁止**对“已经插入 17 个 special tokens 之后”的 expanded `input_ids` 再次调用 [get_rope_index_35(...)](/home/jackson/python/SpatialStack-omega/SpatialStack/src/qwen_vl/data/rope2d.py:427)
  - 因为当前 Qwen3.5 的 MRoPE 构造逻辑在 [rope2d.py](/home/jackson/python/SpatialStack-omega/SpatialStack/src/qwen_vl/data/rope2d.py:475) 到 [rope2d.py](/home/jackson/python/SpatialStack-omega/SpatialStack/src/qwen_vl/data/rope2d.py:518) 用 `input_tokens.index(image_token_id, st)` 和 `st = ed + llm_grid_t * llm_grid_h * llm_grid_w` 线性扫描视觉 span
  - 一旦 expanded sequence 的每帧长度从 `n_patch` 变成 `17 + n_patch`，但这段扫描逻辑仍按原始 `n_patch` 前进，`st` 就会漂移，后续就会触发你描述的那类 “image token 不在 list 里” 错误
- 所以 Phase2 的 position_ids 正确流程必须是：
  1. 用**未扩张**的原始 Qwen visual placeholder 序列计算标准 `position_ids`
  2. 再在 packing helper 中把 17 个 special token 的位置插进去
  3. 后续 forward / eval / generate 只复用这份 expanded `position_ids`，不重新跑 Qwen 原始 rope 扫描

#### 3. 视觉 span 的最终排列
- 每帧内部顺序固定为：
  - `17` 个 alpha special tokens
  - 该帧原始 Qwen merged visual patch tokens
- 这意味着 placeholder 数从：
  - `n_patch`
  - 变成 `17 + n_patch`
- 所有 image spans 都要同步做这件事。

#### 4. video 路径只支持 multi-image 展平
- 当前正式仓本来就主要把视频拆帧后当多图处理。
- Phase2 必须显式沿用这一点，否则 Qwen 原生 temporal merge 会让“每帧插 17 token”的定义失效。

#### 5. `position_ids` 默认采用 `frame_center`，不采用 `h=0,w=0`
- 这次我建议把新增 17 个 token 的默认 MRoPE 扩张规则定为 temp helper 里的 `frame_center`，而不是上一版设想的 `t` 同帧、`h=0,w=0`。
- 理由不是“能不能跑通”，而是“哪种扩张规则更稳”：
  - `h=0,w=0` 本质等价于 temp helper 里的 `frame_first`，会把 17 个 special tokens 和该帧左上角第一个 patch 绑到同一个空间坐标；
  - `frame_center` 仍保持 `t` 与该帧一致，但把 `h/w` 放到该帧 patch run 的中心区域，避免和真实 patch `(0,0)` 发生位置碰撞；
  - temp 实现对 `frame_center` 的说明与插值逻辑在 [cam_distill.py](/home/jackson/python/SpatialStack-omega/SpatialStack_temp/src/qwen_vl/model/cam_distill.py:405) 到 [cam_distill.py](/home/jackson/python/SpatialStack-omega/SpatialStack_temp/src/qwen_vl/model/cam_distill.py:415)，具体赋值在 [cam_distill.py](/home/jackson/python/SpatialStack-omega/SpatialStack_temp/src/qwen_vl/model/cam_distill.py:545) 到 [cam_distill.py](/home/jackson/python/SpatialStack-omega/SpatialStack_temp/src/qwen_vl/model/cam_distill.py:564)
- 需要强调：
  - 你之前 eval 里出现的 “部分 image token 不在 list 里” **更像是 hardcode 的 span 长度 / placeholder 数 / rope 扫描前进量不一致**，而不是 `frame_first` 和 `frame_center` 二选一本身造成的
  - 也就是说，真正会导致 eval 崩的是“expanded sequence 后又跑原始 `get_rope_index_35`”这类结构问题；`frame_center` 只是额外在语义上更稳
- 因此这次 Phase2 的默认策略定为：
  - `t`: 与同帧普通 patch token 一致
  - `h/w`: 取该帧 patch run 的中心坐标
  - 后续如果你想做 ablation，再额外比较 `frame_center` vs `frame_first`

### 不直接采纳 `SpatialStack_temp/` 的模块
- 不迁移 `src/qwen_vl/model/cam_distill.py` 中的注意力学生模块，只借鉴其 packing helpers 的结构拆分。
- 不迁移 `src/qwen_vl/model/cam_distill_loss.py`。
- 不迁移 `src/qwen_vl/model/modeling_qwen3_5_camdistill.py` 的 distill loss 线程，只借鉴其“独立 wrapper”组织方式。
- 不迁移 `scripts/extract/extract_vggt_omega_camscene.py` 作为正式训练依赖；它只提供 special token 切片规则的证据。
- 不迁移 `use_camdistill`、`camdistill_*` 参数体系。

### Test Plan
- Encoder tests
  - `create_geometry_encoder("vggt_omega_alpha")` 返回正确实例。
  - frozen Omega 输出每帧 shape 为 `(17, 2048)`。
  - special token 顺序固定为 `camera + 16 registers`。

- Resize / token alignment tests
  - 对同一 `image_grid_thw`，`vggt` 路径保持 `14 * grid_h/grid_w`。
  - `vggt_omega_alpha` 路径得到 `16 * grid_h/grid_w`，等价于学长方案里的 `32*H, 32*W`。
  - 不允许再出现固定 `196` 或固定 `224` 的假设。

- Packing tests
  - `expand_visual_placeholders(...)` 后，每帧 run 的长度增加 `17`。
  - `expand_image_embeds_with_alpha_tokens(...)` 后，token 数与 expanded placeholder 数严格一致。
  - `build_alpha_only_mask(...)` 能正确标出新增 special token 位置。
  - expanded `position_ids` 长度与 expanded `input_ids` 长度严格一致。
  - expanded 后不再调用原始 `get_rope_index_35(...)`。

- Qwen3.5 forward smoke tests
  - image 单图样本跑通。
  - multi-image 样本跑通。
  - video 样本通过“video -> multi-image”路径跑通。
  - `vggt` / `vggt_omega` 原有 Phase1 forward 行为不变。

- Inference / eval tests
  - `scripts/inference/infer.py` 能加载 alpha wrapper 并生成。
  - `lmms_eval --model qwen3_5` 的 alpha 路径能完成至少一个 benchmark 的前处理和 forward。
  - 覆盖 `vsibench` / `cvbench` 的最小样本，验证不会再出现 `image_token_id` 查找失败。

### 实施顺序
1. 先扩展 `geometry_encoder_type` 注册与 `build_qwen3_5_geometry_inputs(...)` 的 patch-size map。
2. 新建 `vggt_omega_alpha_encoder.py` 和 `vggt_omega_alpha_projector.py`。
3. 新建 `vggt_omega_alpha_packing.py`，只写 placeholder / embed / mask 三个 helper。
4. 新建 `modeling_qwen3_5_vggt_omega_alpha.py`，完成独立 wrapper。
5. 修改 `train_qwen.py`、`infer.py`、`lmms_eval/models/qwen3_5.py` 路由到新 wrapper。
6. 最后补 `TRAINING.md` 和针对 alpha 的最小测试。

### 结论
- 本计划默认采用 `frame_center` 作为 Phase2 的 MRoPE 扩张规则。
- 但真正决定 eval 稳定性的，不是 `center` 还是 `(0,0)`，而是：
  - **不要 hardcode 每帧 token 数**
  - **不要对 expanded visual sequence 重新跑原始 Qwen rope 扫描**
  - **expanded placeholder 数、expanded image embeds 数、expanded position_ids 长度必须三者严格一致**
