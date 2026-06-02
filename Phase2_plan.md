## Phase 2 方案: `geometry_encoder_type=vggt_omega_alpha` 的输入侧 Camera/Scene Token 挂载分支

### Summary
- 目标是在 **不引入任何 feature fusion / feature injection / decoder 内部交互** 的前提下，为 Qwen3.5 新增一个完全独立的 alpha 分支：`geometry_encoder_type=vggt_omega_alpha`。
- 该分支只做输入侧修改，严格按这条顺序执行：
  1. 将送入 `VGGT-Ω` 的 geometry-side frame 强制 resize 到 `224x224`
  2. 对每个 frame 跑 `VGGT-Ω`
  3. 只取每帧的 `1 camera token + 16 scene/register tokens`
  4. 用一个简单的两层 MLP projector 投影到 Qwen3.5 输入 embedding 维度
  5. 将这 17 个 token 拼接进 Qwen3.5 的每个视觉 span 中
  6. 对这 17 个 token 单独设定 MRoPE：`t` 与同帧普通视觉 token 一致，`h=0, w=0`
- 该分支与 Phase1 保持独立：`vggt`、`vggt_omega` 原路径不改语义；alpha 只在 `geometry_encoder_type == "vggt_omega_alpha"` 时生效。
- 论文与本地源码对齐结论：
  - `VGGT-Ω` 的 “scene token” 在实现上就是 16 个 registers，不是单个 token。
  - alpha 分支最终每帧额外插入 **17 个 token**，顺序固定为 `camera + 16 scene tokens`。

### Public / Interface Changes
- [argument.py](/home/jackson/python/SpatialStack-omega/SpatialStack/src/qwen_vl/train/argument.py)
  - 保持 `USE_GEOMETRY_ENCODER=True/False` 为布尔，不改接口。
  - 扩展 `geometry_encoder_type` 的合法值：`vggt | vggt_omega | vggt_omega_alpha | pi3`。
- 内部 batch / model 输入新增 alpha 专用元数据
  - 需要新增一组 alpha 输入侧字段，建议命名为：
    - `alpha_geometry_inputs`
    - `alpha_visual_token_layout`
  - 作用：
    - `alpha_geometry_inputs` 携带给 `VGGT-Ω alpha` 的逐帧 224x224 RGB tensor
    - `alpha_visual_token_layout` 描述每个 image/video span 中普通视觉 token 与 17 个特殊 token 的拼接布局
- 对外训练命令不新增必填 flag
  - 仍然使用 `USE_GEOMETRY_ENCODER=True`
  - 新分支只通过 `GEOMETRY_ENCODER_TYPE=vggt_omega_alpha` 触发
  - 现有 `FEATURE_FUSION_METHOD / GEOMETRY_FUSION_LAYERS / GEOMETRY_ENCODER_LAYERS` 在 alpha 分支中保留兼容，但 **不参与实际逻辑**

### Implementation Changes
- 新增 alpha 专用 encoder wrapper
  - 新建 [vggt_omega_alpha_encoder.py](/home/jackson/python/SpatialStack-omega/SpatialStack/src/qwen_vl/model/geometry_encoders)
  - 主体功能：
    - 运行时桥接 `vggt-omega/`
    - 冻结全部 `VGGT-Ω` 参数
    - 只暴露每帧 `camera_and_register_tokens`
    - 不返回 Omega patch tokens，不返回 dense/camera/text heads
  - 输出 contract：
    - 对每帧输出 `17 x 2048` token 序列
    - token 顺序固定：`camera` 在前，`16 registers(scene tokens)` 在后
  - 加载逻辑：
    - 复用当前 `vggt_omega` 本地 `.pt` checkpoint 解析方式
    - 只加载 alpha wrapper 实际需要的主干权重

- 新增 alpha 专用 projector
  - 新建 [vggt_omega_alpha_projector.py](/home/jackson/python/SpatialStack-omega/SpatialStack/src/qwen_vl/model)
  - 主体功能：
    - 两层 MLP，将 `2048 -> text_hidden -> text_hidden`
    - 激活函数使用 `GELU`
  - 不引入额外 gating、cross-attention、LayerNorm 栈、残差分支
  - alpha 分支中，这是唯一新增的 geometry-side trainable module
  - `VGGT-Ω alpha encoder` 保持 frozen；`projector` 默认 trainable

- 修改 geometry encoder 注册与加载
  - 修改 [factory.py](/home/jackson/python/SpatialStack-omega/SpatialStack/src/qwen_vl/model/geometry_encoders/factory.py) 和 [__init__.py](/home/jackson/python/SpatialStack-omega/SpatialStack/src/qwen_vl/model/geometry_encoders/__init__.py)
  - 新增 `vggt_omega_alpha` 分支
  - 修改 [modeling_qwen3_5.py](/home/jackson/python/SpatialStack-omega/SpatialStack/src/qwen_vl/model/modeling_qwen3_5.py)
    - 新增 alpha path 的模块初始化、checkpoint save/load key 注册
    - `GEOMETRY_STATE_KEYWORDS` 需要纳入 alpha projector 的 state dict 前缀
  - alpha path 不初始化 `language_feature_fusion / feature_fusion / geometry_merger`

- 修改 geometry-side preprocessing，使 alpha 分支强制 224x224
  - 修改 [utils.py](/home/jackson/python/SpatialStack-omega/SpatialStack/src/qwen_vl/data/utils.py)
  - 新增 alpha helper：
    - 对 image path：将每帧原始 RGB resize 到 `224x224` 后送入 `VGGT-Ω`
    - 对 video path：对采样出的每帧逐帧 resize 到 `224x224`
  - 普通 Qwen3.5 processor 路径不改；`224x224` 只作用于 alpha 的 geometry-side 输入
  - 新增显式校验：
    - alpha 分支要求普通 Qwen 视觉 patch 布局与 `14x14 = 196` 对齐
    - 不满足时 fail fast，而不是 silent reshape

- 修改数据构造与 placeholder 扩张逻辑
  - 修改 [data_qwen.py](/home/jackson/python/SpatialStack-omega/SpatialStack/src/qwen_vl/data/data_qwen.py)
  - 训练数据在 image / video 两条路径都要支持 alpha
  - 关键变化：
    - `<image>` 或 `<video>` 展开时，placeholder 数量不再只是普通视觉 token 数
    - alpha 分支变成：`17 * 帧数 + 普通视觉 token 数`
  - 视觉 span 内部顺序固定为：
    - `<|vision_start|>`
    - `camera token`
    - `16 scene tokens`
    - 原始 Qwen visual patch tokens
    - `<|vision_end|>`
  - labels 逻辑保持不变；这些位置仍然属于 multimodal 输入部分，不单独加监督头

- 修改 Qwen3.5 输入拼接逻辑
  - 修改 [modeling_qwen3_5.py](/home/jackson/python/SpatialStack-omega/SpatialStack/src/qwen_vl/model/modeling_qwen3_5.py)
  - alpha 分支不走当前 `image_embeds.masked_scatter(image_mask, image_embeds)` 的单一“纯 patch span”语义
  - 需要新增 alpha-aware visual packing：
    - 先得到原始 Qwen visual patch embeddings
    - 再得到每帧 `17` 个 projected geometry tokens
    - 按每帧顺序拼成新的 visual embedding span
    - 最后再 scatter 到对应 placeholder 区间
  - 该逻辑只在 `geometry_encoder_type == "vggt_omega_alpha"` 时执行
  - 其他 geometry encoder 分支保持现状

- 修改 Qwen3.5 的 MRoPE 计算
  - 修改 [rope2d.py](/home/jackson/python/SpatialStack-omega/SpatialStack/src/qwen_vl/data/rope2d.py)
    - 新增 alpha 专用 `get_rope_index_35_alpha(...)`
    - 训练数据预处理走这条分支
  - 修改 [modeling_qwen3_5.py](/home/jackson/python/SpatialStack-omega/SpatialStack/src/qwen_vl/model/modeling_qwen3_5.py)
    - inference / generate 时不能再直接沿用默认 `compute_3d_position_ids`
    - alpha 分支要调用与训练一致的 alpha MRoPE 逻辑
  - 位置规则固定：
    - 对每帧 `camera + 16 scene`：
      - `t` 复用该帧普通视觉 patch 的时间索引
      - `h = 0`
      - `w = 0`
    - 对原始 patch tokens：
      - 保持现有正常 Qwen3.5 MRoPE 逻辑
  - 这条规则同时适用于 image 和 video 视觉 span

- 修改训练 / 推理 / eval 入口
  - [train_qwen.py](/home/jackson/python/SpatialStack-omega/SpatialStack/src/qwen_vl/train/train_qwen.py)
    - 继续透传 `geometry_encoder_type`
    - alpha 分支下把 alpha 专用 batch 字段送入 model
  - [infer.py](/home/jackson/python/SpatialStack-omega/SpatialStack/scripts/inference/infer.py)
    - 增加 alpha image/video 输入构造
    - 复用同样的 224x224 alpha geometry preprocessing
  - [qwen3_5.py](/home/jackson/python/SpatialStack-omega/SpatialStack/src/lmms_eval/models/qwen3_5.py)
    - eval 路径同步支持 alpha 的 image/video span 构造和 MRoPE
  - 不改 Qwen2.5 adapter，不改 `lmms_eval/models/spatialstack.py`

- 文档更新
  - 修改 [TRAINING.md](/home/jackson/python/SpatialStack-omega/SpatialStack/TRAINING.md) 与 [README.md](/home/jackson/python/SpatialStack-omega/SpatialStack/README.md)
  - 新增 alpha recipe，明确：
    - `GEOMETRY_ENCODER_TYPE=vggt_omega_alpha`
    - alpha 分支只做 input-side modification
    - 每帧额外插入 17 token
    - `VGGT-Ω` 输入固定为 224x224
  - 明确 alpha 与 Phase1 的 `vggt_omega` 分支是两条并行路径，不共用融合逻辑

### Test Plan
- Encoder / projector 单测
  - `create_geometry_encoder("vggt_omega_alpha", ...)` 返回正确实例
  - alpha encoder 对每帧输出 shape 为 `17 x 2048`
  - projector 输出维度等于 `config.text_config.hidden_size`
  - `VGGT-Ω alpha` 参数 frozen，projector 参数 trainable

- 数据与 placeholder 测试
  - image 样本：每帧 placeholder 数 = `17 + 普通 patch token 数`
  - video 样本：每帧都插入 17 个 token，总 span 长度正确
  - visual span 排列顺序固定为 `camera -> 16 scene -> normal patches`

- MRoPE 测试
  - alpha 分支中，17 个特殊 token 的 `h=w=0`
  - 17 个特殊 token 的 `t` 与同帧 patch token 的时间索引一致
  - 普通 patch token 的 position ids 与现有 Qwen3.5 逻辑一致
  - training preprocessing 与 runtime generation 产出的 alpha position ids 一致

- Qwen3.5 前向 smoke test
  - image path 下能跑通 alpha 分支前向
  - video path 下能跑通 alpha 分支前向
  - 不触发任何 `feature_fusion` / `geometry_merger` / `language_feature_fusion` 模块

- 回归测试
  - `geometry_encoder_type=vggt` 和 `vggt_omega` 行为不变
  - 不使用 geometry encoder 的 Qwen3.5 行为不变

### Assumptions / Defaults
- `USE_GEOMETRY_ENCODER` 保持布尔；Phase2 只新增 `geometry_encoder_type=vggt_omega_alpha`。
- alpha 分支保留 **16 个 scene/register tokens**，不把它们压缩成单个 token。
- alpha projector 采用最简单的两层 `Linear + GELU + Linear`，不额外加复杂结构。
- alpha 分支完全绕过 feature fusion；即使配置里保留现有 fusion 相关字段，也不在这条路径中使用。
- 该方案默认同时覆盖 Qwen3.5 的 image 与 video 视觉输入，因为你的架构描述明确要求按“每个 input video 的每个 frame”处理。