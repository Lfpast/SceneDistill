## Phase 1 方案: 在 Qwen3.5 全链路中以可切换挂载层接入 VGGT-Ω

### Summary
- 目标是把当前 `VGGT` 替换为“可选的 `VGGT-Ω` geometry encoder”，但 **不改变** SpatialStack 现有的几何到 LLM 融合语义：仍然是 frozen geometry encoder、仍然只取 patch tokens、仍然走 `geometry_encoder_layers -> geometry_fusion_layers` 的原始映射、仍然只在 prefill 阶段对 vision-token slice 做 language-side residual fusion。
- 第一阶段范围锁定为 **Qwen3.5 全链路**：训练、推理、`lmms_eval` 的 Qwen3.5 入口都支持 `geometry_encoder_type=vggt_omega`；**不扩展 Qwen2.5**，避免把改动面扩到另一条模型实现。
- vggt_omega 的 checkpoint 加载不再限制为本地 .pt，而是同时支持：本地 .pt 文件, 本地目录, HF gated repo id，例如facebook/VGGT-Omega
- 论文与代码锚点：`VGGT-Ω` 的 registers / register attention / single dense head 来自 [arXiv](https://arxiv.org/abs/2605.15195) 和 [project page](https://vggt-omega.github.io/)；本地实现事实来自 `vggt-omega/vggt_omega/models/{vggt_omega.py,aggregator.py}`。

### Public / Interface Changes
- `SpatialStack/src/qwen_vl/train/argument.py`
  - 保留现有 `geometry_encoder_type` 字段，不改接口名；只把合法值从现有的 `vggt|pi3` 扩成 `vggt|vggt_omega|pi3`。
  - `geometry_encoder_path` 的语义补充为：
    - `vggt` 时仍可用现有 HF / local 路径语义。
    - `vggt_omega` 时第一阶段要求 **本地 checkpoint 路径**，指向 `.pt` 文件或包含单个目标 `.pt` 的目录。
- `SpatialStack/src/qwen_vl/model/geometry_encoders/base.py`
  - 不改抽象接口名；继续复用 `BaseGeometryEncoder.encode_layers/get_feature_dim/load_model` 这套 contract。
- 内部接口变更
  - `SpatialStack/src/qwen_vl/data/utils.py::prepare_image_inputs(...)` 增加 geometry encoder context，使其能按 encoder 类型决定 geometry-side resize 逻辑。
  - 这是内部 helper 变更，不是对外 API。

### Implementation Changes
- 复用层，不改语义
  - `SpatialStack/src/qwen_vl/model/feature_fusion.py` 不改融合算法，不新增新的 fusion method。
  - `SpatialStack/src/qwen_vl/model/modeling_qwen3_5.py` 不改 VGGT/LLM 交互语义，不改冻结策略，不改 layer mapping 逻辑；它继续只依赖 `create_geometry_encoder(...)` 返回的统一 encoder 接口。
  - `SpatialStack/src/qwen_vl/train/train_qwen.py` 只做最小 wiring：
    - 把 `geometry_encoder_type` 继续写入 config。
    - 同时把 `data_args.geometry_encoder_type = model_args.geometry_encoder_type`，让数据预处理知道当前 geometry encoder 的 patch size / resize 规则。

- 新增 Omega adapter，不改原 VGGT 实现
  - 新建 `SpatialStack/src/qwen_vl/model/geometry_encoders/vggt_omega_encoder.py`
    - 作用：实现 `BaseGeometryEncoder`，对 `vggt-omega/` 做本地桥接。
    - 导入策略：优先从 sibling repo `vggt-omega/vggt_omega` 导入；若 repo 缺失或不可导入，报清晰错误，不 silently fallback。
    - 构造策略：实例化 `VGGTOmega(enable_camera=False, enable_depth=False, enable_alignment=False)`，只保留 aggregator 主干，不启用 camera/dense/text heads。
    - 冻结策略：与现有 `VGGTEncoder` 完全一致，所有参数 `requires_grad=False`，forward 走 `eval()` + `torch.no_grad()`。
    - 输出策略：
      - `patch_size=16`
      - `feature_dim=2048`，因为 Omega aggregator 输出仍是 `frame_tokens || inter_frame_tokens` 拼接后的 `2 * 1024`
      - `encode_layers(...)` 只返回 patch tokens，丢弃 camera token 和 register tokens
      - 支持的层索引以 Omega aggregator 实际缓存层为准，第一阶段显式支持 `{4, 11, 17, 23}`；SpatialStack recipe 继续使用 `11 17 23`
      - 对 unsupported layer index 直接报错，避免 silent misalignment
    - checkpoint 加载策略：
      - `load_model(path)` 支持本地 `.pt` 文件或本地目录解析到 `.pt`
      - 使用 `torch.load(..., map_location="cpu")` + `load_state_dict`
      - 第一阶段不实现 gated HF 自动下载

- 扩展 geometry encoder 工厂，而不是重写上层逻辑
  - 修改 `SpatialStack/src/qwen_vl/model/geometry_encoders/factory.py`
    - 新增 `encoder_type == "vggt_omega"` 分支，返回 `VGGTOmegaEncoder`
  - 修改 `SpatialStack/src/qwen_vl/model/geometry_encoders/__init__.py`
    - 导出 `VGGTOmegaEncoder`
  - 不新建新的上层“super factory”或第二套 fusion mount；现有 `factory.py` 就是这次切换的挂载点

- 对齐 geometry-side 输入构造，保证 token 数与 Qwen visual grid 对齐
  - 修改 `SpatialStack/src/qwen_vl/data/utils.py`
    - 当前 Qwen3.5 分支把 geometry image resize 到 `grid_h * 14, grid_w * 14`，这是 VGGT 专用逻辑
    - 改为按 `geometry_encoder_type` 派发：
      - `vggt` 用 `14`
      - `vggt_omega` 用 `16`
    - 目标是不改 LLM 融合逻辑的前提下，让 geometry patch grid 继续与 Qwen visual merged grid 一一对齐
  - 修改 `SpatialStack/src/qwen_vl/data/data_qwen.py`
    - `prepare_image_inputs(...)` 调用处传入 `geometry_encoder_type`
  - 修改 `SpatialStack/scripts/inference/infer.py`
    - `build_qwen3_5_geometry_inputs(...)` 取消对 `patch_size=14` 的硬编码，改为按 `config.geometry_encoder_type` 决定 geometry patch size
  - 修改 `SpatialStack/src/lmms_eval/models/qwen3_5.py`
    - 评测路径的 geometry input 构造逻辑同步成同一规则
  - 第一阶段不触碰 `SpatialStack/src/lmms_eval/models/spatialstack.py`，因为它是 Qwen2.5 适配器，不在本阶段 scope

- 配置、脚本、文档更新
  - 修改 `SpatialStack/scripts/train/train.sh`
    - 文档与 env 示例增加 `GEOMETRY_ENCODER_TYPE=vggt_omega`
    - 不改默认值；默认仍保留 `vggt`
  - 修改 `SpatialStack/README.md` 与 `SpatialStack/TRAINING.md`
    - 增加 Omega recipe 示例
    - 明确写出 `GEOMETRY_ENCODER_PATH=/path/to/vggt_omega_1b_512.pt`
    - 明确第一阶段“scene/register/text alignment outputs 不参与 SpatialStack-LLM 融合”
  - 如评测脚本文档有 geometry path 示例，也同步补充 Omega 用法，但不改默认 sbatch 模板行为

- Memory 落盘步骤
  - 本回合是 Plan Mode，不直接写 memory
  - 实施回合应额外写一条 ad-hoc note 到：
    - `/home/jackson/.codex/memories/extensions/ad_hoc/notes/<timestamp>-spatialstack-vggt-llm-omega-phase1.md`
  - 内容只记录两类事实：
    - 现有 SpatialStack 中 VGGT 和 LLM 的交互约束
    - 本阶段已确认的 Omega migration scope / defaults
  - 这一步需要单独处理工作区外写权限

### Test Plan
- Encoder adapter 单测
  - `create_geometry_encoder("vggt_omega", ...)` 能返回正确实例
  - `encode_layers([11,17,23])` 返回 3 个张量，shape 与旧 VGGT path 兼容，且只含 patch tokens
  - 非法层号、缺失 sibling repo、缺失 `.pt` checkpoint 时给出明确错误
- Geometry input alignment 单测
  - 对同一个 Qwen3.5 `image_grid_thw`，`vggt` 生成 `14x` geometry image，`vggt_omega` 生成 `16x` geometry image
  - 两者最终 patch 数都与 language-side 期待的 merged token 数保持一致
- Qwen3.5 前向 smoke test
  - `geometry_encoder_type=vggt_omega` 下，模型能走通 training forward / inference prefill
  - `geometry_layer_features` 仍然按 `11/17/23 -> 0/1/2` 注入，不改 `feature_fusion_method=deepstack_language_add`
- 全链路 smoke test
  - 训练入口能 parse 新 encoder type
  - `scripts/inference/infer.py` 能读取带 `geometry_encoder_type=vggt_omega` 的 config 并生成
  - `lmms_eval` 的 `qwen3_5` adapter 能加载同类 config 并完成 batch_size=1 的 geometry eval
- 回归测试
  - `geometry_encoder_type=vggt` 原路径行为不变
  - 不使用 geometry encoder 的纯 Qwen3.5 路径行为不变

### Assumptions / Defaults
- 第一阶段 **不使用** Omega 的 scene/register token、text alignment embedding、dense head 输出；即使论文强调它们更强，也先保持 SpatialStack 当前“patch-token only”接口不变。
- 第一阶段 **不改变** SpatialStack 的融合层位、融合方式、监督方式、冻结方式，只替换 geometry feature provider。
- 第一阶段默认使用 Omega 本地 checkpoint，而不是实现 HF gated download。
- Qwen2.5 代码路径、额外 architecture 抽象、以及利用 Omega scene token 做新 fusion 的设计，全部留到后续阶段。
