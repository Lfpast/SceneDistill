# VGGT-Omega Direct 原生 Video 链路恢复方案

## Summary

恢复既有 `vggt_omega_direct` 分支，将其接入当前唯一的原生 video dataflow：

```text
原始视频/图片序列
  ├─ Qwen processor → pixel_values_videos + video_grid_thw + MRoPE
  └─ 相同真实帧 → 冻结 VGGT-Omega → camera/scene tokens
                                      ↓ temporal 对齐
                                 direct_projector
                                      ↓
                   front/back 拼接至每个 Qwen temporal group
                                      ↓
                                  Qwen LLM
```

边界固定如下：

- 不新增模型类、配置项、CLI 参数或 adapter hook。
- 不引入 GCTE、Pre/Post、distillation loss 或 decoder injection。
- 保留已有 `direct_projector`，作为唯一必要的 `2048 → Qwen hidden size` 维度桥。
- VGGT-Omega 在训练和评估时都通过 `GEOMETRY_ENCODER_PATH` 外部加载并实际执行，但不写入训练 checkpoint。
- 四个 train/eval 脚本逐字保留，不改变任何参数、默认值、注释或调用方式。

## Implementation Changes

### 1. 将 direct wrapper 完全切换到原生 video

直接重写 [modeling_qwen3_5_vggt_omega_direct.py](/home/jackson/python/SceneDistill/SpatialStack/src/qwen_vl/model/modeling_qwen3_5_vggt_omega_direct.py:133) 中现有 image dataflow：

- 激活条件改为首轮存在 `pixel_values_videos + video_grid_thw`；删除旧 `pixel_values + image_grid_thw` 路径及 native-video `NotImplementedError`。
- Qwen 视觉侧改用：
  ```python
  get_video_features(pixel_values_videos, video_grid_thw)
  ```
- 每个 `[Tq,H,W]` 生成 `Tq` 个大小为 `H×W/merge_size²` 的 merged visual spans。
- 使用 `video_token_id`、video placeholder mask 和原生 `compute_3d_position_ids(..., video_grid_thw=...)`，不恢复手工 RoPE。
- processor 已为每个 temporal group生成独立 timestamp/vision-wrapper run；因此现有 placeholder packing 可在每个 run分别插入 `K` 个 token。该行为与固定的 Transformers 5.4.0 [Qwen3-VL processor](https://github.com/huggingface/transformers/blob/v5.4.0/src/transformers/models/qwen3_vl/processing_qwen3_vl.py#L150-L195) 一致。
- `front`排列为`[direct tokens, Qwen visual tokens]`，`back`排列相反；扩展 label位置继续为`-100`。
- 删除 direct wrapper 中未被任何计算消费的 `_direct_only_mask` 构造；它只服务 SceneDistill injection，direct 分支没有 injection。
- cached decode不再携带首轮 video tensors，因此不会重复运行 Qwen video encoder、VGGT 或 projector。

### 2. 按视频执行 VGGT temporal 对齐

原位改写现有 `_collect_direct_features`，不增加新对齐模块：

```text
geometry_encoder_inputs[i] : [S_i,3,H_i,W_i]
video_grid_thw[i]          : [Tq_i,Hq_i,Wq_i]
VGGT output                : [S_i,K,2048]

K = 1   camera
K = 16  scene16
K = 17  special17
```

每个视频独立处理：

- `S_i == Tq_i`：直接保留。
- `S_i == 2×Tq_i`：reshape为`[Tq_i,2,K,2048]`，相邻两帧取均值。
- `S_i > Tq_i`且不是严格2:1：仅沿时间轴执行 float32 `adaptive_avg_pool1d`。
- `S_i < Tq_i`：拒绝上采样，因为这意味着 Qwen 与 VGGT 没有消费同一批真实帧。
- 多视频分别对齐后再按 placeholder 顺序 concatenate，禁止跨视频 pooling。
- 对齐后的`[sum(Tq_i),K,2048]`进入现有 `direct_projector`，再与对应 Qwen visual spans拼接。

这复用当前 SceneDistill 已验证的时间边界 [modeling_qwen3_5_scene_distill.py](/home/jackson/python/SceneDistill/SpatialStack/src/qwen_vl/model/modeling_qwen3_5_scene_distill.py:200)，但不复用其 Pre/Post 或 loss。

### 3. 评估端在线提供 VGGT 输入

在 [qwen3_5.py](/home/jackson/python/SceneDistill/SpatialStack/src/lmms_eval/models/qwen3_5.py:308) 的现有评估流程中完成，不复制 `generate_until`：

- 保留现有 `_build_sample` 和 processor 原生 video 调用。
- 当 checkpoint 的`geometry_encoder_type == "vggt_omega_direct"`时，把同一批已经采样、resize的`sample_videos`通过现有 `build_geometry_video_inputs`转换成`List[Tensor[S_i,3,H_i,W_i]]`。
- 将该列表移动到当前评估进程的模型设备，并作为`geometry_encoder_inputs`传给`generate`。
- SceneDistill evaluation仍保持 student-only；只有 direct 类型进入这条在线 VGGT 分支。
- 现有 [qwen3_5_vggt_direct.py](/home/jackson/python/SceneDistill/SpatialStack/src/lmms_eval/models/qwen3_5_vggt_direct.py:35) 继续负责 config、token mode、front/back和外部 VGGT路径，不增加构造参数。

### 4. 外挂 VGGT 的 checkpoint 契约

- 将现有“删除冻结 geometry encoder权重”的逻辑移到共用 Qwen3.5 checkpoint职责位置，命名为中性的 geometry-weight过滤函数；SceneDistill和 direct共同调用，避免 direct反向依赖 GCTE模块。
- DeepSpeed和非 DeepSpeed最终保存均排除：
  ```text
  geometry_encoder.*
  model.geometry_encoder.*
  ```
- 保留：
  ```text
  model.direct_projector.*
  model.language_model.*
  lm_head.*
  ```
- 评估时由现有`geometry_encoder_path`重新构建并加载 VGGT-Omega；不增加 key mapping、完整性扫描或自定义 checkpoint错误处理，missing/unexpected keys继续交给 Transformers/PyTorch。
- 不改变其他 geometry分支的保存语义。

### 5. 脚本与文档

以下脚本必须保持当前 SHA256不变：

```text
train_vggt_direct.sh              41db224648c90eccf584d36ed0f3a7bdd60de7d669a520a3444917f2101b7f0d
train_vggt_direct_scene.sh        7de1f1692b8526cca4dcbac02e561e65e2b163458010fe7bc3f0146f35c0456f
eval_qwen35_vggt_direct.sh        aadad7cf016afcc54f70b75c2c16f8dc9d20650a54aa0c5c338f1c1a23f76a7a
eval_qwen35_vggt_direct_scene.sh  07a768156b701b9325dbff58a710020c258cb08e31bbd571f0ead826442b43ca
```

同步更新`Dataflow_refactor.md`中的边界说明：

- SceneDistill评估仍为 student-only。
- `vggt_omega_direct`是明确恢复的例外：train/eval都在线运行外部冻结 VGGT。
- 不修改四个脚本本身。

## Test Plan

### 本地单元与静态测试

- `camera/scene16/special17`分别产生`K=1/16/17`。
- 两个以上 temporal-group runs分别插入 token，而不是每个视频只插一次。
- `front/back`同时验证 embedding、placeholder、label和 position IDs顺序。
- temporal对齐覆盖`S=Tq`、`S=2Tq`、非2:1 adaptive pooling及`S<Tq`。
- 多视频验证各自 pooling后再 concatenate。
- checkpoint验证保留`direct_projector`且不存在 VGGT权重。
- eval验证只有 direct类型添加`geometry_encoder_inputs`，SceneDistill不添加。
- 执行：
  ```bash
  python -m compileall -q <modified Python files>
  PYTHONPATH=SpatialStack/src conda run -n OKT pytest -q SpatialStack/tests/test_scene_distill.py
  bash -n <four direct scripts>
  sha256sum <four direct scripts>
  git diff --check
  ```

### 目标环境 smoke test

- `special17/back`和`scene16/front`各运行一个最小训练 batch。
- 确认16个真实帧产生约8个 Qwen temporal groups，VGGT输出先由16组对齐为8组再拼接。
- 确认 VGGT全部冻结，`direct_projector`与LLM存在梯度，无 distillation loss。
- 保存后检查 checkpoint不含`geometry_encoder` keys。
- direct eval确认外部 VGGT成功加载、首轮生成执行一次、cached decode不重复执行。
- 使用一个奇数帧视频和一个多视频样本验证 adaptive pooling及顺序保持。
- 功能通过后才运行脚本已有六个 benchmark；不在没有结果前声明恢复分支优于SceneDistill。

## Locked Assumptions

- 四个脚本逐字保留，因此现存的默认差异也保留：`train_vggt_direct.sh`默认`back`，对应 eval脚本默认`front`。评估默认训练产物时，调用者必须显式设置：
  ```bash
  GEOMETRY_TOKEN_INSERT_POSITION=back \
  MODEL_PATH=<actual-checkpoint> \
  bash SpatialStack/scripts/evaluation/eval_qwen35_vggt_direct.sh
  ```
- train/eval默认输出路径差异同样不修改，实际评估 checkpoint通过现有`MODEL_PATH`覆盖。
- `direct_projector`是唯一允许保留的附加学习组件；不增加 gate、attention、GCTE或 injection。
- 当前工作树中的`.gitignore`修改和未跟踪的 scene eval脚本视为用户内容，实施时保留。
