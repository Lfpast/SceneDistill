# SceneDistill 原生 Video Dataflow 完全重构方案与实施记录

> 实施状态：已按本方案完成代码重写。下文第 1、3 节中用于说明问题的部分行号属于重构前基线；当前唯一生效实现以本节列出的工作区行号为准。

## 0. 当前实现落点

| 契约 | 当前实现与依据 |
|---|---|
| 完整 processor 原生 video tokenization | [data_qwen.py:76](/home/jackson/python/SceneDistill/SpatialStack/src/qwen_vl/data/data_qwen.py:76) 直接生成 `pixel_values_videos`、`video_grid_thw`、`mm_token_type_ids`；不再生成手工 `position_ids` |
| 单图、已标注图片序列、原始视频统一装载 | [data_qwen.py:302](/home/jackson/python/SceneDistill/SpatialStack/src/qwen_vl/data/data_qwen.py:302) 保留已标注序列的完整帧索引，仅对原始视频/帧目录采样并构造 `VideoMetadata`；[utils.py:21](/home/jackson/python/SceneDistill/SpatialStack/src/qwen_vl/data/utils.py:21) 对同一视频统一 resize |
| annotation 到逻辑视频的规范化 | [data_qwen.py:394](/home/jackson/python/SceneDistill/SpatialStack/src/qwen_vl/data/data_qwen.py:394) 先把旧逐帧 `<image>` markers 折叠成逻辑视频 placeholders，再进入 processor；raw marker 数不再被误当成视频数 |
| 多个独立视频 | [data_qwen.py:400](/home/jackson/python/SceneDistill/SpatialStack/src/qwen_vl/data/data_qwen.py:400) 按 placeholder 顺序生成视频列表；[data_qwen.py:527](/home/jackson/python/SceneDistill/SpatialStack/src/qwen_vl/data/data_qwen.py:527) 按同一顺序展平 grid 与 teacher 输入 |
| 原生 temporal pooling 与 MRoPE | [modeling_qwen3_5_scene_distill.py:315](/home/jackson/python/SceneDistill/SpatialStack/src/qwen_vl/model/modeling_qwen3_5_scene_distill.py:315) 调用 `get_video_features`；[modeling_qwen3_5_scene_distill.py:360](/home/jackson/python/SceneDistill/SpatialStack/src/qwen_vl/model/modeling_qwen3_5_scene_distill.py:360) 调用 Qwen3.5 原生 `compute_3d_position_ids` |
| 每视频 teacher 时间对齐 | [modeling_qwen3_5_scene_distill.py:201](/home/jackson/python/SceneDistill/SpatialStack/src/qwen_vl/model/modeling_qwen3_5_scene_distill.py:201) 实现不变、严格 2:1 相邻平均、非 2:1 adaptive average pooling 和非法上采样报错四种情况 |
| Student-only 原生 video 评估 | [qwen3_5.py:288](/home/jackson/python/SceneDistill/SpatialStack/src/lmms_eval/models/qwen3_5.py:288)、[qwen3_5.py:310](/home/jackson/python/SceneDistill/SpatialStack/src/lmms_eval/models/qwen3_5.py:310)、[qwen3_5.py:367](/home/jackson/python/SceneDistill/SpatialStack/src/lmms_eval/models/qwen3_5.py:367)；[qwen3_5_scene_distill.py:15](/home/jackson/python/SceneDistill/SpatialStack/src/lmms_eval/models/qwen3_5_scene_distill.py:15) 明确不构建 teacher |
| 训练默认值直接覆盖 | [argument.py:33](/home/jackson/python/SceneDistill/SpatialStack/src/qwen_vl/train/argument.py:33) 与 [train.sh:43](/home/jackson/python/SceneDistill/SpatialStack/scripts/train/train.sh:43) 固定为 `16/8/1/262144/12544`；batch、context、LR 不机械翻倍 |
| 依赖版本固定 | [setup.py:10](/home/jackson/python/SceneDistill/SpatialStack/setup.py:10) 固定 `transformers==5.4.0`，[setup.py:60](/home/jackson/python/SceneDistill/SpatialStack/setup.py:60) 固定 `qwen_vl_utils==0.0.14`；Transformers 5.4.0是首个包含 Qwen3.5 video MRoPE `StopIteration` 官方修复的正式版本 |

## 1. 目标、问题与重写原则

重构前，SceneDistill 训练端会先把 `video` 采成多张图片，再把一个 `<video>` 改写为多个 `<image>`；评估端同样把视频帧逐张写成 image content，并调用 `processor(images=..., videos=None)`；模型封装还显式拒绝 `pixel_values_videos`。这三处共同造成旧 SceneDistill 实际运行的是“有序多图”，而不是 Qwen3.5 原生 video。对应旧实现范围是重构前 [data_qwen.py:418–620](/home/jackson/python/SceneDistill/SpatialStack/src/qwen_vl/data/data_qwen.py:418)、[qwen3_5.py:304–421](/home/jackson/python/SceneDistill/SpatialStack/src/lmms_eval/models/qwen3_5.py:304) 和 [modeling_qwen3_5_scene_distill.py:236–262](/home/jackson/python/SceneDistill/SpatialStack/src/qwen_vl/model/modeling_qwen3_5_scene_distill.py:236)。

本次不是增加一条 SceneDistill video 分支，而是直接用原生 video dataflow 覆盖旧数据逻辑：

- 单图、多图序列、单视频、多视频全部转换成 Qwen3.5 原生 video 输入。
- 删除旧的 image packing、手工 visual token 展开、手工 `position_ids`、`second_per_grid_ts` 和不可达 video 分支；不保留兼容开关或 fallback。
- 直接替换现有数据函数、现有配置和现有评估流程；不并存新旧入口，不增加兼容类、adapter hook或配置项。旧函数若名称已经与 video 语义冲突，应删除并以语义正确的同职责函数一对一替换，不保留 alias，函数总数不增加。
- 一个样本可以包含多个独立视频。每个视频拥有自己的 `video_grid_thw` 行、时间轴、timestamp和 GCTE global-attention group。
- 训练默认值直接从 `8/4/2` 改成 `16/8/1`；评估现有 `max_num_frames=32` 保持不变。
- SceneDistill 自己不在采样层补帧，也不把补帧当作 VGGT teacher 对齐策略。必须区分两个层次：Qwen3.5 官方 video processor为了满足 `temporal_patch_size=2` 的 Conv3d patchify，会在奇数帧时于 processor内部重复最后一帧；SceneDistill不重写这一原生行为。VGGT teacher仍保留真实的 $S$ 帧，并在不能严格2:1对齐时采用 CamDistill adaptive average pooling。
- Stage 1/2/3 已实现的 Pre-GCTE、Post-GCTE、internal injection、17-token 顺序、loss和 checkpoint 结构全部冻结，不重新审计或修改。

## 2. 重构后的唯一数据契约

### 2.1 视觉输入统一为 video

每条样本可以包含 $N_v\geq 1$ 个逻辑视频，所有视觉对象按照它们在对话中的出现顺序与 `<video>` placeholder 一一对应：

| 原始输入 | 重写后的含义 |
|---|---|
| 单张 `image` | 一个 1 帧视频 |
| `images: List[image]` | 一个按列表顺序排列的多帧视频 |
| 单个视频文件 | 一个原生视频 |
| 一个帧目录 | 一个按文件名排序的原生视频 |
| `video: List[video_file]` | 多个独立视频，逐个采样和处理 |
| 对话中多个视觉对象 | 多个独立视频，按 placeholder 顺序处理 |
| text-only | 保持纯文本，不激活 SceneDistill |

“多个独立视频”不是异常情况。假设一个样本含三个视频，processor和 collator应保持：

```text
video_grid_thw = [
    [Tq_0, Hq_0, Wq_0],
    [Tq_1, Hq_1, Wq_1],
    [Tq_2, Hq_2, Wq_2],
]

geometry_encoder_inputs = [
    Tensor[S_0, 3, H_0, W_0],
    Tensor[S_1, 3, H_1, W_1],
    Tensor[S_2, 3, H_2, W_2],
]
```

三个视频分别参与 temporal pooling、teacher 对齐和 GCTE global attention，不跨视频做 pooling 或 attention。placeholder规则固定为：

- raw annotation marker 和送入 processor 的逻辑视频 placeholder 是两个层次。旧标注中的多个 `<image>` markers 可以逐帧描述同一个 `images: List[image]`，不能用 marker 数推断逻辑视频数。
- `image`、`images: List[image]` 和标量 `video` 都表示一个逻辑视频。若 annotation 含多个旧 markers，则按对话顺序保留第一个位置并统一为一个 `<video>`，删除其余 marker token但保留周围文本。
- `video: List[video_file]` 表示多个独立视频；若原对话只有一个视觉 marker，就在该位置展开成 $N_v$ 个连续 `<video>` placeholders。
- 若有视觉输入但 annotation 没有 marker，则在第一个 user message前插入 $N_v$ 个 `<video>` placeholders；没有 user message属于无合法挂载位置，直接报错。
- 若多个独立视频的 annotation 已显式提供多个 placeholders，则数量必须与视频对象数一致并按出现顺序映射；既不是1也不是 $N_v$ 的 marker 数具有歧义，直接报错。
- annotation 规范化完成后，placeholder数量、`video_grid_thw` 行数和 geometry input数量最终必须完全一致，否则报错。

### 2.2 帧采样

原始视频文件和未预采样的帧目录独立执行：

```text
target_fps = 1 / base_interval = 1 FPS
target_frames = clamp(round(video_duration × target_fps), 8, 16)
target_frames = min(target_frames, available_frames)
frame_indices = uniform_sample(0, total_frames - 1, target_frames)
```

- `video_min_frames=8` 是有足够原始帧时的采样下限，不通过重复帧把短视频强行补到8帧。
- 单图保持1帧；短视频保留实际可采到的帧数。
- `images: List[image]` 是 annotation 已经选定并排序的完整视频序列，不再执行第二次 `video_max_frames` 采样。SPAR 的 `point_img_idx`、`bbox_img_idx`、问题中的 `Frame-N` 和监督答案都以这个原始列表为索引空间，二次采样会同时破坏视觉标记、文本和答案语义。
- 例如32帧 SPAR 序列仍以32个真实帧进入原生 video processor；Qwen `temporal_patch_size=2` 将其变成16个 temporal groups，VGGT 的32帧 special tokens再按严格2:1相邻平均对齐到这16组。这正是 temporal pooling，不会恢复旧多图路径。
- 视频文件保留原始 FPS 和采样下标，使 timestamp 对应真实时间。
- 帧目录和显式图片列表没有可靠的真实 FPS，使用 `sample_fps=1`。
- SceneDistill采样代码不得主动 `repeat/duplicate` 帧；采样结果原样同时交给 Qwen和 VGGT。

这里必须保留一个不能被文档回避的事实：Transformers 5.4.0 原生 Qwen3-VL video processor在 `T % temporal_patch_size != 0` 时，会在内部重复最后一帧后再做 temporal patchify，[video_processing_qwen3_vl.py:184–239](https://github.com/huggingface/transformers/blob/v5.4.0/src/transformers/models/qwen3_vl/video_processing_qwen3_vl.py#L184-L239)。这不是 SceneDistill新增的策略，而是采用“原生 Qwen video”必然继承的底层行为。若删除它，就必须 fork/改写官方 processor，反而不再是本方案定义的原生 dataflow。

因此奇数帧时两侧语义是：

```text
Qwen student:
    使用官方 processor 返回的 Tq；最后一个 temporal patch可能含内部补齐帧

VGGT teacher:
    始终只编码实际采样的 S 帧；不复制图像
    [S,17,2048] --adaptive_avg_pool1d--> [Tq,17,2048]
```

这是一种依据 CamDistill fallback的近似时间对齐，不是逐 temporal patch严格同构：adaptive pooling的最后几个时间 bin不保证与 Qwen内部补齐后的两帧窗口完全相同。该取舍遵循用户指定的 CamDistill方案，文档和实验报告不得把它描述成“奇数帧下严格逐帧一一对应”。

CamDistill 证明了 Qwen processor 必须接收匹配的 `VideoMetadata`，[qwen.py:321–375](/home/jackson/python/CamDistill/swift/template/templates/qwen.py:321)，但这里不能机械照抄 `qwen_vl_utils.fetch_video`。交叉检查其实现后可以看到：真实视频的 `smart_nframes` 会把采样帧数向 `FRAME_FACTOR=2` 取整，图片帧列表还会直接复制最后一帧补到偶数。这与本方案“SceneDistill采样层不补帧、VGGT保留真实 $S$ 帧”的约束冲突。因此最终实现保留并原位重写现有 `process_video` 来负责解码和真实帧采样，只复用 `qwen_vl_utils.smart_resize` 的原生空间规则，再构造 Transformers `VideoMetadata` 交给 Qwen processor。仓库现有 `qwen_vl_utils` 依赖原位固定为 `==0.0.14`，不增加第二个视频读取依赖，[setup.py](/home/jackson/python/SceneDistill/SpatialStack/setup.py)。

### 2.3 原生 Qwen3.5 processor

CamDistill 的 Qwen3/3.5 模板已经证明应由完整 processor 处理 `videos=`，而不是按图片手工展开 placeholder，[qwen.py:549–598](/home/jackson/python/CamDistill/swift/template/templates/qwen.py:549)。重写后的 SceneDistill 使用同一原则：

1. 对话中的每个视觉对象统一表示为 video content。
2. 重写后的现有 `process_video` 返回每个视频的真实 frames、metadata和采样信息；空间 resize 使用 `qwen_vl_utils.smart_resize`。
3. 完整 Qwen3.5 processor 接收 `videos`、`video_metadata` 和文本。
4. processor生成逐 temporal group 的 timestamp、vision wrapper、video placeholder、`pixel_values_videos`、`video_grid_thw` 和 `mm_token_type_ids`。
5. dataset不再手工生成 `position_ids`；模型依据 `mm_token_type_ids` 和 `video_grid_thw` 调用原生 `compute_3d_position_ids`。当前 Qwen3.5 基类已经支持该接口，[modeling_qwen3_5.py:794–812](/home/jackson/python/SceneDistill/SpatialStack/src/qwen_vl/model/modeling_qwen3_5.py:794)。

processor 调用使用：

```text
videos=<all videos in placeholder order>
video_metadata=<matching metadata list>
images=None
do_sample_frames=False
do_resize=False
return_mm_token_type_ids=True
```

`do_sample_frames=False` 和 `do_resize=False` 的依据是采样已由现有 `process_video` 完成，空间 resize 已按 Qwen `smart_resize` 完成；CamDistill也采用“外部准备完成后禁止 processor 二次采样/resize”的契约，[qwen.py:367–375](/home/jackson/python/CamDistill/swift/template/templates/qwen.py:367)、[qwen.py:568–573](/home/jackson/python/CamDistill/swift/template/templates/qwen.py:568)。

Transformers 5.3.0不能作为这条原生 video dataflow的运行边界。Qwen官方已经用单视频输入复现 `get_rope_index -> next(grid_iters[modality_type]) -> StopIteration`，并确认原因是该版本的原生 video MRoPE实现损坏，[Qwen3.5 issue #58](https://github.com/QwenLM/Qwen3.6/issues/58)；对应修复由 Transformers [PR #44474](https://github.com/huggingface/transformers/pull/44474) 合入并增加 video单元测试。仓库因此直接将原有依赖和训练/评估最低版本改为首个包含修复的正式版本 `transformers==5.4.0`，不在 SceneDistill模型中复制、monkey-patch或绕过第三方 `get_rope_index`。自定义 decoder forward也必须服从同一版本的 linear-attention mask契约：5.4.0的 `_update_linear_attn_mask` 第二个参数是 `past_key_values`，不能继续传入5.3.0使用的 `cache_position`，[modeling_qwen3_5.py:1247–1259](https://github.com/huggingface/transformers/blob/v5.4.0/src/transformers/models/qwen3_5/modeling_qwen3_5.py#L1247-L1259)。

### 2.4 空间预算

旧多图路径把每帧缩放到约 518，再裁成 Qwen `patch_size × merge_size = 32` 的整数倍，[utils.py:78–103](/home/jackson/python/SceneDistill/SpatialStack/src/qwen_vl/data/utils.py:78)、[utils.py:243–255](/home/jackson/python/SceneDistill/SpatialStack/src/qwen_vl/data/utils.py:243)，最大有效尺寸约为 `512×512`。原生 video 重写后继续保持这一空间量：

```text
video_max_frame_pixels = 512 × 512 = 262144
video_min_frame_pixels = 12544
total_pixels = sampled_frames × video_max_frame_pixels
```

Qwen和 VGGT-Omega 必须消费同一批、同一顺序、同一 resize 结果的原始帧。不得分别读取或 resize 两次，否则 teacher/student 即使帧数相同也不再是同一视觉输入。

### 2.5 dataset 与 collator 输出

单样本含 $N_v$ 个视频时：

```text
input_ids:                [L]
labels:                   [L]
mm_token_type_ids:        [L]
pixel_values_videos:      concat(video_0_patches, ..., video_Nv-1_patches)
video_grid_thw:           [Nv, 3]
geometry_encoder_inputs:  List[Tensor[S_i, 3, H_i, W_i]], length=Nv
```

batch collate 后：

```text
input_ids / labels / attention_mask / mm_token_type_ids: [B, Lmax]
pixel_values_videos: 连接 batch 内全部视频 patch
video_grid_thw: [sum(Nv_i), 3]
geometry_encoder_inputs: 按 batch 和 placeholder 顺序展平的 List[Tensor]
```

`geometry_encoder_inputs` 不做 `torch.stack`，因为不同视频可以有不同的 $S_i,H_i,W_i$。当前 collator 强制读取 `position_ids`，[data_qwen.py:646–660](/home/jackson/python/SceneDistill/SpatialStack/src/qwen_vl/data/data_qwen.py:646)，并再次 stack geometry tensors，[data_qwen.py:722–725](/home/jackson/python/SceneDistill/SpatialStack/src/qwen_vl/data/data_qwen.py:722)，这两段直接覆盖为上述唯一契约。

## 3. 删除旧实现并一对一替换

### 3.1 `data_qwen.py`：删除多图路线，不并存兼容入口

这里的约束是“不增加第二套路由”，而不是把错误的旧名字强行保留下来。最终只有一条 SceneDistill video dataflow：

- 删除 `preprocess_qwen_2_visual` 及其按 `grid_thw` 手工重复 `<image_pad>/<video_pad>` 的规则，[data_qwen.py:122–203](/home/jackson/python/SceneDistill/SpatialStack/src/qwen_vl/data/data_qwen.py:122)。在原调用位置以一个 `preprocess_video` 一对一替换：它接收完整 processor的原生 video tokenization结果，并只负责保持现有 assistant label masking。不得同时保留旧函数、alias或 image/video双模式参数。
- 保留并重写现有 `process_video`：让它处理单图、图片序列、视频文件、帧目录和视频列表中的单个视频对象，统一返回 frames、metadata、Qwen video tensors和 teacher tensor。删除旧的 `second_per_grid_ts`，[data_qwen.py:347–380](/home/jackson/python/SceneDistill/SpatialStack/src/qwen_vl/data/data_qwen.py:347)。
- 将 `read_video_images` 的有效读取逻辑合并进重写后的 `process_video`，随后删除 `read_video_images`，不保留两个职责重叠的视频入口。
- 直接重写 `_get_item`：删除 `video → images → 多个 <image>` 改写、整个 image dataflow和不可达的旧 video branch，[data_qwen.py:456–620](/home/jackson/python/SceneDistill/SpatialStack/src/qwen_vl/data/data_qwen.py:456)。新实现按 placeholder 顺序循环处理 $N_v$ 个视频，然后一次调用完整 processor。
- 多视频样本的 `pixel_values_videos`、`video_grid_thw` 和 geometry inputs保持相同顺序；不限制为单视频。
- 静态 length estimate阶段还没有 `video_grid_thw`，不能假装能读取真实 $T_q$。它对原始视频按采样上限估计 temporal-group数，对 `images` 标注序列按完整列表长度估计，并用 `video_max_frame_pixels / (patch_size × merge_size)^2 + 17` 推导每组的视觉 token与注入 token预算，[data_qwen.py:186](/home/jackson/python/SceneDistill/SpatialStack/src/qwen_vl/data/data_qwen.py:186)、[data_qwen.py:244](/home/jackson/python/SceneDistill/SpatialStack/src/qwen_vl/data/data_qwen.py:244)；真实 packing和所有运行时校验只使用processor返回的 `video_grid_thw[:,0]`。
- 直接重写 `DataCollatorForSupervisedDataset`：删除 image字段和手工 `position_ids` collate；加入 `mm_token_type_ids` padding，并展平多视频 geometry list。
- 删除 `DataCollatorForFlattenedSupervisedDataset` 以及 `data_flatten` 分支。SceneDistill codebase只保留一种 collator，不再以报错方式保留旧功能。
- 删除上述重写后不再使用的 image constants、imports和 helper调用，保证文件中只有 SceneDistill native video dataflow。

### 3.2 `utils.py`：用视频语义一对一替换旧 image helper

不能为了满足“原位改写”而让 `prepare_image_inputs` 返回 video tensors；这会制造误导接口。正确边界是旧函数删除、新函数一对一接替，且不保留兼容层：

- 删除 `prepare_image_inputs`，在同一职责位置以 `prepare_video_inputs` 替换；所有调用点同时迁移，不保留旧 image实现或 alias。
- 删除 `load_and_preprocess_images`，以 `load_and_preprocess_video_frames` 一对一替换；复用仍正确的 resize代码，但对一个视频的全部帧统一计算和应用空间尺寸，避免帧间 grid不一致。
- 删除 `build_qwen3_5_geometry_inputs`，以 `build_geometry_video_inputs` 一对一替换；直接接收一个视频共同 resize后的帧 tensor并返回 `[S,3,H,W]`，不再按照 `image_grid_thw` 逐图片二次 resize，[utils.py:204–240](/home/jackson/python/SceneDistill/SpatialStack/src/qwen_vl/data/utils.py:204)。
- 删除非 SceneDistill geometry encoder的 patch-size映射、旧 image-only分支和不再被引用的函数；不为将删除的其他模型保留兼容代码。
- 在上述三个替换函数中完成帧非空、RGB、finite、metadata、H/W被32整除以及共享帧顺序校验，不再为这些校验增加第四个 wrapper。

### 3.3 `modeling_qwen3_5_scene_distill.py`：直接切换到 video

现有 SceneDistill wrapper 原位改写：

- 激活条件从 `pixel_values + image_grid_thw` 改成 `pixel_values_videos + video_grid_thw`。
- 删除 `pixel_values_videos` 的 `NotImplementedError` 和全部 image-only判断。
- `video_sizes` 直接取 `video_grid_thw[:,0]`；每一行代表一个独立视频，不依赖 batch样本数。
- 对每个 `[t,h,w]` 生成 $t$ 个 `frame_sizes=h×w` 和 $t$ 个 `merged_frame_sizes=h×w/merge_size^2`。
- 将 `get_image_features`、`image_token_id` 和 image placeholder mask分别替换为 `get_video_features(..., output_hidden_states=True)`、`video_token_id` 和 video mask。
- 每个 temporal group插入一组17 tokens。现有 packing函数已经按连续 placeholder run工作，[vggt_omega_direct_packing.py:16–32](/home/jackson/python/SceneDistill/SpatialStack/src/qwen_vl/model/vggt_omega_direct_packing.py:16)、[vggt_omega_direct_packing.py:198–289](/home/jackson/python/SceneDistill/SpatialStack/src/qwen_vl/model/vggt_omega_direct_packing.py:198)；CamDistill同样要求 run数等于 `sum(video_grid_thw[:,0])`，[camdistill_plugin.py:674–685](/home/jackson/python/CamDistill/camera_movement_sft/plugins/camdistill_plugin.py:674)。
- MRoPE由原生 `compute_3d_position_ids` 先生成，再由现有 packing position逻辑把17 tokens锚定到对应 temporal group的视觉位置，[vggt_omega_direct_packing.py:35–68](/home/jackson/python/SceneDistill/SpatialStack/src/qwen_vl/model/vggt_omega_direct_packing.py:35)。不增加另一套时间位置编码。
- evaluation不要求 geometry inputs；只有训练且 Pre/Post任一 distillation loss启用时才调用 teacher。

### 3.4 重写现有 teacher 收集，实现 CamDistill 对齐

对齐只有 `_collect_teacher_features` 这一个调用点，拆出额外 helper没有复用价值。直接扩展现有 `_collect_teacher_features`，[modeling_qwen3_5_scene_distill.py:204–214](/home/jackson/python/SceneDistill/SpatialStack/src/qwen_vl/model/modeling_qwen3_5_scene_distill.py:204)：

```text
输入：
    geometry_encoder_inputs[i] -> [S_i, 3, H_i, W_i]
    video_grid_thw[i]           -> [Tq_i, Hq_i, Wq_i]

VGGT-Omega：
    teacher_i -> [S_i, 17, 2048]

对齐：
    S_i == Tq_i:
        teacher_i保持不变

    S_i == 2 × Tq_i:
        teacher_i.reshape(Tq_i, 2, 17, 2048).mean(dim=1)

    S_i > Tq_i 且 S_i != 2 × Tq_i:
        把17×2048视为channel，只在时间轴执行adaptive_avg_pool1d到Tq_i

    S_i < Tq_i:
        说明Qwen和VGGT没有消费同一批帧，直接报契约错误

输出：
    cat(aligned_teacher_i, dim=0)
    -> [sum(Tq_i), 17, 2048]
```

这就是 CamDistill `_align_target_to_t_model` 的核心规则：严格2:1时相邻两帧平均，无法整除时使用 adaptive average pooling，[camdistill_loss.py:81–102](/home/jackson/python/CamDistill/camera_movement_sft/plugins/camdistill_loss.py:81)。CamDistill的注释也明确学生一个 temporal group对应两个 VGGT原始帧，[caminject_model.py:87–102](/home/jackson/python/CamDistill/camera_movement_sft/plugins/caminject_model.py:87)。

关键约束：

- 当 $S_i$ 不是 $2Tq_i$ 时，VGGT teacher不复制任何帧，直接 adaptive average pooling；Qwen processor内部是否为 Conv3d patchify补齐是另一层原生行为。
- runtime中的 `Tq_i` 永远从 `video_grid_thw[i,0]` 读取；SceneDistill不另写一套 `ceil(S_i/2)` 或补帧逻辑。
- 每个视频单独对齐后再 concatenate，绝不跨独立视频边界 pooling。
- pooling使用 float32；camera index 0和 scene indices 1–16只沿时间维平均，token顺序不变。
- teacher只提取和对齐一次，随后同一个 tensor同时供 Pre/Post loss使用，[modeling_qwen3_5_scene_distill.py:299–306](/home/jackson/python/SceneDistill/SpatialStack/src/qwen_vl/model/modeling_qwen3_5_scene_distill.py:299)。

VGGT-Omega 的17-token依据保持不变：aggregator按 `camera + 16 register + patch` 排列，[aggregator.py:81–118](/home/jackson/python/SceneDistill/vggt-omega/vggt_omega/models/aggregator.py:81)；teacher extractor严格取前17个 special tokens，[vggt_omega_direct_encoder.py:92–109](/home/jackson/python/SceneDistill/SpatialStack/src/qwen_vl/model/geometry_encoders/vggt_omega_direct_encoder.py:92)。

### 3.5 评估 dataflow：直接覆盖基础 adapter

不增加 `uses_native_video_inputs()` hook，也不为其他 Qwen3.5评估路径保留旧逻辑。直接重写 [qwen3_5.py](/home/jackson/python/SceneDistill/SpatialStack/src/lmms_eval/models/qwen3_5.py) 中已有函数：

- `_sample_video_frames` 原位改成返回采样帧和匹配 metadata。
- `_build_sample` 原位改成把单图、图片列表、单视频和多视频全部构造成 `type="video"` content，并返回按 placeholder排序的 video list。
- `generate_until` 中把 `sample_images`、`images=...` 和 `videos=None` 全部改成 video字段；processor一次接收 batch内全部视频及其 metadata。
- 删除 `build_qwen3_5_geometry_inputs(sample_images, image_grid_thw)`；evaluation是 student-only，不加载或调用 VGGT-Omega，[qwen3_5_scene_distill.py:38–49](/home/jackson/python/SceneDistill/SpatialStack/src/lmms_eval/models/qwen3_5_scene_distill.py:38)。
- 删除 `uses_geometry_encoder_for_eval` 及非 SceneDistill分支，不增加新的 adapter方法。
- 保留现有 `max_num_frames=32`，[qwen3_5.py:104–119](/home/jackson/python/SceneDistill/SpatialStack/src/lmms_eval/models/qwen3_5.py:104)、[eval_qwen35_scene_distill.sh:7–11](/home/jackson/python/SceneDistill/SpatialStack/scripts/evaluation/eval_qwen35_scene_distill.sh:7)。
- 删除会制造第二套多图语义的 `add_frame_index` 和旧 custom video loader配置；timestamp只来自原生 processor。

## 4. 直接覆盖现有训练配置

不新增 `NATIVE_VIDEO_*`、`SCENE_DISTILL_VIDEO_*` 或任何新配置项。直接修改现有默认值：

| 现有参数 | 旧值 | 覆盖后的值 | 依据 |
|---|---:|---:|---|
| `video_max_frames` / `VIDEO_MAX_FRAMES` | 8 | 16 | temporal pooling后学生时间组数量约保持不变 |
| `video_min_frames` / `VIDEO_MIN_FRAMES` | 4 | 8 | 与最大帧数同比扩大 |
| `base_interval` / `BASE_INTERVAL` | 2 | 1 | 采样提升到约1 FPS |
| `video_max_frame_pixels` / `VIDEO_MAX_FRAME_PIXELS` | 25088 | 262144 | 保持旧多图约512×512的空间量 |
| `video_min_frame_pixels` / `VIDEO_MIN_FRAME_PIXELS` | 3136 | 12544 | 与当前 image最小空间预算一致 |
| eval `max_num_frames` | 32 | 32 | 用户锁定，保持不变 |

修改位置是现有 [argument.py:33–42](/home/jackson/python/SceneDistill/SpatialStack/src/qwen_vl/train/argument.py:33) 和 [train.sh:43–58](/home/jackson/python/SceneDistill/SpatialStack/scripts/train/train.sh:43)。`train_scene_distill.sh` 不再增加一套覆盖值；共享配置本身就是 SceneDistill的唯一配置。现有 `OUTPUT_DIR` 默认值可直接改成新的 native-video run名称，但不增加新的变量或目录配置。

以下参数不乘2：

- `MODEL_MAX_LENGTH=8192`
- `TOTAL_BATCH_SIZE=64`
- `PER_DEVICE_TRAIN_BATCH_SIZE=1`
- gradient accumulation
- learning rate、epoch、warmup、scheduler
- Pre/Post distillation weights
- 每个 temporal group的17-token数量

依据是16个原始帧经过 Qwen原生 temporal pooling后，学生视觉时间组数量目标仍约为旧8-image路径的8组；因此不应同时把 context、batch或优化器参数乘2。VGGT-Omega仍处理16个原始帧，teacher开销会增加，所以“Qwen token量近似不变”不等于整个训练资源严格不变。若后续真实训练 OOM，必须先记录失败位置和峰值显存，不能在本次重构中预先静默修改 batch或分辨率。

## 5. 明确不改的模型结构

以下组件与 dataflow无关，保持原实现：

- [scene_distill_module.py](/home/jackson/python/SceneDistill/SpatialStack/src/qwen_vl/model/scene_distill_module.py)：Frame Cross-Attention本来就依据 `frame_sizes` 分割视觉 token，[scene_distill_module.py:89–144](/home/jackson/python/SceneDistill/SpatialStack/src/qwen_vl/model/scene_distill_module.py:89)；Global Attention依据 `video_sizes` 按独立视频分组，[scene_distill_module.py:147–208](/home/jackson/python/SceneDistill/SpatialStack/src/qwen_vl/model/scene_distill_module.py:147)。只需让 wrapper传入 temporal-group语义，无需编辑 GCTE。
- [modeling_qwen3_5.py](/home/jackson/python/SceneDistill/SpatialStack/src/qwen_vl/model/modeling_qwen3_5.py) 中 Stage 3 Post/injection接口、层号和 KV-cache gate。
- [vggt_omega_direct_encoder.py](/home/jackson/python/SceneDistill/SpatialStack/src/qwen_vl/model/geometry_encoders/vggt_omega_direct_encoder.py) 的 teacher抽取和冻结。
- [aggregator.py](/home/jackson/python/SceneDistill/vggt-omega/vggt_omega/models/aggregator.py) 的 VGGT-Omega模型结构。
- `NUM_SPECIAL_TOKENS=17`、Pre/Post层号、投影器、loss公式、loss权重、internal injection和 checkpoint keys。
- 已实现的 `Distillation_stage1.md`、`Distillation_stage2.md`、`Distillation_stage3.md`。

## 6. 本地可执行的 smoke test

本地没有 SpatialStack/Qwen3.5运行环境，因此不新增 pytest文件，不把完整 forward、MRoPE数值或 GPU训练写成本地可完成的验收项。其他旧测试脚本后续会删除，本次也不为旧 image路径维护测试兼容。

本地只执行基础 smoke test：

```bash
python -m py_compile \
  SpatialStack/src/qwen_vl/data/data_qwen.py \
  SpatialStack/src/qwen_vl/data/utils.py \
  SpatialStack/src/qwen_vl/model/modeling_qwen3_5_scene_distill.py \
  SpatialStack/src/lmms_eval/models/qwen3_5.py \
  SpatialStack/src/lmms_eval/models/qwen3_5_scene_distill.py

bash -n SpatialStack/scripts/train/train.sh
bash -n SpatialStack/scripts/train/train_scene_distill.sh
bash -n SpatialStack/scripts/evaluation/eval_qwen35_scene_distill.sh

git diff --check
```

再执行静态残留检查，目标是确认旧逻辑已经被覆盖而不是隐藏在分支后：

```bash
rg -n "second_per_grid_ts|visual_type=\"image\"|images=sample_images|videos=None|ordered multi-image|only supports videos represented as ordered multi-image" \
  SpatialStack/src/qwen_vl/data \
  SpatialStack/src/qwen_vl/model/modeling_qwen3_5_scene_distill.py \
  SpatialStack/src/lmms_eval/models/qwen3_5.py

rg -n "VIDEO_MAX_FRAMES|VIDEO_MIN_FRAMES|BASE_INTERVAL|VIDEO_MAX_FRAME_PIXELS|VIDEO_MIN_FRAME_PIXELS" \
  SpatialStack/scripts/train/train.sh \
  SpatialStack/src/qwen_vl/train/argument.py
```

第一条残留检查应没有命中；第二条应只显示直接覆盖后的 `16/8/1/262144/12544`。由于缺少目标环境，本地 smoke通过只能证明语法、shell和静态 dataflow约束成立，不能宣称 Qwen3.5 runtime已经验证。

当前实施结果：上述 Python语法检查、三个 shell脚本的 `bash -n`、`git diff --check` 均已通过；旧 SceneDistill dataflow标识和手工 RoPE调用点的定向静态检查无命中。本机缺少 `torch`、`transformers`、`qwen_vl_utils` 组成的 SpatialStack目标环境，因此没有执行 processor、模型 forward或 GPU训练，也不将静态通过表述为运行时通过。

在具备 SpatialStack环境后的首个训练任务中，再检查以下运行时事实，但不将其伪装成本地测试结果：

- 16帧视频的 `video_grid_thw[:,0]` 与模型 temporal groups一致。
- 32帧 SPAR annotation序列保留原始 `point_img_idx` / `bbox_img_idx`，并产生16个 Qwen temporal groups；不得先截断为16帧再访问原索引。
- 多视频样本的 placeholder、grid rows和 geometry tensors一一对应。
- 严格2:1 teacher使用相邻两帧平均；非2:1 teacher使用 adaptive average pooling。
- teacher、Pre和Post输出均为 `[sum(Tq_i),17,2048]`。
- evaluation只运行 student，不构建 VGGT-Omega。
- cached decode不重复执行 video encoder或 Post injection。

## 7. 最终验收标准

实现完成时必须同时满足：

1. SceneDistill视觉输入只存在原生 video dataflow；旧 image和多图包装代码已经删除。
2. 单图、图片序列、单视频和多视频都由同一条流程处理；旧入口被一对一替换，不保留并行兼容路径。
3. 多视频样本按各自 `video_grid_thw` 行独立执行 temporal grouping、teacher对齐和 global attention。
4. SceneDistill采样层不复制帧；Qwen processor保留官方奇数帧内部补齐；非严格2:1的 VGGT teacher使用 CamDistill adaptive average pooling，并明确这是近似对齐。
5. 旧函数删除后只允许语义正确的一对一替换；函数总数不增加，也不增加兼容类、hook或配置开关。
6. GCTE、Stage 3 injection、17-token顺序和 loss不变。
7. 本地只报告实际完成的基础 smoke结果；Qwen3.5和 GPU runtime在目标环境验证前明确标为未验证。
