# Training

All commands run from `SpatialStack/` under `conda activate cm`.

## 8 种训练方案

```bash
bash scripts/train/train_pure.sh                  # 纯 Qwen3.5 SFT
bash scripts/train/train_spatialstack.sh          # VGGT-1B + deepstack_language_add (推荐 baseline)
bash scripts/train/train_spatialstack_omega.sh    # 同上, 编码器换 VGGT-Omega
bash scripts/train/train_vgllm.sh                 # VGGT-1B + post-merger add (VG-LLM 风格)
bash scripts/train/train_vgllm_omega.sh           # 同上, 编码器换 VGGT-Omega
bash scripts/train/train_vggt_direct.sh           # VGGT-Omega, 每帧前拼 17 token (1 cam + 16 reg)
bash scripts/train/train_vggt_direct_scene.sh     # VGGT-Omega, 每帧前拼 16 register token
bash scripts/train/train_scene_distill.sh         # 4-stage Pre + 6-stage Post GCTE 双端在线蒸馏
```

首次跑会自动下载 Qwen3.5-4B (~8GB) + VGGT-1B / VGGT-Omega (~2-3GB) 到 `$HF_HOME/hub`,之后命中缓存不再下。

## 换 base 模型

```bash
MODEL_PATH=Qwen/Qwen3.5-9B bash scripts/train/train_spatialstack.sh
# 或本地路径
MODEL_PATH=/local/path/to/qwen35 bash scripts/train/train_pure.sh
```

## 换输出目录

```bash
OUTPUT_DIR=./output/exp1 bash scripts/train/train_spatialstack.sh
```

`OUTPUT_DIR` 末尾段会自动作为 wandb run name。

## 换数据集

```bash
DATASETS="llava_hound_64k%1" bash scripts/train/train_pure.sh
DATASETS="spar_234k%30,vlm3r_scannet%50" bash scripts/train/train_spatialstack.sh
```

格式: `<dataset_name>%<sampling_rate>`,多个用逗号连。可选 dataset (见 `src/qwen_vl/data/__init__.py:data_dict`):
`spar_234k / llava_hound_64k / vlm3r_scannet / vsi_appr_order` 等。

## 调超参

```bash
LR=5e-6 TOTAL_BATCH_SIZE=32 bash scripts/train/train_spatialstack.sh
```

| Env | 默认 | 作用 |
|---|---|---|
| `LR` | `1e-5` | learning rate |
| `TOTAL_BATCH_SIZE` | `64` | 全局 batch (自动除 world_size 得 grad_accum) |

其它超参 (`num_train_epochs / warmup_ratio / lr_scheduler_type / model_max_length` 等) 直接改 `scripts/train/train.sh`。

## direct injection 特有参数

```bash
# 换 token 模式: camera(1) | scene16(16) | special17(17)
GEOMETRY_DIRECT_TOKEN_MODE=camera bash scripts/train/train_vggt_direct.sh
GEOMETRY_DIRECT_TOKEN_MODE=scene16 bash scripts/train/train_vggt_direct.sh
GEOMETRY_DIRECT_TOKEN_MODE=special17 bash scripts/train/train_vggt_direct.sh

# 换插入位置: front | back
GEOMETRY_TOKEN_INSERT_POSITION=back bash scripts/train/train_vggt_direct.sh
```

`train_scene_distill.sh` 是独立架构路径：Pre 固定使用 Qwen Vision 第 `1/5/9/13` 层，
Post 固定捕获 Qwen LLM 第 `5/9/13/17/21/25` 层；两端均使用 `special17`、首帧参考系和 front insertion。Pre/Post 蒸馏权重可独立覆盖：

```bash
PRE_DISTILL_WEIGHT=0.2 \
POST_DISTILL_WEIGHT=0.05 \
bash scripts/train/train_scene_distill.sh
```

## 限定 GPU

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/train/train_spatialstack.sh
```

不设默认用全部 GPU。`gradient_accumulation_steps` 会按 `TOTAL_BATCH_SIZE / world_size` 自动算。

## 关 wandb

```bash
REPORT_TO=none bash scripts/train/train_pure.sh
```

默认走 wandb project = `gd-llm`,run name = `OUTPUT_DIR` 末尾段。

## 输出位置

```
./output/<recipe>/
  ├── train.rank0.log                  # 全部 stdout
  ├── config.json / model*.safetensors # 训练结束后的最终成果
  ├── trainer_state.json               # step/loss/lr 完整历史
  └── wandb/                           # 本地 wandb cache
```

默认 `save_strategy=no`, 不再保存训练中的 `checkpoint-*` 目录。

## 常见组合

```bash
# 快速 smoke test (只跑 llava_hound)
DATASETS="llava_hound_64k%1" TOTAL_BATCH_SIZE=8 \
  bash scripts/train/train_pure.sh

# 默认不再写中间 checkpoint; 若 OUTPUT_DIR 里已有旧 checkpoint-* 仍会自动 resume
bash scripts/train/train_spatialstack.sh

# 训一版对比: VGGT vs VGGT-Omega, deepstack_language_add
OUTPUT_DIR=./output/spatialstack_vggt bash scripts/train/train_spatialstack.sh
OUTPUT_DIR=./output/spatialstack_omega bash scripts/train/train_spatialstack_omega.sh

# 训 direct 3 档对比
bash scripts/train/train_vggt_direct.sh                                        # special17
bash scripts/train/train_vggt_direct_scene.sh                                  # scene16
GEOMETRY_DIRECT_TOKEN_MODE=camera OUTPUT_DIR=./output/qwen35_vggt_direct_camera \
  bash scripts/train/train_vggt_direct.sh                                      # camera
```

## 全部可覆盖 env

| Env | 默认 | 说明 |
|---|---|---|
| `MODEL_PATH` | `Qwen/Qwen3.5-4B` | base LLM (repo id 或本地路径) |
| `USE_GEOMETRY_ENCODER` | `true` (train.sh) | wrapper 里已固定, 一般不用改 |
| `GEOMETRY_ENCODER_TYPE` | `vggt` (train.sh) | wrapper 里已固定, 一般不用改 |
| `GEOMETRY_ENCODER_PATH` | `facebook/VGGT-1B` 或 `facebook/VGGT-Omega` | 几何编码器 repo id |
| `GEOMETRY_DIRECT_TOKEN_MODE` | `special17` | 仅 vggt_direct: camera/scene16/special17 |
| `GEOMETRY_TOKEN_INSERT_POSITION` | `front` | 仅 vggt_direct: front/back |
| `PRE_DISTILL_WEIGHT` | `0.05`（SceneDistill 母脚本为 `0.2`） | Pre-LLM 蒸馏权重 |
| `POST_DISTILL_WEIGHT` | `0.05` | Post-LLM 蒸馏权重 |
| `OUTPUT_DIR` | `./output/<recipe>` | 最终模型输出目录 |
| `DATASETS` | 4 dataset 混合 | 训练数据集 |
| `LR` | `1e-5` | learning rate |
| `TOTAL_BATCH_SIZE` | `64` | 全局 batch |
| `DATA_FLATTEN` | `False` | 保持 False (Qwen3.5 强制) |
| `REPORT_TO` | `wandb` | 设 `none` 关 wandb |
| `HF_HOME` | `/apdcephfs_gy2/.../hf` | HF cache 根目录 |
| `NPROC_PER_NODE` | 自动检测 | GPU 数量 |
