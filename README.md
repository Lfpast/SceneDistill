# Data Preparation:

### 1. Prepare all Environment Variables
```bash
export REPO_ROOT=YOUR_CODE_BASE_DIR
export SS_ROOT=YOUR_DATASET_BASE_DIR
export HF_HOME=$SS_ROOT/hf_cache
export HUGGINGFACE_HUB_CACHE=$HF_HOME/hub
export HF_XET_HIGH_PERFORMANCE=1
mkdir -p $SS_ROOT
mkdir -p $HF_HOME
Verification:
ls -ld $REPO_ROOT
ls -ld $SS_ROOT
ls -ld $HF_HOME
```

### 2. Login your Huggingface Account
```bash
hf auth login
hf auth whoami
```
然后登录⽹⻚版huggingface确认登录,成功后终端会有输出

### 3. Create Dataset Backbone in your Dataset Root Disk
```bash
# Create Empty Content
mkdir -p $SS_ROOT/data/annotations
mkdir -p $SS_ROOT/data/train
mkdir -p $SS_ROOT/data/media
mkdir -p $SS_ROOT/data/vlm3r/annotations/vsibench_train
mkdir -p $SS_ROOT/data/vlm3r/media/scannet
mkdir -p $SS_ROOT/data/vsi_590k/annotations
mkdir -p $SS_ROOT/data/vsi_590k/media
```

Verify the Root Tree
```bash
find $SS_ROOT/data -maxdepth 3 -type d | sort
```

### 4. Link the data/ Directory inside Codebase to Database
```bash
ln -sfn $SS_ROOT/data $REPO_ROOT/data
```

### 5. Dataset Metadatas Download
```bash
# Download all jsons
hf download Journey9ni/SpatialStackData \--repo-type dataset \--include "*.json" \--local-dir $SS_ROOT/data/annotations
# Build 4 directory linkage
ln -sfn ../annotations/spar_234k.json \
$SS_ROOT/data/train/spar_234k.json
ln -sfn ../annotations/llava_hound_64k.json \
$SS_ROOT/data/train/llava_hound_64k.json
# Verification
ln -sfn ../../../annotations/merged_qa_scannet_train.json \
$SS_ROOT/data/vlm3r/annotations/vsibench_train/merged_qa_scannet_train.json
ln -sfn ../../annotations/vsi_appearance_order_vsibench_scannet.json \
$SS_ROOT/data/vsi_590k/annotations/vsi_appearance_order_vsibench_scannet.json
ls -l $SS_ROOT/data/train
ls -l $SS_ROOT/data/vlm3r/annotations/vsibench_train
ls -l $SS_ROOT/data/vsi_590k/annotations
```
You can see all above directories should exist

### 6. Download SPAR Media Dataset
```bash
# First download all .tar.gz files
mkdir -p $SS_ROOT/data/media/spar
hf download jasonzhango/SPAR-7M \--repo-type dataset \--revision 976c19177468eabe64e9e2dd0f0450cd32dacc1f \--include "spar-*.tar.gz" \--local-dir $SS_ROOT/data/media/spar
# Check all splits existence
ls $SS_ROOT/data/media/spar/spar-*.tar.gz | sort
# Do concatenation unzip operation (very long, recommend to use tmux session >1h)
(
cd $SS_ROOT/data/media
cat \
spar/spar-00.tar.gz spar/spar-01.tar.gz spar/spar-02.tar.gz spar/spar-03.tar.gz \
spar/spar-04.tar.gz spar/spar-05.tar.gz spar/spar-06.tar.gz spar/spar-07.tar.gz \
spar/spar-08.tar.gz spar/spar-09.tar.gz spar/spar-10.tar.gz spar/spar-11.tar.gz \
spar/spar-12.tar.gz spar/spar-13.tar.gz \
| pigz -dc | tar -xf - -C .
)
# Do final check
find $SS_ROOT/data/media/spar -maxdepth 2 -type d | sort | head -n 50
```

### 7. Download LLaVA-Hound Media Dataset
```bash
# First download the train_300k subtree
mkdir -p $SS_ROOT/data/media/llava_hound
hf download ShareGPTVideo/train_video_and_instruction \--repo-type dataset \--include "train_300k/**" \--local-dir $SS_ROOT/data/media/llava_hound
# Check all chunks
find $SS_ROOT/data/media/llava_hound/train_300k -maxdepth 1 -name 'chunk_*.tar.gz' | sort | head
find $SS_ROOT/data/media/llava_hound/train_300k -maxdepth 1 -name 'chunk_*.tar.gz' | wc -l
# Unzip all chunks to frames/
mkdir -p $SS_ROOT/data/media/llava_hound/frames
find $SS_ROOT/data/media/llava_hound/train_300k \-maxdepth 1 -name 'chunk_*.tar.gz' -print0 \
| xargs -0 -P"$(nproc)" -I{} tar -I pigz -x -f "{}" \-C $SS_ROOT/data/media/llava_hound/frames
# Verify the structure
find $SS_ROOT/data/media/llava_hound/frames -maxdepth 2 | head -n 30
```

### 8. Download VLM-3R Dataset
```bash
# Download the dataset. Remember to get access first!!
mkdir -p $SS_ROOT/data/vlm3r/media/scannet
hf download Journey9ni/aweb \--repo-type dataset \--include "ScanNet/videos/train/**" \--local-dir $SS_ROOT/data/vlm3r/media/scannet
# Change the name
mv $SS_ROOT/data/vlm3r/media/scannet/ScanNet/videos/train \
$SS_ROOT/data/vlm3r/media/scannet/videos
# Verify the result
find $SS_ROOT/data/vlm3r/media/scannet -maxdepth 2 -type d | sort | head -n 30
# Make shared entry for VSI-590K
ln -sfn ../../vlm3r/media/scannet/videos \
$SS_ROOT/data/vsi_590k/media/scannet
# Check
ls -l $SS_ROOT/data/vsi_590k/media
readlink -f $SS_ROOT/data/vsi_590k/media/scannet
```

### 9. Final Review
```bash
# Check 4 annotations
ls -l $REPO_ROOT/data/train/spar_234k.json
ls -l $REPO_ROOT/data/train/llava_hound_64k.json
ls -l $REPO_ROOT/data/vlm3r/annotations/vsibench_train/merged_qa_scannet_train.json
ls -l $REPO_ROOT/data/vsi_590k/annotations/vsi_appearance_order_vsibench_scannet.json
# Check 3 media files
ls -ld $REPO_ROOT/data/media
ls -ld $REPO_ROOT/data/media/spar
ls -ld $REPO_ROOT/data/media/llava_hound
ls -ld $REPO_ROOT/data/vlm3r/media
ls -ld $REPO_ROOT/data/vsi_590k/media
# Check soft link paths
readlink -f $REPO_ROOT/data
readlink -f $REPO_ROOT/data/vsi_590k/media/scannet
```

# Training (LayerNorm Version):

The model output path can be modified inside `train.sh` (`OUTPUT_DIR`). 
```bash
MODEL_PATH=Qwen/Qwen3.5-4B \
USE_GEOMETRY_ENCODER=True \
GEOMETRY_ENCODER_TYPE=vggt_omega_alpha \
GEOMETRY_ENCODER_PATH=PATH_TO_VGGT_OMEGA \
DATA_FLATTEN=False \
bash scripts/train/train.sh
```

# Evaluation (LayerNorm Version):

The evaluation result path can be modified inside `eval.sh` (`FORCE_OUTPUT_ROOT` & `FORCE_OUTPUT_PATH`)
```bash
MODEL_PATH=/project/peilab/jys/spatialstack_store/hf_cache/hub/models--nyu-visionx--Cambrian-S-3B/snapshots/9d8cbd75ab3ed53683b171d6fdd888f81f0febd1 \
MODEL_IMPL=cambrians \
MODEL_ARGS_BASE="pretrained=/project/peilab/jys/spatialstack_store/hf_cache/hub/models--nyu-visionx--Cambrian-S-3B/snapshots/9d8cbd75ab3ed53683b171d6fdd888f81f0febd1,max_num_frames=32,max_length=12800,disable_thinking=true" \
BENCHMARKS="cvbench" \
bash scripts/evaluation/eval.sh
```

## Qwen 3.5-4B
```bash
MODEL_PATH=Qwen/Qwen3.5-4B \
MODEL_IMPL=qwen3_5 \
MODEL_ARGS_BASE="pretrained=Qwen/Qwen3.5-4B,use_flash_attention_2=true,max_num_frames=32,max_length=12800,disable_thinking=true" \
BENCHMARKS="vsibench, cvbench, blink_spatial, sparbench, videomme, mmsibench" \
bash scripts/evaluation/eval.sh
```

## SpatialStack
```bash
MODEL_PATH=Journey9ni/SpatialStack-Qwen3.5-4B \
MODEL_IMPL=qwen3_5 \
MODEL_ARGS_BASE="pretrained=Journey9ni/SpatialStack-Qwen3.5-4B,use_flash_attention_2=true,max_num_frames=32,max_length=12800,geometry_encoder_path=facebook/VGGT-1B,disable_thinking=true" \
BENCHMARKS="vsibench, cvbench, sparbench, videomme" \
bash scripts/evaluation/eval.sh
```

## Cambrian-S
```bash
MODEL_PATH=nyu-visionx/Cambrian-S-3B \
MODEL_IMPL=cambrians \
MODEL_ARGS_BASE="pretrained=nyu-visionx/Cambrian-S-3B,max_num_frames=32,max_length=12800,disable_thinking=true" \
BENCHMARKS="vsibench, cvbench, sparbench, videomme" \
bash scripts/evaluation/eval.sh
```

## Baseline
```bash
MODEL_PATH=/project/peilab/jys/qwen3_5_output/baseline \
MODEL_IMPL=qwen3_5 \
MODEL_ARGS_BASE="pretrained=/project/peilab/jys/qwen3_5_output/baseline,use_flash_attention_2=true,max_num_frames=32,max_length=12800,geometry_encoder_path=facebook/VGGT-1B,disable_thinking=true" \
BENCHMARKS="vsibench, cvbench, sparbench, videomme" \
bash scripts/evaluation/eval.sh
```

## VGLLM
```bash
MODEL_PATH=zd11024/vgllm-qa-vggt-4b \
MODEL_IMPL=vgllm \
MODEL_ARGS_BASE="pretrained=zd11024/vgllm-qa-vggt-4b,use_flash_attention_2=true,max_num_frames=32,max_length=12800" \
BENCHMARKS="vsibench, cvbench, sparbench, videomme" \
bash scripts/evaluation/eval.sh
```

## Spatial-MLLM
```bash
MODEL_PATH=Diankun/Spatial-MLLM-v1.1-Instruct-135K \
MODEL_IMPL=spatial_mllm \
MODEL_ARGS_BASE="pretrained=Diankun/Spatial-MLLM-v1.1-Instruct-135K,spatial_mllm_repo_path=Spatial-MLLM,use_flash_attention_2=true,max_num_frames=32,max_length=12800" \
BENCHMARKS="vsibench, cvbench, sparbench, videomme" \
bash scripts/evaluation/eval.sh
```

---

# Probing Experiment

```bash
module load slurm
module load cuda12.2/toolkit/12.2.2
conda activate Probing
export SS_ROOT=/project/peilab/jys/probing_data
export HF_HOME=$SS_ROOT/hf_cache
export HUGGINGFACE_HUB_CACHE=$HF_HOME/hub
export HF_XET_HIGH_PERFORMANCE=1
export PROBING_DATA=/project/peilab/jys/probing_data
cd ~/jiayusheng/SpatialStack-omega/Probing-VLM-VGM
cd /tmp

python download.py \
    --odir "$PROBING_DATA/DL3DV/DL3DV-ALL-960P" \
    --subset 1K \
    --resolution 960P \
    --file_type images+poses \
    --clean_cache

wget https://raw.githubusercontent.com/DL3DV-10K/Dataset/main/scripts/download.py
wget https://kaldir.vc.cit.tum.de/scannet/download-scannet.py


python download-scannet.py -o "$PROBING_DATA/ScanNet"
```


## Frozen Feature Extraction

### Qwen 3.5

**DL3DV**:
```bash
export PROBING_DATA=/project/peilab/jys/probing_data
python -m features.run_dl3dv \
  --vfm qwen35 \
  --vfm-name qwen3.5-4b \
  --subset all \
  --dl3dv-root "$PROBING_DATA/DL3DV/DL3DV-ALL-960P" \
  --processed-root "$PROBING_DATA/DL3DV/DL3DV-processed" \
  --out-root "$PROBING_DATA/DL3DV/FEAT" \
  --model-path "$QWEN35_PATH" \
  --model-type qwen35 \
  --use-query-frame-indices \
  --context-len 76 \
  --query-idx-divisor 4 \
  --output-layers 20

# Visual Encoder
export QWEN35_PATH=/project/peilab/jys/spatialstack_store/hf_cache/hub/models--Qwen--Qwen3.5-4B/snapshots/851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a
python -m features.run_dl3dv \
  --vfm qwen35 \
  --vfm-name qwen3.5-4b-visual \
  --subset all \
  --dl3dv-root "$PROBING_DATA/DL3DV/DL3DV-ALL-960P" \
  --processed-root "$PROBING_DATA/DL3DV/DL3DV-processed" \
  --out-root "$PROBING_DATA/DL3DV/FEAT" \
  --model-path "$QWEN35_PATH" \
  --model-type qwen35-visual \
  --use-query-frame-indices \
  --context-len 76 \
  --query-idx-divisor 4 \
  --output-layers 24
```

**ScanNet**
```bash
export SCANNET_ROOT=/project/peilab/jys/probing_data/ScanNet
export QWEN35_PATH=/project/peilab/jys/spatialstack_store/hf_cache/hub/models--Qwen--Qwen3.5-4B/snapshots/851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a
python -m features.run_scannet \
  --vfm qwen35 \
  --vfm-name qwen3.5-4b \
  --split both \
  --scannet-root "$SCANNET_ROOT/ScanNet-processed" \
  --out-root "$SCANNET_ROOT/FEAT" \
  --model-path "$QWEN35_PATH" \
  --model-type qwen35 \
  --use-query-frame-indices \
  --context-len 76 \
  --query-idx-divisor 4 \
  --output-layers 1

  # Visual Encoder
export QWEN35_PATH=/project/peilab/jys/spatialstack_store/hf_cache/hub/models--Qwen--Qwen3.5-4B/snapshots/851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a
python -m features.run_scannet \
  --vfm qwen35 \
  --vfm-name qwen3.5-4b-visual \
  --split both \
  --scannet-root "$SCANNET_ROOT/ScanNet-processed" \
  --out-root "$SCANNET_ROOT/FEAT" \
  --model-path "$QWEN35_PATH" \
  --model-type qwen35-visual \
  --use-query-frame-indices \
  --context-len 76 \
  --query-idx-divisor 4 \
  --output-layers 1
```

### VGGT-Omega
**DL3DV**
```bash
export VGGT_OMEGA_PATH=/project/peilab/jys/spatialstack_store/hf_cache/hub/models--facebook--VGGT-Omega/vggt_omega_1b_512.pt
python -m features.run_dl3dv \
  --vfm vggt_omega \
  --vfm-name vggt-omega \
  --subset all \
  --dl3dv-root "$PROBING_DATA/DL3DV/DL3DV-ALL-960P" \
  --processed-root "$PROBING_DATA/DL3DV/DL3DV-processed" \
  --out-root "$PROBING_DATA/DL3DV/FEAT" \
  --model-path "$VGGT_OMEGA_PATH" \
  --use-query-frame-indices \
  --context-len 76 \
  --query-idx-divisor 4 \
  --output-layers 24
```

### SpatialStack
```bash
export SPATIALSTACK_PATH=/project/peilab/jys/spatialstack_store/hf_cache/hub/models--Journey9ni--SpatialStack-Qwen3.5-4B/snapshots/777b2289252f7c78628e0f8ac63ffddf50fe7f7b
export VGGT_OMEGA_PATH=/project/peilab/jys/spatialstack_store/hf_cache/hub/models--facebook--VGGT-Omega/vggt_omega_1b_512.pt

python -m features.run_dl3dv \
  --vfm qwen35 \
  --vfm-name spatialstack \
  --subset all \
  --dl3dv-root "$PROBING_DATA/DL3DV/DL3DV-ALL-960P" \
  --processed-root "$PROBING_DATA/DL3DV/DL3DV-processed" \
  --out-root "$PROBING_DATA/DL3DV/FEAT" \
  --model-path "$SPATIALSTACK_PATH" \
  --model-type spatialstack-qwen35 \
  --geometry-encoder-path "$VGGT_OMEGA_PATH" \
  --geometry-encoder-type vggt_omega \
  --use-query-frame-indices \
  --context-len 76 \
  --query-idx-divisor 4 \
  --output-layers 32
```

## 3D Geometry

### Qwen 3.5

```bash
cd ~/jiayusheng/SpatialStack-omega/Probing-VLM-VGM

CUDA_VISIBLE_DEVICES=0,1 python -m probing_vlm_vgm.train \
  experiment=dl3dv/qwen3.5-4b-visual \
  job_name="Layer20" \
  data.data_root="$PROBING_DATA/DL3DV/DL3DV-processed" \
  data.feat_root="$PROBING_DATA/DL3DV/FEAT" \
  feat_postfix="_layer20" \
  seed=42 \
  trainer.devices=2 \
  trainer.max_epochs=60 \
  callbacks.model_checkpoint.save_top_k=0

# Evaluation
cd ~/jiayusheng/SpatialStack-omega/Probing-VLM-VGM
ROOT=/project/peilab/jys/probing/DL3DV/qwen3.5-4b-visual
export PROBING_DATA=/project/peilab/jys/probing_data
TRAIN_RUN=Layer32
EVAL_RUN=${TRAIN_RUN}_eval

CUDA_VISIBLE_DEVICES=0 python -m probing_vlm_vgm.train \
  experiment=dl3dv/qwen3.5-4b \
  job_name="$EVAL_RUN" \
  data.data_root="$PROBING_DATA/DL3DV/DL3DV-processed" \
  data.feat_root="$PROBING_DATA/DL3DV/FEAT" \
  feat_postfix=_layer32 \
  trainer.devices=1 \
  train=false \
  test=true \
  autoresume=false \
  +model.skip_test_viz=true \
  ckpt_path="$ROOT/$TRAIN_RUN/checkpoints/last.ckpt"
```

### VGGT-Omega

```bash
cd ~/jiayusheng/SpatialStack-omega/Probing-VLM-VGM

CUDA_VISIBLE_DEVICES=0,1 python -m probing_vlm_vgm.train \
  experiment=dl3dv/vggt-omega \
  job_name="Layer24" \
  data.data_root="$PROBING_DATA/DL3DV/DL3DV-processed" \
  data.feat_root="$PROBING_DATA/DL3DV/FEAT" \
  feat_postfix=_layer24 \
  trainer.devices=2 \
  trainer.max_epochs=60 \
  callbacks.model_checkpoint.save_top_k=0

# Evaluation
ROOT=/project/peilab/jys/probing/DL3DV/vggt-omega
export PROBING_DATA=/project/peilab/jys/probing_data
TRAIN_RUN=Layer32
EVAL_RUN=${TRAIN_RUN}_eval

CUDA_VISIBLE_DEVICES=0 python -m probing_vlm_vgm.train \
  experiment=dl3dv/vggt-omega \
  job_name="$EVAL_RUN" \
  data.data_root="$PROBING_DATA/DL3DV/DL3DV-processed" \
  data.feat_root="$PROBING_DATA/DL3DV/FEAT" \
  feat_postfix=_layer32 \
  trainer.devices=1 \
  train=false \
  test=true \
  autoresume=false \
  +model.skip_test_viz=true \
  ckpt_path="$ROOT/$TRAIN_RUN/checkpoints/last.ckpt"
```

### SpatialStack
```bash
cd ~/jiayusheng/SpatialStack-omega/Probing-VLM-VGM

CUDA_VISIBLE_DEVICES=0,1 python -m probing_vlm_vgm.train \
  experiment=dl3dv/spatialstack \
  job_name=Layer1 \
  data.data_root="$PROBING_DATA/DL3DV/DL3DV-processed" \
  data.feat_root="$PROBING_DATA/DL3DV/FEAT" \
  feat_postfix=_layer1 \
  trainer.devices=2 \
  trainer.max_epochs=60 \
  callbacks.model_checkpoint.save_top_k=0

# Evaluation
ROOT=/project/peilab/jys/probing/DL3DV
export PROBING_DATA=/project/peilab/jys/probing_data
TRAIN_RUN=Layer32
EVAL_RUN=${TRAIN_RUN}_eval

CUDA_VISIBLE_DEVICES=0 python -m probing_vlm_vgm.train \
  experiment=dl3dv/spatialstack \
  job_name="$EVAL_RUN" \
  data.data_root="$PROBING_DATA/DL3DV/DL3DV-processed" \
  data.feat_root="$PROBING_DATA/DL3DV/FEAT" \
  feat_postfix=_layer32 \
  trainer.devices=1 \
  train=false \
  test=true \
  autoresume=false \
  +model.skip_test_viz=true \
  ckpt_path="$ROOT/$TRAIN_RUN/checkpoints/last.ckpt"
```

## Semantic Tagging

### Qwen 3.5
```bash
cd ~/jiayusheng/SpatialStack-omega/Probing-VLM-VGM

CUDA_VISIBLE_DEVICES=0 python -m probing_vlm_vgm.train \
  experiment=scannet_tagging/qwen3.5-4b \
  job_name=Layer1 \
  feat_postfix=_layer1 \
  trainer.devices=1 \
  callbacks.model_checkpoint.save_top_k=0

# Evaluate
WANDB_MODE=offline CUDA_VISIBLE_DEVICES=0 bash scripts/eval_scannet_tagging.sh \
  --views 8 \
  --runs-dir /project/peilab/jys/probing/ScanNet/qwen3.5-4b/semantic-tagging \
  --run-name Layer1 \
  --vfm qwen3.5-4b \
  --skip-done
```


## Instance Grouping

### Qwen 3.5
```bash
cd ~/jiayusheng/SpatialStack-omega/Probing-VLM-VGM

CUDA_VISIBLE_DEVICES=0,1 python -m probing_vlm_vgm.train \
  experiment=scannet/qwen3.5-4b \
  job_name=qwen3.5-4b_layer20 \
  feat_postfix=_layer20 \
  trainer.devices=2 \
  callbacks.model_checkpoint.save_top_k=0

# Evaluation
WANDB_MODE=offline CUDA_VISIBLE_DEVICES=0,1 python scripts/eval_scannet_instance_sharded.py \
  --views 8 \
  --runs-dir /project/peilab/jys/probing/ScanNet/qwen3.5-4b/instance-grouping \
  --output-root /project/peilab/jys/probing/ScanNet/qwen3.5-4b/instance-grouping \
  --eval-in-run-dir \
  --gpus 0,1 \
  --vfm qwen3.5-4b \
  --project Instance-Grouping \
  --hdbscan-workers 8 \
  --num-workers 2 \
  --skip-done
```