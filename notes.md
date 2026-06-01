Before training, you should do:

**dduab version**
```bash
srun --job-name=qwen35 --nodes=1 --gpus=8 --time=8:00:00 --partition=normal --account=peilab --pty bash
module load slurm
module load cuda12.2/toolkit/12.2.2
export LD_LIBRARY_PATH=$(python -c "import os, glob; paths=[os.path.abspath(x) for x in glob.glob('/home/dduab/.conda/envs/spatialstack/lib/python3.12/site-packages/nvidia/*/lib')]; print(':'.join(paths))"):$LD_LIBRARY_PATH
export REPO_ROOT=/home/dduab/jiayusheng/SpatialStack-omega/SpatialStack
export SS_ROOT=/project/peilab/jys/spatialstack_store
export HF_HOME=$SS_ROOT/hf_cache
export HUGGINGFACE_HUB_CACHE=$HF_HOME/hub
export HF_XET_HIGH_PERFORMANCE=1
export LD_PRELOAD=/home/dduab/.conda/envs/spatialstack/lib/python3.12/site-packages/nvidia/nvjitlink/lib/libnvJitLink.so.12

export PYTHONPATH=$PWD/src:${PYTHONPATH:-}
```

**yjiaag version**
```bash
srun --job-name=qwen35 --nodes=1 --gpus=8 --time=8:00:00 --partition=normal --account=peilab --pty bash
module load slurm
module load cuda12.2/toolkit/12.2.2
export LD_LIBRARY_PATH=$(python -c "import os, glob; paths=[os.path.abspath(x) for x in glob.glob('/home/yjiaag/.conda/envs/spatialstack/lib/python3.12/site-packages/nvidia/*/lib')]; print(':'.join(paths))"):$LD_LIBRARY_PATH
export REPO_ROOT=/home/yjiaag/SpatialStack-omega/SpatialStack
export SS_ROOT=/project/peilab/jys/spatialstack_store
export HF_HOME=$SS_ROOT/hf_cache
export HUGGINGFACE_HUB_CACHE=$HF_HOME/hub
export HF_XET_HIGH_PERFORMANCE=1
export LD_PRELOAD=/home/yjiaag/.conda/envs/spatialstack/lib/python3.12/site-packages/nvidia/nvjitlink/lib/libnvJitLink.so.12

export PYTHONPATH=$PWD/src:${PYTHONPATH:-}
```
补充datasets, deepspeed, opencv-python-headless lib