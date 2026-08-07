#!/bin/bash
#SBATCH --job-name=chmod777
#SBATCH --output=slurm_logs/chmod777.out
#SBATCH --error=slurm_logs/chmod777-%j.err
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --partition=cpu
#SBATCH --account=peilab

set -euo pipefail

DATA="/project/peilab/jys"

echo "Job started at: $(date)"
echo "Running as user: $(whoami)"
echo "Host: $(hostname)"
echo "Target DATA: $DATA"

if [ ! -d "$DATA" ]; then
    echo "ERROR: DATA directory does not exist: $DATA"
    exit 1
fi

echo "Before chmod:"
ls -ld "$DATA"

echo "Start chmod -R 777 ..."
chmod -R 777 "$DATA"

echo "After chmod:"
ls -ld "$DATA"

echo "Checking a few entries:"
find "$DATA" -maxdepth 2 -printf '%M %u %g %p\n' | head -50

echo "Job finished at: $(date)"
EOF