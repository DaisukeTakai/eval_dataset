#!/bin/bash
#SBATCH --job-name=cron_test
#SBATCH --partition=P12
#SBATCH --nodes=1
#SBATCH --gpus-per-node=0
#SBATCH --cpus-per-task=1
#SBATCH --output=/home/Competition2025/P12/%u/%x.out

echo "[INFO] cron_test.sh started at $(date '+%F %T')"
sleep 15
echo "[INFO] cron_test.sh finished at $(date '+%F %T')"
