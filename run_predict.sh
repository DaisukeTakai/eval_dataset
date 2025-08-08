#!/bin/bash
#SBATCH --job-name=predict
#SBATCH --partition=P12
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=40
#SBATCH --time=24:00:00
#SBATCH --output=/home/Competition2025/P12/%u/slurm_logs/%x-%j.out
#SBATCH --error=/home/Competition2025/P12/%u/slurm_logs/%x-%j.err
#--- log用 --------------------------------------------------------
log() {
  echo "$(date '+%Y-%m-%d %H:%M:%S') [${1^^}] ${*:2}"
}
log INFO "JOB開始: ${SLURM_JOB_NAME}-${SLURM_JOB_ID}"

#--- モジュール & Conda -------------------------------------------
module purge
module load cuda/12.6 miniconda/24.7.1-py312
module load cudnn/9.6.0
module load nccl/2.24.3
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate llmbench

# Hugging Face 認証
export HF_TOKEN=""
export HF_HOME=${SLURM_TMPDIR:-$HOME}/.cache/huggingface
export TRANSFORMERS_CACHE=$HF_HOME
export HUGGINGFACE_HUB_TOKEN=$HF_TOKEN
export HF_HUB_ENABLE_HF_TRANSFER=1

export NCCL_SOCKET_IFNAME=enp25s0np0
export NVTE_FUSED_ATTN=0

mkdir -p "$HF_HOME"
echo "HF cache dir : $HF_HOME"                   # デバッグ用

#--- GPU 準備 監視 ------------------------------------------------
export CUDA_VISIBLE_DEVICES=0,1,2,3,4 #,5,6,7

ulimit -v unlimited
ulimit -m unlimited

nvidia-smi -l 3 > nvidia-smi.log &
pid_nvsmi=$!

#--- 必要なディレクトリを作成 -------------------------------------
cd ${SLURM_TMPDIR:-$HOME}/llm_bridge_prod/eval_dataset
mkdir -p predictions
#mkdir -p judged

#--- vLLM 起動（8GPU）---------------------------------------------
vllm serve /home/Competition2025/P12/shareP12/models/Qwen3-32B \
  --tensor-parallel-size 8 \
  --reasoning-parser deepseek_r1 \
  --rope-scaling '{"rope_type":"yarn","factor":4.0,"original_max_position_embeddings":32768}' \
  --max-model-len 131072 \
  --gpu-memory-utilization 0.95 \
  > vllm.log 2>&1 &
pid_vllm=$!

#--- ヘルスチェック -----------------------------------------------
until curl -s http://127.0.0.1:8000/health >/dev/null; do
  echo "$(date +%T) vLLM starting …"
  sleep 10
done
echo "$(date +%T) vLLM READY"

# モデル一覧を取得して変数に入れる（-s はサイレントモード）
models=$(curl -s http://localhost:8000/v1/models)

# 変数の中身を echo で出力
echo "$models"

# hydra-core対策
export PYTHONPATH=$HOME/.conda/envs/llmbench/lib/python3.12/site-packages:$PYTHONPATH

#--- 推論 ---------------------------------------------------------
python predict.py #> predict.log 2>&1

#--- 評価 ---------------------------------------------------------
# python judge.py > judge.log 2>&1
log INFO "JOB正常終了"

#--- 後片付け -----------------------------------------------------
kill $pid_vllm
kill $pid_nvsmi
wait
