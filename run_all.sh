#!/bin/bash
#SBATCH --job-name=predict_judge
#SBATCH --partition=P12
#SBATCH --nodes=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=128
#SBATCH --time=24:00:00
#SBATCH --output=/home/Competition2025/P12/%u/slurm_logs/%x-%j.out
#SBATCH --error=/home/Competition2025/P12/%u/slurm_logs/%x-%j.err
#--- 環境変数設定 --------------------------------------------------
PORT=8010
GPU_NUM=$SLURM_GPUS_PER_NODE

#--- log用 -------------------------------------------------------
log() {
  echo "$(date '+%Y-%m-%d %H:%M:%S') [${1^^}] ${*:2}"
}
log INFO "JOB開始: ${SLURM_JOB_NAME}-${SLURM_JOB_ID}"

#--- モジュール & Conda -------------------------------------------
unset LD_LIBRARY_PATH
unset PYTHONPATH

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
echo "HF cache dir : $HF_HOME"

#--- GPU 準備 監視 ------------------------------------------------
#export CUDA_VISIBLE_DEVICES=$(seq -s, 0 $((GPU_NUM-1)))

ulimit -v unlimited
ulimit -m unlimited

nvidia-smi -l 3 > nvidia-smi.log &
pid_nvsmi=$!

#--- 必要なディレクトリを作成 -------------------------------------
cd "$HOME/llm_bridge_prod/eval_dataset"
mkdir -p predictions
mkdir -p judged

#--- vLLM 起動（推論用）---------------------------------------------
# --rope-scaling '{"rope_type":"yarn","factor":2.0,"original_max_position_embeddings":32768}' \
vllm serve /home/Competition2025/P12/shareP12/models/Qwen3-32B \
  --tensor-parallel-size $GPU_NUM \
  --reasoning-parser deepseek_r1 \
  --kv-cache-dtype fp8 \
  --max-model-len 32768 \
  --gpu-memory-utilization 0.9 \
  --port $PORT \
  > vllm_predict.log 2>&1 &
pid_vllm=$!

#--- ヘルスチェック -----------------------------------------------
until curl -s http://127.0.0.1:${PORT}/health >/dev/null; do
  echo "$(date +%T) vLLM starting …"
  sleep 10
done
echo "$(date +%T) vLLM READY"

# モデル一覧を取得して変数に入れる（-s はサイレントモード）
models=$(curl -s http://localhost:${PORT}/v1/models)

# 変数の中身を echo で出力
echo "$models"

# hydra-core対策
export PYTHONPATH=$HOME/.conda/envs/llmbench/lib/python3.12/site-packages:$PYTHONPATH

#--- 推論 ---------------------------------------------------------
python predict.py #> predict.log 2>&1
kill $pid_vllm
log INFO "推論終了"

#--- 評価準備 -----------------------------------------------------
PORT2=$((PORT+1))

#--- vLLM 起動（評価用）--------------------------------------------
# MoEモデルの実行には--enable-expert-parallelが必要ぽい
# export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
vllm serve /home/Competition2025/P12/shareP12/models/Qwen3-235B-A22B-FP8 \
  --tensor-parallel-size $GPU_NUM \
  --reasoning-parser deepseek_r1 \
  --kv-cache-dtype fp8 \
  --max-model-len 32768 \
  --gpu-memory-utilization 0.9 \
  --enable-expert-parallel \
  --port ${PORT2} \
  > vllm_judge.log 2>&1 &
pid_vllm=$!

#--- ヘルスチェック -------------------------------------------------
until curl -s http://127.0.0.1:${PORT2}/health >/dev/null; do
  echo "$(date +%T) vLLM starting …"
  sleep 10
done
echo "$(date +%T) vLLM READY"

# モデル一覧を取得して確認
models2=$(curl -s http://localhost:${PORT2}/v1/models)
echo "$models2"

#--- 評価 -----------------------------------------------------------
export OPENAI_API_KEY="fakeapikey"
python judge_local.py #> judge.log 2>&1
log INFO "評価終了"

#--- ファイルをcopyして整理 -----------------------------------------
# 現在の日時のdirectoryを作成
timestamp=$(date +%Y%m%d_%H%M%S)

source /home/Competition2025/P12/shareP12/utils/envs/${SLURM_JOB_ID}/env.sh
LOG INFO "FILENAME: $P12_FILENAME_TEMP"

mkdir -p "$HOME/llm_bridge_prod/eval_dataset/predictions/$P12_FILENAME_TEMP/$timestamp"
mkdir -p "$HOME/llm_bridge_prod/eval_dataset/judged/$P12_FILENAME_TEMP/$timestamp"

cp "$HOME/llm_bridge_prod/eval_dataset/predictions/$P12_FILENAME_TEMP.json" \
   "$HOME/llm_bridge_prod/eval_dataset/predictions/$P12_FILENAME_TEMP/$timestamp/$P12_FILENAME_TEMP.json"

cp "$HOME/llm_bridge_prod/eval_dataset/judged/$P12_FILENAME_TEMP.json" \
   "$HOME/llm_bridge_prod/eval_dataset/judged/$P12_FILENAME_TEMP/$timestamp/judged_$P12_FILENAME_TEMP.json"

#--- 次のsbatchのためにscancel_hatakeyama --------------------------
bash /home/Competition2025/P12/shareP12/scancel_hatakeyama.sh "${SLURMD_NODENAME#*-}"
log INFO "JOB正常終了"

#--- 後片付け -----------------------------------------------------
kill $pid_vllm
kill $pid_nvsmi
wait
