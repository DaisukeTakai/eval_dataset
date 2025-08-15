#!/bin/bash
#SBATCH --job-name=cron_job
#SBATCH --partition=P12
#SBATCH --nodelist=osk-gpu86
#SBATCH --nodes=1
#SBATCH --gpus-per-node=0
#SBATCH --cpus-per-task=1
#SBATCH --output=/home/Competition2025/P12/%u/llm_bridge_prod/eval_dataset/cron/%x.out

echo "[INFO] cron.sh started at $(date '+%F %T')"

# === 各自の設定項目 ===
MAX_SUBMIT=5   # ★ ここで上限を設定
MY_JOB_PATH="/home/Competition2025/P12/$USER/llm_bridge_prod/eval_dataset/cron/cron_test.sh"
SLEEP_SEC=600   # job 追加の間隔設定

# === 変数の設定 ===
total_submit=0
TARGET_USER="kan.hatakeyama"
SCANCEL_SCRIPT="/home/Competition2025/P12/shareP12/scancel_hatakeyama.sh"
ENV_PATH="/home/Competition2025/P12/shareP12/utils/envs/env_cron.sh"

# === 初期化（起動時に必ず true） ===
CRON_FLAG=true
echo "CRON_FLAG=$CRON_FLAG" > "$ENV_PATH"

while true; do
  #=== フラグ読み込み ===
  if [[ -f "$ENV_PATH" ]]; then
    source "$ENV_PATH"
  fi

  #=== CRON_FLAG=true 以外の場合はスキップして待機 ===
  if [[ "${CRON_FLAG,,}" != "true" ]]; then
    echo "[INFO] CRON_FLAG!=true → sbatch 投入せず$(( SLEEP_SEC / 60 ))分待機"
    sleep "$SLEEP_SEC"
    continue
  fi

  #=== 現在のhatakeyamaジョブ数 ===
  hatakeyama_cnt=$(squeue -h -u "$TARGET_USER" -t RUNNING -o "%A" | wc -l)

  #=== 現在のPENDINGジョブ数 ===
  pending_cnt=$(squeue -h -t PENDING -o "%A" | wc -l)
  echo "[INFO] 現在のhatakeyamaジョブ数: ${hatakeyama_cnt} / PENDINGジョブ数: ${pending_cnt}"

  #=== 自分のジョブ投入回数 ===
  submit=$(( 3 - pending_cnt + hatakeyama_cnt ))
  (( submit < 0 )) && submit=0

  #=== 最大投入数制限チェック ===
  if (( total_submit + submit > MAX_SUBMIT )); then
    submit=$(( MAX_SUBMIT - total_submit ))
  fi

  #=== 投入 ===
  for ((i=0; i<submit; i++)); do
    sbatch "$MY_JOB_PATH"
    (( total_submit++ ))
  done

  #=== キャンセル ===
  bash "$SCANCEL_SCRIPT" gpu84 gpu85 gpu86

  echo "[INFO] $(date '+%F %T') 投入: ${submit}件 / 累積: ${total_submit}件"

  #=== 上限到達で終了 ===
  if (( total_submit >= MAX_SUBMIT )); then
    echo "[INFO] 累積投入回数 ${total_submit}件 に達したため終了します。"
    break
  fi

  #=== 待機 ===
  sleep "$SLEEP_SEC"
done
