#!/bin/bash
#SBATCH --job-name=cron_manager
#SBATCH --partition=P12
#SBATCH --nodelist=osk-gpu86
#SBATCH --nodes=1
#SBATCH --gpus-per-node=0
#SBATCH --cpus-per-task=1
#SBATCH --output=/home/Competition2025/P12/%u/llm_bridge_prod/eval_dataset/cron/%x.out

# --- cron_job のパス設定 ---
CRON_PATH="/home/Competition2025/P12/$USER/llm_bridge_prod/eval_dataset/cron/cron_job.sh"

# --- ディレクトリ設定 ---
WAIT_DIR="/home/Competition2025/P12/$USER/llm_bridge_prod/eval_dataset/conf_queue/waiting"
FIN_DIR="/home/Competition2025/P12/$USER/llm_bridge_prod/eval_dataset/conf_queue/finished"
BACKUP_DIR="$FIN_DIR/initial_backup_$(date +%Y%m%d_%H%M%S)"
CONF_FILE="/home/Competition2025/P12/$USER/llm_bridge_prod/eval_dataset/conf/config.yaml"

mkdir -p "$WAIT_DIR" "$FIN_DIR" "$BACKUP_DIR"

# --- ログ関数 ---
log() { echo "$(date '+%F %T') [$1] ${*:2}"; }

# --- 最初に実行前の config.yaml を backup に移動 ---
if [[ -s "$CONF_FILE" ]]; then
    mv "$CONF_FILE" "$BACKUP_DIR/config.yaml"
    log INFO "Initial backup saved to $BACKUP_DIR/config.yaml"
fi


while true; do
  # YAML ファイルのリスト取得
  mapfile -t YAML_LIST < <(find "$WAIT_DIR" -maxdepth 1 -type f -name '*.yaml' -printf '%f\n' | sort -V)

  log INFO "Found ${#YAML_LIST[@]} YAML(s): ${YAML_LIST[*]}"

  # WAIT_DIR の YAML ファイルが存在しない場合、終了
  if ((${#YAML_LIST[@]} == 0)); then
    log INFO "No YAML found in $WAIT_DIR."
    break
  fi

  # WAIT_DIR の YAML ファイルを順に処理
  for yaml_base in "${YAML_LIST[@]}"; do
    yaml="$WAIT_DIR/$yaml_base"
    # WAIT_DIR から conf/config.yaml をコピー
    cp -a "$yaml" "$CONF_FILE"

    # cron_job 実行
    bash /home/Competition2025/P12/shareP12/scancel_hatakeyama.sh gpu86
    sbatch --wait "$CRON_PATH"
    log INFO "Cron job was finished."

    # 使用した YAML を FIN_DIR にアーカイブ
    cp -a "$CONF_FILE" "$FIN_DIR/$(date +%Y%m%d_%H%M%S)_$yaml_base"
    rm -f "$yaml"
  done
  sleep 10
done
