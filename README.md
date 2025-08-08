# データセットの評価コード

## 環境構築
運営配布のGithubやNotionを参照してください。
```
#--- モジュール & Conda --------------------------------------------
module purge
module load cuda/12.6 miniconda/24.7.1-py312
module load cudnn/9.6.0  
module load nccl/2.24.3 
conda create -n llmbench python=3.12 -y
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate llmbench

srun --partition=P01 \
     --nodelist=osk-gpu51 \
     --nodes=1 \
     --ntasks=1 \
     --cpus-per-task=8 \
     --gpus-per-node=8 \
     --time=00:30:00 \
     --pty bash -l

conda install -c conda-forge --file requirements.txt
pip install \
  --index-url https://download.pytorch.org/whl/cu126 \
  torch==2.7.1+cu126 torchvision==0.22.1+cu126 torchaudio==2.7.1+cu126 \
  vllm>=0.4.2 \
  --extra-index-url https://pypi.org/simple\
```
## 準備作業
### ファイルの置き場
`$HOME/llm_bridge_prod/`以下に`eval_dataset`フォルダを置いてください。

### 環境に合わせて下記点をご修正ください
`mkdir -p slurm_logs`  
slurmのlogs出力先を指定している行があります。  
各自の環境に合わせて修正してからお使いください。  

## 問題回答用のslurmファイル
'''
sbatch --nodelist=osk-gpu86 run_predict.sh
'''
を実行してください。  
モデル名・データセット名の指定は、`conf/config.yaml`

## 正解判定用のslurmファイル
'''
sbatch --nodelist=osk-gpu86 run_judge.sh
'''
モデル名の指摘は、conf/config.yaml

## 動作確認済みモデル （vLLM対応モデルのみ動作可能です）
- 問題回答: Qwen3 8B
- 正解判定: Qwen3 235B FP8

## configの仕様
`conf/config.yaml`の設定できるパラメーターの説明です。

|フィールド                 |型        |説明                            |
| ----------------------- | -------- | ------------------------------ |
|`dataset`                |string    |評価に使用するベンチマークのデータセットです。全問実施すると時間がかかるため最初は一部の問題のみを抽出して指定してください。|
|`provider`               |string    |評価に使用する推論環境です。vllmを指定した場合、base_urlが必要です。|
|`base_url`               |string    |vllmサーバーのurlです。同じサーバーで実行する場合は初期設定のままで大丈夫です。|
|`model`                  |string    |評価対象のモデルです。vllmサーバーで使われているモデル名を指定してください。|
|`max_completion_tokens`  |int > 0   |最大出力トークン数です。プロンプトが2000トークン程度あるので、vllmサーバー起動時に指定したmax-model-lenより2500ほど引いた値を設定してください。|
|`reasoning`              |boolean   |
|`num_workers`            |int > 1   |同時にリクエストする数です。外部APIを使用時は30程度に、vllmサーバーを使用時は推論効率を高めるため、大きい値に設定してください。|
|`max_samples`            |int > 0   |指定した数の問題をデータセットの前から抽出して、推論します。|
|`judge`                  |string    |LLM評価に使用するOpenAIモデルです。通常はo3-miniを使用ください。|

## Memo
評価結果が`leaderboard`フォルダに書き込まれています。`results.jsonl`と`summary.json`が出力されているかご確認ください。
