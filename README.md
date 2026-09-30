# CobotMagic ROS–Policy Bridge

CobotMagicのROS topicと外部ポリシーサーバーをZeroMQで接続するためのデプロイブリッジです。
このリポジトリの中心はブリッジノードの設定と起動であり、モデル本体や学習コードは含みません。

ブリッジはROSからカメラ画像と関節状態を受け取り、ポリシーサーバーへ観測を送信し、
返されたアクションをCobotMagicの指令topicへpublishします。

```text
CobotMagic ROS topics
  -> ros_bridge_node.py
  -> ZeroMQ
  -> policy server
  -> ZeroMQ
  -> ros_bridge_node.py
  -> CobotMagic command topics
```

主な実装は `cobotmagic_deployment/`、カメラ専用ブリッジは `realsense_bridge/` にあります。

このバージョンで対応するポリシーは **OpenPI、OpenVLA、X-VLA、DreamZero、OpenWAM** です。
対応表は[付録](#付録-対応ポリシー)を参照してください。

OpenWAMの双腕Piper用重み・専用環境・起動手順は [docs/openwam/OPENWAM_PIPER_DEPLOY.md](docs/openwam/OPENWAM_PIPER_DEPLOY.md) を参照してください。

## 1. 環境構築

### 1.1 Dockerコンテナを使う場合

GPUサーバー上で直接環境を構築するか、Dockerコンテナを利用します。
次の例では、GPU、ホストネットワーク、共有メモリを有効にし、ホストの作業ディレクトリを
コンテナの `/workspace` にマウントします。

```bash
docker run -it --gpus all \
  --name <container-name> \
  --network host \
  --shm-size=32g \
  -v <host-workspace>:/workspace \
  <docker-image>
```

- `<container-name>`、`<host-workspace>`、`<docker-image>` は実行環境に合わせて置き換えてください。
- ROS masterや実機と通信するため、ここでは `--network host` を使用します。
- ブリッジだけを動かし、ポリシーサーバーを別環境で動かす場合は、ブリッジ側のGPUは必須ではありません。
- `--shm-size` は同じコンテナで動かすモデルやデータローダーに合わせて調整してください。
- 既存コンテナを再利用する場合は `docker start -ai <container-name>` で起動します。
- チェックポイント、ログ、データはマウントした `/workspace` 配下へ保存してください。

Dockerを使わない場合も、以降は作業ディレクトリを `/workspace` として説明します。
別の場所を使う場合は、コマンドとYAML内の絶対パスを読み替えてください。

### 1.2 リポジトリを取得する

```bash
cd /workspace
git clone https://github.com/Tanichu-Laboratory/cobotmagic_deploy.git
cd cobotmagic_deploy
```

### 1.3 ROSブリッジ用Conda環境

ROSブリッジには、リポジトリに含まれるROS Noetic用環境を使用します。

```bash
cd /workspace/cobotmagic_deploy
conda env create -f aloha.yml
conda activate aloha
```

`aloha` 環境がすでに存在する場合は、定義を更新できます。

```bash
conda env update -n aloha -f aloha.yml --prune
```

ポリシーサーバーはモデルごとにPythonやCUDAの要件が異なるため、`aloha` 環境へ
モデル依存パッケージを追加せず、モデル専用のConda環境または仮想環境で起動してください。

### 1.4 ROSネットワーク

ROS masterと、ブリッジを起動するマシンのIPアドレスを設定します。

```bash
export ROS_MASTER_URI=http://<cobotmagic-ip>:11311
export ROS_IP=<bridge-machine-ip>
```

CobotMagicのIPアドレスが変わった場合は `ROS_MASTER_URI` も更新してください。
継続して使う場合はシェルの設定ファイルや専用セットアップスクリプトに保存します。

ROS masterへ接続できることを確認します。

```bash
rostopic list
```

`/puppet/joint_left`、`/puppet/joint_right`、使用するカメラtopicなどが表示されれば、
ブリッジからROS側を参照できています。

## 2. ブリッジの設定

ブリッジとポリシーサーバーは同じYAMLを参照します。使用する構成に近いYAMLを選び、
ROS topic、ZeroMQ接続先、タスク、初期姿勢を実行環境に合わせて変更してください。

通常の両腕構成では `cobotmagic_deployment/bridges/ros_bridge_node.py`、右単腕構成では
`cobotmagic_deployment/bridges/ros_bridge_node_single_right.py` を使用します。

設定変更時に特に確認する項目は次のとおりです。

| YAML項目 | 用途 |
| --- | --- |
| `task_prompt` | ポリシーへ送るタスク指示 |
| `ros.topics` | カメラ、関節状態、関節指令、enable flagのROS topic |
| `ros.image_type` | `raw` または `compressed` の画像入力 |
| `ros.rate_hz` | ブリッジの基本制御周期 |
| `ros.action_mode` | 関節絶対値、関節差分、EEFなどのアクション解釈 |
| `ros.open_loop_steps` | 1回の応答チャンクから実行するステップ数 |
| `ros.home_position` | 推論開始前に移動する初期姿勢と移動方法 |
| `ros.action_log` | アクションログの有効化と保存先 |
| `zmq.client_connect` | ブリッジが接続するポリシーサーバーのアドレス |
| `zmq.server_bind` | ポリシーサーバーがbindするアドレス |

ブリッジとポリシーサーバーを同一マシンで動かす場合は、通常
`tcp://127.0.0.1:<port>` を使用します。別マシンの場合は、サーバー側を到達可能な
インターフェースへbindし、`zmq.client_connect` にポリシーサーバーのIPアドレスを設定してください。
ポート番号が一致し、ホストやコンテナのファイアウォールを通過できることも確認します。

初期姿勢、`action_mode`、指令topicの誤設定は実機の予期しない動作につながります。
設定を変更した後は、ロボットをすぐ停止できる状態で低速・小さい動作から確認してください。

## 3. CobotMagic側の起動

CobotMagic側でROS master、アーム、カメラを起動します。実際のパスやlaunchファイルは
機体側の構成に合わせてください。

```bash
roscore

cd /workspace/ros_cobotmagic/Piper_ros_private-ros-noetic
bash can_config.sh
roslaunch piper start_ms_piper.launch mode:=1 auto_enable:=true

roslaunch astra_camera multi_camera.launch
```

D405を使う場合は、各機体のserial IDに対応するRealSenseノードを起動します。
ブリッジを起動する前に、ブリッジ側から `rostopic list` を実行し、YAMLの
`ros.topics` に指定したtopicが存在することを確認してください。

## 4. ブリッジノードの起動

コマンドはcloneしたリポジトリのルートから実行します。先にCobotMagic側のROSノードと
ポリシーサーバーを起動してから、別ターミナルでブリッジを起動します。

### 4.1 標準の両腕ブリッジ

```bash
cd /workspace/cobotmagic_deploy
conda activate aloha

python -m cobotmagic_deployment.bridges.ros_bridge_node \
  --config cobotmagic_deployment/configs/<config>.yaml
```

たとえばOpenPI用の設定でブリッジだけを起動する場合は次のとおりです。

```bash
python -m cobotmagic_deployment.bridges.ros_bridge_node \
  --config cobotmagic_deployment/configs/config_openpi.yaml
```

### 4.2 右単腕ブリッジ

```bash
cd /workspace/cobotmagic_deploy
conda activate aloha

python -m cobotmagic_deployment.bridges.ros_bridge_node_single_right \
  --config cobotmagic_deployment/configs/config_single_right_openvla.yaml
```

起動後、ブリッジは必要な画像・関節状態topicを待ち、設定されている場合は初期姿勢へ移動してから
ZeroMQ経由でポリシーへ観測を送ります。観測不足、古いデータ、ポリシー応答のtimeoutは
ROSログへ警告として表示されます。

停止するときは `Ctrl+C` を使用します。標準ブリッジではYAMLの `ros.shutdown_safety` を
有効にすると、終了時に現在姿勢のhold指令やenable flag解除を実行できます。
この処理の有無にかかわらず、実機側の停止手段も常に確保してください。

## 5. 動作確認とトラブルシューティング

最初に次を確認してください。

- YAMLに記載したカメラ・関節状態topicがpublishされている
- `ROS_MASTER_URI` と `ROS_IP` が現在のネットワーク構成に合っている
- `zmq.client_connect` とポリシーサーバーのbind先・ポートが対応している
- 関節状態とカメラ画像が `ros.max_joint_age_sec`、`ros.max_image_age_sec` より新しい
- `home_position`、指令topic、アクション次元が実機構成に合っている

アクションログは、既定でリポジトリ内の `logs/action_commands` に保存されます。
YAMLの `ros.action_log` で有効化と保存先を変更できます。

EEFの微小動作確認には次を使用できます。モデル推論を始める前に実機の安全を確保してください。

```bash
python -m cobotmagic_deployment.tools.eef_motion_smoke_test --help
```

X-VLA互換サーバープロトコルをモデルなしで確認する場合は、mock起動を利用できます。

```bash
python -m cobotmagic_deployment.servers.policy_server_xvla_agilex \
  --config cobotmagic_deployment/configs/config_xvla_agilex.yaml \
  --mock --startup-test
```

## 6. 主なファイル

| パス | 役割 |
| --- | --- |
| `cobotmagic_deployment/bridges/` | ROS–ZeroMQブリッジ |
| `cobotmagic_deployment/servers/` | バックエンド別ポリシーサーバー |
| `cobotmagic_deployment/policies/` | モデル固有のポリシーラッパー |
| `cobotmagic_deployment/common/` | 共通通信処理、Piper IK、OpenWAM変換、グリッパー処理などの共通モジュール |
| `cobotmagic_deployment/configs/` | ROS、通信、アクション、モデル固有のYAML設定 |
| `cobotmagic_deployment/tools/` | smoke testとモデル検証ツール |
| `scripts/` | OpenWAM環境構築・起動スクリプトとログ再生・解析スクリプト |
| `tests/` | IK、OpenWAM変換、グリッパー処理、DreamZeroノイズ適応のテスト |
| `docs/openwam/` | OpenWAMデプロイ手順と調整・診断記録 |
| `realsense_bridge/` | カメラ専用ブリッジ |
| `aloha.yml` | ROSブリッジ用Conda環境 |

## 付録: 対応ポリシー

以下は検証済み設定とポリシーサーバーの例です。ブリッジ自体の基本的な使い方は
ポリシーに依存しません。

| バックエンド | 設定 | ブリッジ | ポリシーサーバー |
| --- | --- | --- | --- |
| OpenPI π₀ / π₀.₅ | `config_openpi.yaml` | `ros_bridge_node.py` | `policy_server_openpi.py` |
| OpenVLA FiLM | `config_openvla_stack_three_film_absolute_stage2.yaml` | `ros_bridge_node.py` | OpenVLA-OFT側サーバー |
| OpenVLA右単腕 | `config_single_right_openvla.yaml` | `ros_bridge_node_single_right.py` | OpenVLA-OFT側サーバー |
| X-VLA | `config_xvla_agilex.yaml` | `ros_bridge_node.py` | `policy_server_xvla_agilex.py` |
| DreamZero | `config_dreamzero_agilex.yaml` | `ros_bridge_node.py` | `policy_server_dreamzero_agilex.py` |
| OpenWAM RoboDojo-Piper | `config_openwam_piper.yaml` | `ros_bridge_node.py`（EEF/IK） | `policy_server_openwam_piper.py` |

### ポリシーサーバー用環境

モデル本体は、たとえば次のように `/workspace/project` 配下へ別リポジトリとして配置します。

- OpenPI: `/workspace/project/openpi`
- OpenVLA-OFT: `/workspace/project/openvla-oft`
- X-VLA: `/workspace/project/X-VLA`
- DreamZero: `/workspace/project/dreamzero`
- OpenWAM: `/workspace/project/OpenWAM`

各モデルリポジトリの手順に従って専用環境を構築し、その環境でポリシーサーバーを起動します。
チェックポイントや外部リポジトリの場所を変える場合は、対応するYAMLのモデル固有セクションも更新してください。

### ポリシーサーバー起動例

OpenPI:

```bash
source /workspace/project/openpi/.venv/bin/activate
OPENPI_REPO_PATH=/workspace/project/openpi \
python -m cobotmagic_deployment.servers.policy_server_openpi \
  --config cobotmagic_deployment/configs/config_openpi.yaml
```

X-VLA:

```bash
python -m cobotmagic_deployment.servers.policy_server_xvla_agilex \
  --config cobotmagic_deployment/configs/config_xvla_agilex.yaml
```

DreamZero:

```bash
python -m cobotmagic_deployment.servers.policy_server_dreamzero_agilex \
  --config cobotmagic_deployment/configs/config_dreamzero_agilex.yaml
```

DreamZeroを複数GPUで動かす場合は、モデル環境に合わせて `torchrun` を使用します。

OpenWAM（専用環境の構築は `scripts/setup_openwam_piper.sh`）:

```bash
bash scripts/serve_openwam_piper.sh
```

OpenVLAのポリシーサーバーは、OpenVLA-OFT側の手順に従って起動してください。

OpenPIの学習、正規化統計、データ変換は外部リポジトリで行います。
詳細は `/workspace/project/openpi/docs/local_mobile_finetune.md` を参照してください。

## 注意

- `policy_server_protocol.py` は共通通信実装であり、直接起動しません。
- ROS用Python環境とモデル用Python環境は分離してください。
- YAML内の絶対パス、GPU番号、ROS topic、初期姿勢は実行環境に合わせて確認してください。
