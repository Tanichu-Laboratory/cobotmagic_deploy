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

主な実装は `cobotmagic_deployment/` にあります。

このバージョンで対応するポリシーは **OpenPI、OpenVLA、X-VLA、DreamZero、OpenWAM** です。
対応表は[付録A](#付録a-対応ポリシー)を参照してください。

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
| `ros.action_mode` | アクションの解釈。`absolute`（関節絶対値）、`velocity`（関節速度を積分）、`eef_absolute`（EEF姿勢をブリッジ側IKで関節指令へ変換） |
| `ros.open_loop_steps` | 1回の応答チャンクから実行するステップ数 |
| `ros.home_position` | 推論開始前に移動する初期姿勢と移動方法 |
| `ros.action_log` | アクションログの有効化と保存先 |
| `zmq.client_connect` | ブリッジが接続するポリシーサーバーのアドレス |
| `zmq.server_bind` | ポリシーサーバーがbindするアドレス |
| `zmq.socket_type` | `req`（REQ/REP、既定）または `pair` |

フィルタ、グリッパー処理、指令の補間、IK、ログなどのオプションは
[付録B: ブリッジのYAMLオプション](#付録b-ブリッジのyamlオプション)にまとめています。

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

右単腕ブリッジは正面・右手首カメラと右腕関節だけを使い、7次元の右腕アクションを
`/master/joint_right` へpublishします。リクエストごとに応答を待つ同期型で、
`open_loop_steps` ステップ実行してから次の観測を送ります。サーバーが
`model_action_representation` と生アクションを返す場合は、`ros.joint_delta_reconstruction`
（既定で有効）により関節差分をpublish時点の実測関節へ足し直し、`openvla.action_delta_scale`
（`action_delta_scale_per_dim` で関節ごとに指定可）で倍率をかけます。

起動後、ブリッジは必要な画像・関節状態topicを待ち、設定されている場合は初期姿勢へ移動してから
ZeroMQ経由でポリシーへ観測を送ります。観測不足、古いデータ、ポリシー応答のtimeoutは
ROSログへ警告として表示されます。

標準ブリッジは非同期で動作します。ポリシー応答を待つ間も手元のチャンクを実行し続け、
前回のリクエスト以降に3台のカメラすべてが更新されたら次の観測を送ります。
`ros.policy_request.when_live_chunk: wait` にすると、現在のチャンクを使い切るまで
次のリクエストを送りません。応答が `ros.policy_response_timeout_sec` を超えた場合は
ZeroMQソケットを作り直します。関節状態が `ros.max_joint_age_sec` より古いときは指令をpublishしません。

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
YAMLの `ros.action_log` で有効化と保存先を変更できます。出力されるファイルは次のとおりです。

| ファイル | 内容 | 有効化 |
| --- | --- | --- |
| `action_commands_<時刻>.csv` | publishした各ステップの生アクション、フィルタ後の値、最終指令、IK結果、遅延など | `action_log.enabled` |
| `request_snapshots_<時刻>/` | リクエストごとの3画像と送信ヘッダー | `action_log.rich.save_request_images` |
| `action_chunks_<時刻>/request_*.npz` | 受信チャンクと各加工段階のチャンク | `action_log.rich.save_action_chunks` |
| `action_commands_<時刻>.csv.rejected.jsonl` | IKが棄却した指令（EEFモード） | `action_log.enabled` |

`ros.rollout_dataset` を有効にすると、観測とサーバーの生アクションをエピソード単位で
HDF5（`episode.hdf5`）またはディレクトリ形式で保存できます。

### 5.1 ロボットなしでの確認

| コマンド | 用途 |
| --- | --- |
| `python -m pytest tests -q` | アクション加工、通信プロトコル、IK、OpenWAM変換、グリッパー処理、DreamZeroノイズ適応の単体テスト（OpenWAM環境などNumPy/SciPy/pyzmq/OpenCVのある環境で実行） |
| `python -m cobotmagic_deployment.servers.policy_server_xvla_agilex --mock --startup-test` | X-VLAサーバーをモデルなしで起動・bind確認 |
| `python -m cobotmagic_deployment.servers.policy_server_xvla_agilex --mock --inference-test` | モックでダミー推論し、アクションの値域を表示 |
| `bash scripts/serve_openwam_piper.sh --mock --startup-test` | OpenWAMの設定とチェックポイント定義をモデルなしで確認 |
| `python -m cobotmagic_deployment.tools.smoke_openwam_piper --requests 2` | 起動中のOpenWAMサーバーへ合成観測を送り、応答を検証 |
| `python -m cobotmagic_deployment.tools.dummy_test_openpi --config ...` | OpenPIポリシーでダミー推論を1回実行 |
| `python -m cobotmagic_deployment.tools.replay_openwam_ik` | 記録済みEEF指令をIKでオフライン再生 |
| `python -m cobotmagic_deployment.tools.download_openwam_piper` | OpenWAM公式重みを固定リビジョンで取得し、ハッシュを検証 |

`scripts/replay_*.py` と `scripts/analyze_gm100_home.py` は、OpenWAM調整時に記録したログを
再生・解析するための診断スクリプトです。`logs/` 以下の記録を入力とし（ファイルはスクリプト内の既定値または引数で指定）、結果も `logs/` 以下へ書き出します。

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
| `cobotmagic_deployment/bridges/` | ROS–ZeroMQブリッジ（`action_logging.py` はログ・データセット出力） |
| `cobotmagic_deployment/servers/` | バックエンド別ポリシーサーバー |
| `cobotmagic_deployment/policies/` | モデル固有のポリシーラッパー |
| `cobotmagic_deployment/common/` | ROS非依存の共通モジュール（[7章](#7-新しいポリシーモデルの追加)の表を参照） |
| `cobotmagic_deployment/configs/` | ROS、通信、アクション、モデル固有のYAML設定 |
| `cobotmagic_deployment/tools/` | smoke testとモデル検証ツール |
| `scripts/` | OpenWAM環境構築・起動スクリプトとログ再生・解析スクリプト |
| `tests/` | ブリッジ部品、サーバー実行基盤、通信、IK、OpenWAM変換、グリッパー処理、DreamZeroノイズ適応のテスト |
| `docs/openwam/` | OpenWAMデプロイ手順と調整・診断記録 |
| `realsense_bridge/` | 旧版の簡易OpenPIブリッジ（同期型・関節速度アクション）。サンプル設定は1台のRealSenseを3カメラ分に流用 |
| `aloha.yml` | ROSブリッジ用Conda環境 |

## 7. 新しいポリシーモデルの追加

ブリッジとサーバーの機能は部品として分かれており、新しいモデルは既存の機能を設定やimportで
そのまま使えます。

### 7.1 手順

1. `servers/policy_server_template.py` を `servers/policy_server_<model>.py` に、
   `configs/config_template.yaml` を `configs/config_<model>.yaml` にコピーします。
2. YAMLの `policy_backend` とモデルセクション名を `<model>` に変え、サーバーの
   `load_server_config(..., "<model>", ...)` と合わせます。
3. `predict(header, images)` にモデル推論を書きます。戻り値は `(T, 14)`（左腕7次元＋右腕7次元）で、
   `action_mode` はブリッジの `ros.action_mode` と同じにします。
   - `absolute`: 関節目標値
   - `eef_absolute`: 各腕 `[x, y, z, roll, pitch, yaw, gripper]`。関節指令への変換はブリッジ側のIK（`eef_ik`）
   - 台車速度も返す場合は `(actions, vel)` を返します。
4. ブリッジはコード変更不要です。YAMLの `ros` セクションで、非同期推論、チャンクの補間・フィルタ、
   ステップごとの加工、グリッパー処理、微分IKなどを `enabled: true` で選びます
   （[付録B](#付録b-ブリッジのyamlオプション)）。

```bash
# モデルなしで起動確認（現在状態をそのまま返す）
python -m cobotmagic_deployment.servers.policy_server_template --mock --startup-test
python -m cobotmagic_deployment.servers.policy_server_template --inference-test
```

サーバー実行基盤（`common/policy_server_runtime.py`）が、設定読み込み、`--config`/`--bind`/`--mock`/
`--startup-test`/`--inference-test`、ウォームアップ、リクエストループ、エラー時の空応答を提供します。
X-VLAサーバーはこの基盤の `run_server` をそのまま使い、OpenPIとOpenWAMはリクエストループ（`serve`）を使っています。
DreamZeroは複数GPU（torchrun）とプロンプト操作用ソケットのため独自ループです。

### 7.2 共通モジュール

`cobotmagic_deployment/common/` のブリッジ部品はROSに依存しないため、別のブリッジやサーバー、
オフライン検証からも直接importできます。標準ブリッジ（`DualArmPolicyBridge`）はこれらを組み合わせたものです。

| モジュール | 主なクラス・関数 | 役割 | 対応するYAML |
| --- | --- | --- | --- |
| `policy_client.py` | `AsyncPolicyClient` | ノンブロッキングのZeroMQ送受信、応答タイムアウト時の再接続 | `zmq.*`、`policy_response_timeout_sec` |
| `chunk_scheduler.py` | `ChunkScheduler`、`PolicyRequestGate` | 非同期推論のタイムライン（実行中チャンク、時間アンサンブル、遅延補償）とリクエスト送信判定 | `temporal_ensemble`、`policy_request`、`initial_action_skip_steps` |
| `chunk_pipeline.py` | `ChunkPipeline`、`parse_action_response` | 応答の解析と、受信時のチャンク加工（ローパス、単調化、平滑化、補間、終点フィルタ） | `action_chunk_*`、`chunk_interpolation`、`chunk_terminal_displacement_filter`、`open_loop_steps` |
| `command_shaper.py` | `CommandShaper`、`PolicyGripperInput`、`VelocityIntegrator` | ステップごとの指令整形とグリッパー処理、ポリシーへ送るグリッパー値、速度の積分 | `action_filter`、`delta_clip`、`gripper_*`、`command_delta_deadband`、`first_action_delta_scale`、`initial_pose_delta_override`、`policy_gripper_input` |
| `ik_commander.py` | `EefIkCommander` | EEF目標から関節指令へのIK（微分IK含む）、棄却判定、手首特異点ガードのホールド | `eef_ik` |
| `command_publisher.py` | `InterpolatedCommandPublisher` | 制御周期間を線形補間する高レートpublishスレッド | `command_publish` |
| `action_processing.py` | 各種関数 | 上記部品が使うNumPyの数値処理（フィルタ、補間、アンサンブル、姿勢変換） | — |
| `piper_ik.py`、`piper_differential_ik.py` | `PiperNumericalIK`、`DifferentialIK` | Piperの数値IKと微分IK | `eef_ik` |
| `gripper_hysteresis.py` | `GripperHysteresis` | グリッパーの2値化（不感帯と連続確認） | `gripper_hysteresis` |
| `policy_server_protocol.py` | `recv_packet`、`send_actions` など | ブリッジとサーバー間の通信形式 | — |
| `policy_server_runtime.py` | `run_server`、`serve`、`EchoStatePolicy` など | ポリシーサーバーの共通実行基盤 | — |
| `bridge_log.py` | `BridgeLog` | 部品共通のログ出力（ROSでも標準loggingでも利用可） | — |

たとえば、別のロボット用ブリッジで補間と微分IKだけを使う場合は次のようになります。

```python
from cobotmagic_deployment.common.chunk_pipeline import ChunkPipeline
from cobotmagic_deployment.common.ik_commander import EefIkCommander

pipeline = ChunkPipeline(cfg['ros'], rate_hz)
chunk = pipeline.process(left, right, None, command_left, command_right, measured_left, measured_right)
ik = EefIkCommander(cfg['ros']['eef_ik'], rate_hz)
results = ik.solve(target_left, target_right, joints_left, joints_right, command_left, command_right)
if results is not None:
    publish(results['left']['joints'], results['right']['joints'])
    ik.commit(results)
```

## 付録A: 対応ポリシー

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
外部リポジトリの場所は、OpenPIでは環境変数 `OPENPI_REPO_PATH`、X-VLA・DreamZero・OpenWAMでは
`xvla.repo_path`、`dreamzero.repo_path`、`openwam.repo_path` で指定します。

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

DreamZeroを複数GPUで動かす場合は `torchrun` で起動します。rank 0だけがZeroMQで待ち受け、
他のrankは推論に参加します。

```bash
torchrun --nproc_per_node=2 -m cobotmagic_deployment.servers.policy_server_dreamzero_agilex \
  --config cobotmagic_deployment/configs/config_dreamzero_agilex.yaml
```

DreamZeroサーバーはタスク指示を実行中に差し替えられます（`zmq.prompt_control_bind`、既定
`tcp://127.0.0.1:5559`）。上書き中はブリッジから届く `task_prompt` より優先されます。

```bash
python -m cobotmagic_deployment.servers.policy_server_dreamzero_agilex --set-task-prompt "pick up the cup"
python -m cobotmagic_deployment.servers.policy_server_dreamzero_agilex --get-task-prompt
python -m cobotmagic_deployment.servers.policy_server_dreamzero_agilex --clear-task-prompt
```

`dreamzero.noise_adaptation: true` にすると、観測した終点誤差から初期ノイズを適応更新します
（1ステップ推論が前提。パラメータは `dreamzero.noise_adaptation_config`）。ブリッジは最初の
リクエストに `episode_start: true` を付け、サーバーは因果推論状態と適応状態をリセットします。

OpenWAM（専用環境の構築は `scripts/setup_openwam_piper.sh`）:

```bash
bash scripts/serve_openwam_piper.sh
```

OpenVLAのポリシーサーバーは、OpenVLA-OFT側の手順に従って起動してください
（YAMLの `openvla` セクションはOpenVLA-OFT側サーバーが読み込みます）。

### ポリシーサーバーのコマンドラインオプション

すべてのサーバーで `--config` を省略すると、各バックエンドの `configs/` 内の既定YAMLを使います。
`--bind` はYAMLの `zmq.server_bind` を上書きします。

| サーバー | オプション |
| --- | --- |
| `policy_server_openpi` | `--config`、`--bind` |
| `policy_server_template` | `--config`、`--bind`、`--mock`、`--startup-test`、`--inference-test`（新モデル用の雛形） |
| `policy_server_xvla_agilex` | `--config`、`--bind`、`--mock`（重みを読まず現在姿勢を返す）、`--startup-test`（bindして終了）、`--inference-test`（ダミー推論1回）、`--local-files-only` |
| `policy_server_dreamzero_agilex` | `--config`、`--bind`、`--startup-test`、`--inference-test`、`--benchmark-runs N`、`--warmup`、`--flash`/`--no-flash`（DiTキャッシュ、既定有効）、`--num-dit-steps`、`--timeout-seconds`、プロンプト操作（`--set-task-prompt`、`--get-task-prompt`、`--clear-task-prompt`、`--prompt-control-bind`、`--prompt-control-connect`、`--prompt-control-timeout-ms`） |
| `policy_server_openwam_piper` | `--config`、`--bind`、`--mock`（重みを読まず現在EEF姿勢を返す）、`--startup-test`（ソケットを開かずに1回推論して終了）、`--denoise-steps` |

`--startup-test` の意味はサーバーごとに異なります。X-VLAとDreamZeroではポートをbindして終了し、
OpenWAMではソケットを開かずに1回推論して終了します。

OpenPIの学習、正規化統計、データ変換は外部リポジトリで行います。
詳細は `/workspace/project/openpi/docs/local_mobile_finetune.md` を参照してください。

## 注意

- `common/` のモジュール（`policy_server_protocol.py` など）はライブラリであり、直接起動しません。
- ROS用Python環境とモデル用Python環境は分離してください。
- YAML内の絶対パス、GPU番号、ROS topic、初期姿勢は実行環境に合わせて確認してください。

## 付録B: ブリッジのYAMLオプション

`ros_bridge_node.py` が読む `ros` セクションのオプションです。記載のない項目は既定値で動作し、
多くの加工は `enabled: false` が既定です。値の例は `cobotmagic_deployment/configs/` を参照してください。

### 処理の順序

応答チャンクを受信したとき（チャンク単位）:

1. `action_chunk_lowpass` → `action_chunk_monotonic` → `action_chunk_smoothing`
2. `chunk_interpolation`（`linear` または `adaptive_delta`。後者は `initial_bridge_enabled` でチャンク境界も補間）
3. `chunk_terminal_displacement_filter` → `chunk_action_skip_steps`

各制御ステップ（`rate_hz`）:

1. `absolute`/`eef_absolute`: `temporal_ensemble`、`velocity`: 速度を積分（グリッパーは絶対値）
2. `action_filter` → `delta_clip` → `gripper_delta_scale` → `gripper_hysteresis` → `gripper_threshold`
3. `command_delta_deadband` → `first_action_delta_scale`（各チャンクの最初のステップのみ） → `initial_pose_delta_override`
4. `eef_absolute` では `eef_ik` で関節指令へ変換
5. `command_publish` に従ってpublish

### 基本・通信

| 項目 | 内容 |
| --- | --- |
| `rate_hz`、`open_loop_steps` | 制御周期と、1チャンクから実行する最大ステップ数 |
| `image_type`、`jpeg_quality` | `raw` はブリッジでJPEG化（品質指定）、`compressed` は `<topic>/compressed` をそのまま転送 |
| `joint_names` | 指令 `JointState` の関節名（既定 `joint0`〜`joint6`） |
| `topics` | `img_front/left/right`、`puppet_arm_left/right`、`cmd_joint_left/right`、`enable_flag`、EEFモードでは `puppet_arm_left_pose`/`puppet_arm_right_pose` |
| `publish_enable_flag` | 起動時に `enable_flag` へ `True` をpublish |
| `use_robot_base`、`robot_base_topics`、`clip` | 移動台車のodom入力と `cmd_vel` 出力（`v_max`/`w_max` でクリップ） |
| `policy_response_timeout_sec`、`max_joint_age_sec`、`max_image_age_sec` | 応答タイムアウトと観測の鮮度しきい値 |
| `policy_request.when_live_chunk` | `allow`（実行中でも新しい観測で要求）または `wait`（チャンクを使い切ってから要求） |
| `initial_action_skip_steps`、`chunk_action_skip_steps` | 最初の応答／各応答の先頭ステップを読み飛ばす数 |
| `policy_gripper_input.mode` | ポリシーへ送るグリッパー状態。`measured`、`commanded`（前回指令）、`hybrid`（差が `hybrid_max_error` 以内なら指令値） |

### チャンクの加工

| 項目 | 内容 |
| --- | --- |
| `action_chunk_lowpass` | チャンク全体へのゼロ位相Butterworthローパス（`cutoff_hz`、`sample_rate_hz`、`order`、`preserve_endpoints`、`include_gripper`） |
| `action_chunk_monotonic` | 関節ごとに開始点から終点方向への単調な軌道へ射影し、チャンク内の往復を抑制（`arms`、`strength`、`min_terminal_delta`、`include_gripper`） |
| `action_chunk_smoothing` | 3次スプラインでアップサンプリングしてSavitzky-Golay平滑化（`upsample_factor`、`window_length`、`polyorder`） |
| `chunk_interpolation` | `mode: linear` は `factor` 倍に補間。`mode: adaptive_delta` は関節変化量が `joint_threshold` を超える区間を `min_factor`〜`max_factor` 倍に補間し、`source_steps_to_execute`/`source_overlap_steps` を元チャンクのステップ数で指定。`initial_bridge_enabled`/`initial_bridge_max_factor` で現在指令から最初のアクションまでを補間 |
| `chunk_terminal_displacement_filter` | 実行終点での変位が `left_threshold`/`right_threshold` 未満の関節を、チャンク全体で要求時の姿勢に固定 |

### ステップごとの加工

| 項目 | 内容 |
| --- | --- |
| `temporal_ensemble` | 重なった過去チャンクを指数重み（`exp_decay`）で平均。`max_history_chunks`、`max_candidate_age`、`min_action_index`、`overlap_steps`、`latency_compensation`（`measured`/`fixed` で遅延分のステップを読み飛ばす） |
| `action_filter` | 目標値のEMA（`ema_alpha`）と、前回指令からの微小変化を無視する `deadband` |
| `delta_clip` | 1ステップの関節変化量を `max_delta` に制限。`reference: command`（前回指令基準）または `current`（実測関節基準） |
| `gripper_delta_scale` | 要求時の実測値からのグリッパー変化量を左右別の倍率（`left_gain`/`right_gain`）で拡大縮小し、`clip_min`/`clip_max` でクリップ |
| `gripper_hysteresis` | グリッパーを開/閉の2値にし、しきい値の不感帯と `confirm_steps` 回の連続確認で切り替え。`request_relative` で要求時の開度に応じてしきい値を調整。`gripper_threshold` と同時には使えず、`command_publish.mode: direct` が必要 |
| `gripper_threshold` | 開/閉しきい値を超えた値を固定値へ置換（旧名 `gripper_binary` も可） |
| `command_delta_deadband` | 前回指令からの変化が左右別のしきい値未満の関節を前回指令のまま保持 |
| `first_action_delta_scale` | 各チャンク最初のステップの変化量を `coefficient`（0〜1）倍にする |
| `initial_pose_delta_override` | 指定した腕（`arms`）を要求時の実測姿勢に固定（`include_gripper` でグリッパーも） |

### 出力・IK

| 項目 | 内容 |
| --- | --- |
| `command_publish` | `mode: direct` は制御周期ごとに1回publish。`mode: interpolated` は `rate_hz`（`ros.rate_hz` の整数倍に丸め）で前回指令から線形補間してpublish（`eef_absolute` では使えません） |
| `eef_ik` | `eef_absolute` で必須。`solver` は `differential`（速度・加速度制限付きの微分IK、`differential.*`）、`least_squares`、`damped_least_squares`（既定）。`urdf_path`、`base_link`、`tip_link`、`joint_signs`、`calibration_mode`（`base`/`tool`）、`max_joint_delta_rad`、`publish_on_failure`、`wrist_singularity_avoidance` など。詳細は [docs/openwam/OPENWAM_DIFFERENTIAL_IK.md](docs/openwam/OPENWAM_DIFFERENTIAL_IK.md) |

### 初期姿勢・終了処理・記録

| 項目 | 内容 |
| --- | --- |
| `home_position` | `left`/`right` の目標へ移動。`mode: step`（`arm_steps_length` ずつ）または `linear`（`num_steps` 分割）、`rate_hz`、`gripper_move_start_fraction`（linearでグリッパーを動かし始める進捗）、`settle`/`settle_tolerance`/`settle_hold_sec`/`settle_timeout_sec`/`settle_rate_hz`（実測関節の収束待ち）、`require_settle`（収束しなければ推論せず終了）、`post_sleep_sec` |
| `home_position.sequence` | 複数フェーズのリスト。各フェーズは `name`、`left`、`right` と上記の項目を個別に上書き可能 |
| `shutdown_safety` | 終了時に実測関節のhold指令を `hold_current_sec` 秒publishし、`publish_enable_false` で `enable_flag=False`、`zero_base_velocity` で台車を停止 |
| `action_log` | `dir`、`flush_every_rows`、`rich.enabled`/`save_request_images`/`save_action_chunks`/`request_snapshot_every_n` |
| `rollout_dataset` | `dir`、`run_name`、`format`（`hdf5`/`directory`）、`save_images` |

