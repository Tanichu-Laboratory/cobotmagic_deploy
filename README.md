# CobotMagic Multi-Policy Deployment

CobotMagicで複数のVLAバックエンドを動かすためのROS・ZeroMQブリッジです。
モデル本体や学習コードはこのリポジトリに含めず、`/workspace/project` 配下の各モデルリポジトリを利用します。

## 対応バックエンド

| バックエンド | 設定 | ROSブリッジ | ポリシーサーバー |
| --- | --- | --- | --- |
| OpenPI π₀ / π₀.₅ | `config_openpi.yaml` | `ros_bridge_node.py` | `policy_server_openpi.py` |
| OpenVLA FiLM | `config_openvla_stack_three_film_absolute_stage2.yaml` | `ros_bridge_node.py` | OpenVLA-OFT側サーバー |
| SemanticVLA | `config_semanticvla_stack_three_bs8_3k.yaml` | `ros_bridge_node.py` | OpenVLA-OFT側サーバー |
| OpenVLA右単腕 | `config_single_right_openvla.yaml` | `ros_bridge_node_single_right.py` | OpenVLA-OFT側サーバー |
| X-VLA | `config_xvla_agilex.yaml` | `ros_bridge_node.py` | `policy_server_xvla_agilex.py` |
| Hy-VLA EEF | `config_hy_vla_eef.yaml` | `ros_bridge_node.py` | `policy_server_xvla_agilex.py` |
| DreamZero | `config_dreamzero_agilex.yaml` | `ros_bridge_node.py` | `policy_server_dreamzero_agilex.py` |

実装は `cobotmagic_deployment/`、カメラ専用ブリッジは `realsense_bridge/` にあります。

## セットアップ

### ROS環境

```bash
conda env create -n aloha -f aloha.yml
conda activate aloha
```

ROS masterのアドレスは環境に合わせて設定します。

```bash
export ROS_MASTER_URI=http://<cobotmagic-ip>:11311
export ROS_IP=<gpu-server-ip>
```

### モデル環境

各ポリシーサーバーは、それぞれのモデルリポジトリの環境で起動します。

- OpenPI: `/workspace/project/openpi`
- OpenVLA-OFT: `/workspace/project/openvla-oft`
- X-VLA: `/workspace/project/X-VLA`
- Hy-VLA: `/workspace/project/Hy-Embodied-0.5-VLA`
- DreamZero: `/workspace/project/dreamzero`

チェックポイントや外部リポジトリの場所を変える場合は、対応するYAMLを更新してください。

## CobotMagic側

```bash
roscore
cd /workspace/ros_cobotmagic/Piper_ros_private-ros-noetic
bash can_config.sh
roslaunch piper start_ms_piper.launch mode:=1 auto_enable:=true
roslaunch astra_camera multi_camera.launch
```

D405を使う場合は、機体のserial IDに合わせてRealSenseノードを起動します。

## 起動方法

以下のコマンドはcloneしたリポジトリのルートから実行します。
ROSブリッジとポリシーサーバーは別ターミナルで起動してください。

### OpenPI

```bash
# ROS環境
python cobotmagic_deployment/ros_bridge_node.py \
  --config cobotmagic_deployment/config_openpi.yaml

# OpenPI環境
source /workspace/project/openpi/.venv/bin/activate
OPENPI_REPO_PATH=/workspace/project/openpi \
python cobotmagic_deployment/policy_server_openpi.py \
  --config cobotmagic_deployment/config_openpi.yaml
```

OpenPIの学習・正規化統計・データ変換は外部リポジトリで行います。
詳細は `/workspace/project/openpi/docs/local_mobile_finetune.md` を参照してください。

### X-VLA

```bash
python cobotmagic_deployment/ros_bridge_node.py \
  --config cobotmagic_deployment/config_xvla_agilex.yaml
python cobotmagic_deployment/policy_server_xvla_agilex.py \
  --config cobotmagic_deployment/config_xvla_agilex.yaml
```

### Hy-VLA EEF

```bash
python cobotmagic_deployment/ros_bridge_node.py \
  --config cobotmagic_deployment/config_hy_vla_eef.yaml
python cobotmagic_deployment/policy_server_xvla_agilex.py \
  --config cobotmagic_deployment/config_hy_vla_eef.yaml
```

### DreamZero

```bash
python cobotmagic_deployment/ros_bridge_node.py \
  --config cobotmagic_deployment/config_dreamzero_agilex.yaml
python cobotmagic_deployment/policy_server_dreamzero_agilex.py \
  --config cobotmagic_deployment/config_dreamzero_agilex.yaml
```

複数GPUでDreamZeroを使う場合は、環境に合わせて `torchrun` を使用します。

### OpenVLA右単腕

```bash
python cobotmagic_deployment/ros_bridge_node_single_right.py \
  --config cobotmagic_deployment/config_single_right_openvla.yaml
```

OpenVLA-OFT側のポリシーサーバーは外部モデルリポジトリから起動してください。

## 動作確認

EEFの微小動作確認には次を使用できます。実機の安全を確保して実行してください。

```bash
python cobotmagic_deployment/eef_motion_smoke_test.py --help
```

モデルをロードせずX-VLAサーバーのプロトコルだけを確認する場合:

```bash
python cobotmagic_deployment/policy_server_xvla_agilex.py \
  --config cobotmagic_deployment/config_xvla_agilex.yaml \
  --mock --startup-test
```

## 注意

- `policy_server_protocol.py` はX-VLA/Hy-VLA/DreamZero共通の通信実装で、直接起動しません。
- ROS用Python環境とモデル用Python環境は分離してください。
- YAML内の絶対パス、GPU番号、ROS topic、初期姿勢は実行環境に合わせて確認してください。
- ログは既定でリポジトリ内の `logs/action_commands` に保存されます。
