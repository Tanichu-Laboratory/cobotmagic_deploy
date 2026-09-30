# OpenWAM × 双腕Piper デプロイ

## 使用する重みと環境

- 公式モデル: [OpenWAM/OpenWAM-Alpha-Real-RoboDojo-Piper](https://huggingface.co/OpenWAM/OpenWAM-Alpha-Real-RoboDojo-Piper)
- 固定revision: `12bfca6fb289174ab5a0f4f7eb363dc77ed265f0`
- 重み: `checkpoint_step_30000.safetensors`、24,813,767,464 bytes
- SHA-256: `ba9b2696624b12676610ac7bdc18882d088f819d9993b37d05d5d13f6466f004`
- 保存先: `/workspace/project/OpenWAM/assets/openwam_ckpt/openwam_alpha/OpenWAM-Alpha-Real-RoboDojo-Piper`
- OpenWAMソース: `/workspace/project/OpenWAM`、基準commit `a8992b613fdb5b1c4649f53b3a914379c49e3892`
- Python: `/workspace/project/OpenWAM/.venv-piper/bin/python`（3.11.14）
- PyTorch: 2.7.1+cu128、GPU: RTX PRO 6000 Blackwell、既定 `cuda:0`
- 全依存: `scripts/openwam_piper_requirements.lock`

既存 `.venv` は3.11用site-packagesに対しPython実体が3.13だったため、別環境を作成した。
ヘッドレスコンテナの `libGL.so.1` 不足には専用環境のOpenCVをheadless版へ置換して対応した。
OpenWAMパッケージの依存名は `opencv-python` のため、`uv pip check` ではこの置換について未充足表示となる。
推論・画像デコードは実動作で確認する。

再構築と再取得（既存の取得済みファイルは再利用し、LFSのSHA-256を照合）:

```bash
cd /workspace/cobotmagic_deploy
bash scripts/setup_openwam_piper.sh
```

## 接続経路

```text
3台のRGB + 左右 /puppet/end_pose_* + /puppet/joint_*
  → 既存 ros_bridge_node（EEFモード）
  → ZeroMQ REQ/REP tcp://127.0.0.1:5557
  → policy_server_openwam_piper
  → OpenWAM ObsPreprocessor → JointInferenceEngine（実重み）
  → 生EEF20 → xyz/rpy/グリッパーの双腕14D
  → 既存ブリッジの数値IK・関節角差分制限
  → /master/joint_left, /master/joint_right
```

設定は `cobotmagic_deployment/configs/config_openwam_piper.yaml` に集約した。

- 学習と同じ画像配置: 正面を上、左手首を左下、右手首を右下。キャンバス384×320。
- EEFは各腕のベース座標系、位置m、ROSのextrinsic xyz Euler角rad。
- 入力20Dは左 `[xyz, rotation第1列, rotation第2列, grip]`、右も同じ順。
  X-VLAの交互列6Dを流用しない。関節角からの偽のEEF補完も行わない。
- 20D→80D展開、min-max正規化、出力の逆変換はチェックポイント付属統計でOpenWAM本体が行う。
- グリッパーはユーザー申告の全閉 `0.0`、全開 `0.06558`（左右共通）を `[0,1]` に変換。
  出力も実機単位へ戻すため、ブリッジの二値閾値変換は無効。実機で端点は再確認する。
- タスク文には学習時と同じ `A video recorded from a robot's point of view executing the following instruction: ` を付加。
- サーバーは生成した全ステップ（現チェックポイントは32）を返す。
  ブリッジだけが `min(open_loop_steps, 受信ステップ数)` を実行し、残りを破棄する。
  `open_loop_steps: null` なら受信全体を実行する。変更時はブリッジのみ再起動すればよい。
- 初期設定は10 denoise steps、compile無効、5Hz、現在32ステップ実行、応答timeout 120秒。
  5Hzはチャンク内の指令周期で、推論待ち時間を含む実効制御周期を保証しない。
- 初期姿勢はユーザー指定の左右7関節値をconfigに設定。約3秒で線形移動し、到達確認後に推論開始。URDFは既存ローカル資産
  `/workspace/project/piper_description_cobotmagic_local_20260529.urdf` を利用する。

## 起動

ポリシーサーバーのみ（起動時に合成観測でwarmupし、READY後に接続受付）:

```bash
cd /workspace/cobotmagic_deploy
bash scripts/serve_openwam_piper.sh
```

モデル初期化に約2分かかる。このサーバーはROSのpublishを行わない。
別マシンから接続する場合はYAMLの `server_bind` と `client_connect` を到達可能なIPに変更する。
同一マシンなら既定のloopbackを使用する。

実機を使わないネットワーク検証（READY後、ROSブリッジと同時使用しない）:

```bash
cd /workspace/cobotmagic_deploy
/workspace/project/OpenWAM/.venv-piper/bin/python \
  -m cobotmagic_deployment.tools.smoke_openwam_piper --requests 2
```

単発推論検証 / モデルなし変換検証:

```bash
bash scripts/serve_openwam_piper.sh --startup-test
bash scripts/serve_openwam_piper.sh --mock --startup-test
```

実機側でアーム・カメラ・ROS master起動後、別ターミナルでブリッジを起動:

```bash
cd /workspace/cobotmagic_deploy
source /opt/ros/noetic/setup.bash
conda activate aloha
source scripts/env_openwam_piper.sh  # master 10.228.162.56 / bridge 10.228.162.225
rostopic list
python -m cobotmagic_deployment.bridges.ros_bridge_node \
  --config cobotmagic_deployment/configs/config_openwam_piper.yaml
```

この最後のコマンドは実機のenableと関節指令publishを行う。
`/puppet/end_pose_left` と `...right` が学習と同じ各腕ベース座標の末端姿勢であり、
前・左右手首のカメラ配置が対応していることを確認して使う。
現在のタスクは `put cubes into the box`。`task_prompt` を実際の作業に合わせる。

## 検証記録（2026-09-17）

- 公式重み、正規化統計、tokenizerを取得。全ファイルサイズおよびLFS SHA-256一致。
- 実重みGPU推論: 10 denoise steps、出力 `(8,14)`、全要素finite、初回2.73秒。
- `tests/test_openwam_piper.py`: 12 tests passed。
  公式OpenWAMとの非自明回転の照合、グリッパー端点、異常入力拒否、ZMQエラー後の回復を含む。
- 実モデルZMQ往復: 2回成功、各 `(8,14)`、全要素finite、約0.377秒 / 0.367秒。
  合成黒画像・同一プロンプト・warmup後・DiT/prompt cache有効での参考値であり、実観測の速度保証ではない。
- 推論サーバーは `tcp://127.0.0.1:5557` で検証済み。検証用の常駐プロセスはポート競合解消のため停止済み。
- ROS環境で既存ブリッジをimportし、ローカルURDFの6関節IKをオフライン検証。
- 設定済みROS master `http://10.228.162.29:11311` は接続拒否。
  実カメラ入力、実機移動、タスク成功率は未検証。公開Piper重みの本機・本タスクへの転移性能は別途評価が必要。

テスト再実行:

```bash
cd /workspace/cobotmagic_deploy
/workspace/project/OpenWAM/.venv-piper/bin/python -m pytest tests/test_openwam_piper.py -q
```

ログ: `logs/openwam_piper_smoke.log`, `logs/openwam_piper_server.log`, `logs/openwam_piper_zmq_smoke.log`。
検証用バックグラウンドサーバーは停止し、そのPIDファイルは削除済み。通常は上記コマンドでフォアグラウンド起動する。
停止する場合はPIDとプロセス名を確認したうえでSIGTERMを送る。再起動時は既存サーバーを先に停止する。

## 接続・初期姿勢の更新

ROS masterを `http://10.228.162.56:11311`、このマシンのROS_IPを `10.228.162.225` に更新。
`scripts/env_openwam_piper.sh` をsourceすると両方を設定する。
新masterへの読み取り接続と3台のカメラtopicを確認済み。更新時点では腕のjoint/EEF topicは未公開。
実機側のPiperノードを起動してからブリッジを起動する。
左右の初期姿勢は指定値をそのまま `ros.home_position` に保存。
グリッパーの全閉0／全開0.06558は維持し、測定値の端点外れを実機単位で0.005まで許容して[0,1]にクリップする。
これにより初期値の -0.0037 / -0.0009 はモデル入力では閉状態になる。

ポート競合時はモデルロード前に即時終了するようサーバーを修正済み。起動完了はTCPポートの開通ではなくログの `READY` で確認する。

全チャンク送信へのコード更新を反映するため、既存サーバーは一度再起動する。更新後のstartup-testは `(32,14)` を返し、open_loop_stepsに依存しない。過去の検証記録の `(8,14)` は旧版の結果。

## EEF差分制限の撤廃

現在は `ros.delta_clip.enabled: false`。モデルのEEF位置・姿勢を差分クリップせずIKへ渡す。
このスイッチはグリッパー成分の事前差分クリップも無効にするが、サーバー側の開閉端点への出力制限は維持する。
IK後は実測関節角を基準にJ1〜3を±0.18rad、J4〜6を±0.24radに制限する（従来の1.5倍）。
これは1指令ごとの変化量制限で、速度・加速度制限ではない。
URDF可動域、IK残差設定、publish_on_failure=falseは維持。
`max_step_rad: 0.18` はIK内部の反復計算幅であり、送信する関節差分上限とは別。
変更反映はROSブリッジの再起動のみ。既知のFK/実機EEF不整合やIK採用判定の問題を解決する変更ではない。

## publish前のIK精度改善（2026-09-17）

OpenWAM設定の `eef_ik.solver: least_squares` で、URDF関節範囲内の位置・姿勢残差を直接最小化する。
現在関節角へ引き戻す正則化は0。従来の `damping` / `max_step_rad` は旧ソルバーの反復用パラメータであり、新ソルバーでは使用しない。
関節変化量上限 `[0.18,0.18,0.18,0.24,0.24,0.24]` rad は従来どおり実測関節を基準に適用する。
制限が必要なら、その同じ範囲内で位置・姿勢誤差を再最小化する。EEF目標の差分クリップは無効のまま。

- 制限前のIK解: 位置5 mm以下、姿勢0.05 rad以下であることを確認。
- 制限後の送信角度: 既存の位置40 mm・姿勢0.8 rad上限を再確認。制限前の収束でこの確認を省略しない。
- 上限超過時は左右ともpublishしない。未送信の目標でcommand履歴を更新しない。
- チャンクは引き続き1 publishにつき1 step消費する。関節制限時に同じ目標を繰り返して到達を待つ機能は今回追加していない。
- CSVの `ik_seed_joint_left/right` は各solveで使った実測関節角。
  `ik_diagnostics_left/right` に制限前解・制限前残差・制限発動・制限内再最適化・送信角度のFK・計算時間を保存。
  既存の `ik_*_position_error_m` / `ik_*_orientation_error_rad` は送信するfloat32角度の残差。

反映にはブリッジの再起動が必要。モデルサーバーの再起動は不要。今回の検証では実機publish・再起動は行っていない。
実機起動コマンドは上記と同じ。安全制約を保ったまま大半の数値残差を改善したが、
関節変化量制限中の残差と、URDF FK／実機end_pose対応の不整合は別の問題として残る。

ROSも実機も不要な再現コマンド:

```bash
cd /workspace/cobotmagic_deploy
/workspace/project/OpenWAM/.venv-piper/bin/python -m pytest tests/test_piper_ik.py tests/test_openwam_piper.py -q
/workspace/project/OpenWAM/.venv-piper/bin/python -m cobotmagic_deployment.tools.replay_openwam_ik
```

36テスト成功。ROS Python 3.8でもブリッジimportと全62ステップのオフライン再生を確認。
結果と仮定は `OPENWAM_SMALL_MOTION_DIAGNOSIS.md` の追記を参照。


## 基部ヨー回転のフレーム修正（092132ログ以降）

上記の数値IK改善に加えて、実機とURDFの座標対応を修正した。
このローカルURDFではROS関節インデックス3・5の符号を反転してFKへ入力し、
link6と実機EEFの差を移動する末端フレームの補正として右乗算する。
OpenWAM configの `joint_signs: [1,1,1,-1,1,-1]` / `calibration_mode: tool` が該当設定。
既存の関節変化量上限・ユーザー指定初期角度・モデルEEF目標は変更していない。

初期観測とは別の実測20姿勢で検証済み。41テストとROS Python 3.8でのオフライン再生に成功。
反映にはブリッジを再起動し、新たに校正を行う。モデルサーバーの再起動は不要。
起動ログの `EEF IK calibrated` に `mode=tool` と上記signsが出ることを確認できる。
ただし現初期姿勢は横変位に敏感で、修正後もモデル目標によって大きな基部回転が生じ得る。
この修正だけで基部回転がなくなった、または実機追従を確認したとは扱わない。
詳細は `OPENWAM_SMALL_MOTION_DIAGNOSIS.md` の092132節を参照。

最新8ステップのオフライン再生:

```bash
/workspace/project/OpenWAM/.venv-piper/bin/python -m cobotmagic_deployment.tools.replay_openwam_ik \
  --fixture tests/data/openwam_ik_092132.json \
  --output logs/openwam_ik_diagnosis/frame_fix_092132.json
```


## グリッパー開き動作の強調

`openwam.gripper.action_open_normalized: 0.75` を設定。
モデルが返す[0,1]開度を `clip(g / 0.75, 0, 1)` としてから、実機の全閉0〜全開0.06558へ変換する。
左右共通で、0.375→0.03279、0.75以上→0.06558。飽和前の開度指令は従来の約1.33倍。
全閉0と物理上限0.06558は維持する。これは出力の開度強調であり、実測入力の正規化や学習済み統計は変更しない。
省略時は1.0で従来動作。今回は「正規化開度0.75で全開」と解釈して適用している。
反映にはモデルサーバーの再起動が必要。稼働中サーバーの設定自動再読み込みはない。


## joint5固定pitchの補正

OpenWAMのIKは `cobotmagic_deployment/assets/piper_description_cobotmagic_joint5_pitch0.urdf` を使用する。
元URDFからjoint5（ROSインデックス4）の固定origin pitchだけを−5°から0°へ補正した。
独立した112観測で実測EEFとの位置誤差中央値が左0.437 mm・右0.893 mmとなることを確認。
関節インデックス3・5の符号反転は維持する。元URDFは比較用に保持している。
ブリッジを通常のコマンドで再起動すると新URDFを読み込み、末端補正を再校正する。
今回の検証はオフラインで、実物の把持精度向上は未確認。


## 開閉のメリハリを強める閾値設定

`ros.gripper_threshold` を有効化。左右とも、サーバーの0.75開度強調を適用した後の実機単位で:

- 指令値0.025以下 → 全閉0.0
- 指令値0.040以上 → 全開0.06558
- 0.025より大きく0.040未満 → モデル由来の指令値を維持

既存の `threshold_gripper_targets` を使用する。これは端点へのスナップであり、
中間域で前回の開閉状態を保持するヒステリシス方式ではない。
実測入力の正規化、EEF位置・姿勢、実機全開上限は従来どおり。
サーバー側の `action_open_normalized: 0.75` も引き続き適用する。
元の正規化モデル出力で約0.286以下が全閉、約0.457以上が全開に相当する。
反映にはブリッジを再起動する。実機への送信を伴う検証は行っていない。


## 2026-09-24: モデル開度に対する状態保持型の開閉強調

現在の有効設定は、サーバー `action_open_normalized: 1.0`、旧 `gripper_threshold.enabled: false`、新 `gripper_hysteresis.enabled: true`。
以前の `/0.75` 開き増幅および実機単位0.040以上の全開化は使用しない。

- モデル正規化開度0.55以下が2指令連続: 全閉0.0。
- モデル正規化開度0.70以上が2指令連続: 全開0.06558。
- 0.55〜0.70の中間域: 直前の開閉状態を維持。
- 左右独立。初回は実測開度50%を境に状態を初期化。チャンク境界では状態をリセットしない。
- 一時的な閾値超えは切り替えず、条件が途切れたら確認カウントをリセット。
- IKで送信を見送った指令は状態・確認カウントに反映しない。
- 新しいCSV列 `gripper_hysteresis` に正規化入力、前後の状態、切替フラグ、確認カウントを記録。
- 入力側の実測グリッパー正規化は従来どおり。腕6関節の目標やIK設定は変更しない。

閉閾値0.55は、失敗ログで最小約0.495まで下がった中間開度を閉動作へ変換する暫定値。
開閾値0.70は閉閾値との間に状態保持域を設け、旧ログで0.75以上が飽和していた区間も曖昧さなく検証できる値。
把持タイミングや物体接触を検出する制御ではない。次回実機で到達前の早閉じ・持ち上げ中の再開きを確認する。

過去ログ再生（既存モデル出力を固定、旧サーバー変換を逆変換）:
- 033828右: step33で開、150で閉、161で再開。以前はstep27以降ずっと全開。
- 110643左: step157で開、235で閉、273で再開。
- 033603と111658: 元々閉じたままの系列は新処理でも閉を維持。
- 閉動作によって次の観測とモデル出力が変わる実際の閉ループ結果は未検証。

再現: `scripts/replay_gripper_hysteresis.py`。
数値: `logs/openwam_tracking_diagnosis/gripper_hysteresis_replay.json`。
状態遷移・未送信時の保持・左右独立・ノイズ・設定検証を含む関連テスト57件成功。ROS用Python 3.8で構文検証済み。

**反映にはポリシーサーバーとROSブリッジの両方を再起動する。** サーバーは旧倍率をメモリに保持するため、ブリッジだけの再起動では正規化値が一致しない。
この変更作業では実機操作や稼働プロセスの再起動は行っていない。

### 035100実行後の右手閉閾値調整

最新ログでは初回の左右実測グリッパー値は0.0で、初期化後は閉状態。
推論中の右開度はrequest4で最小0.576453、request5で0.599906となり、閉閾値0.55に届かなかった。
右のみclose_threshold_normalizedを0.65へ変更（左0.55、開閾値左右0.70、確認2指令は維持）。
固定ログの再生では初回の開く時刻は左右とも変わらず、右step127と159で全閉に切り替わる。
ただし次チャンクのモデル出力は1.0へ戻るため、step129と161で再開する。実機で閉じた後の観測・出力は変わるので、把持保持の成功は未検証。
検証結果: logs/openwam_tracking_diagnosis/gripper_retune_035100.json。状態遷移テスト10件成功。
この調整の反映はブリッジ再起動のみ。実機操作は実施していない。


## 2026-09-24: 両腕の手首特異姿勢回避

`ros.eef_ik.wrist_singularity_avoidance` を有効化。詳細は [診断・実装記録](OPENWAM_WRIST_SINGULARITY_AVOIDANCE.md)。位置誤差3 mm以内、姿勢誤差8°以内で手首の曲がりと送信角の連続性を保つ。制約が両立しなければ当該周期の送信を見送る。ブリッジ再起動で反映。実機未検証。

### 040452後：手首以外の特異姿勢も判定

全6軸Jacobian条件数35以上でも回避を作動し、40以下を目指す制約を追加。
位置・姿勢を許容内に収められない場合、その腕のみ最後の特異性が低い送信姿勢へ保持し、他腕は継続する。
保持時はモデル目標に未到達である旨を警告・ログ記録。安全条件を満たす保持姿勢もなければ送信を見送る。
詳細と限界は OPENWAM_WRIST_SINGULARITY_AVOIDANCE.md の040452修正欄。
ブリッジ再起動で反映。

## 2026-09-24 現行IK

旧条件数guardを微分IKに置換。現在の設定・検証・適用方法は [OPENWAM_DIFFERENTIAL_IK.md](OPENWAM_DIFFERENTIAL_IK.md) を参照。以下の過去の絶対IK残差閾値・wrist guardの説明より、この新方式が優先されます。

## 2026-09-24 グリッパ開閉バランス再調整

左右の close_threshold_normalized を [0.75, 0.75]、
open_threshold_normalized を [0.90, 0.90] に変更。
全閉0.0・全開0.06558、2 publish連続確認は維持。
開状態では正規化開度<=0.75が2指令続くと閉じ、
閉状態では>=0.90が2指令続くと開く。中間帯は前状態を維持する。

最新043908ログには、右のモデル開度が0.66〜0.69でも旧close閾値0.65を
下回らず全開維持になる区間があった。同じ入力列のリプレイでは、
右の閉指令数が64/844から129/844に増え、従来閉じなかった2区間でも閉じる。
これは把持成功の測定ではない。モデルが開度1.0を出し続ける区間では開いたまま。
開動作も遅くなるため、実機では接近時の開きタイミングとリリースを確認する。

再現: PYTHONPATH=. /workspace/project/OpenWAM/.venv-piper/bin/python scripts/replay_gripper_rebalance.py
結果: logs/openwam_tracking_diagnosis/gripper_rebalance_20260924.json
適用にはブリッジのみ再起動。サーバー再起動は不要。実機操作は実施していない。
