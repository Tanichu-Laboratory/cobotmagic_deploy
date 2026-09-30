# Piper 微分IKへの変更（2026-09-24）

現在の有効設定は config_openwam_piper.yaml の solver: differential。
旧 wrist_singularity_avoidance の閾値調整方式を置き換えた。
実機の起動・publishはこの変更作業では実施していない。

## 調査と原因
公式SDKは関節指令と末端姿勢指令の両方を提供する。
公式ROS資料も解なし・特異点・関節制限の状態を区別しており、
末端姿勢指令に切り替えるだけで到達性が保証されるわけではない。
またファームウェア世代によりJ2/J3のDHオフセットが異なる。
実機のファームウェア版・独自ROSドライバを未確認のまま、
単位や座標系を変更したり、ファームウェアCartesian制御に切り替えることはしていない。

- [公式Cartesian指令デモ](https://github.com/agilexrobotics/piper_sdk/blob/master/piper_sdk/demo/V2/piper_ctrl_end_pose.py)
- [公式関節指令デモ](https://github.com/agilexrobotics/piper_sdk/blob/master/piper_sdk/demo/V2/piper_ctrl_joint.py)
- [公式ROS noetic資料](https://github.com/agilexrobotics/piper_ros/tree/noetic)
- [公式FKとDH定義](https://github.com/agilexrobotics/piper_sdk/blob/master/piper_sdk/kinematics/piper_fk.py)
- [Pinkの制約付き微分IK実装](https://github.com/pink-kinematics/pink/blob/main/pink/solve_ik.py)
- [Pinkの速度制約](https://github.com/pink-kinematics/pink/blob/main/pink/limits/velocity_limit.py)

最新 action_commands_20260924_041814.csv は112行。
旧右腕guardは inactive 28、track_relaxed_pose 68、hold_last_safe 16。
最終保持の条件数は40.0000007357、位置残差20.987 mm。
絶対IKには解があるが、その先の条件数40制約が移動を妨げていた。
これはログからの診断であり、実機故障やファームウェアの特異点停止を確認したものではない。

## 新解法
独自実装のbox制約付き減衰微分IK。Pinkへの依存は追加していない。
校正済みtool座標系の幾何JacobianとSO(3)回転残差を使い、
各周期の関節変位をSciPy BVLSで解く。最大8回の局所線形化。
SVDの小さい特異値に対応する方向だけ正則化を強める。
位置と姿勢は重み付きsoft task（姿勢換算係数0.15 m/rad）。
正則化には前回publish姿勢と校正時姿勢への弱いバイアスを使う。

最適化の内部で次の制約を同時に適用する。
- URDF関節角度範囲
- 実測角度との差: J1–3 ±0.18 rad、J4–6 ±0.24 rad
- 前回publishからの速度: J1–3 0.6 rad/s、J4–6 0.8 rad/s
- 指令列の加速度: J1–3 3 rad/s²、J4–6 4 rad/s²

実測追従制限と加速度制限が両立しない場合、角度・速度制限を優先し、
acceleration_override=trueを記録する。角度・速度制約も両立しなければ
acceptable=falseとしてpublishを拒否する。機械の実加速度を保証するものではない。

実際のpublish完了後だけ履歴をcommit。左右両方の指令が許容されなければ履歴は進めない。
計算前までの経過時間を使用し、推論待ちによる長い間隔でもdtは0.2秒を上限とする。
長い中断後は前回速度をゼロとして再始動する。
既存joint_signs・tool校正・URDF補正・グリッパ処理は維持。

## 診断の意味と限界
acceptable は「有限かつ関節制約内の指令」の意味。
target_reached は位置3 mm以内かつ姿勢0.14 rad以内。
solution_* はこの周期のbounded解であり、別途計算した絶対IK解ではない。
残差が大きいだけで全体を停止せず、boundedな途中指令を許容する。
differential_ik にmode、速度、条件数、最小特異値、加速度例外、誤差を保存する。
進めなくなった場合はstalledとして警告する。旧condition=40でのholdは使用しない。

特異姿勢を完全に排除するアルゴリズムではなく、特異点近傍で
過大な関節変位や解枝の飛び移りを抑えるアルゴリズム。
到達不能姿勢・局所停滞はあり得る。衝突回避や大域経路計画は実装していない。
有限の位置・姿勢誤差と追従遅れを許容するため、把持成功は実機で別途評価が必要。

## 検証
Python 3.11: 関連81テスト成功。
新規テストは収束、速度/加速度制約、未publish履歴不変、
中断後の再始動、特異点近傍、制約不整合、非正の時間、未到達診断を確認。
ROS Python 3.8にはpytestがないため、直接ログ再計算と構文チェックで検証。

scripts/replay_openwam_differential_ik.pyで過去5エピソードを再計算。
初回実測で校正し、以降は新指令を理想追従する反実仮想。
追加で毎周期差分の70%だけ追従する簡易遅れモデルも評価。
後者は実機モデルの同定ではない。どちらもモデル入力・目標は旧記録のままで、
画像を再入力する閉ループ評価ではない。

理想追従の左右合計1,434指令で拒否0・stalled 0・加速度例外0。
最新右腕: 位置誤差95%点4.43 mm、最大5.38 mm、
姿勢誤差95%点1.15°、最大1.32°。
5エピソード全体の最大位置誤差8.44 mm。
片腕計算はPython 3.11測定で概ね4–7 ms。
特異点近傍の条件数は大きい値を取り得るが、これを停止閾値にはしていない。
詳細: logs/openwam_differential_ik/replay.json。

## 適用
モデルサーバーはそのままで、ブリッジのみ停止後に再起動する。
起動に伴う既存の初期姿勢移動がある。

    cd /workspace/cobotmagic_deploy
    source /opt/ros/noetic/setup.bash
    conda activate aloha
    source scripts/env_openwam_piper.sh
    python -m cobotmagic_deployment.bridges.ros_bridge_node --config cobotmagic_deployment/configs/config_openwam_piper.yaml

新ログではpublish前残差と、実測EEFとのpublish後残差を分けて評価する。

## 同日後続の精度調整

正則化を過去姿勢へのバイアスから各反復の修正量への減衰に変更しました。上記の初期姿勢へのバイアスの説明は旧実装です。制限別の診断と再検証結果は [OPENWAM_DELTA_LIMIT_ADJUSTMENT.md](OPENWAM_DELTA_LIMIT_ADJUSTMENT.md) を参照してください。
