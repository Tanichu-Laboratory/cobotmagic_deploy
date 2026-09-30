# GM100に合わせた初期姿勢

/workspace/dataset/GM100 の106タスク・15,982エピソードを集計。
各parquetのframe_index=0にある observation.state.arm.position（左右各6関節）
および observation.state.effector.position（左右グリッパ）を使用。
action列ではなく実測state列を採用した。

タスクごとの関節中央値を求め、さらにタスク間の中央値を計算。
エピソード数の多いタスクだけに偏らない代表値とした。
その12関節代表値へのユークリッド距離が最小で、
現在のURDF関節範囲・グリッパ物理範囲内にある実在の開始フレームを選定。
合成した平均姿勢ではない。対象タスク限定の初期姿勢ではなく、全体の代表姿勢。

採用元:
 /workspace/dataset/GM100/task_00110/data/chunk-000/episode_000026.parquet
 frame_index=0

left:
 [-0.021682892, 0.0053029759, -0.0041167838, -0.0205490328, 0.2520832419, -0.0147401802, 0.0005]
right:
 [0.046383597, 0.0066984962, -0.001186192, 0.0358823091, 0.3131895661, 0.0, 0.0002]

先頭6値はROSの関節角度rad、最後はグリッパ値。
IK内部のURDF用符号変換をこのROS関節指令へ重ねて適用しない。
旧初期姿勢はconfig内コメントとして残した。
移動2秒・グリッパを終盤で動かす設定・settle設定は維持。
今回はグリッパもデータに合わせてほぼ閉じた姿勢となる。

検証: YAMLロード、全要素finite、採用フレームとの一致、関節範囲、
グリッパ範囲、移動時間等の設定不変。
実機の移動は実施していない。データに存在する姿勢であることは、
現在の設置環境での衝突回避や特異点回避を保証するものではない。

再集計:
 PYTHONPATH=. /workspace/project/OpenWAM/.venv-piper/bin/python scripts/analyze_gm100_home.py
集計結果:
 logs/openwam_tracking_diagnosis/gm100_initial_pose.json

ブリッジ再起動時の初期姿勢移動から反映。モデルサーバー再起動は不要。
