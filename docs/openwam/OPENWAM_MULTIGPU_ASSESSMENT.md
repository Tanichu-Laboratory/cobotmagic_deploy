# OpenWAM Piper 複数GPU推論の検討

## 結論

単一観測の応答時間短縮は、tensor/sequence parallelを専用実装すれば候補になる。
現在のコードには、そのまま有効化できる分散推論経路はない。
まず単一GPUのcompile最適化を測定し、その後2GPUの同一入力比較を行う順序が妥当。
4GPU化による高速化率は未測定であり、保証しない。
稼働サーバー・ROSブリッジ・重み・デプロイconfigは今回変更していない。

## 確認した構成

- RTX PRO 6000 Blackwell Max-Q、約96GB × 4。
- GPU間は全組み合わせ NODE（同一NUMA内のPCIe host bridge経由）、NVLink接続なし。
- 全GPU間でP2P read/writeはサポート。
- GPU0の既存サーバーは約25GB使用し、モデルは1枚に収まる。
- 画像384×320、学習窓33、video_stride=4で動画9フレーム。
  VAE空間圧縮16・時間圧縮4、patch=(1,2,2)から、動画側は360 tokensと算出。
  一般的な高解像度・長時間の動画生成より、sequence parallelの分割粒度が小さい。

## 実装の確認

- OpenWAM `scripts/deploy.sh` の `NUM_GPUS` はGPUごとの独立WebSocketサーバー起動。
  処理件数を増やす機能で、1リクエストの分散計算ではない。
- Piperモデルは dual_system/joint_self_attn、mutual attention。
  `openwam/model/architectures/dual_system/mot_driver.py` は各層で動画と行動のQ/K/Vを結合してSDPAを行う。
  左右の腕も同一行動モデルから同時に生成するため、左右各GPUへの単純分割はできない。
- sequence parallelのコードは `wan/_reference.py` にあるが、現在のMoT推論経路には接続されていない。
- Wan推論ではCFGを使わない（cfg_scale=1）。conditional/unconditionalを2GPUへ分ける案は使えない。
- text embedding cache、DiT velocity cache、動画decode省略は既に有効。compileは現在無効。

## 方式の比較

| 方式 | 単一応答時間への見込み | 制約 |
|---|---|---|
| GPUごとに独立モデル | 通常短縮しない | 多台数ロボット・評価並列向け |
| 左右腕ごとにGPU | 単純適用不可 | 双腕を同一モデルで同時生成 |
| 動画GPU / 行動GPU | 優先度低 | 各層の結合attention、動画側との計算量不均衡 |
| 層を連続区間で分割 | 優先度低 | batch=1では層の依存関係が直列、主にメモリ分散用途 |
| CFG並列 | 現構成では不可 | CFG分岐がない |
| Tensor parallel | 応答時間短縮の候補 | QKV、出力投影、FFNの分割とcollectiveが必要 |
| Sequence parallel | 応答時間短縮の候補 | MoT、mask、RoPE、行動・proprioの配置を整合させる必要 |

xDiTはWan2.2 TI2V対応を公開しているが、OpenWAMの独自行動expert・結合attentionへの対応を意味しない。
利用する場合もOpenWAM向けの統合作業が必要。

## 通信の参考実測

空きGPU2→3、torch Tensor.copy_、5回warmup後50回、両GPU同期を含む平均:

| サイズ | 転送時間 | 帯域 |
|---|---:|---:|
| 1 MiB | 0.040 ms | 26.5 GB/s |
| 8 MiB | 0.231 ms | 36.3 GB/s |
| 32 MiB | 0.911 ms | 36.8 GB/s |

これは一方向bulk copyであり、NCCL all-reduce/all-to-all性能や分散モデルの高速化率ではない。
データ: `logs/openwam_multigpu/p2p_gpu2_to_gpu3.json`。

## 推奨評価手順

1. 同じ実観測セット・seed・10 denoise stepsで1GPU baselineを測る。
   初回とwarmup後を区別し、画像ごとのDiT cache skip数、p50/p95、GPU時間、前処理時間を記録。
2. 1GPUのcompileを有効にした別プロセスで比較。コンパイル成功・graph break・出力差も確認。
3. 不十分なら2GPU tensor/sequence parallelの小規模試作。
   特に最終行動20D、グリッパー、mask、非活性次元の処理を単一GPUと照合する。
4. 実観測でp95応答時間が十分改善し、出力整合性を確認できてから4GPUを比較する。

制御と推論の非同期化は制御中の待ちを隠す別の選択肢だが、1回の計算自体を速くするものではない。
現在のCobotMagic adapterは同期chunk生成なので、非同期化にはadapter/bridge統合の検討が必要。
Denoise steps削減は出力品質に影響し得るため、別の品質評価項目として扱う。

## 一次資料

- https://github.com/OpenWAM-Official/OpenWAM/blob/main/assets/openwam_usage_docs/train-and-deploy.md
- https://github.com/xdit-project/xDiT/blob/main/docs/runner/runner.md
- https://docs.pytorch.org/docs/stable/distributed.tensor.parallel

## 単一GPUの参考実測

空きGPU1、既存と同じ重み・10 denoise steps・compile無効・動画decode省略。
最初は黒画像、その後は毎回異なるseedのランダムRGB画像3枚、同一タスク・EEF状態で実行。
各呼び出しの前後でGPU1を明示同期したwall time:

- 初回: 1.877秒（モデルロード時間を除く）。
- warmup後4回: 0.360 / 0.359 / 0.362 / 0.360秒、中央値約0.360秒。
- 各回10 denoise steps中6回のDiT再計算をキャッシュで省略。
- 出力は全回8×14、finite。ピークallocated memory約25.10 GB。
- ROS/ZMQ/IKの時間を含まず、実画像・実機性能や分散高速化率を示す結果ではない。
- ログ内の内部stage profilingは既定CUDAデバイスを同期する実装なので、
  GPU1測定ではstage別時間を定量根拠に使わず、明示同期した全体時間のみ採用した。
- データ: `logs/openwam_multigpu/single_gpu_results.json`。

今回は分散モデル自体の実装・2/4GPUでのモデル速度比較は行っていない。
