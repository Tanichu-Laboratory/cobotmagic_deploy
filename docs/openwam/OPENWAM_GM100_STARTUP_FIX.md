# GM100 step96000 起動対応（2026-09-24）

指定されたチェックポイント:
`/workspace/dataset/checkpoints/gm100_openwam_step96000/checkpoint_step_96000.safetensors`

実起動で再現したエラー:

    ValueError: incompatible checkpoint dataloader.type: 'gm100'

common/openwam_piper.py の validate_checkpoint が dataloader.type=robodojo
だけを認めていたため、重みをロードする前に失敗していた。
robodojoとgm100を受け入れるよう修正した。
real / piper / eef、unify_action、左右の次元配置[0-9,34-43]、
head/left/rightのカメラ順の検証は維持。
チェックポイント側のconfig.yamlや重みは変更していない。
旧チェックポイント指定はデプロイ設定内のコメントとして保持。

関連35テスト成功。gm100でも異なるロボット・joint指令・左右逆配置・
カメラ順違いは拒否するテストを追加した。

--startup-testで実際にCUDA:0にロードし、
denoise_steps=10、mock=falseの推論成功。
出力32×14、全要素有限。初回推論約1.73秒。
モデルの構築・読み込み時間はこれに含まれない（約2分）。
正規化統計とtokenizerもチェックポイントの同梱資産から読み込めた。

起動検証ログ:
logs/openwam_96000_startup_verify.log

通常サーバーの通信検証は本番5557と分離した127.0.0.1:15557で実施。
サーバーログ: logs/openwam_96000_server_verify.log
クライアント結果: logs/openwam_96000_client_verify.log

起動コマンド:

    cd /workspace/cobotmagic_deploy
    bash scripts/serve_openwam_piper.sh

実機を動かすブリッジはこの検証では起動しない。

通信検証結果: READY到達後、連続2リクエストとも32×14・全要素有限、
グリッパ指令も設定範囲内。応答時間は約0.384秒／0.376秒
（同一の合成観測を使用。実デプロイの速度・把持成功を示すものではない）。
検証用サーバーは終了済み。本番ポート5557にはサーバーを残していない。
