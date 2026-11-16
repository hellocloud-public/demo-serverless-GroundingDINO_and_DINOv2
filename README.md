# GroundingDINO + DINOv2 サーバーレス GPU 推論サンプル

テキストで指定した物体を [GroundingDINO](https://huggingface.co/IDEA-Research/grounding-dino-tiny) で検出して数え、
参照画像 (お手本) があれば [DINOv2](https://huggingface.co/facebook/dinov2-small) の埋め込み類似度で絞り込む
GPU 推論 API のサンプルです。

[HelloCloud Serverless](https://hello.ai/cloud/) (Modal SDK 互換) 向けに書いていますが、[Modal](https://modal.com) でもそのまま動きます。

## 構成

```
client.py / scripts/smoke_test.sh
    │  HTTPS  POST /count  (multipart、Bearer トークン)
    ▼
CountModel.serve (FastAPI)   GPU コンテナ: RTX PRO 6000 x1、リクエストに応じて自動スケール、5 分アイドルで 0 台
    ├─ 1. GroundingDINO  テキストで指定した物体を検出
    ├─ 2. 切り出し        検出枠ごとに画像を切り出す
    └─ 3. DINOv2         参照画像との類似度で絞り込む (参照画像を渡したときだけ)
```

| ファイル | 役割 |
| --- | --- |
| `app.py` | アプリ本体。イメージ定義、GPU 推論クラス、FastAPI、`modal run` 用の疎通確認 |
| `client.py` | デプロイ済み API を呼ぶクライアントのサンプル |
| `scripts/smoke_test.sh` | デプロイ済み API の疎通確認 (curl のみ) |
| `.github/workflows/deploy.yml` | main へのマージ時に `modal deploy` する GitHub Actions |
| `requirements.txt` | ローカル / CI 用 (`modal==1.6.1`, `requests`) |

| 項目 | 内容 |
| --- | --- |
| GPU | `gpu="RTX-PRO-6000"` (NVIDIA RTX PRO 6000 Blackwell、VRAM 96GB)。環境変数 `GPU_TYPE` で変更可 |
| 実行環境 | `debian_slim` + Python 3.11、PyTorch 2.8 (CUDA 12.8)、transformers 4.51、FastAPI |
| モデル | `IDEA-Research/grounding-dino-tiny` + `facebook/dinov2-small`。重み (約 0.8GB) はビルド時にイメージへ同梱 |
| 認証 | Bearer トークン。値は Secret に入れ、アプリ側で検証 |

## 動かし方

### 前提

Python 3.10 以上で SDK を入れます。

```bash
pip install -r requirements.txt   # modal==1.6.1
```

**HelloCloud Serverless の場合**: 発行した API キーと API サーバーの URL を環境変数に設定します。以降のコマンドは Modal SDK と同じです。

```bash
export MODAL_SERVER_URL=<HelloCloud Serverless の API サーバーの URL>
export MODAL_TOKEN_ID=<API キーの ID>
export MODAL_TOKEN_SECRET=<API キーのシークレット>
```

**Modal の場合**: `modal setup` でログインします (`MODAL_SERVER_URL` は設定しません)。

### 1. Bearer トークンを Secret に登録

```bash
export COUNT_API_TOKEN=$(openssl rand -hex 24)   # 4. で API を呼ぶときにも使う
modal secret create grounding-dino-dinov2-api COUNT_API_TOKEN=$COUNT_API_TOKEN
```

### 2. 疎通確認 (一時起動)

```bash
modal run app.py
```

初回はイメージをビルドします (PyTorch の取得と重みの同梱)。その後サンプル画像
(COCO val2017 の猫 2 匹 + リモコン 2 個) を SDK 経由で 2 回推論し、コールド → ウォームの応答時間と結果を表示します。
手元の画像で試す場合:

```bash
modal run app.py --image shelf.jpg --prompt "a cardboard box"
modal run app.py --image shelf.jpg --prompt "a bottle" --reference exemplar.jpg   # お手本で絞り込み
```

### 3. 常設デプロイ

```bash
modal deploy app.py
```

`POST /count` を受ける URL が表示されます (Modal では `https://<workspace>--grounding-dino-dinov2-countmodel-serve.modal.run` の形)。
既定では 0 台まで縮退するので、アイドル中は GPU 課金が発生しません。止めるときは `modal app stop grounding-dino-dinov2` です。

### 4. API を呼ぶ

```bash
export COUNT_API_URL=<3. で表示された URL>
export COUNT_API_TOKEN=<1. で Secret に入れた値>
python client.py --image shelf.jpg --prompt "a cardboard box"
```

まとめて疎通確認する場合 (コールド/ウォーム、お手本画像、health、401 を順に実行):

```bash
bash scripts/smoke_test.sh                          # サンプル画像で実行
bash scripts/smoke_test.sh shelf.jpg "a bottle"     # 手元の画像で実行
```

curl の場合:

```bash
curl -sS -X POST "$COUNT_API_URL/count" \
  -H "Authorization: Bearer $COUNT_API_TOKEN" \
  -F image=@shelf.jpg -F prompt="a cardboard box"
```

### 5. GitHub Actions から自動デプロイ

リポジトリの Secrets に `MODAL_TOKEN_ID` と `MODAL_TOKEN_SECRET` を登録すると、main へ push したときに
`modal deploy app.py` が走ります (未登録ならスキップ)。HelloCloud Serverless にデプロイする場合は、
Secrets に `MODAL_SERVER_URL` (API サーバーの URL) も登録してください。未登録なら Modal にデプロイします。

## API

### `POST /count` (multipart/form-data, `Authorization: Bearer <token>` 必須)

| フィールド | 型 | 既定値 | 説明 |
| --- | --- | --- | --- |
| `image` | file | 必須 | 数えたい画像 |
| `prompt` | text | `object` | 数えたい対象。`"a cat. a remote control."` のように複数可 |
| `reference` | file | なし | お手本 (参照画像)。指定すると検出枠を切り出して DINOv2 の類似度で絞り込む |
| `threshold` | float | 0.35 | 検出スコアのしきい値 |
| `text_threshold` | float | 0.25 | テキスト一致のしきい値 |
| `similarity_threshold` | float | 0.5 | 参照画像との類似度のしきい値 (`reference` 指定時) |

サンプル画像 + `prompt="a cat. a remote control."` のレスポンス例 (`container` は省略):

```json
{
  "count": 3,
  "prompt": "a cat. a remote control.",
  "image_size": [640, 480],
  "detections": [
    {"box": [344.7, 23.1, 637.2, 374.3], "score": 0.4785, "label": "a cat"},
    {"box": [12.3, 51.9, 316.9, 472.4], "score": 0.4381, "label": "a cat"},
    {"box": [38.6, 70.0, 176.8, 118.2], "score": 0.4759, "label": "a remote control"}
  ],
  "timing_ms": {"detect_ms": 306, "total_ms": 311},
  "container": {"...": "..."}
}
```

右の猫の切り抜きを `reference` に渡すと、猫 2 匹 (`similarity` 約 0.98 / 0.60) だけが残り、リモコンは除外されます。

| フィールド | 説明 |
| --- | --- |
| `count` | 絞り込み後の検出数 |
| `detections[].box` | `[x1, y1, x2, y2]` (ピクセル) |
| `detections[].similarity` | 参照画像との類似度 (`reference` 指定時のみ) |
| `timing_ms` | 推論内部の内訳 (`detect_ms`、`embed_ms`、`total_ms`) |
| `container.cold` | そのコンテナで最初のリクエストなら `true`。コールドスタート全体の時間は呼び出し側で計測する (`client.py` が表示) |
| `container.gpu` / `container.model_load_seconds` | コンテナの GPU 名とモデルのロード時間 |

`GET /health` は認証なしで `{"status": "ok", ...}` を返します。

## カスタマイズ

| 変えたいもの | 変更箇所 |
| --- | --- |
| GPU | `GPU_TYPE=L4 modal deploy app.py` のように環境変数で指定 (既定は `RTX-PRO-6000`) |
| モデル | `DETECTOR_ID` を `IDEA-Research/grounding-dino-base` に、`EMBEDDER_ID` を `facebook/dinov2-base` / `-large` などに |
| スケーリング | `SCALEDOWN_WINDOW` (アイドル停止までの秒数)、`MAX_CONTAINERS`、`min_containers` (1 以上で常時起動。その分 GPU 課金が続く) |

RTX PRO 6000 は Blackwell 世代 (sm_120) のため、CUDA 12.8 以降向けの PyTorch が必要です。
PyTorch のバージョンを変える場合も `TORCH_INDEX_URL` は `cu128` 以降のままにしてください。

## ライセンス

使用しているモデルは `IDEA-Research/grounding-dino-tiny`、`facebook/dinov2-small` ともに Apache-2.0 です。
サンプル画像は COCO val2017 の画像を実行時にダウンロードして使います (リポジトリには含みません)。
