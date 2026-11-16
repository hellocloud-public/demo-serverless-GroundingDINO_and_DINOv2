"""
GroundingDINO + DINOv2 で「テキストで指定した物体を検出して数える」GPU 推論 API のサンプル。

HelloCloud Serverless (Modal SDK 互換) 向けに書いていますが、Modal でもそのまま動きます。

  処理         GroundingDINO でテキスト指定の物体検出
               → 参照画像があれば検出枠を切り出し、DINOv2 の埋め込み類似度で絞り込む
  GPU          NVIDIA RTX PRO 6000 x1 / コンテナ (環境変数 GPU_TYPE で変更可)
  実行環境      Python 3.11 + PyTorch 2.8 (CUDA 12.8) + transformers 4.51 + FastAPI
  モデル        IDEA-Research/grounding-dino-tiny + facebook/dinov2-small (重みはイメージに同梱)
  API          POST /count (multipart)。Bearer トークンはアプリ側で検証
  スケーリング   リクエストに応じて自動スケール。5 分アイドルで 0 台

使い方:
  modal secret create grounding-dino-dinov2-api COUNT_API_TOKEN=<任意の文字列>
  modal run app.py       # 一時起動してサンプル画像で疎通確認 (コールド→ウォームの 2 回)
  modal deploy app.py    # 常設デプロイ。表示された URL を client.py から呼ぶ
"""

import hmac
import io
import os
import re
import time

import modal

# ---------------------------------------------------------------------------
# 設定
# ---------------------------------------------------------------------------
APP_NAME = "grounding-dino-dinov2"

# HelloCloud Serverless / Modal ともに "RTX-PRO-6000" (Blackwell, VRAM 96GB) を指定できる。
# Modal で別の GPU を使う場合はデプロイ時に GPU_TYPE=L4 などを指定する
GPU = os.environ.get("GPU_TYPE", "RTX-PRO-6000")
PYTHON_VERSION = "3.11"
TORCH_INDEX_URL = "https://download.pytorch.org/whl/cu128"  # Blackwell (sm_120) は CUDA 12.8 以降が必要

MODEL_DIR = "/models"  # 重みの置き場所 (イメージに同梱する)
DETECTOR_ID = "IDEA-Research/grounding-dino-tiny"  # テキスト指定の物体検出
EMBEDDER_ID = "facebook/dinov2-small"  # 検出した領域と参照画像 (お手本) を埋め込みで照合

SECRET_NAME = "grounding-dino-dinov2-api"  # COUNT_API_TOKEN を持つ Secret
TOKEN_ENV = "COUNT_API_TOKEN"

SCALEDOWN_WINDOW = 5 * 60  # 5 分アイドルで停止 (scale-to-zero)
MAX_CONTAINERS = 2  # バースト時の上限 (デモ用)
REQUEST_TIMEOUT = 120  # 推論 1 件の上限秒

SAMPLE_IMAGE_URL = "http://images.cocodataset.org/val2017/000000039769.jpg"  # 猫 2 匹 + リモコン 2 個


# ---------------------------------------------------------------------------
# 推論ロジック (Modal に依存しない部分)
# ---------------------------------------------------------------------------
def load_models(device: str) -> dict:
    """検出器と埋め込みモデルを読み込む。"""
    from transformers import (
        AutoImageProcessor,
        AutoModel,
        AutoModelForZeroShotObjectDetection,
        AutoProcessor,
    )

    return {
        "device": device,
        "detector_processor": AutoProcessor.from_pretrained(DETECTOR_ID),
        "detector": AutoModelForZeroShotObjectDetection.from_pretrained(DETECTOR_ID).to(device).eval(),
        "embedder_processor": AutoImageProcessor.from_pretrained(EMBEDDER_ID),
        "embedder": AutoModel.from_pretrained(EMBEDDER_ID).to(device).eval(),
    }


def normalize_prompt(prompt: str) -> str:
    """GroundingDINO が期待する "a cat. a remote control." の形式に揃える。"""
    parts = [p.strip().lower() for p in re.split(r"[.,。、]", prompt) if p.strip()]
    return ". ".join(parts) + "." if parts else "object."


def embed_images(models: dict, images: list) -> "torch.Tensor":
    """DINOv2 の CLS 埋め込みを L2 正規化して返す。"""
    import torch

    inputs = models["embedder_processor"](images=images, return_tensors="pt").to(models["device"])
    with torch.inference_mode():
        outputs = models["embedder"](**inputs)
    return torch.nn.functional.normalize(outputs.pooler_output, dim=-1)


def count_objects(
    models: dict,
    image_bytes: bytes,
    prompt: str,
    reference_bytes: bytes | None = None,
    threshold: float = 0.35,
    text_threshold: float = 0.25,
    similarity_threshold: float = 0.5,
) -> dict:
    """画像 1 枚 (+ 参照画像) から対象物を検出し、件数と検出位置を返す。

    1. 物体検出    : プロンプトで指定した対象を GroundingDINO で検出
    2. 領域切り出し : 検出枠ごとに画像を切り出す
    3. 画像埋め込み : 参照画像がある場合、切り出し画像と参照画像の類似度で絞り込む
    """
    import torch
    from PIL import Image

    device = models["device"]
    timing = {}
    started = time.perf_counter()

    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    text = normalize_prompt(prompt)

    # 1. 物体検出
    t = time.perf_counter()
    processor = models["detector_processor"]
    inputs = processor(images=image, text=text, return_tensors="pt").to(device)
    with torch.inference_mode():
        outputs = models["detector"](**inputs)
    result = processor.post_process_grounded_object_detection(
        outputs,
        inputs.input_ids,
        threshold=threshold,
        text_threshold=text_threshold,
        target_sizes=[image.size[::-1]],
    )[0]
    labels = result.get("text_labels") or result["labels"]
    detections = [
        {
            "box": [round(v, 1) for v in box.tolist()],  # [x1, y1, x2, y2] (ピクセル)
            "score": round(float(score), 4),
            "label": str(label),
        }
        for box, score, label in zip(result["boxes"], result["scores"], labels)
    ]
    timing["detect_ms"] = round((time.perf_counter() - t) * 1000)

    # 2. 領域切り出し + 3. 画像埋め込み (参照画像がある場合のみ)
    if reference_bytes is not None and detections:
        t = time.perf_counter()
        reference = Image.open(io.BytesIO(reference_bytes)).convert("RGB")
        crops = []
        for d in detections:
            x1, y1, x2, y2 = (int(v) for v in d["box"])
            crops.append(image.crop((x1, y1, max(x2, x1 + 1), max(y2, y1 + 1))))
        embeddings = embed_images(models, [reference, *crops])
        similarities = (embeddings[1:] @ embeddings[0]).tolist()
        for d, s in zip(detections, similarities):
            d["similarity"] = round(s, 4)
        detections = [d for d in detections if d["similarity"] >= similarity_threshold]
        timing["embed_ms"] = round((time.perf_counter() - t) * 1000)

    timing["total_ms"] = round((time.perf_counter() - started) * 1000)
    return {
        "count": len(detections),
        "prompt": text,
        "image_size": list(image.size),
        "detections": detections,
        "timing_ms": timing,
    }


# ---------------------------------------------------------------------------
# コンテナイメージ (ビルド時に重みを同梱する)
# ---------------------------------------------------------------------------
def download_models() -> None:
    """重み (約 0.8GB) をイメージに同梱し、コールドスタート時のダウンロードをなくす。"""
    from huggingface_hub import snapshot_download

    for repo_id in (DETECTOR_ID, EMBEDDER_ID):
        snapshot_download(repo_id, ignore_patterns=["*.bin", "*.h5", "*.msgpack", "*.onnx", "*.ot"])


image = (
    modal.Image.debian_slim(python_version=PYTHON_VERSION)
    .env(
        {
            "HF_HOME": f"{MODEL_DIR}/hf",
            "HF_HUB_DISABLE_TELEMETRY": "1",
            "TOKENIZERS_PARALLELISM": "false",
        }
    )
    .pip_install("torch==2.8.0", "torchvision==0.23.0", index_url=TORCH_INDEX_URL)
    .pip_install(
        "transformers==4.51.3",
        "huggingface_hub>=0.30,<1.0",
        "fastapi[standard]>=0.115,<1.0",
        "pillow>=10",
    )
    .run_function(download_models)
    .env({"HF_HUB_OFFLINE": "1"})  # 実行時は Hub に問い合わせず同梱した重みだけを使う
)

secret = modal.Secret.from_name(SECRET_NAME, required_keys=[TOKEN_ENV])

app = modal.App(APP_NAME)

with image.imports():
    import torch
    from fastapi import FastAPI, File, Form, Header, HTTPException, UploadFile


# ---------------------------------------------------------------------------
# GPU 推論 API
# ---------------------------------------------------------------------------
@app.cls(
    image=image,
    gpu=GPU,
    secrets=[secret],
    scaledown_window=SCALEDOWN_WINDOW,
    min_containers=0,  # 平常時は 0 台。常時 1 台起動しておきたい場合は 1 にする (その分 GPU 課金が続く)
    max_containers=MAX_CONTAINERS,
    timeout=REQUEST_TIMEOUT,
)
class CountModel:
    @modal.enter()
    def load(self) -> None:
        """コンテナ起動時に一度だけ実行。コールドスタート時間の大半はイメージ取得とここ。"""
        t = time.perf_counter()
        self.models = load_models("cuda")
        self.model_load_seconds = round(time.perf_counter() - t, 2)
        self.started_at = time.time()
        self.requests_served = 0
        self.container_id = os.environ.get("MODAL_TASK_ID", "unknown")
        self.gpu_name = torch.cuda.get_device_name(0)

    def _count(self, image_bytes: bytes, prompt: str, reference_bytes: bytes | None = None, **thresholds) -> dict:
        self.requests_served += 1
        result = count_objects(self.models, image_bytes, prompt, reference_bytes, **thresholds)
        result["container"] = {
            "id": self.container_id,
            "gpu": self.gpu_name,
            "model_load_seconds": self.model_load_seconds,
            "uptime_seconds": round(time.time() - self.started_at, 1),
            "requests_served": self.requests_served,
            "cold": self.requests_served == 1,  # このコンテナで最初のリクエストなら True
        }
        return result

    @modal.method()
    def count(self, image_bytes: bytes, prompt: str, reference_bytes: bytes | None = None) -> dict:
        """Modal SDK から直接呼ぶ用 (HTTP を介さない)。"""
        return self._count(image_bytes, prompt, reference_bytes)

    @modal.asgi_app()
    def serve(self):
        """HTTPS エンドポイント。client.py / scripts/smoke_test.sh はここを呼ぶ。"""
        web = FastAPI(title="GroundingDINO + DINOv2 count API")

        def check_bearer(authorization: str | None) -> None:
            scheme, _, token = (authorization or "").partition(" ")
            if scheme.lower() != "bearer" or not hmac.compare_digest(token, os.environ[TOKEN_ENV]):
                raise HTTPException(status_code=401, detail="invalid token")

        @web.get("/health")
        def health() -> dict:
            return {"status": "ok", "gpu": self.gpu_name, "requests_served": self.requests_served}

        @web.post("/count")
        def count(
            image: UploadFile = File(...),
            reference: UploadFile | None = File(None),
            prompt: str = Form("object"),
            threshold: float = Form(0.35),
            text_threshold: float = Form(0.25),
            similarity_threshold: float = Form(0.5),
            authorization: str | None = Header(None),
        ) -> dict:
            check_bearer(authorization)
            return self._count(
                image.file.read(),
                prompt,
                reference.file.read() if reference else None,
                threshold=threshold,
                text_threshold=text_threshold,
                similarity_threshold=similarity_threshold,
            )

        return web


# ---------------------------------------------------------------------------
# 疎通確認 (`modal run app.py`)
# ---------------------------------------------------------------------------
@app.local_entrypoint()
def main(image: str = "", prompt: str = "a cat. a remote control.", reference: str = "", repeat: int = 2) -> None:
    """同じリクエストを SDK 経由で繰り返し、コールド→ウォームの応答時間を比べる。"""
    import json
    import urllib.request

    image_bytes = open(image, "rb").read() if image else urllib.request.urlopen(SAMPLE_IMAGE_URL).read()
    reference_bytes = open(reference, "rb").read() if reference else None

    model = CountModel()
    for i in range(repeat):
        t = time.perf_counter()
        result = model.count.remote(image_bytes, prompt, reference_bytes)
        elapsed = round(time.perf_counter() - t, 2)
        state = "cold" if result["container"]["cold"] else "warm"
        print(f"\n--- request {i + 1} ({state}) : count={result['count']} elapsed={elapsed}s ---")
        print(json.dumps(result, ensure_ascii=False, indent=2))

    print(f"\nHTTP endpoint (POST /count): {model.serve.get_web_url()}")
