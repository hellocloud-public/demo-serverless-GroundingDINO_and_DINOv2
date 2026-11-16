"""
デプロイ済みの推論 API を HTTPS + Bearer + multipart で呼ぶクライアントのサンプル。

  pip install requests
  export COUNT_API_URL=<modal deploy で表示された URL>
  export COUNT_API_TOKEN=<modal secret に入れた値>
  python client.py --image shelf.jpg --prompt "a cardboard box"
  python client.py --image shelf.jpg --prompt "a bottle" --reference exemplar.jpg
"""

import argparse
import json
import os
import sys
import time

import requests


def count(url: str, token: str, image_path: str, prompt: str, reference_path: str | None, timeout: float) -> dict:
    files = {"image": open(image_path, "rb")}
    if reference_path:
        files["reference"] = open(reference_path, "rb")
    response = requests.post(
        f"{url.rstrip('/')}/count",
        headers={"Authorization": f"Bearer {token}"},
        files=files,
        data={"prompt": prompt},
        timeout=timeout,  # コールドスタートを含めて待つ秒数
    )
    response.raise_for_status()
    return response.json()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--url", default=os.environ.get("COUNT_API_URL"), help="推論 API の URL (env: COUNT_API_URL)")
    parser.add_argument("--token", default=os.environ.get("COUNT_API_TOKEN"), help="Bearer トークン (env: COUNT_API_TOKEN)")
    parser.add_argument("--image", required=True, help="数えたい画像")
    parser.add_argument("--prompt", default="object", help='数えたい対象 (例: "a cat. a remote control.")')
    parser.add_argument("--reference", help="お手本 (参照画像)。指定すると類似度で絞り込む")
    parser.add_argument("--timeout", type=float, default=60.0)
    args = parser.parse_args()
    if not args.url or not args.token:
        sys.exit("--url と --token (または COUNT_API_URL / COUNT_API_TOKEN) を指定してください")

    started = time.perf_counter()
    result = count(args.url, args.token, args.image, args.prompt, args.reference, args.timeout)
    elapsed = time.perf_counter() - started

    container = result["container"]
    print(f"count={result['count']}  elapsed={elapsed:.2f}s  "
          f"({'cold' if container['cold'] else 'warm'}, container={container['id']})")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
