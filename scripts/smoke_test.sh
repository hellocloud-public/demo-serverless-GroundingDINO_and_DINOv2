#!/usr/bin/env bash
# デプロイ済みの推論 API に対する疎通テスト。
#
#   export COUNT_API_URL=<modal deploy で表示された URL>
#   export COUNT_API_TOKEN=<Secret grounding-dino-dinov2-api の COUNT_API_TOKEN の値>
#   bash scripts/smoke_test.sh                         # COCO のサンプル画像 (猫2匹+リモコン2個) で実行
#   bash scripts/smoke_test.sh shelf.jpg "a bottle"    # 手元の画像とプロンプトで実行
#
# 必要なもの: curl と python3 (JSON の整形用)。macOS なら sips でお手本画像を切り出すテストも行う。
set -euo pipefail

: "${COUNT_API_URL:?COUNT_API_URL を設定してください}"
: "${COUNT_API_TOKEN:?COUNT_API_TOKEN を設定してください}"

WORK=$(mktemp -d)
IMAGE="${1:-}"
PROMPT="${2:-a cat. a remote control.}"

if [ -z "$IMAGE" ]; then
  IMAGE="$WORK/cats.jpg"
  curl -sS -o "$IMAGE" http://images.cocodataset.org/val2017/000000039769.jpg
fi

summary() {
  python3 - "$1" <<'PY'
import json, sys
d = json.load(open(sys.argv[1])); c = d["container"]
print(f'  count={d["count"]}  cold={c["cold"]}  timing_ms={d["timing_ms"]}  container={c["id"]}')
for det in d["detections"]:
    extra = f'  similarity={det["similarity"]:.3f}' if "similarity" in det else ""
    print(f'    {det["label"]:20s} score={det["score"]:.3f}{extra}')
PY
}

post() {  # post <出力ファイル> [curl の追加オプション...]
  local out=$1; shift
  curl -sS -o "$out" -w "  HTTP %{http_code}  elapsed %{time_total}s\n" -X POST "$COUNT_API_URL/count" \
    -H "Authorization: Bearer $COUNT_API_TOKEN" -F "image=@$IMAGE" -F "prompt=$PROMPT" "$@"
}

echo "== 1. POST /count 1回目 (0台からならコールドスタートで数十秒かかる) =="
post "$WORK/r1.json"; summary "$WORK/r1.json"

echo "== 2. POST /count 2回目 (ウォーム) =="
post "$WORK/r2.json"; summary "$WORK/r2.json"

if [ -z "${1:-}" ] && command -v sips >/dev/null; then
  echo "== 3. POST /count + お手本画像 (右の猫を切り出して渡す → 猫だけ残りリモコンは除外) =="
  sips -c 351 292 --cropOffset 23 345 "$IMAGE" --out "$WORK/cat_ref.jpg" >/dev/null
  post "$WORK/r3.json" -F "reference=@$WORK/cat_ref.jpg"; summary "$WORK/r3.json"
fi

echo "== 4. GET /health (認証なし) =="
curl -sS -w "  HTTP %{http_code}\n" "$COUNT_API_URL/health"

echo "== 5. 認証エラー (どちらも 401 なら OK) =="
curl -sS -o /dev/null -w "  wrong token -> HTTP %{http_code}\n" -X POST "$COUNT_API_URL/count" \
  -H "Authorization: Bearer wrong" -F "image=@$IMAGE" -F "prompt=$PROMPT"
curl -sS -o /dev/null -w "  no token    -> HTTP %{http_code}\n" -X POST "$COUNT_API_URL/count" \
  -F "image=@$IMAGE" -F "prompt=$PROMPT"

echo "== レスポンス全文 (1回目) =="
python3 -m json.tool "$WORK/r1.json"
