"""
homepage_collector.py 결과를 파인튜닝용 JSONL로 변환
- training_data → ft 포맷 변환
- 성공한 쿼리만 필터링

사용법:
  python homepage_convert.py
"""

import json
import re
from pathlib import Path

TRAINING_LOG = Path("./data/homepage/training_data_gemini-2.5-flash.jsonl")
RESULTS_JSON = Path("./data/homepage/results/gemini-2.5-flash_final.json")
OUTPUT_ALL   = Path("./data/homepage/ft_homepage.jsonl")
OUTPUT_OK    = Path("./data/homepage/ft_homepage_success.jsonl")


def convert():
    if not TRAINING_LOG.exists():
        print(f"학습 로그 없음: {TRAINING_LOG}")
        return

    raw = []
    with open(TRAINING_LOG, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                raw.append(json.loads(line))
    print(f"학습 로그 로드: {len(raw)}개")

    converted = []
    for d in raw:
        messages = [
            {"role": m["role"], "content": m["content"]}
            for m in d["input"]
        ]
        messages.append({
            "role": "assistant",
            "content": json.dumps(d["output"], ensure_ascii=False),
        })
        converted.append({"messages": messages})

    OUTPUT_ALL.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_ALL, "w", encoding="utf-8") as f:
        for item in converted:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"전체 변환 저장: {OUTPUT_ALL} ({len(converted)}개)")

    if not RESULTS_JSON.exists():
        print("결과 JSON 없음, 성공 필터링 생략")
        return

    with open(RESULTS_JSON, encoding="utf-8") as f:
        results = json.load(f)
    success_queries = {r["query"] for r in results if r.get("success")}
    print(f"성공 쿼리 수: {len(success_queries)}개")

    kept = []
    for item in converted:
        query = _extract_query(item)
        if query and query in success_queries:
            kept.append(item)

    with open(OUTPUT_OK, "w", encoding="utf-8") as f:
        for item in kept:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"성공 필터링 저장: {OUTPUT_OK} ({len(kept)}개)")


def _extract_query(item: dict) -> str | None:
    for msg in item["messages"]:
        content = msg.get("content", "")
        if isinstance(content, str) and "[질문]" in content:
            match = re.search(r"\[질문\]\n(.+?)(?:\n|$)", content)
            if match:
                return match.group(1).strip()
    return None


if __name__ == "__main__":
    convert()
