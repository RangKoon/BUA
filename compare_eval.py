import json
from pathlib import Path

EVAL_DIR = Path("./data/eval_results")
MODELS = ["gemini-2.5-flash", "qwen-base", "qwen-ft"]


def load(model):
    path = EVAL_DIR / f"{model}_final.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else []


def stats(records):
    total = len(records)
    if not total:
        return {}
    success = sum(1 for r in records if r["success"])
    step_times = [r["step_duration_avg"] for r in records if r.get("steps", 0) > 0]
    return {
        "total": total,
        "success": success,
        "success_rate": round(success / total * 100, 1),
        "avg_duration": round(sum(r["duration_seconds"] for r in records) / total, 1),
        "avg_steps": round(sum(r["steps"] for r in records) / total, 1),
        "avg_step_time": round(sum(step_times) / len(step_times), 1) if step_times else 0,
    }


def main():
    all_data = {m: load(m) for m in MODELS}
    available = [m for m in MODELS if all_data[m]]

    if not available:
        print("결과 없음. eval_runner.py 먼저 실행하세요.")
        return

    col = 22
    sep = "=" * (20 + col * len(available))
    print(f"\n{sep}\n 모델 성능 비교 (테스트셋)\n{sep}")
    print(f"{'지표':<20}" + "".join(f"{m:>{col}}" for m in available))
    print("-" * (20 + col * len(available)))

    all_stats = {m: stats(all_data[m]) for m in available}
    for label, key, unit in [
        ("총 쿼리 수",       "total",        "개"),
        ("성공 수",          "success",       "개"),
        ("성공률",           "success_rate",  "%"),
        ("평균 소요 시간",   "avg_duration",  "초"),
        ("평균 스텝 수",     "avg_steps",     ""),
        ("평균 스텝당 시간", "avg_step_time", "초"),
    ]:
        row = f"{label:<20}" + "".join(
            f"{f'{all_stats[m].get(key, chr(45))}{unit}':>{col}}" for m in available
        )
        print(row)
    print(sep)

    # 쿼리별 성공/실패
    print(f"\n{'#':<4} {'쿼리':<38}" + "".join(f"{m:>{16}}" for m in available))
    print("-" * (42 + 16 * len(available)))
    by_index: dict[int, dict] = {}
    for m in available:
        for r in all_data[m]:
            idx = r["index"]
            if idx not in by_index:
                by_index[idx] = {"query": r["query"]}
            by_index[idx][m] = "✅" if r["success"] else "❌"

    for idx in sorted(by_index.keys()):
        e = by_index[idx]
        q = e["query"][:36] + ".." if len(e["query"]) > 38 else e["query"]
        print(f"{idx:<4} {q:<38}" + "".join(f"{e.get(m, '-'):>{16}}" for m in available))

    # 파인튜닝 효과
    if "qwen-base" in available and "qwen-ft" in available:
        improved = [i for i, e in by_index.items() if e.get("qwen-ft") == "✅" and e.get("qwen-base") == "❌"]
        degraded = [i for i, e in by_index.items() if e.get("qwen-ft") == "❌" and e.get("qwen-base") == "✅"]
        print(f"\n[FT 후 개선 {len(improved)}개]")
        for i in sorted(improved):
            print(f"  #{i}: {by_index[i]['query']}")
        print(f"\n[FT 후 저하 {len(degraded)}개]")
        for i in sorted(degraded):
            print(f"  #{i}: {by_index[i]['query']}")


if __name__ == "__main__":
    main()
