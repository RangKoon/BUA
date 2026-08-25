"""
충남대학교 컴퓨터융합학부 홈페이지 데이터 수집 스크립트
대상: https://computer.cnu.ac.kr/computer/index.do
모델: gemini-2.5-flash
용도: Qwen2.5-32B 파인튜닝용 학습 데이터 생성

사용법:
  python homepage_collector.py                      # 전체 130개
  python homepage_collector.py 0 50                 # 0~49번
  python homepage_collector.py 50 100               # 50~99번
"""

import asyncio
import json
import os
import sys
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from browser_use import BrowserSession, BrowserProfile
from browser_use.llm.google.chat import ChatGoogle
from logged_agent import LoggedAgent

load_dotenv()

HOMEPAGE_URL = "https://computer.cnu.ac.kr/computer/index.do"
MODEL_NAME = "gemini-2.5-flash"
DATA_DIR = Path("/NHNHOME/WORKSPACE/26moe002_H/mhjin/BUA/data/homepage")


def build_task(user_query: str) -> str:
    return f"""충남대학교 컴퓨터융합학부 홈페이지({HOMEPAGE_URL})에서 아래 질문에 답해줘.

[질문]
{user_query}

[지침]
- 반드시 홈페이지 내 실제 페이지를 직접 탐색하여 정보를 찾아라.
- URL을 추측하거나 임의로 생성하지 말고, 메뉴 링크를 클릭하여 이동하라.
- 찾은 정보를 명확하고 구체적으로 정리하여 답변하라.
- 페이지에 해당 정보가 없으면 "해당 정보를 찾을 수 없습니다"라고 답하라.
- 게시판 조회 시 페이지 목록을 충분히 탐색하라.
"""


async def run_query(user_query: str, llm, log_path: str) -> tuple[str | None, dict]:
    browser = BrowserSession(browser_profile=BrowserProfile(headless=True))
    agent = LoggedAgent(
        task=build_task(user_query),
        llm=llm,
        log_path=log_path,
        browser=browser,
        use_vision=False,
        max_failures=3,
        step_timeout=60,
        extend_system_message="모든 사고 과정과 응답은 반드시 한국어로만 작성하라. 영어 사용 금지.",
        initial_actions=[{"navigate": {"url": HOMEPAGE_URL}}],
    )
    history = await agent.run(max_steps=25)

    input_tokens  = getattr(history, "total_input_tokens",  lambda: None)()
    output_tokens = getattr(history, "total_output_tokens", lambda: None)()

    metrics = {
        "steps": history.number_of_steps(),
        "duration_seconds": round(history.total_duration_seconds(), 2),
        "success": history.is_successful() or False,
        "errors": [e for e in history.errors() if e is not None],
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": (input_tokens + output_tokens) if (input_tokens and output_tokens) else None,
    }
    return history.final_result(), metrics


async def run_batch(start: int = 0, end: int | None = None):
    from homepage_queries import HOMEPAGE_QUERIES

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    (DATA_DIR / "results").mkdir(exist_ok=True)

    safe = MODEL_NAME.replace(":", "-").replace("/", "-")
    log_path = str(DATA_DIR / f"training_data_{safe}.jsonl")
    progress_path = DATA_DIR / "results" / f"{safe}_progress.json"

    llm = ChatGoogle(model=MODEL_NAME)

    targets = HOMEPAGE_QUERIES[start:end]
    total = len(targets)
    actual_end = start + total
    print(f"\n모델: {MODEL_NAME}")
    print(f"범위: {start}번 ~ {actual_end - 1}번 (총 {total}개)")
    print(f"로그: {log_path}\n")

    all_results: list[dict] = []
    if progress_path.exists():
        with open(progress_path, encoding="utf-8") as f:
            all_results = json.load(f)
        print(f"기존 결과 {len(all_results)}개 로드, 이어서 실행\n")

    done_indices = {r["index"] for r in all_results}

    for i, query in enumerate(targets):
        global_idx = start + i
        if global_idx in done_indices:
            print(f"[{i+1}/{total}] SKIP ({global_idx}): {query[:50]}")
            continue

        print(f"[{i+1}/{total}] ({global_idx}) {query}")

        try:
            result, metrics = await run_query(query, llm, log_path)
        except Exception as e:
            metrics = {"steps": 0, "duration_seconds": 0, "success": False, "errors": [str(e)]}
            result = None

        step_avg = round(metrics["duration_seconds"] / metrics["steps"], 2) if metrics["steps"] else 0
        record = {
            "timestamp": datetime.now().isoformat(),
            "model": MODEL_NAME,
            "index": global_idx,
            "query": query,
            "success": metrics["success"],
            "steps": metrics["steps"],
            "duration_seconds": metrics["duration_seconds"],
            "step_duration_avg": step_avg,
            "errors": metrics["errors"],
            "input_tokens": metrics["input_tokens"],
            "output_tokens": metrics["output_tokens"],
            "total_tokens": metrics["total_tokens"],
            "result": result,
        }
        all_results.append(record)

        status = "✅" if metrics["success"] else "❌"
        token_str = f" | 토큰 {metrics['total_tokens']}개" if metrics["total_tokens"] else ""
        print(f"  {status} {metrics['duration_seconds']}초 | {metrics['steps']}스텝 | 스텝당 {step_avg}초{token_str}")
        if result:
            print(f"  결과: {result[:80]}...")

        if (i + 1) % 10 == 0:
            _save(all_results, safe, "progress")
            print(f"  >>> 중간 저장 완료 ({len(all_results)}개)\n")

    _save(all_results, safe, "final")
    _print_summary(all_results)


def _save(records: list[dict], safe: str, suffix: str):
    path = DATA_DIR / "results" / f"{safe}_{suffix}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)


def _print_summary(records: list[dict]):
    total = len(records)
    if not total:
        return
    success = sum(1 for r in records if r["success"])
    avg_time = sum(r["duration_seconds"] for r in records) / total
    avg_steps = sum(r["steps"] for r in records) / total
    step_times = [r["step_duration_avg"] for r in records if r["steps"] > 0]
    avg_step_time = sum(step_times) / len(step_times) if step_times else 0

    token_records = [r for r in records if r.get("total_tokens")]
    total_tokens = sum(r["total_tokens"] for r in token_records)
    avg_tokens = total_tokens // len(token_records) if token_records else 0

    print(f"\n{'='*60}")
    print(f" 수집 결과: {MODEL_NAME}")
    print(f"{'='*60}")
    print(f" 총 쿼리 수       : {total}")
    print(f" 성공률           : {success}/{total} ({success/total*100:.1f}%)")
    print(f" 평균 소요 시간   : {avg_time:.1f}초")
    print(f" 평균 스텝 수     : {avg_steps:.1f}")
    print(f" 평균 스텝당 시간 : {avg_step_time:.1f}초")
    if token_records:
        print(f" 총 사용 토큰     : {total_tokens:,}개")
        print(f" 쿼리당 평균 토큰 : {avg_tokens:,}개")
        print(f" 130개 예상 토큰  : {avg_tokens * 130:,}개")
    print(f" 학습 데이터 로그 : {DATA_DIR}/training_data_{MODEL_NAME}.jsonl")
    print(f"{'='*60}")


if __name__ == "__main__":
    start = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    end   = int(sys.argv[2]) if len(sys.argv) > 2 else None
    asyncio.run(run_batch(start, end))
