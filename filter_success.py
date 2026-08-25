"""
ft_train.jsonl에서 성공한 에피소드(done action으로 끝나는 것)만 필터링하여
ft_train_success.jsonl로 저장합니다.

성공 기준: 에피소드의 마지막 스텝 assistant의 action에 'done' 키가 포함된 경우
"""

import json
import re
from pathlib import Path

INPUT_PATH = "./data/ft_train.jsonl"
OUTPUT_PATH = "./data/ft_train_success.jsonl"


def get_step_num(sample: dict) -> int | None:
    for msg in sample["messages"]:
        if msg["role"] == "user":
            match = re.search(r"Step(\d+)", msg["content"])
            if match:
                return int(match.group(1))
    return None


def is_done_action(sample: dict) -> bool:
    for msg in sample["messages"]:
        if msg["role"] == "assistant":
            try:
                parsed = json.loads(msg["content"])
                action = parsed.get("action", [])
                if isinstance(action, list):
                    return any("done" in a for a in action)
            except Exception:
                pass
    return False


def split_into_episodes(lines: list[str]) -> list[list[dict]]:
    episodes = []
    current: list[dict] = []
    for line in lines:
        sample = json.loads(line)
        if get_step_num(sample) == 1 and current:
            episodes.append(current)
            current = []
        current.append(sample)
    if current:
        episodes.append(current)
    return episodes


def main():
    with open(INPUT_PATH, encoding="utf-8") as f:
        lines = [l for l in f if l.strip()]

    episodes = split_into_episodes(lines)
    print(f"전체 에피소드: {len(episodes)}개 / 전체 스텝: {sum(len(e) for e in episodes)}개")

    success_episodes = [ep for ep in episodes if is_done_action(ep[-1])]
    fail_episodes = [ep for ep in episodes if not is_done_action(ep[-1])]
    print(f"성공 에피소드: {len(success_episodes)}개 / 스텝: {sum(len(e) for e in success_episodes)}개")
    print(f"실패 에피소드: {len(fail_episodes)}개 / 스텝: {sum(len(e) for e in fail_episodes)}개")

    Path(OUTPUT_PATH).parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        for ep in success_episodes:
            for sample in ep:
                f.write(json.dumps(sample, ensure_ascii=False) + "\n")

    print(f"\n저장 완료: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
