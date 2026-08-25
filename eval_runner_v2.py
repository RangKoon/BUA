"""
BUA 테스트셋(50개) 평가 스크립트
사용:
  python eval_runner.py gemini-2.5-flash          <- with 도메인 (기본값)
  python eval_runner.py qwen-base with
  python eval_runner.py qwen-base home
  python eval_runner.py qwen-base home 0 25       <- 범위 지정
  python eval_runner.py qwen-ft
"""

import asyncio
import json
import os
import sys
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from browser_use import BrowserSession, BrowserProfile, ChatOpenAI
from browser_use.llm.google.chat import ChatGoogle
from browser_use.llm.anthropic.chat import ChatAnthropic
from logged_agent import LoggedAgent
import logging
logging.getLogger("browser_use").setLevel(logging.ERROR)
# 스크린샷 타임아웃 에러만 추가로 억제
logging.getLogger("browser_use.browser.watchdogs").setLevel(logging.CRITICAL)
logging.getLogger("browser_use.browser.session").setLevel(logging.CRITICAL)

load_dotenv()

# ── 모델 설정 ────────────────────────────────────────────────
MODEL_CONFIGS = {
    "gemini-2.5-flash-20test_v3": {
        "llm_factory": lambda: ChatGoogle(model="gemini-2.5-flash"),
    },
    "qwen-base_v5": {
        "llm_factory": lambda: ChatOpenAI(
            model="Qwen/Qwen2.5-32B-Instruct",
            base_url=os.getenv("QWEN_BASE_URL", "http://localhost:8000/v1"),
            api_key="EMPTY",
        ),
    },
    "qwen-ft_v5": {
        "llm_factory": lambda: ChatOpenAI(
            model=os.getenv("QWEN_FT_MODEL_NAME", "/NHNHOME/WORKSPACE/26moe002_H/mhjin/BUA/output/qwen2.5-32b-cnu-merged"),
            base_url=os.getenv("QWEN_FT_BASE_URL", "http://localhost:8001/v1"),
            api_key="EMPTY",
        ),
    },
    "claude-sonnet": {
        "llm_factory": lambda: ChatAnthropic(
            model="claude-sonnet-4-5",
            api_key=os.getenv("ANTHROPIC_API_KEY"),
        ),
    },
    "qwen-vl-base_v5": {
        "llm_factory": lambda: ChatOpenAI(
            model="/NHNHOME/WORKSPACE/26moe002_H/mwchoi/hf_cache/hub/models--Qwen--Qwen2.5-VL-32B-Instruct/snapshots/7cfb30d71a1f4f49a57592323337a4a4727301da",
            base_url=os.getenv("QWEN_VL_BASE_URL", "http://localhost:8002/v1"),
            api_key="EMPTY",
        ),
    },
    "qwen-vl-ft_v5": {
        "llm_factory": lambda: ChatOpenAI(
            model="qwen-vl-ft",
            base_url=os.getenv("QWEN_VL_FT_BASE_URL", "http://localhost:8002/v1"),
            api_key="EMPTY",
        ),
    },
}

# ── 도메인 설정 ──────────────────────────────────────────────
WITH_URL = "https://with.cnu.ac.kr/non/index.do"
HOME_URL = "https://computer.cnu.ac.kr/computer/index.do"

DOMAIN_CONFIGS = {
    "with": {
        "start_url": WITH_URL,
        "query_var": "WITH_QUERIES",
        "query_module": "with_20_queries",
    },
    "home": {
        "start_url": HOME_URL,
        "query_var": "HOMEPAGE_QUERIES",
        "query_module": "homepage_20_queries",
    },
    "home_50": {
        "start_url": HOME_URL,
        "query_var": "COMPUTER_QUERIES",
        "query_module": "test_queries_home",
    },
}


# ── extend_msg (도메인별 시스템 프롬프트) ────────────────────
EXTEND_MSG_WITH = """
모든 사고 과정과 응답은 반드시 한국어로만 작성하라.
같은 액션이 연속으로 실패하면 즉시 전략을 바꿔라. 루프에 빠지지 마라.
CSS selector에 :contains()를 절대 사용하지 마라.
에러 발생 후 같은 방법을 2회 이상 재시도하지 마라.
로그인 시 아이디→비밀번호→버튼 순서를 반드시 지켜라.

[로그인 직후 팝업 처리 - 최우선]
- 로그인 후 공지 팝업(jQuery UI dialog)이 자동으로 나타남
- 팝업이 뜨면 반드시 먼저 처리한 후 다른 작업 진행
- 팝업 닫기 버튼 우선순위:
  1) '오늘하루열지않기' 버튼 클릭 (재방문 시 방해 방지)
  2) 없으면 '닫기' 버튼 클릭
- 팝업을 닫지 않으면 뒤에 있는 요소가 클릭되지 않을 수 있음

[메뉴 탐색 규칙]
- 상단 GNB 메뉴는 hover 시 서브메뉴가 나타나며 해당 li에 class="on"이 추가됨
- 서브메뉴 클릭 절차:
  1) 부모 메뉴(예: '비교과')에 마우스를 올려 class="on"이 추가될 때까지 대기
  2) 서브메뉴가 나타나면 원하는 항목 클릭
  3) hover 없이 바로 클릭하면 서브메뉴가 열리지 않음
- 일부 메뉴는 3단계 구조(예: 비교과 > 그룹비교과 > 모집공고/신청)이므로
  각 단계마다 hover 후 클릭 필요
- 모든 페이지 이동은 /menu/redirect.do?url= 패턴으로 이루어짐

[페이지 로딩 대응]
- 대시보드 위젯들은 AJAX로 비동기 로드됨 (addDiv() 함수 사용)
- 페이지 진입 후 'LOADING...' 표시가 사라질 때까지 대기
- 데이터가 보이지 않으면 3초 대기 후 재확인
- '조회된 데이터가 없습니다.' 텍스트가 보이면 해당 섹션은 데이터 없음으로 판단

[목록/카드 탐색 규칙]
- 비교과 프로그램, e클래스 과목 등은 Swiper 캐러셀로 표시됨
- 캐러셀 내 더 많은 항목을 보려면 우측 화살표(swiper-button-next) 클릭
- swiper-button-disabled 클래스가 있으면 더 이상 항목 없음
- 전체 목록이 필요하면 각 섹션의 'MORE' 버튼 클릭하여 목록 페이지로 이동

[상세 페이지 이동 규칙]
- 비교과 프로그램 카드는 class="detailBtn"이며 data-params 속성에 파라미터 보유
- detailBtn 클릭 시 내부적으로 $.location() 함수로 이동 → 일반 클릭으로 처리
- 클릭이 실패하면 data-params의 encSddpbSeq 값을 확인하여
  /ptfol/imng/icmpNsbjtPgm/findIcmpNsbjtPgmInfo.do 로 직접 파라미터 전달

[목록 페이지 데이터 조회 규칙]
- 총 건수(예: '총 44건')가 표시되면 페이지를 넘기며 전체 탐색
- 한 페이지 기본 10개 표시
- 페이지 이동: 목록 하단 페이지 번호 클릭
- 검색 기능 있으면 키워드 입력 후 검색 버튼 클릭

[세션 관리]
- 세션 유효시간 60분, 우측 상단에 남은 시간 표시
- 작업 중 세션 만료 경고 팝업이 뜨면 '세션 시간을 연장하시겠습니까?' 확인 버튼 클릭
- 세션 만료 전 '시간연장' 버튼 클릭하여 세션 유지 가능

[에러 대응 전략 - 반드시 준수]
- 같은 액션이 2회 연속 실패하면 즉시 완전히 다른 방법을 시도하라. 절대 동일한 실패를 반복하지 마라.
- Navigation timeout 발생 시: 뒤로 가거나 메뉴를 직접 클릭하여 다른 경로로 접근하라.
- element를 찾지 못할 때(index -1): 페이지를 위아래로 스크롤한 후 재시도하거나, 텍스트로 요소를 찾아라.
- 클릭이 실패할 때: 요소가 화면 밖에 있을 수 있으니 스크롤 후 재시도하라.
- 3회 이상 같은 방식으로 실패하면 해당 접근 방법을 포기하고 완전히 다른 경로를 선택하라.

[기술적 주의사항]
- CSS selector에서 :contains() 같은 jQuery 전용 선택자는 절대 사용하지 마라 (표준 CSS 미지원, 오류 발생).
- JavaScript로 hidden/detached 요소에 접근하지 마라.
- URL은 추측하거나 생성하지 말 것 (현재 페이지의 링크와 메뉴만 사용).
- 결과를 명확하게 정리하여 반환하라.
"""

EXTEND_MSG_HOME = f"""
모든 사고 과정과 응답은 반드시 한국어로만 작성하라.
같은 액션이 연속으로 실패하면 즉시 전략을 바꿔라. 루프에 빠지지 마라.
CSS selector에 :contains()를 절대 사용하지 마라.
에러 발생 후 같은 방법을 2회 이상 재시도하지 마라.

[페이지 로딩 문제 대응 - 최우선]
- 이 사이트는 JavaScript 렌더링 기반으로, 페이지 로드 후 콘텐츠가 비어있는 경우가 자주 발생함
- 빈 페이지 판단 기준: 본문 영역에 텍스트나 링크가 전혀 없는 상태
- 빈 페이지 대응 순서:
  1) 즉시 재시도하지 말고 3초 대기
  2) 3초 후에도 비어있으면, 현재 URL의 href 링크를 직접 탐색하여 목적 페이지로 이동
  3) 메인 페이지(index.do)가 비어있는 경우 메인을 거치지 말고 목적 페이지로 바로 이동
  4) 동일 페이지에서 3회 이상 빈 화면이 반복되면 다른 경로로 우회

[메뉴 탐색 규칙]
- 상단 GNB 메뉴(학부소개/교육/대학원/교육대학원/공지사항/커뮤니티/자료실/수행사업)는 hover 기반으로 작동함
- 서브메뉴 클릭 절차:
  1) 부모 메뉴(예: '공지사항')에 마우스를 올려 서브메뉴가 나타날 때까지 대기
  2) 서브메뉴가 화면에 나타난 것을 확인한 후 원하는 항목 클릭
  3) hover 없이 바로 클릭 시도하면 서브메뉴가 열리지 않아 실패함
- hover 인터랙션이 불안정하면 메뉴 링크의 href를 직접 URL로 접근하는 것이 더 안정적

[게시판 목록 탐색 규칙]
- 게시판 목록 진입 시 반드시 총 게시글 수(예: '총 44건') 확인
- 한 페이지에 기본 10개 항목이 표시됨
- 총 게시글 수가 10개 초과이면 반드시 페이지를 넘기며 전체 탐색
- 페이지 이동 방법:
  1) 목록 하단의 페이지 번호 링크를 직접 클릭
  2) 클릭 후 새 목록이 로드될 때까지 대기
- 화면에 보이는 항목만으로 답하는 것은 오답 가능성이 높으므로 금지
- 관련 공지가 최신순으로 정렬되어 있으므로 1페이지부터 순서대로 탐색

[게시글 상세 보기 규칙]
- 목록에서 게시글 제목 링크를 클릭하여 상세 페이지로 이동
- 상세 페이지에서 제목, 작성자, 등록일, 조회수, 본문 내용을 모두 확인
- 본문이 길면 스크롤하여 전체 내용 확인
- 첨부파일이 있어도 다운로드하지 말고 텍스트 내용만 추출

[검색 기능 사용 규칙]
- 특정 키워드 관련 공지 탐색 시 게시판 내 검색창 활용
- 검색 절차:
  1) 검색창 클릭 후 키워드 입력
  2) '검색' 버튼 클릭 또는 Enter
  3) 검색 결과 로드 대기
  4) 결과가 없으면 유사 키워드로 재시도

[요소 클릭 실패 대응]
- JavaScript($) 방식으로 요소를 클릭할 때 인덱스 -1 오류가 자주 발생함
- 오류 발생 시 대응 순서:
  1) JavaScript 재시도 금지
  2) 해당 요소의 href 속성 확인
  3) href를 직접 URL로 입력하여 이동
- jQuery($) 사용 자제: 이 사이트에서 요소 탐색 시 오작동 가능성 높음
- 요소가 화면에 보이지만 클릭이 안 될 경우 인접한 다른 클릭 가능 요소 시도

[JavaScript 실행 오류 대응]
- 'Object reference chain is too long' 오류 발생 시:
  1) JavaScript 실행을 즉시 중단
  2) 직접 URL 이동 또는 일반 클릭 방식으로 전환
- JavaScript 실행은 일반 클릭이나 URL 이동으로 해결 불가한 경우에만 최후 수단으로 사용

[정보 추출 규칙]
- 찾는 정보가 현재 화면에 보이면 즉시 done 액션 호출 (추가 탐색 금지)
- 테이블 형태의 정보는 행/열을 모두 읽어 빠짐없이 추출
- 스크롤은 현재 화면 아래에 더 많은 콘텐츠가 있는 것이 확실한 경우에만 사용
- 스크롤을 반복해도 새 콘텐츠가 나타나지 않으면 즉시 중단
- 페이지에 해당 정보가 없으면 다른 메뉴/게시판으로 이동하여 탐색 시도

[탐색 전략 우선순위]
1. 목적에 맞는 페이지로 직접 URL 이동
2. 메인 페이지 메뉴 hover → 서브메뉴 클릭
3. 게시판 검색창 활용
4. 페이지 내 링크 탐색

[에러 대응 전략 - 반드시 준수]
- 같은 액션이 2회 연속 실패하면 즉시 완전히 다른 방법을 시도하라. 절대 동일한 실패를 반복하지 마라.
- element를 찾지 못할 때(index -1): 페이지 최상단으로 스크롤 후 재시도하거나, hover 방식으로 메뉴를 다시 열어라.
- Navigation timeout 발생 시: 홈({HOME_URL})으로 직접 이동한 뒤 hover 방식으로 재탐색하라.
- Connection error 발생 시: 같은 URL로 navigate를 재시도하라. 3회 이상 실패 시 작업을 종료하라.
- 3회 이상 같은 방식으로 실패하면 해당 접근 방법을 포기하고 완전히 다른 메뉴 경로를 선택하라.

[기술적 주의사항]
- CSS selector에서 :contains() 같은 jQuery 전용 선택자는 절대 사용하지 마라 (표준 CSS 미지원, 오류 발생).
- JavaScript로 hidden/detached 요소에 접근하지 마라.
- URL은 추측하거나 생성하지 말 것 (현재 페이지의 링크와 메뉴만 사용).
- 결과를 명확하게 정리하여 반환하라.
"""

EXTEND_MSGS = {
    "with": EXTEND_MSG_WITH,
    "home": EXTEND_MSG_HOME,
    "home_50": EXTEND_MSG_HOME,
}


# ── 도메인별 task 프롬프트 ───────────────────────────────────
def build_task_with(user_query: str) -> str:
    return f"""
충남대학교 학사지원시스템 {WITH_URL} 에서 다음 작업을 수행해줘:
{user_query}

[로그인 절차 - 로그인이 필요한 경우 반드시 이 순서를 따라라]
1. 아이디 입력란을 클릭하고 x_username을 입력한다.
2. 비밀번호 입력란을 클릭하고 x_password를 입력한다.
3. 로그인 버튼을 클릭한다.
4. 로그인 후 페이지가 완전히 로딩될 때까지 기다린다.
5. 로그인 성공 후 목표 작업 페이지로 직접 메뉴를 탐색한다.
- 로그인 버튼을 찾지 못하면 페이지를 스크롤하거나 iframe 내부를 확인하라.
- 로그인 후 목표 페이지로 자동 이동되지 않으면 좌측 메뉴나 상단 메뉴에서 직접 탐색하라.
"""


def build_task_home(user_query: str) -> str:
    return f"""
충남대학교 컴퓨터융합학부 홈페이지 {HOME_URL} 에서 다음 정보를 찾아줘:
{user_query}
"""


TASK_BUILDERS = {
    "with": build_task_with,
    "home": build_task_home,
    "home_50": build_task_home,
}


# ── 브라우저 설정 ────────────────────────────────────────────
def build_browser(domain: str) -> BrowserSession:
    return BrowserSession(browser_profile=BrowserProfile(
        headless=True,                            # ← 변경
        viewport={"width": 1280, "height": 720}, # ← 추가
        args=[
            "--disable-dev-shm-usage",             # ← 추가
            "--no-sandbox",                        # ← 추가
            "--disable-gpu",                  
            "--use-gl=swiftshader",        # 소프트웨어 GL 렌더러
            "--window-size=1280,720",      
        ],
        wait_for_network_idle_page_load_time=0,
        minimum_wait_page_load_time=0,             # ← 0.5 → 0
        cross_origin_iframes=True,
    ))


# ── 쿼리 실행 ────────────────────────────────────────────────
async def run_query(user_query: str, llm, log_path: str, domain: str) -> tuple[str | None, dict]:
    task = TASK_BUILDERS[domain](user_query)
    start_url = DOMAIN_CONFIGS[domain]["start_url"]

    sensitive_data = {}
    if domain == "with":
        sensitive_data["with.cnu.ac.kr"] = {
            "x_username": os.getenv("CNU_ID"),
            "x_password": os.getenv("CNU_PASSWORD"),
        }

    extend_msg = EXTEND_MSGS[domain]

    browser = build_browser(domain)
    agent = LoggedAgent(
        task=task,
        llm=llm,
        log_path=log_path,
        browser=browser,
        sensitive_data=sensitive_data,
        use_vision=True,
        max_failures=3,
        step_timeout=30,
        extend_system_message=extend_msg,
        initial_actions=[{"navigate": {"url": start_url}}],
    )
    history = await agent.run(max_steps=20)
    metrics = {
        "steps": history.number_of_steps(),
        "duration_seconds": round(history.total_duration_seconds(), 2),
        "success": history.is_successful() or False,
        "errors": [e for e in history.errors() if e is not None],
    }
    return history.final_result(), metrics


# ── 평가 실행 ────────────────────────────────────────────────
async def run_eval(model_key: str, domain: str = "with", start: int = 0, end: int | None = None):
    cfg = DOMAIN_CONFIGS[domain]
    module = __import__(cfg["query_module"])
    queries = getattr(module, cfg["query_var"])

    llm = MODEL_CONFIGS[model_key]["llm_factory"]()
    safe = model_key.replace(":", "-").replace("/", "-")
    Path("./data/eval_results").mkdir(parents=True, exist_ok=True)
    log_path      = f"./data/eval_logs_{safe}_{domain}.jsonl"
    progress_path = Path(f"./data/eval_results/{safe}_{domain}_progress.json")

    targets = queries[start:end]
    total = len(targets)
    print(f"\n모델: {model_key} | 도메인: {domain} | 총 {total}개 쿼리 평가 시작\n")

    all_results = []
    if progress_path.exists():
        with open(progress_path, encoding="utf-8") as f:
            all_results = json.load(f)
        print(f"기존 결과 {len(all_results)}개 로드, 이어서 실행\n")

    done_indices = {r["index"] for r in all_results}

    for i, query in enumerate(targets):
        global_idx = start + i
        if global_idx in done_indices:
            print(f"[{i+1}/{total}] SKIP: {query[:40]}")
            continue

        print(f"[{i+1}/{total}] {query}")
        try:
            result, metrics = await run_query(query, llm, log_path, domain)
        except Exception as e:
            metrics = {"steps": 0, "duration_seconds": 0, "success": False, "errors": [str(e)]}
            result = None

        step_avg = round(metrics["duration_seconds"] / metrics["steps"], 2) if metrics["steps"] else 0
        record = {
            "timestamp": datetime.now().isoformat(),
            "model": model_key,
            "domain": domain,
            "index": global_idx,
            "query": query,
            "success": metrics["success"],
            "steps": metrics["steps"],
            "duration_seconds": metrics["duration_seconds"],
            "step_duration_avg": step_avg,
            "errors": metrics["errors"],
            "result": result,
        }
        all_results.append(record)

        status = "✅" if metrics["success"] else "❌"
        print(f"  {status} {metrics['duration_seconds']}초 | {metrics['steps']}스텝 | 스텝당 {step_avg}초")

        if (i + 1) % 10 == 0:
            _save(all_results, safe, domain, "progress")

    _save(all_results, safe, domain, "final")
    _print_summary(all_results, model_key, domain)


def _save(records: list[dict], safe: str, domain: str, suffix: str):
    path = Path(f"./data/eval_results/{safe}_{domain}_{suffix}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)


def _print_summary(records: list[dict], model_key: str, domain: str):
    total = len(records)
    if not total:
        return
    success = sum(1 for r in records if r["success"])
    avg_time = sum(r["duration_seconds"] for r in records) / total
    avg_steps = sum(r["steps"] for r in records) / total
    print(f"\n{'='*50}\n 평가 결과: {model_key} [{domain}]\n{'='*50}")
    print(f" 총 쿼리  : {total}")
    print(f" 성공률   : {success}/{total} ({success/total*100:.1f}%)")
    print(f" 평균 시간: {avg_time:.1f}초")
    print(f" 평균 스텝: {avg_steps:.1f}")
    print("=" * 50)


if __name__ == "__main__":
    valid_models = list(MODEL_CONFIGS.keys())
    valid_domains = list(DOMAIN_CONFIGS.keys())

    if len(sys.argv) < 2:
        print(f"사용법: python eval_runner_v2.py <model>[,<model2>] [domain] [start] [end]")
        print(f"  model  옵션: {valid_models}")
        print(f"  domain 옵션: {valid_domains} (기본값: with)")
        print(f"  예시: python eval_runner_v2.py qwen-vl-ft_v3,qwen-vl-base home_50")
        sys.exit(1)

    # 콤마로 여러 모델 지정 가능 (예: qwen-vl-ft_v3,qwen-vl-base)
    models = sys.argv[1].split(",")
    for m in models:
        if m not in MODEL_CONFIGS:
            print(f"알 수 없는 모델: {m}  (옵션: {valid_models})")
            sys.exit(1)

    domain = "with"
    arg_offset = 2
    if len(sys.argv) > 2 and sys.argv[2] in DOMAIN_CONFIGS:
        domain = sys.argv[2]
        arg_offset = 3

    start = int(sys.argv[arg_offset])     if len(sys.argv) > arg_offset     else 0
    end   = int(sys.argv[arg_offset + 1]) if len(sys.argv) > arg_offset + 1 else None

    for model in models:
        print(f"\n{'='*50}\n 시작: {model} [{domain}]\n{'='*50}")
        asyncio.run(run_eval(model, domain, start, end))

# if __name__ == "__main__":
#     if len(sys.argv) < 2 or sys.argv[1] not in MODEL_CONFIGS:
#         print(f"사용법: python eval_runner.py <model> [domain] [start] [end]")
#         print(f"  model  옵션: {list(MODEL_CONFIGS.keys())}")
#         print(f"  domain 옵션: with (기본값), home")
#         sys.exit(1)

#     model = sys.argv[1]

#     # 두 번째 인자가 도메인인지 숫자(start)인지 판별
#     domain = "with"
#     arg_offset = 2
#     if len(sys.argv) > 2 and sys.argv[2] in DOMAIN_CONFIGS:
#         domain = sys.argv[2]
#         arg_offset = 3

#     start = int(sys.argv[arg_offset])     if len(sys.argv) > arg_offset     else 0
#     end   = int(sys.argv[arg_offset + 1]) if len(sys.argv) > arg_offset + 1 else None

#     asyncio.run(run_eval(model, domain, start, end))
