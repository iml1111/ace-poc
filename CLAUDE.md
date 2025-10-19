# CLAUDE.md

**Last Updated**: 2025-10-20

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**Agentic Context Engineering (ACE) Framework POC**

ACE 프레임워크는 LLM이 가중치 수정 없이 입력 컨텍스트(프롬프트, 메모리, 증거)를 **플레이북(playbook)**으로 관리하여 시간에 따라 성능을 자가 개선하는 방법론입니다.

### Core Concept: Triple-Agent Architecture

ACE는 컨텍스트를 전면 재작성하는 대신 **역할 분담을 통한 증분 업데이트**로 지식을 누적·정제합니다:

1. **Generator (생성자)**: 새로운 컨텍스트 후보를 생성
2. **Reflector (반성자)**: 생성된 컨텍스트를 평가하고 개선점 도출
3. **Curator (큐레이터)**: 델타 업데이트를 적용하여 플레이북 정제

이 접근법은 다음 문제를 해결합니다:
- **Brevity bias (요약 편향)**: 반복적 요약으로 세부 정보 손실
- **Context collapse (컨텍스트 붕괴)**: 도메인 특화 지식과 전략의 소실
- **Accuracy/Cost trade-off**: 정확도·속도·비용 동시 개선

### Playbook Evolution

플레이북은 정적 프롬프트가 아닌 **살아있는 지식 베이스**:
- 작업 수행 경험을 통해 도메인 지식 축적
- 델타 업데이트로 점진적 개선 (전면 재작성 회피)
- 증거 기반 의사결정으로 환각(hallucination) 최소화

## Development Environment

### Python Setup (pyenv)
- **Python Version**: 3.12.11 (pyenv로 관리)
- **가상환경**: `ace-poc` (pyenv-virtualenv)
- `.python-version` 파일로 자동 활성화
- 프로젝트 디렉토리 진입 시 자동으로 `ace-poc` 가상환경 활성화됨

### 환경 설정 명령어
```bash
# 가상환경 생성 (최초 1회)
pyenv virtualenv 3.12.11 ace-poc

# 가상환경 활성화 (자동 - .python-version 존재)
cd /path/to/ace-poc  # 자동 활성화

# 수동 활성화 (필요시)
pyenv activate ace-poc

# 의존성 설치 (requirements.txt 생성 후)
pip install -r requirements.txt
```

### 디렉토리 구조
- 소스 코드: `src/ace/` 또는 `ace/`
- 테스트: `tests/`
- 유틸리티 스크립트: `scripts/`
- 임시 실험 코드: `scripts/` (작업 후 정리 필수)

### 주요 의존성
**Core Dependencies** (requirements.txt):
- `anthropic>=0.18.0` - Claude API 클라이언트
- `pydantic>=2.0.0` - 데이터 검증 및 타입 안전성
- `orjson>=3.9.0` - 빠른 JSON 직렬화
- `python-dotenv>=1.0.0` - 환경 변수 관리
- `numpy>=1.24.0` - 수치 연산
- `tqdm>=4.65.0` - 진행 표시

**Optional Dependencies** (requirements-semantic.txt):
- `sentence-transformers>=2.2.0` - 의미 기반 중복 제거 (선택적)
  - 메모리: ~100MB
  - 활성화: `ACE_USE_SEMANTIC_DEDUP=true`

## Architecture Components

### 1. Agent Roles
- **Generator**: 컨텍스트 후보 생성 로직
- **Reflector**: 평가 및 피드백 생성
- **Curator**: 델타 패치 적용 및 플레이북 버전 관리

### 2. Playbook Management
- 플레이북 저장/로드 메커니즘
- 버전 관리 및 롤백 기능
- 델타 업데이트 적용 알고리즘

### 3. Evaluation Pipeline
- 정확도 측정 (도메인별 메트릭)
- 비용 추적 (토큰 사용량)
- 속도 벤치마크 (응답 시간)

### 4. Evidence Management
- 증거 수집 및 저장
- 컨텍스트 검색 (RAG 패턴)
- 관련성 스코어링

## Code Standards

### Import Organization
- 모든 `from`/`import` 구문은 파일 최상단에 배치
- 순서: 표준 라이브러리 → 서드파티 → 로컬 모듈

### Testing
- 테스트 타임아웃 설정 최대화 또는 미설정 (LLM 호출 고려)
- 단위 테스트: 각 에이전트 역할별 독립 테스트
- 통합 테스트: Generator → Reflector → Curator 파이프라인

### Temporary Scripts
- 모든 임시 테스트/실험 코드는 `scripts/`에 생성
- 실험 완료 후 스크립트 정리

## Design Principles

### Delta Updates Over Rewrites
- 플레이북을 전면 재작성하지 않고 증분 변경
- 변경 이력 추적으로 회귀 방지
- 작은 개선의 누적으로 안정성 확보

### Evidence-Based Context
- 모든 컨텍스트 업데이트는 증거 기반
- 추측이나 환각을 플레이북에 포함하지 않음
- 검증 가능한 사실만 누적

### Role Separation
- Generator는 생성만, 평가하지 않음
- Reflector는 평가만, 수정하지 않음
- Curator는 검증된 델타만 적용

### Iterative Refinement
- 일회성 최적화가 아닌 지속적 개선
- 각 작업마다 플레이북이 조금씩 성장
- 성능 저하 시 롤백 메커니즘

## Implementation Notes

### Playbook Format (구현 완료)
플레이북은 구조화된 JSON 파일 (storage/playbook.json):
- **PlaybookItem** 구조:
  - `item_id`: SHA-256 기반 결정론적 ID
  - `category`: strategy/formula/pitfall/checklist/example
  - `title`: 간결한 제목
  - `content`: 1-6문장의 실행 가능한 내용
  - `tags`: 분류 태그 리스트
  - `helpful_count` / `harmful_count`: 유용성 추적
  - `created_at` / `updated_at`: 타임스탬프

### Deduplication System (2025-10-17 구현)
중복 제거 메커니즘:
- **기본 모드** (difflib): 문자열 기반 유사도 계산
  - 빠르고 의존성 없음
  - Threshold: 0.92
- **의미 기반 모드** (sentence-transformers): 선택적 활성화
  - 의미적 유사도 계산 (cosine similarity)
  - 더 정확한 중복 감지
  - 환경 변수: `ACE_USE_SEMANTIC_DEDUP=true`
  - Graceful degradation: 의존성 없으면 자동 fallback

### Delta Update Structure (구현 완료)
델타는 명확한 변경 단위 (src/ace/models.py):
- **add**: 새로운 플레이북 아이템 추가
  - category: strategy/formula/pitfall/checklist/example
  - 자동 중복 제거 (similarity >= 0.92)
- **amend**: 기존 아이템 수정
  - content_append: 내용 추가
  - tags_add: 태그 추가
  - bullet_id로 대상 지정
- **deprecate**: 유해한 아이템 표시
  - harmful_count 증가
  - "deprecated" 태그 자동 추가
  - 임계값 초과 시 serving에서 제외

### Evaluation Loop
1. 작업 수행 (현재 플레이북 사용)
2. 성능 측정 (정확도, 비용, 속도)
3. Generator: 개선 후보 생성
4. Reflector: 후보 평가 및 피드백
5. Curator: 검증된 델타 적용
6. 플레이북 업데이트 및 다음 이터레이션

### 프롬프트 진화 메커니즘 (Context Engineering)

**핵심 원리**: "Context Engineering, Not Prompt Engineering"

ACE에서 프롬프트는 두 부분으로 구성:
1. **System Prompt (고정)**: src/ace/prompts.py에 정의, 영구 불변
2. **User Prompt (동적)**: playbook + reflection + question으로 구성

**진화 대상**: User Prompt의 Playbook 부분만 진화

**예시** (src/ace/prompts.py):

```python
# System Prompt - 영원히 고정
GENERATOR_SYSTEM_PROMPT = """You are an analysis expert tasked with solving tasks using:
1. Your general knowledge
2. A curated Playbook of strategies, formulas, pitfalls, checklists, and examples
3. An optional Reflection summarizing prior mistakes and fixes
..."""

# User Prompt - Playbook 부분이 진화
def create_generator_user_prompt(
    playbook_items: list,  # 이 부분이 0개 → 20개로 성장
    reflection: Dict[str, Any] | None,
    question: Dict[str, Any]
) -> str:
    return json.dumps({
        "playbook": {"items": playbook_items},  # 진화 대상
        "reflection": reflection,
        "question": question
    })
```

**델타 업데이트 예시**:

```python
# add: 새 전략 추가
{
    "operation": "add",
    "item_id": "321fe6e22562",
    "category": "strategy",
    "title": "Test-First Span Extraction",
    "content": "Before calculating positions, first identify exact target strings...",
    "tags": ["span_extraction", "validation", "test_first"]
}

# amend: 기존 항목 개선
{
    "operation": "amend",
    "bullet_id": "b7b314eff6e4",
    "content_append": "9. Check annotation guidelines for entity-specific boundary rules",
    "tags_add": ["guidelines", "entity_types"]
}

# deprecate: 유해한 전략 표시
{
    "operation": "deprecate",
    "bullet_id": "f959d4fd77dd",
    "reason": "Manual character counting is error-prone, use string.find() instead"
}
```

**결과**:
- System prompt는 교과서처럼 영구 고정
- Playbook은 참고 노트처럼 계속 성장
- 이전 지식 보존 (덮어쓰기 없음)
- 버전 관리 및 롤백 가능

## Research Context

이 POC는 ACE 논문의 핵심 가설을 검증:
- **가설 1**: 델타 업데이트가 전면 재작성보다 정보 보존에 효과적
- **가설 2**: 역할 분담이 단일 에이전트보다 품질 높은 컨텍스트 생성
- **가설 3**: 플레이북 진화가 few-shot 프롬프팅보다 장기적으로 우수

실험 설계 시 다음을 측정:
- Brevity bias 발생 빈도
- Context collapse 사례
- 누적 성능 개선 곡선
- 도메인 특화 지식 보존율

## Experimental Results

### NER 진화 실험 (2025-10-19)

**실험 설정**:
- 작업: Named Entity Recognition (개체명 인식)
- 데이터셋: CoNLL-2003 기반 토이 샘플 (5개)
- 목표: 빈 플레이북에서 시작하여 자동 진화 관찰
- 실행 방식: `PYTHONPATH=src python -m ace offline --task ner --epochs 5`

**진화 타임라인**:
- **Epoch 1**: 0 → 8 items (기초 전략 발견)
  - Validate Span Extraction, Character Position Counting Rules
  - 첫 실패로부터 검증 체크리스트 생성
- **Epoch 2**: 8 → 14 items (고급 전략 추가)
  - Token Boundary Consistency, Test-First Approach
  - 경계 처리 문제 해결을 위한 전략 축적
- **Epoch 3**: 14 → 20 items (Early Stopping)
  - Entity-Specific Boundary Rules, Multiple Entity Handling
  - 수렴 감지로 조기 종료

**프롬프트 크기 변화**:
- 초기 프롬프트: ~40 토큰 (빈 플레이북)
- 최종 프롬프트: ~900 토큰 (20개 항목)
- 성장률: 22.5배 증가

**주요 발견**:
1. **창발적 전략**: string.find() 사용, test-first 접근법이 자연스럽게 발견됨
2. **자기 교정**: 이전 실수를 반성하여 체크리스트와 pitfall 항목 자동 생성
3. **지식 축적**: deprecated 태그로 유해한 전략 표시, 유용한 전략만 보존
4. **카테고리 균형**: strategy 7개, formula 8개, checklist 1개, pitfall 3개, example 1개

**아카이브 위치**: `experiments/ner-evolution-20251019_154015/`
- `prompts_comparison.md`: 초기/최종 프롬프트 상세 비교
- `initial_prompt_example.txt`: 빈 플레이북 프롬프트
- `final_prompt_example.txt`: 20개 항목 최종 프롬프트
- `final_playbook.json`: 진화된 플레이북
- `offline_run_final.log`: 전체 실행 로그
- `evolution_summary.txt`: Epoch별 요약

**실험 보고서**: `report.md` (한글 종합 분석)
