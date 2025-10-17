# ACE 프레임워크 사용 가이드

**Agentic Context Engineering (ACE)** 프레임워크를 사용하여 자신의 데이터로 LLM 성능을 개선하는 완전한 가이드입니다.

## 목차

1. [개요](#개요)
2. [환경 설정](#환경-설정)
3. [데이터셋 준비](#데이터셋-준비)
4. [실행 가이드](#실행-가이드)
5. [성능 비교](#성능-비교)
6. [실전 예제](#실전-예제)
7. [문제 해결](#문제-해결)

---

## 개요

### ACE가 하는 일

ACE는 LLM이 **맥락(context)을 학습**하여 성능을 개선하는 프레임워크입니다:

```
초기 상태 (빈 플레이북)
  ↓
학습 (오프라인 적응)
  ↓
진화된 플레이북 (전략, 공식, 함정 등)
  ↓
개선된 성능 (온라인 추론)
```

### 기대 효과

- **정확도 향상**: 33% → 100% (테스트 결과 기준)
- **도메인 지식 축적**: 작업별 전략과 패턴 학습
- **점진적 개선**: 데이터가 추가될수록 계속 발전

---

## 환경 설정

### 1. Python 환경 (pyenv)

```bash
# 현재 설정된 Python 버전 확인
cat .python-version
# 출력: ace-poc (Python 3.12.11 기반)

# 환경 활성화 (자동 활성화되지 않은 경우)
pyenv activate ace-poc
```

### 2. 의존성 설치

```bash
pip install -r requirements.txt
```

**requirements.txt 내용**:
```
pydantic>=2.0.0
orjson>=3.9.0
python-dotenv>=1.0.0
anthropic>=0.18.0
httpx>=0.24.0
numpy>=1.24.0
tqdm>=4.65.0
pytest>=7.4.0
```

### 3. API 키 설정

```bash
# .env 파일 생성
cp .env.example .env

# .env 파일 편집
# ANTHROPIC_API_KEY=your_key_here
```

**.env 파일 예제**:
```bash
# API Configuration
ANTHROPIC_API_KEY=sk-ant-api03-xxxx
ACE_MODEL=claude-3-5-sonnet-latest
ACE_MAX_TOKENS=2048
ACE_TEMPERATURE=0.0
ACE_SEED=42

# Storage
ACE_STORAGE_DIR=./storage
ACE_RUNS_DIR=./runs

# Playbook Tuning
ACE_HARMFUL_THRESHOLD=3
ACE_DEDUP_SIMILARITY=0.92
ACE_MAX_OPERATIONS_PER_CURATOR=20

# Semantic Deduplication (선택사항)
ACE_USE_SEMANTIC_DEDUP=false
ACE_EMBEDDING_MODEL=all-MiniLM-L6-v2
```

#### 고급 기능: 의미 기반 중복 제거 (선택사항)

ACE는 선택적으로 **의미 기반 임베딩**을 사용한 향상된 중복 제거를 지원합니다.

**기본 모드 vs 의미 기반 모드**:
- **기본 모드** (difflib): 문자열 기반 유사도 비교 (빠름, 추가 설치 불필요)
- **의미 기반 모드** (sentence-transformers): 의미 기반 유사도 비교 (더 정확, ~100MB 메모리 사용)

**의미 기반 모드의 장점**:
```
"check authentication" ≈ "verify auth"
→ 기본 모드: 0.35 유사도 (중복 감지 안됨)
→ 의미 모드: 0.94 유사도 (중복 감지됨)
```

**설치 방법** (선택사항):
```bash
# 1. 의미 기반 의존성 설치
pip install -r requirements-semantic.txt

# 2. .env 파일에서 활성화
ACE_USE_SEMANTIC_DEDUP=true
```

**참고**:
- 메모리 사용량: ~100MB 추가
- 설치하지 않으면 자동으로 기본 모드(difflib) 사용
- 대부분의 경우 기본 모드로도 충분함

### 4. 설치 확인

```bash
# CLI 동작 확인
python -m ace list-datasets

# 출력 예시:
# ============================================================
# AVAILABLE DATASETS
# ============================================================
#
# labeling:
#   Description: Named entity recognition (FiNER-like)
#   Train: 3 samples
#   Test:  2 samples
#   Total: 5 samples
```

---

## 데이터셋 준비

### 데이터 형식 이해

ACE 데이터셋은 **question**과 **ground_truth** 쌍으로 구성됩니다:

```python
{
    "question": {
        # 작업 정의 및 입력 데이터
    },
    "ground_truth": {
        # 정답 또는 기대 출력
    }
}
```

### 지원하는 태스크 타입

#### 1. LABELING (개체명 인식)

**용도**: 텍스트에서 특정 개체(조직, 돈, 날짜 등)를 추출하고 라벨링

**데이터 형식**:
```python
{
    "question": {
        "task": "label_spans",
        "text": "Microsoft acquired LinkedIn for $26.2B in June 2016.",
        "labels": ["ORG", "MONEY", "DATE"]
    },
    "ground_truth": {
        "spans": [
            {"text": "Microsoft", "label": "ORG", "start": 0, "end": 9},
            {"text": "LinkedIn", "label": "ORG", "start": 19, "end": 27},
            {"text": "$26.2B", "label": "MONEY", "start": 32, "end": 38},
            {"text": "June 2016", "label": "DATE", "start": 42, "end": 51}
        ]
    }
}
```

#### 2. NUMERIC (수식 계산)

**용도**: 금융 공식, 통계 계산 등 수치 연산

**데이터 형식**:
```python
{
    "question": {
        "task": "finance_compute",
        "formula": "simple_interest",
        "inputs": {"principal": 1000, "rate_pct": 5, "years": 2}
    },
    "ground_truth": {
        "answer": 100.0,
        "formula": "principal * (rate_pct / 100) * years"
    }
}
```

#### 3. CODE_AGENT (리스트 연산)

**용도**: 데이터 집계, 변환 등 프로그래밍 작업

**데이터 형식**:
```python
{
    "question": {
        "task": "list_aggregate",
        "input": [3, 7, 7, 10],
        "op": "mode"
    },
    "ground_truth": {
        "answer": 7,
        "explanation": "Most frequent element"
    }
}
```

### 커스텀 데이터셋 추가하기

#### Step 1: 데이터 정의

`src/ace/datasets.py` 파일을 열고 새로운 데이터셋을 추가합니다.

**예제: 고객 지원 티켓 분류**

```python
# ============================================================================
# Dataset 4: TICKET_CLASSIFICATION (Customer Support)
# ============================================================================

TICKET_TRAIN = [
    {
        "question": {
            "task": "classify_ticket",
            "text": "I can't log in to my account. Keep getting 'invalid password' error.",
            "categories": ["technical", "billing", "account", "feature_request"]
        },
        "ground_truth": {
            "category": "account",
            "priority": "high",
            "reasoning": "Login issues require immediate attention"
        }
    },
    {
        "question": {
            "task": "classify_ticket",
            "text": "How do I export my data to CSV format?",
            "categories": ["technical", "billing", "account", "feature_request"]
        },
        "ground_truth": {
            "category": "feature_request",
            "priority": "low",
            "reasoning": "User asking about existing feature usage"
        }
    },
    {
        "question": {
            "task": "classify_ticket",
            "text": "I was charged twice this month. Please refund the duplicate charge.",
            "categories": ["technical", "billing", "account", "feature_request"]
        },
        "ground_truth": {
            "category": "billing",
            "priority": "high",
            "reasoning": "Billing issues require immediate resolution"
        }
    },
]

TICKET_TEST = [
    {
        "question": {
            "task": "classify_ticket",
            "text": "The mobile app crashes when I try to upload photos.",
            "categories": ["technical", "billing", "account", "feature_request"]
        },
        "ground_truth": {
            "category": "technical",
            "priority": "high",
            "reasoning": "App crashes are critical bugs"
        }
    },
    {
        "question": {
            "task": "classify_ticket",
            "text": "Can you add dark mode to the dashboard?",
            "categories": ["technical", "billing", "account", "feature_request"]
        },
        "ground_truth": {
            "category": "feature_request",
            "priority": "low",
            "reasoning": "Feature enhancement request"
        }
    },
]
```

#### Step 2: 데이터셋 등록

같은 파일의 `DATASETS` 딕셔너리에 추가:

```python
DATASETS = {
    "labeling": {
        "train": LABELING_TRAIN,
        "test": LABELING_TEST,
        "description": "Named entity recognition (FiNER-like)"
    },
    "numeric": {
        "train": NUMERIC_TRAIN,
        "test": NUMERIC_TEST,
        "description": "Formula-based calculations (finance-like)"
    },
    "code_agent": {
        "train": CODE_AGENT_TRAIN,
        "test": CODE_AGENT_TEST,
        "description": "List operations (AppWorld-like)"
    },
    "ticket": {  # 새로운 데이터셋 추가
        "train": TICKET_TRAIN,
        "test": TICKET_TEST,
        "description": "Customer support ticket classification"
    },
}
```

#### Step 3: CLI에 추가 (선택사항)

`src/ace/cli.py`의 `choices` 리스트를 업데이트:

```python
# 287-290번 줄과 319-322번 줄
choices=["labeling", "numeric", "code_agent", "ticket", "all"],
```

#### Step 4: 평가 로직 추가

`src/ace/evaluator.py`에 평가 함수 추가:

```python
def evaluate_ticket(pred: dict, gt: dict) -> Tuple[bool, float]:
    """
    Evaluate ticket classification.

    Args:
        pred: {"category": str, "priority": str, "reasoning": str}
        gt: Same format as pred

    Returns:
        (is_correct, score)
    """
    # Category 정확도 (가중치 0.7)
    category_correct = pred.get("category", "").lower() == gt.get("category", "").lower()

    # Priority 정확도 (가중치 0.3)
    priority_correct = pred.get("priority", "").lower() == gt.get("priority", "").lower()

    # 최종 점수 계산
    score = (0.7 if category_correct else 0.0) + (0.3 if priority_correct else 0.0)
    is_correct = (score >= 0.7)  # 70% 이상이면 정답으로 판정

    return is_correct, score
```

그리고 `evaluate_sample` 함수에 태스크 추가:

```python
def evaluate_sample(sample: Dict[str, Any], prediction: Dict[str, Any]) -> EvaluationResult:
    """Evaluate a single prediction against ground truth."""
    question = sample["question"]
    ground_truth = sample["ground_truth"]
    task = question.get("task")

    # ... 기존 코드 ...

    elif task == "classify_ticket":
        is_correct, score = evaluate_ticket(prediction, ground_truth)

    # ... 나머지 코드 ...
```

#### Step 5: 동작 확인

```bash
# 새 데이터셋이 등록되었는지 확인
python -m ace list-datasets

# 출력에 새로운 항목이 나타나야 함:
# ticket:
#   Description: Customer support ticket classification
#   Train: 3 samples
#   Test:  2 samples
#   Total: 5 samples
```

---

## 실행 가이드

### 전체 프로세스 개요

```
1. Baseline 테스트 (빈 플레이북)
   ↓
2. Offline 학습 (플레이북 진화)
   ↓
3. 플레이북 확인 (학습 내용 검증)
   ↓
4. Online 테스트 (진화된 플레이북)
   ↓
5. 성능 비교 (개선도 측정)
```

### Step 1: Baseline 테스트 (초기 성능 측정)

빈 플레이북으로 테스트하여 초기 성능을 측정합니다.

```bash
# 플레이북 초기화 (비어있는 상태로 만들기)
rm -rf storage/playbook.json

# 테스트 데이터로 추론 실행 (학습 없이)
python -m ace online --dataset labeling

# 또는 모든 데이터셋에 대해
python -m ace online --dataset all
```

**예상 출력**:
```
============================================================
RESULTS
============================================================
Run ID:     baseline_20241017_143022
Run Dir:    ./runs/baseline_20241017_143022

Accuracy:   40.00%
Avg Score:  0.450
Correct:    2 / 5

Playbook:
  Total items:   0
  Serving items: 0
```

> **중요**: 이 결과를 **반드시 기록**하세요. 나중에 개선도를 비교할 기준점입니다.

### Step 2: Offline 학습 (플레이북 진화)

학습 데이터로 플레이북을 진화시킵니다.

```bash
# 특정 데이터셋으로 학습 (2 에포크)
python -m ace offline --dataset labeling --epochs 2

# 모든 데이터셋으로 학습
python -m ace offline --dataset all --epochs 3

# 처음부터 다시 시작 (기존 플레이북 리셋)
python -m ace offline --dataset all --epochs 2 --reset
```

**실행 중 화면**:
```
2024-10-17 14:35:12 [INFO] ace.pipeline: Starting offline adaptation...
2024-10-17 14:35:12 [INFO] ace.pipeline: Epoch 1/2
2024-10-17 14:35:15 [INFO] ace.pipeline: Sample 1/3 - Score: 0.85
2024-10-17 14:35:18 [INFO] ace.pipeline: Sample 2/3 - Score: 1.00
2024-10-17 14:35:21 [INFO] ace.pipeline: Sample 3/3 - Score: 0.92

Epoch 1 - Accuracy: 91.67%
  Playbook operations applied: 5 items added

2024-10-17 14:35:22 [INFO] ace.pipeline: Epoch 2/2
...
```

**학습 완료 후 출력**:
```
============================================================
RESULTS
============================================================
Run ID:     offline_labeling_20241017_143522
Run Dir:    ./runs/offline_labeling_20241017_143522

Accuracy:   100.00%
Avg Score:  1.000
Correct:    3 / 3

Playbook:
  Total items:   8
  Serving items: 8
```

### Step 3: 플레이북 확인 (학습 내용 검증)

플레이북에 어떤 지식이 축적되었는지 확인합니다.

```bash
# 기본 통계 확인
python -m ace stats

# 상세 정보 확인 (각 항목의 내용까지)
python -m ace stats --verbose
```

**기본 통계 출력**:
```
============================================================
PLAYBOOK STATISTICS
============================================================
Total items:      8
Serving items:    8
Deprecated items: 0
Harmful items:    0

By category:
  strategy    : 3
  formula     : 2
  pitfall     : 1
  checklist   : 1
  example     : 1
```

**상세 정보 출력** (`--verbose`):
```
============================================================
ITEM DETAILS
============================================================

[a3f9e2b4c1d8] strategy - active
  Title: Four-digit Year Recognition
  Helpful: 2 | Harmful: 0
  Tags: labeling, date, pattern

  Content: When encountering four consecutive digits in text (e.g., "2024"),
  classify them as DATE entities. This pattern is reliable for year recognition
  in financial and news contexts.

[b7c4d8f2e1a9] formula - active
  Title: Simple Interest Formula
  Helpful: 3 | Harmful: 0
  Tags: finance, numeric, interest

  Content: Simple Interest = Principal × (Rate / 100) × Years
  Example: $1,000 at 5% for 2 years = 1000 × 0.05 × 2 = $100

[c8e1a9b4f3d2] pitfall - active
  Title: Off-by-one in List Indexing
  Helpful: 1 | Harmful: 0
  Tags: code_agent, list, indexing

  Content: Remember that list indices start at 0, not 1. When asked for
  "the 3rd element", use index 2, not 3.

...
```

### Step 4: Online 테스트 (진화된 플레이북로 추론)

학습된 플레이북을 사용하여 테스트 데이터로 추론합니다.

```bash
# 학습된 플레이북으로 테스트
python -m ace online --dataset labeling

# 추론 중에도 학습 활성화 (incremental learning)
python -m ace online --dataset labeling --enable-learning
```

**출력 예시**:
```
============================================================
RESULTS
============================================================
Run ID:     online_labeling_20241017_144022
Run Dir:    ./runs/online_labeling_20241017_144022

Accuracy:   100.00%
Avg Score:  1.000
Correct:    2 / 2

Playbook:
  Total items:   8
  Serving items: 8
```

---

## 성능 비교

### 메트릭 수집

각 실행 단계에서 다음 정보를 기록하세요:

| 단계 | Accuracy | Correct | Total | Playbook Items |
|------|----------|---------|-------|----------------|
| Baseline (Before) | 40.00% | 2 | 5 | 0 |
| After Training | 100.00% | 5 | 5 | 8 |

### 개선도 계산

```python
# 절대 개선도
absolute_improvement = accuracy_after - accuracy_before
# 100.00% - 40.00% = +60.00%

# 상대 개선도
relative_improvement = (accuracy_after / accuracy_before - 1) * 100
# (100.00 / 40.00 - 1) × 100 = +150%
```

### 자동화된 비교 스크립트

`scripts/compare_performance.py` 파일을 생성:

```python
"""
성능 비교 스크립트
"""
import json
from pathlib import Path
from typing import Dict, List


def load_run_results(run_dir: str) -> Dict:
    """Load results from a run directory."""
    metadata_path = Path(run_dir) / "run_metadata.json"
    with open(metadata_path) as f:
        return json.load(f)


def compare_runs(baseline_dir: str, trained_dir: str):
    """Compare baseline and trained runs."""
    baseline = load_run_results(baseline_dir)
    trained = load_run_results(trained_dir)

    baseline_acc = baseline["final_metrics"]["accuracy"]
    trained_acc = trained["final_metrics"]["accuracy"]

    abs_improvement = trained_acc - baseline_acc
    rel_improvement = (trained_acc / baseline_acc - 1) * 100 if baseline_acc > 0 else 0

    print("\n" + "="*60)
    print("PERFORMANCE COMPARISON")
    print("="*60)
    print(f"\nBaseline Performance:")
    print(f"  Accuracy: {baseline_acc:.2%}")
    print(f"  Correct:  {baseline['final_metrics']['correct']}/{baseline['final_metrics']['total']}")
    print(f"  Playbook: {baseline['playbook_stats']['total_items']} items")

    print(f"\nTrained Performance:")
    print(f"  Accuracy: {trained_acc:.2%}")
    print(f"  Correct:  {trained['final_metrics']['correct']}/{trained['final_metrics']['total']}")
    print(f"  Playbook: {trained['playbook_stats']['total_items']} items")

    print(f"\nImprovement:")
    print(f"  Absolute: {abs_improvement:+.2%}")
    print(f"  Relative: {rel_improvement:+.1f}%")
    print(f"  Playbook Growth: +{trained['playbook_stats']['total_items'] - baseline['playbook_stats']['total_items']} items")
    print()


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 3:
        print("Usage: python scripts/compare_performance.py <baseline_run_dir> <trained_run_dir>")
        sys.exit(1)

    compare_runs(sys.argv[1], sys.argv[2])
```

**사용 방법**:
```bash
python scripts/compare_performance.py \
    ./runs/baseline_20241017_143022 \
    ./runs/online_labeling_20241017_144022
```

---

## 실전 예제

### 시나리오: 법률 문서 개체명 인식

법률 문서에서 당사자, 날짜, 금액 등을 추출하는 시스템을 만들어봅시다.

#### 1. 데이터 준비

`src/ace/datasets.py`에 추가:

```python
LEGAL_TRAIN = [
    {
        "question": {
            "task": "label_spans",
            "text": "On March 15, 2024, John Doe filed a lawsuit against ABC Corp for $500,000.",
            "labels": ["DATE", "PERSON", "ORG", "MONEY"]
        },
        "ground_truth": {
            "spans": [
                {"text": "March 15, 2024", "label": "DATE", "start": 3, "end": 17},
                {"text": "John Doe", "label": "PERSON", "start": 19, "end": 27},
                {"text": "ABC Corp", "label": "ORG", "start": 49, "end": 57},
                {"text": "$500,000", "label": "MONEY", "start": 62, "end": 70}
            ]
        }
    },
    {
        "question": {
            "task": "label_spans",
            "text": "The defendant, Jane Smith, must pay damages by December 31, 2024.",
            "labels": ["PERSON", "DATE"]
        },
        "ground_truth": {
            "spans": [
                {"text": "Jane Smith", "label": "PERSON", "start": 15, "end": 25},
                {"text": "December 31, 2024", "label": "DATE", "start": 47, "end": 64}
            ]
        }
    },
]

LEGAL_TEST = [
    {
        "question": {
            "task": "label_spans",
            "text": "XYZ Inc. agreed to settle with Robert Johnson for €2.5M on July 1st.",
            "labels": ["ORG", "PERSON", "MONEY", "DATE"]
        },
        "ground_truth": {
            "spans": [
                {"text": "XYZ Inc.", "label": "ORG", "start": 0, "end": 8},
                {"text": "Robert Johnson", "label": "PERSON", "start": 31, "end": 45},
                {"text": "€2.5M", "label": "MONEY", "start": 50, "end": 55},
                {"text": "July 1st", "label": "DATE", "start": 59, "end": 67}
            ]
        }
    },
]

# DATASETS에 등록
DATASETS["legal"] = {
    "train": LEGAL_TRAIN,
    "test": LEGAL_TEST,
    "description": "Legal document entity extraction"
}
```

#### 2. Baseline 측정

```bash
rm -rf storage/playbook.json
python -m ace online --dataset legal
```

**예상 결과**: 50-60% 정확도 (법률 도메인 특수성으로 낮음)

#### 3. 학습 실행

```bash
python -m ace offline --dataset legal --epochs 3
```

#### 4. 재테스트

```bash
python -m ace online --dataset legal
```

**예상 결과**: 80-100% 정확도 (법률 용어 패턴 학습)

#### 5. 학습 내용 확인

```bash
python -m ace stats --verbose
```

**예상되는 학습 내용**:
- "유로 화폐 기호(€) 인식 전략"
- "법률 문서의 날짜 표현 패턴"
- "회사명 식별 체크리스트 (Inc., Corp, Ltd 등)"

---

## 문제 해결

### 흔한 오류와 해결책

#### 1. `ModuleNotFoundError: No module named 'anthropic'`

**원인**: 의존성이 설치되지 않음

**해결**:
```bash
pip install -r requirements.txt
```

#### 2. `Error: ANTHROPIC_API_KEY not found`

**원인**: API 키가 설정되지 않음

**해결**:
```bash
# .env 파일 생성 및 편집
cp .env.example .env
# ANTHROPIC_API_KEY=your_key_here 추가
```

#### 3. `FileNotFoundError: [Errno 2] No such file or directory: './storage/playbook.json'`

**원인**: 플레이북 파일이 없음 (온라인 모드를 먼저 실행한 경우)

**해결**:
```bash
# 먼저 오프라인 학습 실행
python -m ace offline --dataset labeling --epochs 2

# 그 다음 온라인 추론
python -m ace online --dataset labeling
```

#### 4. 정확도가 개선되지 않음

**가능한 원인**:
1. **학습 데이터 부족**: 최소 5개 이상의 train 샘플 필요
2. **에포크 부족**: `--epochs 3` 이상으로 실행
3. **데이터 품질**: ground_truth가 정확한지 확인

**해결**:
```bash
# 더 많은 에포크로 재학습
python -m ace offline --dataset your_dataset --epochs 5 --reset

# 플레이북 상태 확인
python -m ace stats --verbose
```

#### 5. API 비용 제어

**문제**: 학습 중 API 비용이 걱정됨

**해결**:
1. 작은 데이터셋(3-5개 샘플)로 시작
2. `ACE_MAX_TOKENS`를 낮게 설정 (1024)
3. Mock 테스트로 프로세스 먼저 검증 (`scripts/test_playbook_evolution.py` 참고)

```bash
# .env에 토큰 제한 추가
ACE_MAX_TOKENS=1024
```

#### 6. 플레이북이 너무 커짐

**문제**: 플레이북 아이템이 100개 이상

**원인**: 중복 제거 임계값이 너무 낮음

**해결**:
```bash
# .env에서 임계값 조정
ACE_DEDUP_SIMILARITY=0.95  # 더 엄격한 중복 제거 (기본값: 0.92)
ACE_MAX_OPERATIONS_PER_CURATOR=10  # 한 번에 추가되는 항목 제한 (기본값: 20)
```

---

## 다음 단계

### 추가 학습 자료

- **README.md**: 전체 아키텍처 및 CLI 명령어
- **CLAUDE.md**: 개발 가이드라인
- **report.md**: 진화 테스트 결과 및 성능 분석

### 고급 기능

1. **Incremental Learning**: 추론 중에도 학습 활성화
   ```bash
   python -m ace online --dataset labeling --enable-learning
   ```

2. **Early Stopping**: 수렴 시 자동 종료
   ```bash
   python -m ace offline --dataset labeling --epochs 10 --patience 3
   ```

3. **Custom Evaluator**: 자신만의 평가 로직 구현
   - `src/ace/evaluator.py`의 `evaluate_sample` 함수 수정

### 프로덕션 배포

프로덕션 환경에서 사용하려면:

1. **플레이북 버전 관리**: Git으로 `storage/playbook.json` 추적
2. **A/B 테스트**: 빈 플레이북 vs 진화된 플레이북 성능 비교
3. **모니터링**: `runs/` 디렉토리의 로그 분석
4. **지속적 개선**: 새로운 데이터로 주기적 재학습

---

## 요약

ACE 프레임워크 사용 흐름:

```
1. 환경 설정 → pyenv + pip install + API key
2. 데이터 준비 → datasets.py에 question/ground_truth 추가
3. Baseline → python -m ace online (빈 플레이북)
4. 학습 → python -m ace offline --epochs 3
5. 검증 → python -m ace stats --verbose
6. 재테스트 → python -m ace online (진화된 플레이북)
7. 비교 → 성능 메트릭 분석
```

**핵심 원칙**:
- 작은 데이터셋으로 시작 (3-5 샘플)
- 반드시 baseline 측정 후 학습
- 플레이북 내용을 정기적으로 검토
- 도메인 지식이 축적되는 과정을 관찰

이제 자신의 데이터로 ACE 프레임워크를 사용해보세요! 🚀
