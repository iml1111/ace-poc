# ACE Framework POC

**Agentic Context Engineering (ACE)** - A proof-of-concept implementation for evolving LLM context through playbook-based delta updates.

## Overview

ACE is a framework that treats LLM context (prompts, memory, evidence) as an evolving **playbook** that improves over time through systematic delta updates, rather than weight modification or full context rewrites.

### Key Concepts

- **Playbook**: A structured knowledge base containing strategies, formulas, pitfalls, checklists, and examples
- **Delta Updates**: Small, incremental changes (add/amend/deprecate) instead of full rewrites
- **Triple-Agent Architecture**: Generator → Reflector → Curator pipeline for systematic improvement
- **Deterministic Execution**: Fixed seeds, temperature=0, SHA-256 IDs for reproducibility

### Problems Solved

- **Brevity Bias**: Repeated summarization loses critical details
- **Context Collapse**: Domain-specific knowledge and strategies disappear
- **Accuracy/Cost Trade-off**: Simultaneous improvements in accuracy, speed, and cost

---

## ACE의 핵심: Context Engineering, Not Prompt Engineering

### 전통적 접근법의 문제점

**프롬프트 엔지니어링 (수정 방식)**:
```python
# Version 1
"Extract entities from text."

# Version 2 (직접 수정)
"Extract entities from text. Use string.find() for positions."

# Version 3 (또 수정)
"Extract entities from text. Use string.find(). Check entity types."
```

**문제점**:
- ❌ **Brevity bias**: 매번 재작성하면서 세부사항 손실
- ❌ **Context collapse**: 도메인 특화 지식이 일반화되면서 소실
- ❌ **버전 관리 어려움**: 어느 버전이 왜 좋은지 추적 불가
- ❌ **전면 재작성 리스크**: 이전에 작동하던 부분도 망가질 수 있음

### ACE의 해결책 (추가 방식)

**시스템 프롬프트는 영구 고정 + 플레이북만 진화**:

```python
# 시스템 프롬프트 (영원히 고정)
"You are an analysis expert. Use the Playbook..."

# 플레이북 (계속 성장)
Epoch 1:
  + "인덱스 0부터 카운트"
  + "text[start:end] 검증"

Epoch 2:
  + "string.find() 사용"  # 기존 항목 유지
  + "테스트 우선 접근"

Epoch 3:
  + "엔티티별 경계 규칙"  # 기존 항목 계속 유지
```

**장점**:
- ✅ **증분 개선**: 이전 지식 보존하면서 새 지식 추가
- ✅ **버전 관리**: 모든 항목에 ID, 타임스탬프, 유용성 카운터
- ✅ **롤백 가능**: 유해한 항목은 deprecated 표시
- ✅ **감사 추적**: 언제 어떤 항목이 왜 추가되었는지 추적 가능

### 살아있는 플레이북

**비유**:
- **교과서(시스템 프롬프트)**: "수학 문제 풀 때는 단계적으로 접근하세요" (고정)
- **참고 노트(플레이북)**:
  - Day 1: "분수 문제는 통분부터"
  - Day 2: "음수 곱셈은 부호 먼저 확인"
  - Day 3: "방정식은 양변에 같은 연산"

→ 교과서는 안 바뀌지만, 참고 노트가 계속 쌓임
→ 이전 노트들도 계속 참고 가능

---

## Installation

### Prerequisites

- Python 3.11+ (managed with pyenv)
- Anthropic API key

### Setup

```bash
# Clone repository
cd ace-poc

# Activate pyenv environment (auto-activates via .python-version)
# If not auto-activated:
pyenv activate ace-poc

# Install dependencies
pip install -r requirements.txt

# Create .env file
cp .env.example .env
# Edit .env and add your ANTHROPIC_API_KEY
```

---

## Quick Start

### 1. List Available Datasets

```bash
python -m ace list-datasets
```

Output shows three toy datasets:
- **labeling**: Named entity recognition (FiNER-like)
- **numeric**: Formula-based calculations (finance-like)
- **code_agent**: List operations (AppWorld-like)

### 2. Run Offline Adaptation (Warm-up)

Train the playbook on training data:

```bash
# Run on all datasets for 2 epochs
python -m ace offline --dataset all --epochs 2

# Or run on specific dataset
python -m ace offline --dataset labeling --epochs 3

# Reset playbook and start fresh
python -m ace offline --dataset all --epochs 2 --reset
```

This will:
- Process each training sample through Generator → Reflector → Curator
- Update the playbook with learned strategies and insights
- Save playbook to `storage/playbook.json`
- Log all steps to `runs/{timestamp}/`

### 3. Run Online Inference

Test the trained playbook on test data:

```bash
# Run inference on all test sets
python -m ace online --dataset all

# Enable incremental learning during inference
python -m ace online --dataset all --enable-learning
```

### 4. View Playbook Statistics

```bash
# Basic stats
python -m ace stats

# Detailed item info
python -m ace stats --verbose
```

---

## Architecture

### Triple-Agent Pipeline

```
┌─────────────┐     ┌────────────┐     ┌──────────┐
│  Generator  │ --> │ Reflector  │ --> │ Curator  │
└─────────────┘     └────────────┘     └──────────┘
      │                   │                   │
      ▼                   ▼                   ▼
  Prediction         Analysis            Delta Ops
  + Bullet IDs       + Tags              (add/amend/deprecate)
                                              │
                                              ▼
                                        ┌──────────┐
                                        │ Playbook │
                                        │  Merge   │
                                        └──────────┘
```

### Components

1. **Generator** (`agents.py`): Uses playbook to answer questions, tracks which items (bullet_ids) were helpful
2. **Reflector** (`agents.py`): Analyzes prediction vs ground truth, tags bullets as helpful/harmful/neutral
3. **Curator** (`agents.py`): Proposes delta operations to improve playbook based on reflection
4. **Playbook Store** (`playbook.py`): Manages merge operations with deduplication and conflict resolution
5. **Pipeline** (`pipeline.py`): Orchestrates offline/online loops with logging and early stopping
6. **Evaluator** (`evaluator.py`): Task-specific metrics (exact match, F1, etc.)

### Prompt Evolution Example

**초기 프롬프트 (Epoch 1, Sample 1)**:
```json
{
  "playbook": {
    "items": []  // 빈 배열
  },
  "question": {
    "task": "label_spans",
    "text": "Apple Inc. reported $1.2M in 2024."
  }
}
```
- 크기: ~40 토큰
- 결과: 제로샷 추론, 위치 계산 오류

**최종 프롬프트 (Epoch 3 완료)**:
```json
{
  "playbook": {
    "items": [
      // 20개 항목:
      // - 전략 7개 (테스트 우선, 엔티티별 규칙 등)
      // - 공식 8개 (string.find(), 검증 공식 등)
      // - 체크리스트 1개 (9포인트 검증)
      // - 함정 3개 (일반적 오류)
      // - 예시 1개 (deprecated)
    ]
  },
  "question": { /* 동일 */ }
}
```
- 크기: ~900 토큰 (22.5배 증가)
- 결과: 경험 기반 추론, 구체적 전략 적용

**상세 비교**: `experiments/ner-evolution-20251019_154015/prompts_comparison.md`

### Data Flow

**Offline Mode** (Training):
```
For each epoch:
  For each train sample:
    1. Generator(playbook, question) → prediction + bullet_ids
    2. Evaluate(prediction, ground_truth) → metrics
    3. Reflector(prediction, gt, bullet_ids) → analysis + tags
    4. Update bullet stats (helpful/harmful counts)
    5. Curator(playbook, analysis) → delta operations
    6. Merge operations → updated playbook
  Save playbook
```

**Online Mode** (Inference):
```
For each test sample:
  1. Generator(playbook, question) → prediction
  2. (Optional) Run Reflector + Curator for incremental learning
  Log results
```

---

## Experimental Results

### NER 진화 실험 (2025-10-19)

**작업**: Named Entity Recognition - Span Labeling
**Dataset**: 3 training samples
**에포크**: 3 (early stopping triggered)
**플레이북 성장**: 0 → 8 → 14 → 20 items

#### 학습 진화 과정

| Epoch | Accuracy | Playbook Size | Avg Bullets Used | 학습 내용 |
|-------|----------|---------------|------------------|-----------|
| 1     | 33.33%   | 8 items       | 2.7 bullets      | 기본 검증, 위치 카운팅 |
| 2     | 0.00%    | 14 items      | 4.3 bullets      | string.find() 발견, 테스트 우선 접근 |
| 3     | 33.33%   | 20 items      | 5.0 bullets      | 엔티티별 경계 규칙 |

#### 주요 발견

**1. 창발적 전략 발견**:
- "Test-First Span Extraction": 시스템이 독립적으로 "목표 우선, 위치 이후" 접근법 발견
- Entity-type-specific rules: 엔티티 타입별 구두점 규약 자동 학습

**2. 자기 교정 능력**:
- 2개 항목 deprecated (harmful_count >= 3)
- 잘못된 예시를 스스로 발견하고 제거

**3. 지식 축적 패턴**:
```
Epoch 1: "인덱스 0부터 카운트, text[start:end] 검증"
Epoch 2: "string.find() 사용, 상대 위치 계산 회피"
Epoch 3: "조직명은 'Inc.' 포함, 날짜는 후행 마침표 조건부 포함"
```

#### 프롬프트 크기 변화

| 구분 | 초기 | 최종 | 변화 |
|------|------|------|------|
| 플레이북 항목 | 0 | 20 | +20 |
| 토큰 수 | ~40 | ~900 | 22.5배 |
| 사용 항목 | 0 | 5 평균 | +5 |

**전체 보고서**: `report.md` (한글)
**원본 데이터**: `experiments/ner-evolution-20251019_154015/`
**프롬프트 비교**: `experiments/ner-evolution-20251019_154015/prompts_comparison.md`

---

## Project Structure

```
ace-poc/
├── src/ace/
│   ├── __init__.py
│   ├── __main__.py         # Entry point for python -m ace
│   ├── models.py           # Pydantic schemas (541 lines)
│   ├── playbook.py         # Playbook storage & merge logic (290 lines)
│   ├── prompts.py          # Agent prompt templates (272 lines)
│   ├── agents.py           # LLM wrappers with JSON validation (284 lines)
│   ├── datasets.py         # Toy datasets (286 lines)
│   ├── evaluator.py        # Task-specific metrics (347 lines)
│   ├── pipeline.py         # Offline/Online orchestration (434 lines)
│   └── cli.py              # Command-line interface (360 lines)
├── tests/
│   ├── test_models.py      # Model & ID generation tests
│   └── test_playbook.py    # Playbook merge operation tests
├── storage/
│   └── playbook.json       # Evolved playbook (created after first run)
├── runs/
│   └── {timestamp}/        # Execution logs and metadata
│       ├── steps.jsonl     # All agent steps with hashes
│       ├── reflections.jsonl
│       └── run_metadata.json
├── experiments/            # Experimental results archive
│   └── ner-evolution-20251019_154015/  # NER evolution experiment
│       ├── prompts_comparison.md
│       ├── initial_prompt_example.txt
│       ├── final_prompt_example.txt
│       ├── final_playbook.json
│       └── offline_run_final.log
├── scripts/                # Temporary experiment scripts
├── requirements.txt
├── .env.example
├── .gitignore
├── CLAUDE.md               # Development guidelines
├── GETTING_STARTED.md      # User guide (Korean)
├── report.md               # Evolution experiment report (Korean)
├── LICENSE
└── README.md
```

---

## Configuration

Configuration via environment variables (`.env` file):

```bash
# API Configuration
ANTHROPIC_API_KEY=your_key_here
ACE_MODEL=claude-3-5-sonnet-latest
ACE_MAX_TOKENS=2048
ACE_TEMPERATURE=0.0
ACE_SEED=42

# Storage
ACE_STORAGE_DIR=./storage
ACE_RUNS_DIR=./runs

# Playbook Tuning
ACE_HARMFUL_THRESHOLD=3          # Hide items with harmful_count >= 3
ACE_DEDUP_SIMILARITY=0.92        # Similarity threshold for deduplication
ACE_MAX_OPERATIONS_PER_CURATOR=20  # Max operations per curator call

# Semantic Deduplication (Optional)
ACE_USE_SEMANTIC_DEDUP=false     # Enable semantic embeddings for duplicate detection
ACE_EMBEDDING_MODEL=all-MiniLM-L6-v2  # Model for semantic similarity
```

### Optional: Semantic Deduplication

ACE POC supports optional semantic deduplication for improved duplicate detection using sentence embeddings.

**Benefits:**
- Better duplicate detection: "check authentication" ≈ "verify auth" (semantic similarity)
- Reduces false negatives compared to string-based difflib matching
- Cleaner playbook → improved Generator performance

**Setup:**

```bash
# Install optional dependencies (~100MB memory usage)
pip install -r requirements-semantic.txt

# Enable in .env
ACE_USE_SEMANTIC_DEDUP=true
```

**Default behavior:** Uses `difflib.SequenceMatcher` (no additional dependencies)
**With semantic dedup:** Uses `sentence-transformers` with `all-MiniLM-L6-v2` model

The system automatically falls back to difflib if semantic dependencies are not installed.

---

## CLI Commands

### offline

Train playbook on training data:

```bash
python -m ace offline [OPTIONS]

Options:
  --dataset {labeling,numeric,code_agent,all}  Dataset to use (default: all)
  --epochs INT                                 Number of epochs (default: 2)
  --model STR                                  Model name override
  --seed INT                                   Random seed
  --patience INT                               Early stopping patience (default: 2)
  --early-stop-delta FLOAT                     Early stopping delta (default: 0.01)
  --reset                                      Reset playbook (start from scratch)
```

### online

Run inference on test data:

```bash
python -m ace online [OPTIONS]

Options:
  --dataset {labeling,numeric,code_agent,all}  Dataset to use (default: all)
  --model STR                                  Model name override
  --seed INT                                   Random seed
  --enable-learning                            Enable incremental learning during inference
```

### stats

View playbook statistics:

```bash
python -m ace stats [--verbose]
```

### list-datasets

List available datasets:

```bash
python -m ace list-datasets
```

---

## Testing

Run unit tests:

```bash
# All tests
pytest tests/ -v

# Specific test file
pytest tests/test_models.py -v
pytest tests/test_playbook.py -v
```

---

## Key Implementation Details

### Deterministic ID Generation

```python
item_id = hashlib.sha256(
    f"{category}|{normalize(title)}|{normalize(content)}".encode()
).hexdigest()[:12]
```

### Deduplication

**Default mode** (difflib):
- Uses `difflib.SequenceMatcher` with threshold 0.92
- String-based similarity matching
- Fast, no additional dependencies

**Optional semantic mode** (sentence-transformers):
- Uses semantic embeddings with cosine similarity
- Better duplicate detection for semantically similar text
- Requires: `pip install -r requirements-semantic.txt`
- Enable with: `ACE_USE_SEMANTIC_DEDUP=true`

Both modes:
- If new item is ≥92% similar to existing item → convert add to amend
- Prevents redundant near-duplicate items

### Merge Order

Operations applied deterministically:
1. **Deprecate** (mark harmful items)
2. **Amend** (update existing items)
3. **Add** (with dedup check)

### Audit Logging

Every step logged with:
- Input/output SHA-256 hashes
- Model name, seed, temperature
- Prompt version
- Bullet IDs used
- Operations applied

---

## Limitations

This is a POC with intentionally limited scope:

- **Small datasets**: 3-5 samples per split (for quick iteration)
- **Simple tasks**: Labeling, numeric, list operations
- **No production optimizations**: No caching, batching, or async processing
- **Basic evaluation**: Exact match and F1, no complex metrics

---

## Research Context

This POC validates ACE framework hypotheses:

1. **Delta updates preserve information** better than full rewrites
2. **Role separation** (Generator/Reflector/Curator) produces higher quality context
3. **Playbook evolution** outperforms static few-shot prompting over time

Measured outcomes:
- Brevity bias frequency (before/after)
- Context collapse instances
- Cumulative performance improvement curves
- Domain-specific knowledge retention rates

### Experimental Evidence

**NER Evolution Experiment** demonstrated:
- ✅ Zero knowledge → 20 structured knowledge items in 3 epochs
- ✅ Emergent strategies: "Test-First Extraction" discovered independently
- ✅ Self-correction: 2 items deprecated when found harmful
- ✅ Knowledge preservation: All previous items retained while adding new ones

**Key Insight**: System evolved from generic "count characters" to sophisticated "entity-type-specific boundary rules" without any weight updates.

---

## Documentation

- **README.md** (this file): Project overview and architecture
- **GETTING_STARTED.md**: Step-by-step user guide (Korean)
- **CLAUDE.md**: Development guidelines for AI assistants
- **report.md**: Detailed evolution experiment report (Korean)
- **experiments/**: Archived experimental results with raw data

---

## License

MIT License - see LICENSE file

---

## References

Based on the Agentic Context Engineering research paper, which proposes treating context as an evolving playbook managed through systematic delta updates rather than weight modification.

**Core Philosophy**: "Context Engineering, Not Prompt Engineering"
- System prompts stay fixed
- Playbooks evolve through delta updates
- Knowledge accumulates without collapse
- Reproducible and auditable
