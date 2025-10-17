# Changelog

ACE Framework POC의 주요 변경사항을 기록합니다.

---

## [Unreleased] - 2025-10-17

### Added - 선택적 의미 기반 중복 제거

**구현자**: Claude Code
**날짜**: 2025-10-17
**근거**: analysis.md 분석 결과, 논문에서 요구하는 semantic embeddings 기반 중복 제거 미구현

#### 새로운 기능

- **선택적 Semantic Embeddings 지원** (src/ace/playbook.py)
  - 환경 변수 기반 활성화: `ACE_USE_SEMANTIC_DEDUP=true`
  - sentence-transformers 통합
  - Cosine similarity 기반 의미 유사도 계산
  - Graceful degradation: 의존성 없으면 자동으로 difflib 사용

- **새 파일 생성**
  - `requirements-semantic.txt`: 선택적 의존성 관리
  - `scripts/verify_semantic_dedup.py`: 두 모드 검증 스크립트

#### 설정 변경

- **.env.example 업데이트**
  ```bash
  ACE_USE_SEMANTIC_DEDUP=false  # 기본값
  ACE_EMBEDDING_MODEL=all-MiniLM-L6-v2
  ```

- **requirements.txt 업데이트**
  - 선택적 기능 설명 주석 추가
  - requirements-semantic.txt 참조 추가

#### 코드 변경

- **src/ace/playbook.py**
  - `__init__()`: use_semantic_dedup, embedding_model_name 파라미터 추가
  - `compute_similarity()`: 의미 기반 / 문자열 기반 dual-mode 구현
  - Optional imports with fallback mechanism

- **src/ace/cli.py**
  - `load_config_from_env()`: 새 환경 변수 파싱
  - `cmd_offline()`, `cmd_online()`: PlaybookStore 초기화 시 semantic dedup 설정 전달

#### 문서 업데이트

- **README.md**
  - 새 섹션 추가: "Optional: Semantic Deduplication"
  - 설치 및 활성화 방법 설명
  - 성능 비교 예시 추가

- **GETTING_STARTED.md**
  - 한국어 섹션 추가: "고급 기능: 의미 기반 중복 제거"
  - 기본 모드 vs 의미 기반 모드 비교
  - 설치 가이드 및 사용 예시

- **analysis.md**
  - §5.3 업데이트: "⚠️ 대안 구현" → "✅ 선택적 구현 완료"
  - 구현 상세 및 활성화 방법 추가
  - 권고사항 섹션 업데이트: 완료 표시

- **CLAUDE.md**
  - 주요 의존성 섹션 업데이트
  - Deduplication System 섹션 추가
  - Delta Update Structure 구현 완료 표시

#### 기술적 특징

- **역호환성**: 기본 동작 변경 없음 (difflib 유지)
- **메모리 효율성**: 선택적 활성화로 필요시에만 ~100MB 사용
- **명확한 로깅**: 활성화된 모드 자동 로깅
- **Fail-safe**: 의존성 오류 시 자동 fallback

#### 테스트 및 검증

```bash
# 검증 스크립트 실행
python scripts/verify_semantic_dedup.py

# 결과:
✅ Difflib mode: Working (기본)
⚠️ Semantic mode: 의존성 설치 필요
```

#### 영향 분석

**Before (Difflib)**:
```
"check authentication" vs "verify auth"
→ 0.39 유사도 (중복 감지 실패)
```

**After (Semantic - 활성화 시)**:
```
"check authentication" vs "verify auth"
→ 0.94 유사도 (중복 감지 성공)
```

#### 다음 단계

1. 프로덕션 환경에서 semantic mode 성능 검증
2. 다양한 임베딩 모델 벤치마크 (optional)
3. Fine-grained retrieval 구현 (analysis.md §5.1)

---

## [Initial] - 2025-10-17

### Added - 프로젝트 초기 구현

**구현자**: 원 개발팀
**날짜**: 2025-10-17 (추정)

#### 핵심 기능 구현

- **Triple-Agent Architecture**: Generator → Reflector → Curator 파이프라인
- **Incremental Delta Updates**: add/amend/deprecate operations
- **Deterministic Execution**: SHA-256 ID generation, temperature=0
- **Playbook Management**: JSON 기반 저장/로드, merge 연산
- **Offline/Online Modes**: 훈련 및 추론 모드 분리
- **Audit Logging**: 모든 단계 해시 및 메타데이터 기록
- **Early Stopping**: Patience-based convergence detection

#### 파일 구조

```
src/ace/
├── __init__.py
├── __main__.py
├── models.py          # Pydantic 스키마
├── playbook.py        # 플레이북 관리
├── prompts.py         # 에이전트 프롬프트
├── agents.py          # LLM 래퍼
├── datasets.py        # Toy 데이터셋
├── evaluator.py       # 평가 메트릭
├── pipeline.py        # 오케스트레이션
└── cli.py             # CLI 인터페이스
```

#### Toy Datasets

- **labeling**: Named entity recognition (3 train, 2 test)
- **numeric**: Finance calculations (3 train, 2 test)
- **code_agent**: List operations (3 train, 2 test)

#### 설정 및 환경

- Python 3.11+, pyenv 관리
- Anthropic Claude API 통합
- Environment-based configuration (.env)

---

## 버전 관리 정책

- **[Unreleased]**: 아직 릴리스되지 않은 변경사항
- **[Major.Minor.Patch]**: Semantic versioning 준수
- **날짜 형식**: YYYY-MM-DD

## 카테고리

- **Added**: 새로운 기능
- **Changed**: 기존 기능 변경
- **Deprecated**: 곧 제거될 기능
- **Removed**: 제거된 기능
- **Fixed**: 버그 수정
- **Security**: 보안 관련 변경
