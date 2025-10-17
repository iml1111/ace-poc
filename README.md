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
├── scripts/                # Temporary experiment scripts
├── requirements.txt
├── .env.example
├── .gitignore
├── CLAUDE.md               # Development guidelines
├── LICENSE
└── README.md
```

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

## Testing

Run unit tests:

```bash
# All tests
pytest tests/ -v

# Specific test file
pytest tests/test_models.py -v
pytest tests/test_playbook.py -v
```

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

## Limitations

This is a POC with intentionally limited scope:

- **Small datasets**: 3-5 samples per split (for quick iteration)
- **Simple tasks**: Labeling, numeric, list operations
- **No production optimizations**: No caching, batching, or async processing
- **Basic evaluation**: Exact match and F1, no complex metrics

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

## License

MIT License - see LICENSE file

## References

Based on the Agentic Context Engineering research paper, which proposes treating context as an evolving playbook managed through systematic delta updates rather than weight modification.
