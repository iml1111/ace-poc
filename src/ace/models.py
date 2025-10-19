"""
Pydantic models and schemas for the ACE framework.

All models use strict validation and provide deterministic serialization
for reproducibility and audit logging.
"""

import hashlib
import re
from datetime import datetime
from typing import Any, Dict, List, Literal, Optional, Union

import orjson
from pydantic import BaseModel, Field, field_validator


def normalize_text(text: str) -> str:
    """
    Normalize text for deterministic comparison and ID generation.

    - Lowercase
    - Strip leading/trailing whitespace
    - Collapse multiple whitespace to single space
    """
    return re.sub(r'\s+', ' ', text.strip().lower())


def generate_item_id(category: str, title: str, content: str) -> str:
    """
    Generate deterministic item_id from category, title, and content.

    Uses SHA-256 hash of normalized concatenation to ensure:
    - Same content always generates same ID
    - Order-invariant (only content matters)
    - Collision-resistant

    Returns first 12 hex characters for readability.
    """
    normalized = f"{category}|{normalize_text(title)}|{normalize_text(content)}"
    return hashlib.sha256(normalized.encode('utf-8')).hexdigest()[:12]


def compute_hash(obj: Any) -> str:
    """
    Compute deterministic SHA-256 hash of any JSON-serializable object.

    Uses orjson with sorted keys for stable serialization.
    """
    serialized = orjson.dumps(
        obj,
        option=orjson.OPT_SORT_KEYS | orjson.OPT_SERIALIZE_NUMPY
    )
    return hashlib.sha256(serialized).hexdigest()


class StrictBaseModel(BaseModel):
    """Base model with strict validation and deterministic serialization."""

    class Config:
        extra = "forbid"  # Reject unknown fields
        validate_assignment = True

    def model_dump_json(self, **kwargs) -> str:
        """Deterministic JSON serialization using orjson."""
        return orjson.dumps(
            self.model_dump(**kwargs),
            option=orjson.OPT_SORT_KEYS
        ).decode('utf-8')

    def compute_hash(self) -> str:
        """Compute hash of this model instance."""
        return compute_hash(self.model_dump())


# ============================================================================
# Playbook Models
# ============================================================================

class PlaybookItem(StrictBaseModel):
    """
    A single item in the Playbook.

    Categories:
    - strategy: High-level approach or heuristic
    - formula: Mathematical or computational pattern
    - pitfall: Known error or failure mode to avoid
    - checklist: Step-by-step validation or procedure
    - example: Concrete instance or pattern
    """
    item_id: str = Field(..., description="Deterministic SHA-256 derived ID")
    category: Literal["strategy", "formula", "pitfall", "checklist", "example"]
    title: str = Field(..., min_length=1, max_length=200)
    content: str = Field(..., min_length=1, max_length=1000, description="1-6 actionable sentences")
    tags: List[str] = Field(default_factory=list)
    helpful_count: int = Field(default=0, ge=0)
    harmful_count: int = Field(default=0, ge=0)
    created_at: str = Field(default_factory=lambda: datetime.utcnow().isoformat())
    updated_at: str = Field(default_factory=lambda: datetime.utcnow().isoformat())

    @field_validator('tags')
    @classmethod
    def validate_tags(cls, v: List[str]) -> List[str]:
        """Ensure tags are non-empty strings."""
        return [tag.strip() for tag in v if tag and tag.strip()]


class Playbook(StrictBaseModel):
    """Collection of PlaybookItems forming the evolving knowledge base."""
    items: List[PlaybookItem] = Field(default_factory=list)

    def get_item_by_id(self, item_id: str) -> Optional[PlaybookItem]:
        """Retrieve item by ID, returns None if not found."""
        for item in self.items:
            if item.item_id == item_id:
                return item
        return None

    def filter_serving_items(self, harmful_threshold: int = 3) -> List[PlaybookItem]:
        """
        Get items suitable for serving to Generator.

        Excludes items with harmful_count >= threshold or 'deprecated' tag.
        """
        return [
            item for item in self.items
            if item.harmful_count < harmful_threshold
            and "deprecated" not in item.tags
        ]


class PlaybookItemDraft(StrictBaseModel):
    """Draft item for curator operations (no ID or metadata yet)."""
    category: Literal["strategy", "formula", "pitfall", "checklist", "example"]
    title: str = Field(..., min_length=1, max_length=200)
    content: str = Field(..., min_length=1, max_length=1000)
    tags: List[str] = Field(default_factory=list)


# ============================================================================
# Generator Models
# ============================================================================

class GenInput(StrictBaseModel):
    """Input to Generator agent."""
    playbook: Playbook
    reflection: Optional[Dict[str, Any]] = None  # Previous ReflOutput as dict
    question: Dict[str, Any]  # Task-specific question format


class GenOutput(StrictBaseModel):
    """Output from Generator agent."""
    reasoning: str = Field(..., description="Concise step-by-step analysis")
    bullet_ids: List[str] = Field(default_factory=list, description="Playbook item IDs used")
    final_answer: Union[str, Dict[str, Any], List[Any]] = Field(..., description="Final answer (text, dict, or list)")


# ============================================================================
# Reflector Models
# ============================================================================

class BulletTag(StrictBaseModel):
    """Tag for a specific bullet's usefulness."""
    bullet_id: str
    tag: Literal["helpful", "harmful", "neutral"]


class ReflInput(StrictBaseModel):
    """Input to Reflector agent."""
    question: Dict[str, Any]
    predicted_answer: Union[str, Dict[str, Any], List[Any]]
    ground_truth: Optional[Union[str, Dict[str, Any], List[Any]]] = None
    env_feedback: Optional[Dict[str, Any]] = None  # pass/fail, errors, traces
    gen_reasoning: str
    used_bullet_ids: List[str]
    playbook_subset: List[PlaybookItem]


class ReflOutput(StrictBaseModel):
    """Output from Reflector agent."""
    reasoning_summary: str
    error_identification: str
    root_cause_analysis: str
    correct_approach: str
    key_insight: str
    bullet_tags: List[BulletTag] = Field(default_factory=list)


# ============================================================================
# Curator Models
# ============================================================================

class CurOpAdd(StrictBaseModel):
    """Add a new playbook item."""
    op: Literal["add"] = "add"
    item: PlaybookItemDraft


class CurOpAmend(StrictBaseModel):
    """Amend an existing playbook item."""
    op: Literal["amend"] = "amend"
    bullet_id: str
    delta: Dict[str, Any] = Field(
        ...,
        description="Delta changes: content_append (str), tags_add (List[str])"
    )


class CurOpDeprecate(StrictBaseModel):
    """Deprecate (soft-delete) a playbook item."""
    op: Literal["deprecate"] = "deprecate"
    bullet_id: str
    reason: str


# Union type for all curator operations
CuratorOperation = Union[CurOpAdd, CurOpAmend, CurOpDeprecate]


class CurInput(StrictBaseModel):
    """Input to Curator agent."""
    token_budget: int = Field(..., ge=0)
    question_context: Dict[str, Any]
    current_playbook: Playbook
    final_generated: GenOutput
    reflection: ReflOutput


class CurOutput(StrictBaseModel):
    """Output from Curator agent."""
    operations: List[CuratorOperation] = Field(default_factory=list)

    @field_validator('operations', mode='before')
    @classmethod
    def parse_operations(cls, v: Any) -> List[CuratorOperation]:
        """Parse operations from dicts or objects."""
        if not isinstance(v, list):
            v = [v]

        parsed = []
        for item in v:
            if isinstance(item, dict):
                op_type = item.get('op')
                if op_type == 'add':
                    parsed.append(CurOpAdd(**item))
                elif op_type == 'amend':
                    parsed.append(CurOpAmend(**item))
                elif op_type == 'deprecate':
                    parsed.append(CurOpDeprecate(**item))
                else:
                    raise ValueError(f"Unknown operation type: {op_type}")
            else:
                parsed.append(item)
        return parsed


# ============================================================================
# Audit and Logging Models
# ============================================================================

class StepLog(StrictBaseModel):
    """Log entry for a single step (gen/refl/cur)."""
    step_type: Literal["generator", "reflector", "curator"]
    sample_id: str
    timestamp: str = Field(default_factory=lambda: datetime.utcnow().isoformat())
    input_hash: str
    output_hash: str
    model_name: str
    seed: Optional[int] = None
    temperature: float
    prompt_version: str
    used_bullet_ids: List[str] = Field(default_factory=list)
    operations_applied: Optional[List[str]] = None  # For curator steps


class RunMetadata(StrictBaseModel):
    """Metadata for an entire run (offline or online)."""
    run_id: str
    run_type: Literal["offline", "online"]
    start_time: str
    end_time: Optional[str] = None
    config: Dict[str, Any]
    dataset_name: str
    num_samples: int
    playbook_hash_before: str
    playbook_hash_after: Optional[str] = None
    metrics: Dict[str, Any] = Field(default_factory=dict)
