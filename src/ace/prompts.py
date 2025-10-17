"""
Prompt templates for Generator, Reflector, and Curator agents.

All prompts enforce pure JSON output and track bullet_ids for
deterministic, auditable playbook evolution.
"""

import json
from typing import Any, Dict

import orjson


# Version for audit logging
PROMPT_VERSION = "v1.0.0"


def safe_json_dumps(obj: Any) -> str:
    """Safely serialize object to JSON string with sorted keys."""
    return orjson.dumps(
        obj,
        option=orjson.OPT_SORT_KEYS | orjson.OPT_INDENT_2
    ).decode('utf-8')


# ============================================================================
# Generator Prompts
# ============================================================================

GENERATOR_SYSTEM_PROMPT = """You are an analysis expert tasked with solving tasks using:
1. Your general knowledge
2. A curated Playbook of strategies, formulas, pitfalls, checklists, and examples
3. An optional Reflection summarizing prior mistakes and fixes

Instructions:
- Read the Playbook carefully and apply only relevant items
- Avoid known pitfalls explicitly
- If formulas or code snippets are relevant, use them appropriately
- Double-check your logic before providing the final answer
- Track which playbook item_ids you actually use in your reasoning
- Output ONLY valid JSON with no markdown code fences

Output format:
{
  "reasoning": "concise step-by-step analysis (no rambling)",
  "bullet_ids": ["<item_id>", "..."],
  "final_answer": "<string or JSON object>"
}"""


def create_generator_user_prompt(
    playbook_items: list,
    reflection: Dict[str, Any] | None,
    question: Dict[str, Any]
) -> str:
    """Create Generator user prompt with playbook, reflection, and question."""

    # Serialize playbook items
    playbook_data = {
        "items": [
            {
                "item_id": item.get("item_id") or item.get("item_id", ""),
                "category": item.get("category", ""),
                "title": item.get("title", ""),
                "content": item.get("content", ""),
                "tags": item.get("tags", [])
            }
            for item in playbook_items
        ]
    }

    prompt_data = {
        "playbook": playbook_data,
        "reflection": reflection if reflection else None,
        "question": question
    }

    return safe_json_dumps(prompt_data)


# ============================================================================
# Reflector Prompts
# ============================================================================

REFLECTOR_SYSTEM_PROMPT = """You are an expert analyst and educator. Your task is to:
1. Diagnose the gap between the model's prediction and ground truth (or environment feedback)
2. Identify concrete conceptual, calculation, or strategy issues
3. State what should have been done instead
4. Distill a portable key insight that can improve future performance
5. Tag each used playbook item as 'helpful', 'harmful', or 'neutral'

Instructions:
- Be specific about what went wrong and why
- Identify root causes, not just symptoms
- Provide actionable guidance for the correct approach
- Extract insights that generalize beyond this specific example
- Output ONLY valid JSON with no markdown code fences

Output format:
{
  "reasoning_summary": "brief summary of what happened",
  "error_identification": "what went wrong (if anything)",
  "root_cause_analysis": "why it happened",
  "correct_approach": "what to do next time",
  "key_insight": "portable principle",
  "bullet_tags": [
    {"bullet_id": "<id>", "tag": "helpful"},
    {"bullet_id": "<id>", "tag": "harmful"},
    {"bullet_id": "<id>", "tag": "neutral"}
  ]
}"""


def create_reflector_user_prompt(
    question: Dict[str, Any],
    predicted_answer: Any,
    ground_truth: Any | None,
    env_feedback: Dict[str, Any] | None,
    gen_reasoning: str,
    used_bullet_ids: list[str],
    playbook_subset: list
) -> str:
    """Create Reflector user prompt with all evaluation context."""

    prompt_data = {
        "question": question,
        "predicted_answer": predicted_answer,
        "ground_truth": ground_truth if ground_truth is not None else None,
        "env_feedback": env_feedback if env_feedback else None,
        "gen_reasoning": gen_reasoning,
        "used_bullet_ids": used_bullet_ids,
        "playbook_subset": [
            {
                "item_id": item.get("item_id", ""),
                "category": item.get("category", ""),
                "title": item.get("title", ""),
                "content": item.get("content", "")
            }
            for item in playbook_subset
        ]
    }

    return safe_json_dumps(prompt_data)


# ============================================================================
# Curator Prompts
# ============================================================================

CURATOR_SYSTEM_PROMPT = """You are a master curator of knowledge. Your task is to:
1. Analyze the current Playbook and the Reflection
2. Propose ONLY NEW items or precise AMENDMENTS that improve future predictions
3. Avoid redundancy - do NOT regenerate the entire Playbook
4. Be concise and actionable
5. Deprecate items that are proven harmful

Operation types:
- add: Create a new playbook item (will be checked for duplicates)
- amend: Update an existing item by ID (append content or add tags)
- deprecate: Mark an item as harmful/outdated

Instructions:
- Propose small, incremental improvements (delta updates)
- Each item should be 1-6 sentences, actionable and specific
- Prioritize quality over quantity
- Respond with VALID JSON ONLY (no markdown code fences)

Output format:
{
  "operations": [
    {
      "op": "add",
      "item": {
        "category": "strategy|formula|pitfall|checklist|example",
        "title": "concise title",
        "content": "actionable content (1-6 sentences)",
        "tags": ["tag1", "tag2"]
      }
    },
    {
      "op": "amend",
      "bullet_id": "<item_id>",
      "delta": {
        "content_append": "additional insight",
        "tags_add": ["new_tag"]
      }
    },
    {
      "op": "deprecate",
      "bullet_id": "<item_id>",
      "reason": "explanation of why this is harmful"
    }
  ]
}"""


def create_curator_user_prompt(
    token_budget: int,
    question_context: Dict[str, Any],
    current_playbook: Dict[str, Any],
    final_generated: Dict[str, Any],
    reflection: Dict[str, Any]
) -> str:
    """Create Curator user prompt with all context for delta generation."""

    prompt_data = {
        "token_budget": token_budget,
        "question_context": question_context,
        "current_playbook": current_playbook,
        "final_generated": final_generated,
        "reflection": reflection
    }

    return safe_json_dumps(prompt_data)


# ============================================================================
# JSON Repair Prompt
# ============================================================================

JSON_REPAIR_SYSTEM_PROMPT = """You are a JSON repair specialist. Your task is to:
1. Fix malformed JSON to make it valid
2. Preserve the original content and structure as much as possible
3. Do NOT add or remove fields
4. Do NOT change the meaning
5. Output ONLY the fixed JSON with no markdown code fences or explanations"""


def create_json_repair_prompt(broken_json: str) -> str:
    """Create prompt to repair broken JSON output."""
    return f"""The following JSON is malformed. Fix it and output ONLY the valid JSON:

{broken_json}"""


# ============================================================================
# Helper Functions
# ============================================================================

def estimate_token_count(text: str) -> int:
    """
    Rough estimate of token count.

    Uses simple heuristic: ~4 characters per token.
    For production, use tiktoken or similar.
    """
    return len(text) // 4


def truncate_playbook_for_budget(
    playbook_items: list,
    budget: int,
    prioritize_recent: bool = True
) -> list:
    """
    Truncate playbook items to fit within token budget.

    Args:
        playbook_items: List of playbook items (dicts)
        budget: Approximate token budget
        prioritize_recent: If True, keep most recently updated items

    Returns:
        Truncated list of items
    """
    if prioritize_recent:
        # Sort by updated_at descending
        sorted_items = sorted(
            playbook_items,
            key=lambda x: x.get("updated_at", ""),
            reverse=True
        )
    else:
        sorted_items = playbook_items[:]

    selected = []
    current_tokens = 0

    for item in sorted_items:
        item_text = json.dumps(item)
        item_tokens = estimate_token_count(item_text)

        if current_tokens + item_tokens > budget:
            break

        selected.append(item)
        current_tokens += item_tokens

    return selected
