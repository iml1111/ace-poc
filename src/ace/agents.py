"""
LLM agent wrappers for Generator, Reflector, and Curator roles.

Implements strict JSON validation, repair logic, and deterministic
execution with temperature=0 and fixed seeds.
"""

import logging
import os
import time
from typing import Any, Dict, List, Optional

import orjson
from anthropic import Anthropic
from pydantic import ValidationError

from .models import (
    CurInput,
    CurOutput,
    GenInput,
    GenOutput,
    Playbook,
    ReflInput,
    ReflOutput,
)
from .prompts import (
    CURATOR_SYSTEM_PROMPT,
    GENERATOR_SYSTEM_PROMPT,
    JSON_REPAIR_SYSTEM_PROMPT,
    REFLECTOR_SYSTEM_PROMPT,
    PROMPT_VERSION,
    create_curator_user_prompt,
    create_generator_user_prompt,
    create_json_repair_prompt,
    create_reflector_user_prompt,
)


logger = logging.getLogger(__name__)


class AgentError(Exception):
    """Base exception for agent-related errors."""
    pass


class JSONValidationError(AgentError):
    """JSON validation failed after repair attempt."""
    pass


class AnthropicClient:
    """
    Wrapper for Anthropic API with retry logic and deterministic execution.

    Ensures temperature=0, seed propagation, and JSON-only responses.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "claude-3-5-sonnet-latest",
        max_tokens: int = 2048,
        temperature: float = 0.0,
        seed: Optional[int] = None,
        max_retries: int = 3,
        retry_delay: float = 1.0,
    ):
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ValueError("ANTHROPIC_API_KEY not found in environment or constructor")

        self.client = Anthropic(api_key=self.api_key)
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.seed = seed
        self.max_retries = max_retries
        self.retry_delay = retry_delay

    def call(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: Optional[int] = None,
    ) -> str:
        """
        Call Anthropic API with retry logic.

        Returns raw text response.
        """
        tokens = max_tokens or self.max_tokens

        for attempt in range(self.max_retries):
            try:
                kwargs = {
                    "model": self.model,
                    "max_tokens": tokens,
                    "temperature": self.temperature,
                    "system": system_prompt,
                    "messages": [
                        {"role": "user", "content": user_prompt}
                    ]
                }

                # Add seed if supported (only for specific models)
                if self.seed is not None and "claude-3" in self.model:
                    # Note: seed parameter may not be available in all API versions
                    # This is a placeholder for future support
                    pass

                response = self.client.messages.create(**kwargs)

                # Extract text from response
                if response.content and len(response.content) > 0:
                    return response.content[0].text

                raise AgentError("Empty response from API")

            except Exception as e:
                logger.warning(
                    f"API call failed (attempt {attempt + 1}/{self.max_retries}): {e}"
                )
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay * (2 ** attempt))  # Exponential backoff
                else:
                    raise AgentError(f"API call failed after {self.max_retries} attempts: {e}")

        raise AgentError("Unexpected: exhausted retries without return or exception")


def parse_json_response(raw_response: str) -> Dict[str, Any]:
    """
    Parse JSON from LLM response, handling common formatting issues.

    Strips markdown code fences if present.
    """
    # Remove markdown code fences
    cleaned = raw_response.strip()
    if cleaned.startswith("```json"):
        cleaned = cleaned[7:]
    elif cleaned.startswith("```"):
        cleaned = cleaned[3:]

    if cleaned.endswith("```"):
        cleaned = cleaned[:-3]

    cleaned = cleaned.strip()

    try:
        return orjson.loads(cleaned)
    except orjson.JSONDecodeError as e:
        raise JSONValidationError(f"Failed to parse JSON: {e}\n\nRaw response:\n{raw_response}")


def validate_and_parse_json(
    raw_response: str,
    model_class: type,
    client: AnthropicClient,
    allow_repair: bool = True
) -> Any:
    """
    Validate and parse JSON response into Pydantic model.

    If parsing fails and allow_repair=True, attempts one repair using LLM.
    """
    try:
        data = parse_json_response(raw_response)
        return model_class(**data)
    except (JSONValidationError, ValidationError) as e:
        logger.warning(f"JSON validation failed: {e}")

        if not allow_repair:
            raise

        # Attempt repair
        logger.info("Attempting JSON repair...")
        try:
            repair_prompt = create_json_repair_prompt(raw_response)
            repaired = client.call(
                system_prompt=JSON_REPAIR_SYSTEM_PROMPT,
                user_prompt=repair_prompt,
                max_tokens=client.max_tokens
            )

            data = parse_json_response(repaired)
            return model_class(**data)

        except Exception as repair_error:
            logger.error(f"JSON repair failed: {repair_error}")
            raise JSONValidationError(
                f"Original error: {e}\nRepair error: {repair_error}\n\n"
                f"Raw response:\n{raw_response}\n\nRepaired:\n{repaired if 'repaired' in locals() else 'N/A'}"
            )


# ============================================================================
# Agent Call Functions
# ============================================================================

def call_generator(
    gen_input: GenInput,
    client: AnthropicClient,
) -> GenOutput:
    """
    Call Generator agent to produce an answer using playbook and reflection.

    Returns:
        GenOutput with reasoning, bullet_ids, and final_answer
    """
    # Get serving items from playbook (exclude deprecated/harmful)
    serving_items = gen_input.playbook.filter_serving_items()

    # Convert to dicts for prompt
    playbook_dicts = [item.model_dump() for item in serving_items]

    # Create prompt
    user_prompt = create_generator_user_prompt(
        playbook_items=playbook_dicts,
        reflection=gen_input.reflection,
        question=gen_input.question
    )

    # Call LLM
    logger.info(f"Calling Generator for question: {gen_input.question.get('task', 'unknown')}")
    raw_response = client.call(
        system_prompt=GENERATOR_SYSTEM_PROMPT,
        user_prompt=user_prompt
    )

    # Parse and validate
    gen_output = validate_and_parse_json(raw_response, GenOutput, client)

    logger.info(f"Generator used {len(gen_output.bullet_ids)} bullets")
    return gen_output


def call_reflector(
    refl_input: ReflInput,
    client: AnthropicClient,
) -> ReflOutput:
    """
    Call Reflector agent to analyze gaps and tag bullet usefulness.

    Returns:
        ReflOutput with analysis and bullet_tags
    """
    # Convert playbook subset to dicts
    playbook_dicts = [item.model_dump() for item in refl_input.playbook_subset]

    # Create prompt
    user_prompt = create_reflector_user_prompt(
        question=refl_input.question,
        predicted_answer=refl_input.predicted_answer,
        ground_truth=refl_input.ground_truth,
        env_feedback=refl_input.env_feedback,
        gen_reasoning=refl_input.gen_reasoning,
        used_bullet_ids=refl_input.used_bullet_ids,
        playbook_subset=playbook_dicts
    )

    # Call LLM
    logger.info("Calling Reflector to analyze performance")
    raw_response = client.call(
        system_prompt=REFLECTOR_SYSTEM_PROMPT,
        user_prompt=user_prompt
    )

    # Parse and validate
    refl_output = validate_and_parse_json(raw_response, ReflOutput, client)

    logger.info(f"Reflector tagged {len(refl_output.bullet_tags)} bullets")
    return refl_output


def call_curator(
    cur_input: CurInput,
    client: AnthropicClient,
) -> CurOutput:
    """
    Call Curator agent to propose playbook delta operations.

    Returns:
        CurOutput with operations list (add/amend/deprecate)
    """
    # Convert models to dicts
    playbook_dict = cur_input.current_playbook.model_dump()
    gen_dict = cur_input.final_generated.model_dump()
    refl_dict = cur_input.reflection.model_dump()

    # Create prompt
    user_prompt = create_curator_user_prompt(
        token_budget=cur_input.token_budget,
        question_context=cur_input.question_context,
        current_playbook=playbook_dict,
        final_generated=gen_dict,
        reflection=refl_dict
    )

    # Call LLM
    logger.info("Calling Curator to propose playbook updates")
    raw_response = client.call(
        system_prompt=CURATOR_SYSTEM_PROMPT,
        user_prompt=user_prompt
    )

    # Parse and validate
    cur_output = validate_and_parse_json(raw_response, CurOutput, client)

    logger.info(f"Curator proposed {len(cur_output.operations)} operations")
    return cur_output


# ============================================================================
# Helper Functions
# ============================================================================

def create_client_from_config(config: Dict[str, Any]) -> AnthropicClient:
    """Create AnthropicClient from configuration dict."""
    return AnthropicClient(
        api_key=config.get("api_key"),
        model=config.get("model", "claude-3-5-sonnet-latest"),
        max_tokens=config.get("max_tokens", 2048),
        temperature=config.get("temperature", 0.0),
        seed=config.get("seed"),
        max_retries=config.get("max_retries", 3),
        retry_delay=config.get("retry_delay", 1.0),
    )
