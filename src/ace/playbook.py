"""
Playbook management: storage, merge operations, and deduplication logic.

Implements deterministic merge operations for evolving the Playbook
through delta updates from the Curator agent.
"""

import json
import logging
import os
from datetime import datetime
from difflib import SequenceMatcher
from pathlib import Path
from typing import List, Optional, Tuple

import orjson

from .models import (
    CurOpAdd,
    CurOpAmend,
    CurOpDeprecate,
    CurOutput,
    Playbook,
    PlaybookItem,
    PlaybookItemDraft,
    generate_item_id,
    normalize_text,
)

# Optional: Semantic similarity using sentence-transformers
try:
    from sentence_transformers import SentenceTransformer
    import numpy as np
    SEMANTIC_AVAILABLE = True
except ImportError:
    SEMANTIC_AVAILABLE = False

logger = logging.getLogger(__name__)


class PlaybookStore:
    """
    Manages Playbook storage and evolution through curator operations.

    Ensures deterministic, auditable updates with deduplication and
    harmful item filtering.
    """

    def __init__(
        self,
        storage_path: str = "./storage/playbook.json",
        dedup_similarity_threshold: float = 0.92,
        harmful_threshold: int = 3,
        use_semantic_dedup: Optional[bool] = None,
        embedding_model_name: str = "all-MiniLM-L6-v2",
    ):
        self.storage_path = Path(storage_path)
        self.dedup_threshold = dedup_similarity_threshold
        self.harmful_threshold = harmful_threshold
        self.playbook: Playbook = Playbook()

        # Ensure storage directory exists
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)

        # Determine whether to use semantic deduplication
        if use_semantic_dedup is None:
            # Read from environment variable
            use_semantic_str = os.getenv("ACE_USE_SEMANTIC_DEDUP", "false").lower()
            use_semantic_dedup = use_semantic_str in ("true", "1", "yes")

        self.use_semantic_dedup = use_semantic_dedup
        self.embedding_model = None

        # Initialize embedding model if requested and available
        if self.use_semantic_dedup:
            if not SEMANTIC_AVAILABLE:
                logger.warning(
                    "Semantic deduplication requested but sentence-transformers not installed. "
                    "Falling back to difflib-based deduplication. "
                    "Install with: pip install sentence-transformers"
                )
                self.use_semantic_dedup = False
            else:
                logger.info(f"Loading embedding model: {embedding_model_name}")
                try:
                    self.embedding_model = SentenceTransformer(embedding_model_name)
                    logger.info("✓ Semantic deduplication enabled (using sentence-transformers)")
                except Exception as e:
                    logger.warning(
                        f"Failed to load embedding model: {e}. "
                        f"Falling back to difflib-based deduplication."
                    )
                    self.use_semantic_dedup = False
        else:
            logger.info("Using difflib-based deduplication (default)")

    def load(self) -> Playbook:
        """Load playbook from storage, returns empty playbook if not found."""
        if not self.storage_path.exists():
            self.playbook = Playbook()
            return self.playbook

        try:
            with open(self.storage_path, 'rb') as f:
                data = orjson.loads(f.read())
                self.playbook = Playbook(**data)
            return self.playbook
        except Exception as e:
            raise RuntimeError(f"Failed to load playbook from {self.storage_path}: {e}")

    def save(self, playbook: Optional[Playbook] = None) -> None:
        """Save playbook to storage with deterministic formatting."""
        if playbook is not None:
            self.playbook = playbook

        try:
            with open(self.storage_path, 'wb') as f:
                f.write(
                    orjson.dumps(
                        self.playbook.model_dump(),
                        option=orjson.OPT_INDENT_2 | orjson.OPT_SORT_KEYS
                    )
                )
        except Exception as e:
            raise RuntimeError(f"Failed to save playbook to {self.storage_path}: {e}")

    def compute_similarity(self, text1: str, text2: str) -> float:
        """
        Compute similarity between two texts.

        Uses semantic embeddings if enabled and available, otherwise falls back to difflib.

        Returns ratio in [0.0, 1.0] where 1.0 is identical.
        """
        norm1 = normalize_text(text1)
        norm2 = normalize_text(text2)

        # Try semantic embedding if enabled
        if self.use_semantic_dedup and self.embedding_model is not None:
            try:
                # Encode texts to embeddings
                emb1 = self.embedding_model.encode([norm1], convert_to_numpy=True)
                emb2 = self.embedding_model.encode([norm2], convert_to_numpy=True)

                # Compute cosine similarity
                dot_product = float(np.dot(emb1[0], emb2[0]))
                norm_product = float(np.linalg.norm(emb1[0]) * np.linalg.norm(emb2[0]))
                similarity = dot_product / norm_product if norm_product > 0 else 0.0

                return similarity
            except Exception as e:
                logger.warning(
                    f"Semantic similarity computation failed: {e}. Using fallback."
                )
                # Fall through to difflib fallback

        # Fallback: difflib-based similarity
        return SequenceMatcher(None, norm1, norm2).ratio()

    def find_duplicate(self, draft: PlaybookItemDraft) -> Optional[PlaybookItem]:
        """
        Find duplicate item in playbook based on content similarity.

        Returns the first item with similarity >= threshold, or None.
        """
        for item in self.playbook.items:
            if item.category != draft.category:
                continue

            similarity = self.compute_similarity(item.content, draft.content)
            if similarity >= self.dedup_threshold:
                return item

        return None

    def _apply_add(self, op: CurOpAdd) -> Tuple[bool, str]:
        """
        Apply add operation with deduplication.

        If duplicate found, converts to amend operation.
        Returns (success, message).
        """
        draft = op.item
        duplicate = self.find_duplicate(draft)

        if duplicate:
            # Convert to amend operation
            new_tags = [tag for tag in draft.tags if tag not in duplicate.tags]
            if new_tags:
                duplicate.tags.extend(new_tags)
                duplicate.updated_at = datetime.utcnow().isoformat()
                return True, f"Converted add to amend for duplicate item {duplicate.item_id}"
            return False, f"Duplicate item {duplicate.item_id} already has all tags, skipping"

        # Create new item
        item_id = generate_item_id(draft.category, draft.title, draft.content)
        new_item = PlaybookItem(
            item_id=item_id,
            category=draft.category,
            title=draft.title,
            content=draft.content,
            tags=draft.tags,
            helpful_count=0,
            harmful_count=0,
        )
        self.playbook.items.append(new_item)
        return True, f"Added new item {item_id}"

    def _apply_amend(self, op: CurOpAmend) -> Tuple[bool, str]:
        """
        Apply amend operation to existing item.

        Supports content_append and tags_add.
        Returns (success, message).
        """
        item = self.playbook.get_item_by_id(op.bullet_id)
        if not item:
            return False, f"Item {op.bullet_id} not found for amend"

        if "deprecated" in item.tags:
            return False, f"Cannot amend deprecated item {op.bullet_id}"

        modified = False

        # Append content if provided
        if "content_append" in op.delta and op.delta["content_append"]:
            append_text = op.delta["content_append"].strip()
            # Check if content already contains this text (avoid redundancy)
            if append_text and append_text not in item.content:
                item.content = f"{item.content} {append_text}"
                modified = True

        # Add new tags
        if "tags_add" in op.delta and op.delta["tags_add"]:
            new_tags = [tag for tag in op.delta["tags_add"] if tag not in item.tags]
            if new_tags:
                item.tags.extend(new_tags)
                modified = True

        if modified:
            item.updated_at = datetime.utcnow().isoformat()
            return True, f"Amended item {op.bullet_id}"

        return False, f"No changes for item {op.bullet_id}"

    def _apply_deprecate(self, op: CurOpDeprecate) -> Tuple[bool, str]:
        """
        Apply deprecate operation.

        Marks item as deprecated and increments harmful_count.
        Returns (success, message).
        """
        item = self.playbook.get_item_by_id(op.bullet_id)
        if not item:
            return False, f"Item {op.bullet_id} not found for deprecation"

        if "deprecated" not in item.tags:
            item.tags.append("deprecated")

        item.harmful_count += 1
        item.updated_at = datetime.utcnow().isoformat()

        return True, f"Deprecated item {op.bullet_id}: {op.reason}"

    def merge_operations(
        self,
        cur_output: CurOutput,
        max_operations: int = 20
    ) -> List[str]:
        """
        Apply curator operations to playbook in deterministic order.

        Order:
        1. Deprecations (mark harmful items)
        2. Amendments (update existing items)
        3. Additions (with dedup check)

        Returns list of operation result messages.
        """
        if len(cur_output.operations) > max_operations:
            raise ValueError(
                f"Too many operations ({len(cur_output.operations)}), "
                f"max allowed: {max_operations}"
            )

        results: List[str] = []

        # Separate operations by type
        deprecations = [op for op in cur_output.operations if isinstance(op, CurOpDeprecate)]
        amendments = [op for op in cur_output.operations if isinstance(op, CurOpAmend)]
        additions = [op for op in cur_output.operations if isinstance(op, CurOpAdd)]

        # Apply in order: deprecate -> amend -> add
        for op in deprecations:
            success, msg = self._apply_deprecate(op)
            results.append(f"{'✓' if success else '✗'} {msg}")

        for op in amendments:
            success, msg = self._apply_amend(op)
            results.append(f"{'✓' if success else '✗'} {msg}")

        for op in additions:
            success, msg = self._apply_add(op)
            results.append(f"{'✓' if success else '✗'} {msg}")

        return results

    def update_bullet_stats(
        self,
        bullet_tags: List[dict]
    ) -> None:
        """
        Update helpful/harmful counts based on reflector tags.

        bullet_tags: List[{"bullet_id": str, "tag": "helpful"|"harmful"|"neutral"}]
        """
        for tag_info in bullet_tags:
            bullet_id = tag_info.get("bullet_id")
            tag = tag_info.get("tag")

            if not bullet_id or not tag:
                continue

            item = self.playbook.get_item_by_id(bullet_id)
            if not item:
                continue

            if tag == "helpful":
                item.helpful_count += 1
            elif tag == "harmful":
                item.harmful_count += 1
            # neutral: no change

            item.updated_at = datetime.utcnow().isoformat()

    def filter_serving_items(self) -> List[PlaybookItem]:
        """
        Get items suitable for serving to Generator.

        Excludes deprecated items and those with harmful_count >= threshold.
        """
        return self.playbook.filter_serving_items(self.harmful_threshold)

    def get_stats(self) -> dict:
        """Get playbook statistics."""
        total = len(self.playbook.items)
        serving = len(self.filter_serving_items())
        deprecated = sum(1 for item in self.playbook.items if "deprecated" in item.tags)
        harmful = sum(
            1 for item in self.playbook.items
            if item.harmful_count >= self.harmful_threshold
        )

        categories = {}
        for item in self.playbook.items:
            categories[item.category] = categories.get(item.category, 0) + 1

        return {
            "total_items": total,
            "serving_items": serving,
            "deprecated_items": deprecated,
            "harmful_items": harmful,
            "categories": categories,
        }

    def prune_deprecated(self, keep_threshold: int = 5) -> int:
        """
        Remove deprecated items with harmful_count > keep_threshold.

        Returns number of items removed.
        """
        before_count = len(self.playbook.items)
        self.playbook.items = [
            item for item in self.playbook.items
            if not ("deprecated" in item.tags and item.harmful_count > keep_threshold)
        ]
        removed = before_count - len(self.playbook.items)
        return removed

    def sort_items(self) -> None:
        """Sort items deterministically by (category, title, item_id)."""
        self.playbook.items.sort(
            key=lambda item: (
                item.category,
                normalize_text(item.title),
                item.item_id
            )
        )
