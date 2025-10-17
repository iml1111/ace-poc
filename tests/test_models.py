"""
Tests for ACE models and schemas.
"""

import pytest

from src.ace.models import (
    Playbook,
    PlaybookItem,
    PlaybookItemDraft,
    generate_item_id,
    normalize_text,
)


def test_normalize_text():
    """Test text normalization."""
    assert normalize_text("  Hello   World  ") == "hello world"
    assert normalize_text("HELLO") == "hello"
    assert normalize_text("Multiple   spaces") == "multiple spaces"


def test_generate_item_id_deterministic():
    """Test that item ID generation is deterministic."""
    id1 = generate_item_id("strategy", "Test Title", "Test content here")
    id2 = generate_item_id("strategy", "Test Title", "Test content here")
    assert id1 == id2

    # Different content should produce different IDs
    id3 = generate_item_id("strategy", "Test Title", "Different content")
    assert id1 != id3


def test_generate_item_id_normalized():
    """Test that ID generation uses normalization."""
    id1 = generate_item_id("strategy", "  Test  Title  ", "Test content")
    id2 = generate_item_id("strategy", "Test Title", "Test content")
    assert id1 == id2

    id3 = generate_item_id("strategy", "TEST TITLE", "TEST CONTENT")
    id4 = generate_item_id("strategy", "test title", "test content")
    assert id3 == id4


def test_playbook_item_creation():
    """Test PlaybookItem creation with validation."""
    item = PlaybookItem(
        item_id="abc123",
        category="strategy",
        title="Test Strategy",
        content="This is a test strategy for the framework.",
        tags=["test", "example"],
    )

    assert item.item_id == "abc123"
    assert item.category == "strategy"
    assert item.helpful_count == 0
    assert item.harmful_count == 0
    assert "test" in item.tags


def test_playbook_item_draft():
    """Test PlaybookItemDraft without ID."""
    draft = PlaybookItemDraft(
        category="formula",
        title="Test Formula",
        content="A = B * C",
        tags=["math"],
    )

    assert draft.category == "formula"
    assert draft.title == "Test Formula"


def test_playbook_filter_serving_items():
    """Test filtering of serving items."""
    playbook = Playbook(
        items=[
            PlaybookItem(
                item_id="1", category="strategy", title="Good", content="Good item",
                helpful_count=5, harmful_count=0
            ),
            PlaybookItem(
                item_id="2", category="strategy", title="Bad", content="Bad item",
                helpful_count=0, harmful_count=5, tags=["deprecated"]
            ),
            PlaybookItem(
                item_id="3", category="strategy", title="OK", content="OK item",
                helpful_count=2, harmful_count=1
            ),
        ]
    )

    serving = playbook.filter_serving_items(harmful_threshold=3)
    assert len(serving) == 2
    assert "1" in [item.item_id for item in serving]
    assert "3" in [item.item_id for item in serving]
    assert "2" not in [item.item_id for item in serving]


def test_playbook_get_item_by_id():
    """Test retrieving items by ID."""
    item1 = PlaybookItem(
        item_id="abc", category="strategy", title="Test", content="Content"
    )
    playbook = Playbook(items=[item1])

    found = playbook.get_item_by_id("abc")
    assert found is not None
    assert found.item_id == "abc"

    not_found = playbook.get_item_by_id("xyz")
    assert not_found is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
