"""
Tests for playbook management.
"""

import tempfile
from pathlib import Path

import pytest

from src.ace.models import CurOpAdd, CurOpAmend, CurOpDeprecate, CurOutput, PlaybookItemDraft
from src.ace.playbook import PlaybookStore


@pytest.fixture
def temp_store():
    """Create temporary playbook store for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        store = PlaybookStore(
            storage_path=str(Path(tmpdir) / "playbook.json"),
            dedup_similarity_threshold=0.92,
            harmful_threshold=3,
        )
        yield store


def test_playbook_store_init(temp_store):
    """Test playbook store initialization."""
    assert len(temp_store.playbook.items) == 0
    assert temp_store.dedup_threshold == 0.92
    assert temp_store.harmful_threshold == 3


def test_playbook_save_load(temp_store):
    """Test saving and loading playbook."""
    # Add item
    draft = PlaybookItemDraft(
        category="strategy",
        title="Test",
        content="Test content",
        tags=["test"]
    )
    op = CurOpAdd(item=draft)
    cur_output = CurOutput(operations=[op])
    temp_store.merge_operations(cur_output)

    # Save
    temp_store.save()

    # Create new store and load
    new_store = PlaybookStore(
        storage_path=temp_store.storage_path,
        dedup_similarity_threshold=0.92,
    )
    new_store.load()

    assert len(new_store.playbook.items) == 1
    assert new_store.playbook.items[0].title == "Test"


def test_merge_add_operation(temp_store):
    """Test adding new items."""
    draft = PlaybookItemDraft(
        category="formula",
        title="Simple Interest",
        content="I = P * r * t",
        tags=["finance"]
    )
    op = CurOpAdd(item=draft)
    cur_output = CurOutput(operations=[op])

    results = temp_store.merge_operations(cur_output)
    assert len(results) == 1
    assert "Added new item" in results[0]
    assert len(temp_store.playbook.items) == 1


def test_merge_duplicate_detection(temp_store):
    """Test deduplication of similar items."""
    # Add first item
    draft1 = PlaybookItemDraft(
        category="strategy",
        title="Test",
        content="This is a test strategy",
        tags=["test"]
    )
    op1 = CurOpAdd(item=draft1)
    temp_store.merge_operations(CurOutput(operations=[op1]))

    # Try to add very similar item
    draft2 = PlaybookItemDraft(
        category="strategy",
        title="Test",
        content="This is a test strategy.",  # Very similar (only punctuation different)
        tags=["example"]
    )
    op2 = CurOpAdd(item=draft2)
    results = temp_store.merge_operations(CurOutput(operations=[op2]))

    # Should convert to amend
    assert "amend" in results[0].lower() or "duplicate" in results[0].lower()
    assert len(temp_store.playbook.items) == 1


def test_merge_amend_operation(temp_store):
    """Test amending existing items."""
    # Add item first
    draft = PlaybookItemDraft(
        category="strategy",
        title="Test",
        content="Original content",
        tags=["test"]
    )
    op1 = CurOpAdd(item=draft)
    temp_store.merge_operations(CurOutput(operations=[op1]))
    item_id = temp_store.playbook.items[0].item_id

    # Amend it
    op2 = CurOpAmend(
        bullet_id=item_id,
        delta={"content_append": "Additional info.", "tags_add": ["updated"]}
    )
    results = temp_store.merge_operations(CurOutput(operations=[op2]))

    assert "Amended" in results[0]
    item = temp_store.playbook.items[0]
    assert "Additional info" in item.content
    assert "updated" in item.tags


def test_merge_deprecate_operation(temp_store):
    """Test deprecating items."""
    # Add item first
    draft = PlaybookItemDraft(
        category="strategy",
        title="Bad Strategy",
        content="This doesn't work",
    )
    op1 = CurOpAdd(item=draft)
    temp_store.merge_operations(CurOutput(operations=[op1]))
    item_id = temp_store.playbook.items[0].item_id

    # Deprecate it
    op2 = CurOpDeprecate(bullet_id=item_id, reason="Proven harmful")
    results = temp_store.merge_operations(CurOutput(operations=[op2]))

    assert "Deprecated" in results[0]
    item = temp_store.playbook.items[0]
    assert "deprecated" in item.tags
    assert item.harmful_count == 1


def test_update_bullet_stats(temp_store):
    """Test updating helpful/harmful counts."""
    # Add item
    draft = PlaybookItemDraft(
        category="strategy",
        title="Test",
        content="Test content",
    )
    op = CurOpAdd(item=draft)
    temp_store.merge_operations(CurOutput(operations=[op]))
    item_id = temp_store.playbook.items[0].item_id

    # Update stats
    bullet_tags = [
        {"bullet_id": item_id, "tag": "helpful"},
    ]
    temp_store.update_bullet_stats(bullet_tags)

    item = temp_store.playbook.items[0]
    assert item.helpful_count == 1
    assert item.harmful_count == 0


def test_filter_serving_items(temp_store):
    """Test filtering serving items."""
    # Add helpful item
    draft1 = PlaybookItemDraft(category="strategy", title="Good", content="Good content")
    temp_store.merge_operations(CurOutput(operations=[CurOpAdd(item=draft1)]))

    # Add harmful item
    draft2 = PlaybookItemDraft(category="strategy", title="Bad", content="Bad content")
    temp_store.merge_operations(CurOutput(operations=[CurOpAdd(item=draft2)]))
    temp_store.playbook.items[1].harmful_count = 5

    serving = temp_store.filter_serving_items()
    assert len(serving) == 1
    assert serving[0].title == "Good"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
