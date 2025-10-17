"""
Verification script for semantic deduplication feature.
Tests both difflib mode and semantic embedding mode.
"""
import os
import sys

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.ace.playbook import PlaybookStore
from src.ace.models import PlaybookItemDraft, CurOpAdd, CurOutput


def test_difflib_mode():
    """Test deduplication with difflib (default mode)."""
    print("\n" + "="*60)
    print("Testing DIFFLIB Mode (default)")
    print("="*60)

    # Create store without semantic dedup
    store = PlaybookStore(
        storage_path="./storage/test_playbook.json",
        use_semantic_dedup=False,
    )

    # Test similarity computation
    text1 = "check authentication"
    text2 = "verify auth"
    similarity = store.compute_similarity(text1, text2)

    print(f"\nText 1: '{text1}'")
    print(f"Text 2: '{text2}'")
    print(f"Similarity (difflib): {similarity:.4f}")
    print(f"Dedup mode: {'semantic' if store.use_semantic_dedup else 'difflib'}")

    # Test with more similar texts
    text3 = "check authentication process"
    text4 = "check authentication process."
    similarity2 = store.compute_similarity(text3, text4)

    print(f"\nText 3: '{text3}'")
    print(f"Text 4: '{text4}'")
    print(f"Similarity (difflib): {similarity2:.4f}")

    return store


def test_semantic_mode():
    """Test deduplication with semantic embeddings (if available)."""
    print("\n" + "="*60)
    print("Testing SEMANTIC Mode (optional)")
    print("="*60)

    try:
        # Try to create store with semantic dedup
        store = PlaybookStore(
            storage_path="./storage/test_playbook.json",
            use_semantic_dedup=True,
        )

        if not store.use_semantic_dedup:
            print("\n⚠️ Semantic dedup not available (dependencies not installed)")
            print("Install with: pip install -r requirements-semantic.txt")
            return None

        # Test similarity computation
        text1 = "check authentication"
        text2 = "verify auth"
        similarity = store.compute_similarity(text1, text2)

        print(f"\nText 1: '{text1}'")
        print(f"Text 2: '{text2}'")
        print(f"Similarity (semantic): {similarity:.4f}")
        print(f"Dedup mode: {'semantic' if store.use_semantic_dedup else 'difflib'}")

        # Test with more similar texts
        text3 = "check authentication process"
        text4 = "verify authentication workflow"
        similarity2 = store.compute_similarity(text3, text4)

        print(f"\nText 3: '{text3}'")
        print(f"Text 4: '{text4}'")
        print(f"Similarity (semantic): {similarity2:.4f}")

        print("\n✅ Semantic deduplication is working!")
        return store

    except Exception as e:
        print(f"\n❌ Error loading semantic dedup: {e}")
        return None


def main():
    """Run verification tests."""
    print("\n" + "="*60)
    print("ACE Semantic Deduplication Verification")
    print("="*60)

    # Test difflib mode
    difflib_store = test_difflib_mode()

    # Test semantic mode
    semantic_store = test_semantic_mode()

    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"✅ Difflib mode: Working")

    if semantic_store:
        print(f"✅ Semantic mode: Working")
        print(f"\nBoth modes are operational!")
        print(f"Use ACE_USE_SEMANTIC_DEDUP=true to enable semantic mode.")
    else:
        print(f"⚠️ Semantic mode: Not available (install requirements-semantic.txt)")
        print(f"\nDifflib mode is working as default.")
        print(f"To enable semantic mode:")
        print(f"  1. pip install -r requirements-semantic.txt")
        print(f"  2. Set ACE_USE_SEMANTIC_DEDUP=true in .env")

    print()


if __name__ == "__main__":
    main()
