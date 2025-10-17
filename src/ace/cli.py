"""
Command-line interface for ACE framework.

Provides commands for offline adaptation, online inference,
and playbook management.
"""

import argparse
import logging
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

from .agents import AnthropicClient
from .datasets import get_all_datasets, get_dataset, get_dataset_info
from .pipeline import Pipeline, PipelineConfig
from .playbook import PlaybookStore


# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)


def load_config_from_env() -> dict:
    """Load configuration from environment variables."""
    load_dotenv()

    # Parse semantic dedup setting
    use_semantic_str = os.getenv("ACE_USE_SEMANTIC_DEDUP", "false").lower()
    use_semantic_dedup = use_semantic_str in ("true", "1", "yes")

    return {
        "api_key": os.getenv("ANTHROPIC_API_KEY"),
        "model": os.getenv("ACE_MODEL", "claude-3-5-sonnet-latest"),
        "max_tokens": int(os.getenv("ACE_MAX_TOKENS", "2048")),
        "temperature": float(os.getenv("ACE_TEMPERATURE", "0.0")),
        "seed": int(os.getenv("ACE_SEED", "42")) if os.getenv("ACE_SEED") else None,
        "storage_dir": os.getenv("ACE_STORAGE_DIR", "./storage"),
        "runs_dir": os.getenv("ACE_RUNS_DIR", "./runs"),
        "harmful_threshold": int(os.getenv("ACE_HARMFUL_THRESHOLD", "3")),
        "dedup_similarity": float(os.getenv("ACE_DEDUP_SIMILARITY", "0.92")),
        "max_operations": int(os.getenv("ACE_MAX_OPERATIONS_PER_CURATOR", "20")),
        "use_semantic_dedup": use_semantic_dedup,
        "embedding_model": os.getenv("ACE_EMBEDDING_MODEL", "all-MiniLM-L6-v2"),
    }


def cmd_offline(args):
    """Run offline adaptation (warm-up) on training data."""
    logger.info("Starting offline adaptation...")

    # Load config
    env_config = load_config_from_env()

    # Override with CLI args
    config = PipelineConfig(
        model=args.model or env_config["model"],
        max_tokens=args.max_tokens or env_config["max_tokens"],
        temperature=env_config["temperature"],
        seed=args.seed if args.seed is not None else env_config["seed"],
        storage_dir=env_config["storage_dir"],
        runs_dir=env_config["runs_dir"],
        harmful_threshold=env_config["harmful_threshold"],
        dedup_similarity=env_config["dedup_similarity"],
        max_operations=env_config["max_operations"],
        early_stop_patience=args.patience,
        early_stop_delta=args.early_stop_delta,
    )

    # Create client
    client = AnthropicClient(
        api_key=env_config["api_key"],
        model=config.model,
        max_tokens=config.max_tokens,
        temperature=config.temperature,
        seed=config.seed,
    )

    # Create/load playbook store
    playbook_path = Path(config.storage_dir) / "playbook.json"
    store = PlaybookStore(
        storage_path=str(playbook_path),
        dedup_similarity_threshold=config.dedup_similarity,
        harmful_threshold=config.harmful_threshold,
        use_semantic_dedup=env_config["use_semantic_dedup"],
        embedding_model_name=env_config["embedding_model"],
    )

    if args.reset:
        logger.warning("Resetting playbook (starting fresh)")
        store.playbook.items = []
    else:
        logger.info("Loading existing playbook...")
        store.load()
        logger.info(f"Loaded {len(store.playbook.items)} items")

    # Load dataset
    if args.dataset == "all":
        logger.info("Running on all datasets")
        all_train = get_all_datasets("train")
        for dataset_name, train_data in all_train.items():
            logger.info(f"\n{'='*60}\nDataset: {dataset_name}\n{'='*60}")
            pipeline = Pipeline(config, client, store)
            result = pipeline.run_offline(
                train_data=train_data, dataset_name=dataset_name, epochs=args.epochs
            )
            print_results(result)
    else:
        train_data = get_dataset(args.dataset, "train")
        pipeline = Pipeline(config, client, store)
        result = pipeline.run_offline(
            train_data=train_data, dataset_name=args.dataset, epochs=args.epochs
        )
        print_results(result)

    logger.info(f"\nPlaybook saved to: {playbook_path}")


def cmd_online(args):
    """Run online adaptation (inference) on test data."""
    logger.info("Starting online inference...")

    # Load config
    env_config = load_config_from_env()

    config = PipelineConfig(
        model=args.model or env_config["model"],
        max_tokens=args.max_tokens or env_config["max_tokens"],
        temperature=env_config["temperature"],
        seed=args.seed if args.seed is not None else env_config["seed"],
        storage_dir=env_config["storage_dir"],
        runs_dir=env_config["runs_dir"],
        harmful_threshold=env_config["harmful_threshold"],
        dedup_similarity=env_config["dedup_similarity"],
        max_operations=env_config["max_operations"],
    )

    # Create client
    client = AnthropicClient(
        api_key=env_config["api_key"],
        model=config.model,
        max_tokens=config.max_tokens,
        temperature=config.temperature,
        seed=config.seed,
    )

    # Load playbook store
    playbook_path = Path(config.storage_dir) / "playbook.json"
    store = PlaybookStore(
        storage_path=str(playbook_path),
        dedup_similarity_threshold=config.dedup_similarity,
        harmful_threshold=config.harmful_threshold,
        use_semantic_dedup=env_config["use_semantic_dedup"],
        embedding_model_name=env_config["embedding_model"],
    )

    logger.info("Loading playbook...")
    store.load()
    logger.info(f"Loaded {len(store.playbook.items)} items")

    if len(store.playbook.items) == 0:
        logger.warning(
            "Playbook is empty! Consider running offline adaptation first "
            "to warm up the playbook."
        )

    # Load dataset
    if args.dataset == "all":
        logger.info("Running on all datasets")
        all_test = get_all_datasets("test")
        for dataset_name, test_data in all_test.items():
            logger.info(f"\n{'='*60}\nDataset: {dataset_name}\n{'='*60}")
            pipeline = Pipeline(config, client, store)
            result = pipeline.run_online(
                test_data=test_data,
                dataset_name=dataset_name,
                enable_learning=args.enable_learning,
            )
            print_results(result)
    else:
        test_data = get_dataset(args.dataset, "test")
        pipeline = Pipeline(config, client, store)
        result = pipeline.run_online(
            test_data=test_data,
            dataset_name=args.dataset,
            enable_learning=args.enable_learning,
        )
        print_results(result)


def cmd_stats(args):
    """Show playbook statistics."""
    env_config = load_config_from_env()
    playbook_path = Path(env_config["storage_dir"]) / "playbook.json"

    store = PlaybookStore(storage_path=str(playbook_path))
    store.load()

    stats = store.get_stats()

    print("\n" + "=" * 60)
    print("PLAYBOOK STATISTICS")
    print("=" * 60)
    print(f"Total items:      {stats['total_items']}")
    print(f"Serving items:    {stats['serving_items']}")
    print(f"Deprecated items: {stats['deprecated_items']}")
    print(f"Harmful items:    {stats['harmful_items']}")
    print("\nBy category:")
    for category, count in sorted(stats["categories"].items()):
        print(f"  {category:12s}: {count}")

    if args.verbose:
        print("\n" + "=" * 60)
        print("ITEM DETAILS")
        print("=" * 60)
        for item in store.playbook.items:
            status = "deprecated" if "deprecated" in item.tags else "active"
            print(
                f"\n[{item.item_id}] {item.category} - {status}\n"
                f"  Title: {item.title}\n"
                f"  Helpful: {item.helpful_count} | Harmful: {item.harmful_count}\n"
                f"  Tags: {', '.join(item.tags) if item.tags else 'none'}"
            )


def cmd_list_datasets(args):
    """List available datasets."""
    info = get_dataset_info()

    print("\n" + "=" * 60)
    print("AVAILABLE DATASETS")
    print("=" * 60)

    for name, details in info.items():
        print(f"\n{name}:")
        print(f"  Description: {details['description']}")
        print(f"  Train: {details['train_size']} samples")
        print(f"  Test:  {details['test_size']} samples")
        print(f"  Total: {details['total_size']} samples")


def print_results(result: dict):
    """Print formatted results."""
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"Run ID:     {result['run_id']}")
    print(f"Run Dir:    {result['run_dir']}")

    if "final_metrics" in result:
        metrics = result["final_metrics"]
    elif "metrics" in result:
        metrics = result["metrics"]
    else:
        metrics = {}

    if metrics:
        print(f"\nAccuracy:   {metrics.get('accuracy', 0):.2%}")
        print(f"Avg Score:  {metrics.get('average_score', 0):.3f}")
        print(f"Correct:    {metrics.get('correct', 0)} / {metrics.get('total', 0)}")

        if "by_task" in metrics:
            print("\nBy Task:")
            for task, task_metrics in metrics["by_task"].items():
                print(
                    f"  {task:20s}: {task_metrics['accuracy']:.2%} "
                    f"({task_metrics['correct']}/{task_metrics['total']})"
                )

    if "playbook_stats" in result:
        stats = result["playbook_stats"]
        print(f"\nPlaybook:")
        print(f"  Total items:   {stats['total_items']}")
        print(f"  Serving items: {stats['serving_items']}")


def main():
    """Main entry point for CLI."""
    parser = argparse.ArgumentParser(
        description="ACE Framework - Agentic Context Engineering",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Offline command
    offline_parser = subparsers.add_parser(
        "offline", help="Run offline adaptation (warm-up) on training data"
    )
    offline_parser.add_argument(
        "--dataset",
        type=str,
        default="all",
        choices=["labeling", "numeric", "code_agent", "all"],
        help="Dataset to use (default: all)",
    )
    offline_parser.add_argument(
        "--epochs", type=int, default=2, help="Number of epochs (default: 2)"
    )
    offline_parser.add_argument(
        "--model", type=str, help="Model name (default: from env or claude-3-5-sonnet-latest)"
    )
    offline_parser.add_argument("--max-tokens", type=int, help="Max tokens per response")
    offline_parser.add_argument("--seed", type=int, help="Random seed for reproducibility")
    offline_parser.add_argument(
        "--patience", type=int, default=2, help="Early stopping patience (default: 2)"
    )
    offline_parser.add_argument(
        "--early-stop-delta",
        type=float,
        default=0.01,
        help="Early stopping delta (default: 0.01)",
    )
    offline_parser.add_argument(
        "--reset", action="store_true", help="Reset playbook (start from scratch)"
    )
    offline_parser.set_defaults(func=cmd_offline)

    # Online command
    online_parser = subparsers.add_parser(
        "online", help="Run online adaptation (inference) on test data"
    )
    online_parser.add_argument(
        "--dataset",
        type=str,
        default="all",
        choices=["labeling", "numeric", "code_agent", "all"],
        help="Dataset to use (default: all)",
    )
    online_parser.add_argument(
        "--model", type=str, help="Model name (default: from env or claude-3-5-sonnet-latest)"
    )
    online_parser.add_argument("--max-tokens", type=int, help="Max tokens per response")
    online_parser.add_argument("--seed", type=int, help="Random seed for reproducibility")
    online_parser.add_argument(
        "--enable-learning",
        action="store_true",
        help="Enable incremental learning during inference",
    )
    online_parser.set_defaults(func=cmd_online)

    # Stats command
    stats_parser = subparsers.add_parser("stats", help="Show playbook statistics")
    stats_parser.add_argument("--verbose", "-v", action="store_true", help="Show detailed item info")
    stats_parser.set_defaults(func=cmd_stats)

    # List datasets command
    list_parser = subparsers.add_parser("list-datasets", help="List available datasets")
    list_parser.set_defaults(func=cmd_list_datasets)

    # Parse args
    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    # Run command
    try:
        args.func(args)
    except Exception as e:
        logger.error(f"Command failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
