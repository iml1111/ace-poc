"""
Pipeline orchestration for ACE framework.

Implements offline (warm-up) and online (inference) adaptation modes
with full Generator → Reflector → Curator loops and audit logging.
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import orjson
from tqdm import tqdm

from .agents import AnthropicClient, call_curator, call_generator, call_reflector
from .datasets import add_sample_ids
from .evaluator import compute_aggregate_metrics, evaluate_sample
from .models import (
    CurInput,
    GenInput,
    Playbook,
    ReflInput,
    RunMetadata,
    StepLog,
    compute_hash,
)
from .playbook import PlaybookStore
from .prompts import PROMPT_VERSION


logger = logging.getLogger(__name__)


class PipelineConfig:
    """Configuration for pipeline execution."""

    def __init__(
        self,
        model: str = "claude-3-5-sonnet-latest",
        max_tokens: int = 2048,
        temperature: float = 0.0,
        seed: Optional[int] = 42,
        storage_dir: str = "./storage",
        runs_dir: str = "./runs",
        harmful_threshold: int = 3,
        dedup_similarity: float = 0.92,
        max_operations: int = 20,
        token_budget: int = 8000,
        early_stop_patience: int = 2,
        early_stop_delta: float = 0.01,
    ):
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.seed = seed
        self.storage_dir = storage_dir
        self.runs_dir = runs_dir
        self.harmful_threshold = harmful_threshold
        self.dedup_similarity = dedup_similarity
        self.max_operations = max_operations
        self.token_budget = token_budget
        self.early_stop_patience = early_stop_patience
        self.early_stop_delta = early_stop_delta

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging."""
        return {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "seed": self.seed,
            "storage_dir": self.storage_dir,
            "runs_dir": self.runs_dir,
            "harmful_threshold": self.harmful_threshold,
            "dedup_similarity": self.dedup_similarity,
            "max_operations": self.max_operations,
            "token_budget": self.token_budget,
            "early_stop_patience": self.early_stop_patience,
            "early_stop_delta": self.early_stop_delta,
        }


class Pipeline:
    """Main pipeline orchestrator for ACE framework."""

    def __init__(
        self,
        config: PipelineConfig,
        client: AnthropicClient,
        playbook_store: PlaybookStore,
    ):
        self.config = config
        self.client = client
        self.store = playbook_store

        # Setup logging directory
        self.run_id = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        self.run_dir = Path(config.runs_dir) / self.run_id
        self.run_dir.mkdir(parents=True, exist_ok=True)

        # Log files
        self.steps_log_path = self.run_dir / "steps.jsonl"
        self.reflections_log_path = self.run_dir / "reflections.jsonl"
        self.metadata_path = self.run_dir / "run_metadata.json"

        logger.info(f"Pipeline initialized: run_id={self.run_id}")

    def log_step(self, step_log: StepLog) -> None:
        """Append step log to JSONL file."""
        with open(self.steps_log_path, "ab") as f:
            f.write(orjson.dumps(step_log.model_dump()) + b"\n")

    def log_reflection(self, sample_id: str, reflection: Dict[str, Any]) -> None:
        """Append reflection to JSONL file."""
        entry = {
            "sample_id": sample_id,
            "timestamp": datetime.utcnow().isoformat(),
            "reflection": reflection,
        }
        with open(self.reflections_log_path, "ab") as f:
            f.write(orjson.dumps(entry) + b"\n")

    def process_sample(
        self,
        sample: Dict[str, Any],
        last_reflection: Optional[Dict[str, Any]] = None,
        run_curator: bool = True,
    ) -> Tuple[Any, Dict[str, Any], Optional[Dict[str, Any]]]:
        """
        Process a single sample through Generator → Reflector → Curator.

        Args:
            sample: Sample with 'question', 'ground_truth', 'sample_id'
            last_reflection: Previous reflection (optional)
            run_curator: Whether to run curator and update playbook

        Returns:
            (predicted_answer, evaluation_result, reflection_output)
        """
        sample_id = sample.get("sample_id", "unknown")
        question = sample["question"]
        ground_truth = sample.get("ground_truth")

        # Step 1: Generator
        gen_input = GenInput(
            playbook=self.store.playbook,
            reflection=last_reflection,
            question=question,
        )
        gen_output = call_generator(gen_input, self.client)

        # Log generator step
        gen_log = StepLog(
            step_type="generator",
            sample_id=sample_id,
            input_hash=compute_hash(gen_input.model_dump()),
            output_hash=compute_hash(gen_output.model_dump()),
            model_name=self.config.model,
            seed=self.config.seed,
            temperature=self.config.temperature,
            prompt_version=PROMPT_VERSION,
            used_bullet_ids=gen_output.bullet_ids,
        )
        self.log_step(gen_log)

        # Step 2: Evaluate
        eval_result = evaluate_sample(
            sample_id=sample_id,
            question=question,
            predicted=gen_output.final_answer,
            ground_truth=ground_truth,
        )

        # Prepare env_feedback for reflector
        env_feedback = {
            "correct": eval_result.correct,
            "score": eval_result.score,
            "details": eval_result.details,
        }

        # Get playbook subset (items actually used by generator)
        playbook_subset = [
            self.store.playbook.get_item_by_id(bid)
            for bid in gen_output.bullet_ids
        ]
        playbook_subset = [item for item in playbook_subset if item is not None]

        # Step 3: Reflector
        refl_input = ReflInput(
            question=question,
            predicted_answer=gen_output.final_answer,
            ground_truth=ground_truth,
            env_feedback=env_feedback,
            gen_reasoning=gen_output.reasoning,
            used_bullet_ids=gen_output.bullet_ids,
            playbook_subset=playbook_subset,
        )
        refl_output = call_reflector(refl_input, self.client)

        # Log reflector step
        refl_log = StepLog(
            step_type="reflector",
            sample_id=sample_id,
            input_hash=compute_hash(refl_input.model_dump()),
            output_hash=compute_hash(refl_output.model_dump()),
            model_name=self.config.model,
            seed=self.config.seed,
            temperature=self.config.temperature,
            prompt_version=PROMPT_VERSION,
            used_bullet_ids=gen_output.bullet_ids,
        )
        self.log_step(refl_log)

        # Log reflection
        self.log_reflection(sample_id, refl_output.model_dump())

        # Update bullet stats based on reflector tags
        bullet_tags_dict = [tag.model_dump() for tag in refl_output.bullet_tags]
        self.store.update_bullet_stats(bullet_tags_dict)

        # Step 4: Curator (if enabled)
        curator_output = None
        if run_curator:
            cur_input = CurInput(
                token_budget=self.config.token_budget,
                question_context=question,
                current_playbook=self.store.playbook,
                final_generated=gen_output,
                reflection=refl_output,
            )
            curator_output = call_curator(cur_input, self.client)

            # Log curator step
            cur_log = StepLog(
                step_type="curator",
                sample_id=sample_id,
                input_hash=compute_hash(cur_input.model_dump()),
                output_hash=compute_hash(curator_output.model_dump()),
                model_name=self.config.model,
                seed=self.config.seed,
                temperature=self.config.temperature,
                prompt_version=PROMPT_VERSION,
                operations_applied=[
                    f"{op.op}:{getattr(op, 'bullet_id', 'new')}"
                    for op in curator_output.operations
                ],
            )
            self.log_step(cur_log)

            # Apply operations to playbook
            try:
                merge_results = self.store.merge_operations(
                    curator_output, max_operations=self.config.max_operations
                )
                logger.info(f"Merge results for {sample_id}:")
                for result in merge_results:
                    logger.info(f"  {result}")
            except Exception as e:
                logger.error(f"Failed to merge operations for {sample_id}: {e}")

        return gen_output.final_answer, eval_result.to_dict(), refl_output.model_dump()

    def run_offline(
        self,
        train_data: List[Dict[str, Any]],
        dataset_name: str = "unknown",
        epochs: int = 1,
    ) -> Dict[str, Any]:
        """
        Run offline adaptation (warm-up) on training data.

        Iterates through training samples, updating playbook based on
        reflections and curator operations.

        Args:
            train_data: List of training samples
            dataset_name: Name of dataset for logging
            epochs: Number of epochs to run

        Returns:
            Summary with metrics and playbook statistics
        """
        logger.info(f"Starting offline adaptation: {len(train_data)} samples, {epochs} epochs")

        # Add sample IDs if not present
        if not train_data or "sample_id" not in train_data[0]:
            train_data = add_sample_ids(train_data, dataset_name, "train")

        # Initial playbook hash
        playbook_hash_before = self.store.playbook.compute_hash()

        # Track metrics per epoch
        epoch_metrics = []
        best_accuracy = 0.0
        patience_counter = 0

        for epoch in range(epochs):
            logger.info(f"\n{'='*60}\nEpoch {epoch + 1}/{epochs}\n{'='*60}")

            epoch_results = []
            last_reflection = None

            # Process each sample
            pbar = tqdm(train_data, desc=f"Epoch {epoch + 1}")
            for sample in pbar:
                predicted, eval_result, reflection = self.process_sample(
                    sample, last_reflection=last_reflection, run_curator=True
                )
                epoch_results.append(eval_result)
                last_reflection = reflection

                # Update progress bar
                correct = sum(1 for r in epoch_results if r["correct"])
                accuracy = correct / len(epoch_results)
                pbar.set_postfix({"accuracy": f"{accuracy:.2%}"})

            # Compute epoch metrics
            from .evaluator import EvaluationResult

            eval_objects = [
                EvaluationResult(
                    sample_id=r["sample_id"],
                    task=r["task"],
                    predicted=r["predicted"],
                    ground_truth=r["ground_truth"],
                    correct=r["correct"],
                    score=r["score"],
                    details=r["details"],
                )
                for r in epoch_results
            ]
            metrics = compute_aggregate_metrics(eval_objects)
            epoch_metrics.append(metrics)

            logger.info(f"Epoch {epoch + 1} results:")
            logger.info(f"  Accuracy: {metrics['accuracy']:.2%}")
            logger.info(f"  Avg Score: {metrics['average_score']:.3f}")
            logger.info(f"  Playbook size: {len(self.store.playbook.items)} items")

            # Early stopping check
            if metrics["accuracy"] > best_accuracy + self.config.early_stop_delta:
                best_accuracy = metrics["accuracy"]
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= self.config.early_stop_patience:
                    logger.info(f"Early stopping at epoch {epoch + 1}")
                    break

            # Save playbook after each epoch
            self.store.save()

        # Final playbook hash
        playbook_hash_after = self.store.playbook.compute_hash()

        # Create run metadata
        metadata = RunMetadata(
            run_id=self.run_id,
            run_type="offline",
            start_time=datetime.utcnow().isoformat(),
            config=self.config.to_dict(),
            dataset_name=dataset_name,
            num_samples=len(train_data),
            playbook_hash_before=playbook_hash_before,
            playbook_hash_after=playbook_hash_after,
            metrics={"epoch_metrics": epoch_metrics, "final": epoch_metrics[-1] if epoch_metrics else {}},
        )

        # Save metadata
        with open(self.metadata_path, "wb") as f:
            f.write(
                orjson.dumps(
                    metadata.model_dump(), option=orjson.OPT_INDENT_2 | orjson.OPT_SORT_KEYS
                )
            )

        logger.info(f"\nOffline adaptation complete!")
        logger.info(f"Results saved to: {self.run_dir}")
        logger.info(f"Playbook stats: {self.store.get_stats()}")

        return {
            "run_id": self.run_id,
            "run_dir": str(self.run_dir),
            "final_metrics": epoch_metrics[-1] if epoch_metrics else {},
            "playbook_stats": self.store.get_stats(),
        }

    def run_online(
        self,
        test_data: List[Dict[str, Any]],
        dataset_name: str = "unknown",
        enable_learning: bool = False,
    ) -> Dict[str, Any]:
        """
        Run online adaptation (inference) on test data.

        Performs inference with optional incremental learning.

        Args:
            test_data: List of test samples
            dataset_name: Name of dataset for logging
            enable_learning: If True, run curator and update playbook

        Returns:
            Summary with metrics and predictions
        """
        logger.info(
            f"Starting online adaptation: {len(test_data)} samples, "
            f"learning={'enabled' if enable_learning else 'disabled'}"
        )

        # Add sample IDs if not present
        if not test_data or "sample_id" not in test_data[0]:
            test_data = add_sample_ids(test_data, dataset_name, "test")

        playbook_hash_before = self.store.playbook.compute_hash()

        results = []
        predictions = []
        last_reflection = None

        pbar = tqdm(test_data, desc="Online Inference")
        for sample in pbar:
            predicted, eval_result, reflection = self.process_sample(
                sample, last_reflection=last_reflection, run_curator=enable_learning
            )
            results.append(eval_result)
            predictions.append({"sample_id": sample.get("sample_id"), "predicted": predicted})
            last_reflection = reflection if enable_learning else None

            # Update progress bar
            correct = sum(1 for r in results if r["correct"])
            accuracy = correct / len(results)
            pbar.set_postfix({"accuracy": f"{accuracy:.2%}"})

        # Compute final metrics
        from .evaluator import EvaluationResult

        eval_objects = [
            EvaluationResult(
                sample_id=r["sample_id"],
                task=r["task"],
                predicted=r["predicted"],
                ground_truth=r["ground_truth"],
                correct=r["correct"],
                score=r["score"],
                details=r["details"],
            )
            for r in results
        ]
        metrics = compute_aggregate_metrics(eval_objects)

        playbook_hash_after = self.store.playbook.compute_hash()

        # Save playbook if learning was enabled
        if enable_learning:
            self.store.save()

        # Create run metadata
        metadata = RunMetadata(
            run_id=self.run_id,
            run_type="online",
            start_time=datetime.utcnow().isoformat(),
            config=self.config.to_dict(),
            dataset_name=dataset_name,
            num_samples=len(test_data),
            playbook_hash_before=playbook_hash_before,
            playbook_hash_after=playbook_hash_after,
            metrics=metrics,
        )

        # Save metadata
        with open(self.metadata_path, "wb") as f:
            f.write(
                orjson.dumps(
                    metadata.model_dump(), option=orjson.OPT_INDENT_2 | orjson.OPT_SORT_KEYS
                )
            )

        logger.info(f"\nOnline adaptation complete!")
        logger.info(f"Accuracy: {metrics['accuracy']:.2%}")
        logger.info(f"Results saved to: {self.run_dir}")

        return {
            "run_id": self.run_id,
            "run_dir": str(self.run_dir),
            "metrics": metrics,
            "predictions": predictions,
            "playbook_stats": self.store.get_stats(),
        }
