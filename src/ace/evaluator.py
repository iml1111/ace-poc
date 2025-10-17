"""
Evaluation logic for ACE framework.

Provides task-specific metrics and accuracy tracking across
different dataset types.
"""

import logging
from collections import Counter
from statistics import mode as stats_mode, median as stats_median
from typing import Any, Dict, List, Optional, Tuple


logger = logging.getLogger(__name__)


class EvaluationResult:
    """Result of evaluating a single sample."""

    def __init__(
        self,
        sample_id: str,
        task: str,
        predicted: Any,
        ground_truth: Any,
        correct: bool,
        score: float,
        details: Optional[Dict[str, Any]] = None
    ):
        self.sample_id = sample_id
        self.task = task
        self.predicted = predicted
        self.ground_truth = ground_truth
        self.correct = correct
        self.score = score  # 0.0 to 1.0
        self.details = details or {}

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging."""
        return {
            "sample_id": self.sample_id,
            "task": self.task,
            "predicted": self.predicted,
            "ground_truth": self.ground_truth,
            "correct": self.correct,
            "score": self.score,
            "details": self.details
        }


# ============================================================================
# Task-Specific Evaluators
# ============================================================================

def evaluate_labeling(
    predicted: Any,
    ground_truth: Dict[str, Any]
) -> Tuple[bool, float, Dict[str, Any]]:
    """
    Evaluate named entity labeling task.

    Uses set-based comparison of (text, label) tuples for partial credit.
    """
    try:
        gt_spans = ground_truth.get("spans", [])
        pred_spans = []

        # Extract spans from prediction
        if isinstance(predicted, dict) and "spans" in predicted:
            pred_spans = predicted["spans"]
        elif isinstance(predicted, str):
            # Try to parse as simple format
            logger.warning(f"Predicted answer is string, not structured spans: {predicted}")
            return False, 0.0, {"error": "Invalid format"}

        # Create sets of (text, label) tuples for comparison
        gt_set = {(span["text"], span["label"]) for span in gt_spans}
        pred_set = {(span["text"], span["label"]) for span in pred_spans if "text" in span and "label" in span}

        # Calculate metrics
        if not gt_set:
            return True if not pred_set else False, 1.0 if not pred_set else 0.0, {}

        true_positives = len(gt_set & pred_set)
        false_positives = len(pred_set - gt_set)
        false_negatives = len(gt_set - pred_set)

        precision = true_positives / len(pred_set) if pred_set else 0.0
        recall = true_positives / len(gt_set) if gt_set else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        correct = gt_set == pred_set
        score = f1  # Use F1 as score

        details = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "tp": true_positives,
            "fp": false_positives,
            "fn": false_negatives
        }

        return correct, score, details

    except Exception as e:
        logger.error(f"Error evaluating labeling task: {e}")
        return False, 0.0, {"error": str(e)}


def evaluate_numeric(
    predicted: Any,
    ground_truth: Dict[str, Any],
    tolerance: float = 1e-6
) -> Tuple[bool, float, Dict[str, Any]]:
    """
    Evaluate numeric calculation task.

    Uses absolute tolerance for floating point comparison.
    """
    try:
        gt_answer = ground_truth.get("answer")
        pred_answer = None

        # Extract numeric answer from prediction
        if isinstance(predicted, dict):
            pred_answer = predicted.get("answer")
        elif isinstance(predicted, (int, float)):
            pred_answer = predicted
        elif isinstance(predicted, str):
            try:
                pred_answer = float(predicted)
            except ValueError:
                logger.warning(f"Could not parse predicted answer as number: {predicted}")
                return False, 0.0, {"error": "Invalid numeric format"}

        if pred_answer is None:
            return False, 0.0, {"error": "No answer found in prediction"}

        # Compare with tolerance
        difference = abs(float(pred_answer) - float(gt_answer))
        correct = difference <= tolerance

        # Score based on relative error
        if gt_answer != 0:
            relative_error = difference / abs(gt_answer)
            score = max(0.0, 1.0 - relative_error)
        else:
            score = 1.0 if correct else 0.0

        details = {
            "predicted_answer": pred_answer,
            "ground_truth_answer": gt_answer,
            "difference": difference,
            "tolerance": tolerance
        }

        return correct, score, details

    except Exception as e:
        logger.error(f"Error evaluating numeric task: {e}")
        return False, 0.0, {"error": str(e)}


def evaluate_code_agent(
    predicted: Any,
    ground_truth: Dict[str, Any],
    question: Optional[Dict[str, Any]] = None
) -> Tuple[bool, float, Dict[str, Any]]:
    """
    Evaluate code agent list operation task.

    Uses exact match for discrete answers. Can also execute operations
    to verify correctness.
    """
    try:
        gt_answer = ground_truth.get("answer")
        pred_answer = None

        # Extract answer from prediction
        if isinstance(predicted, dict):
            pred_answer = predicted.get("answer")
        elif isinstance(predicted, (int, float, list)):
            pred_answer = predicted
        elif isinstance(predicted, str):
            try:
                pred_answer = int(predicted)
            except ValueError:
                try:
                    pred_answer = float(predicted)
                except ValueError:
                    logger.warning(f"Could not parse predicted answer: {predicted}")
                    return False, 0.0, {"error": "Invalid answer format"}

        if pred_answer is None:
            return False, 0.0, {"error": "No answer found in prediction"}

        # Exact match for integers, close match for floats
        if isinstance(gt_answer, int) and isinstance(pred_answer, (int, float)):
            correct = int(pred_answer) == gt_answer
        elif isinstance(gt_answer, float):
            correct = abs(pred_answer - gt_answer) < 1e-6
        else:
            correct = pred_answer == gt_answer

        score = 1.0 if correct else 0.0

        details = {
            "predicted_answer": pred_answer,
            "ground_truth_answer": gt_answer
        }

        # Optionally verify by re-executing operation
        if question and not correct:
            op = question.get("op")
            input_list = question.get("input", [])
            try:
                verified_answer = execute_list_operation(input_list, op)
                details["verified_answer"] = verified_answer
                if verified_answer == pred_answer:
                    details["note"] = "Prediction matches verified execution (GT may be wrong)"
            except Exception as exec_error:
                details["execution_error"] = str(exec_error)

        return correct, score, details

    except Exception as e:
        logger.error(f"Error evaluating code agent task: {e}")
        return False, 0.0, {"error": str(e)}


def execute_list_operation(input_list: List, op: str) -> Any:
    """Execute a list operation for verification."""
    if op == "sum":
        return sum(input_list)
    elif op == "max":
        return max(input_list)
    elif op == "min":
        return min(input_list)
    elif op == "mean":
        return sum(input_list) / len(input_list)
    elif op == "median":
        return stats_median(input_list)
    elif op == "mode":
        counter = Counter(input_list)
        return counter.most_common(1)[0][0]
    else:
        raise ValueError(f"Unknown operation: {op}")


# ============================================================================
# Main Evaluator
# ============================================================================

def evaluate_sample(
    sample_id: str,
    question: Dict[str, Any],
    predicted: Any,
    ground_truth: Any
) -> EvaluationResult:
    """
    Evaluate a single sample based on task type.

    Routes to appropriate task-specific evaluator.
    """
    task = question.get("task", "unknown")

    if task == "label_spans":
        correct, score, details = evaluate_labeling(predicted, ground_truth)
    elif task == "finance_compute":
        correct, score, details = evaluate_numeric(predicted, ground_truth)
    elif task == "list_aggregate":
        correct, score, details = evaluate_code_agent(predicted, ground_truth, question)
    else:
        logger.warning(f"Unknown task type: {task}, using exact match")
        correct = predicted == ground_truth.get("answer", ground_truth)
        score = 1.0 if correct else 0.0
        details = {}

    return EvaluationResult(
        sample_id=sample_id,
        task=task,
        predicted=predicted,
        ground_truth=ground_truth,
        correct=correct,
        score=score,
        details=details
    )


def compute_aggregate_metrics(results: List[EvaluationResult]) -> Dict[str, Any]:
    """
    Compute aggregate metrics across all results.

    Returns accuracy, average score, and per-task breakdown.
    """
    if not results:
        return {
            "total": 0,
            "accuracy": 0.0,
            "average_score": 0.0,
            "by_task": {}
        }

    total = len(results)
    correct = sum(1 for r in results if r.correct)
    accuracy = correct / total if total > 0 else 0.0
    avg_score = sum(r.score for r in results) / total if total > 0 else 0.0

    # Group by task
    by_task = {}
    for result in results:
        task = result.task
        if task not in by_task:
            by_task[task] = {
                "total": 0,
                "correct": 0,
                "scores": []
            }

        by_task[task]["total"] += 1
        if result.correct:
            by_task[task]["correct"] += 1
        by_task[task]["scores"].append(result.score)

    # Compute task-specific metrics
    for task, stats in by_task.items():
        stats["accuracy"] = stats["correct"] / stats["total"] if stats["total"] > 0 else 0.0
        stats["average_score"] = sum(stats["scores"]) / len(stats["scores"]) if stats["scores"] else 0.0
        del stats["scores"]  # Remove raw scores to keep output clean

    return {
        "total": total,
        "correct": correct,
        "accuracy": accuracy,
        "average_score": avg_score,
        "by_task": by_task
    }


def compare_metrics(
    baseline: Dict[str, Any],
    current: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Compare current metrics against baseline.

    Returns improvement percentages and deltas.
    """
    comparison = {
        "accuracy_delta": current["accuracy"] - baseline["accuracy"],
        "accuracy_improvement_pct": (
            (current["accuracy"] - baseline["accuracy"]) / baseline["accuracy"] * 100
            if baseline["accuracy"] > 0 else 0.0
        ),
        "score_delta": current["average_score"] - baseline["average_score"],
    }

    return comparison
