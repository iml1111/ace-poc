"""
Toy datasets for ACE framework evaluation.

Provides three task types with train/test splits:
1. LABELING: Named entity recognition (FiNER-like)
2. NUMERIC: Formula-based calculations (finance-like)
3. CODE_AGENT: List operations (AppWorld-like)
"""

from typing import Any, Dict, List, Tuple


# ============================================================================
# Dataset 1: LABELING (Named Entity Recognition)
# ============================================================================

LABELING_TRAIN = [
    {
        "question": {
            "task": "label_spans",
            "text": "Revenue grew to $1.2M in 2024.",
            "labels": ["MONEY", "DATE"]
        },
        "ground_truth": {
            "spans": [
                {"text": "$1.2M", "label": "MONEY", "start": 16, "end": 21},
                {"text": "2024", "label": "DATE", "start": 25, "end": 29}
            ]
        }
    },
    {
        "question": {
            "task": "label_spans",
            "text": "Apple Inc. reported Q3 earnings of $89.5B on July 28, 2024.",
            "labels": ["ORG", "MONEY", "DATE"]
        },
        "ground_truth": {
            "spans": [
                {"text": "Apple Inc.", "label": "ORG", "start": 0, "end": 10},
                {"text": "$89.5B", "label": "MONEY", "start": 35, "end": 41},
                {"text": "July 28, 2024", "label": "DATE", "start": 45, "end": 58}
            ]
        }
    },
    {
        "question": {
            "task": "label_spans",
            "text": "The CEO will visit Tokyo on March 15th to meet investors.",
            "labels": ["LOCATION", "DATE"]
        },
        "ground_truth": {
            "spans": [
                {"text": "Tokyo", "label": "LOCATION", "start": 19, "end": 24},
                {"text": "March 15th", "label": "DATE", "start": 28, "end": 38}
            ]
        }
    },
]

LABELING_TEST = [
    {
        "question": {
            "task": "label_spans",
            "text": "Microsoft acquired LinkedIn for $26.2B in June 2016.",
            "labels": ["ORG", "MONEY", "DATE"]
        },
        "ground_truth": {
            "spans": [
                {"text": "Microsoft", "label": "ORG", "start": 0, "end": 9},
                {"text": "LinkedIn", "label": "ORG", "start": 19, "end": 27},
                {"text": "$26.2B", "label": "MONEY", "start": 32, "end": 38},
                {"text": "June 2016", "label": "DATE", "start": 42, "end": 51}
            ]
        }
    },
    {
        "question": {
            "task": "label_spans",
            "text": "The conference in Paris starts on October 5th with €500 tickets.",
            "labels": ["LOCATION", "DATE", "MONEY"]
        },
        "ground_truth": {
            "spans": [
                {"text": "Paris", "label": "LOCATION", "start": 18, "end": 23},
                {"text": "October 5th", "label": "DATE", "start": 34, "end": 45},
                {"text": "€500", "label": "MONEY", "start": 51, "end": 55}
            ]
        }
    },
]


# ============================================================================
# Dataset 2: NUMERIC (Formula-based Calculations)
# ============================================================================

NUMERIC_TRAIN = [
    {
        "question": {
            "task": "finance_compute",
            "formula": "simple_interest",
            "inputs": {"principal": 1000, "rate_pct": 5, "years": 2}
        },
        "ground_truth": {
            "answer": 100.0,
            "formula": "principal * (rate_pct / 100) * years"
        }
    },
    {
        "question": {
            "task": "finance_compute",
            "formula": "compound_interest",
            "inputs": {"principal": 1000, "rate_pct": 5, "years": 2, "n": 1}
        },
        "ground_truth": {
            "answer": 102.5,
            "formula": "principal * ((1 + rate_pct/100/n)**(n*years)) - principal"
        }
    },
    {
        "question": {
            "task": "finance_compute",
            "formula": "profit_margin",
            "inputs": {"revenue": 1000, "cost": 600}
        },
        "ground_truth": {
            "answer": 40.0,
            "formula": "(revenue - cost) / revenue * 100"
        }
    },
    {
        "question": {
            "task": "finance_compute",
            "formula": "return_on_investment",
            "inputs": {"gain": 500, "cost": 2000}
        },
        "ground_truth": {
            "answer": 25.0,
            "formula": "(gain - cost) / cost * 100"
        }
    },
]

NUMERIC_TEST = [
    {
        "question": {
            "task": "finance_compute",
            "formula": "simple_interest",
            "inputs": {"principal": 5000, "rate_pct": 3.5, "years": 3}
        },
        "ground_truth": {
            "answer": 525.0,
            "formula": "principal * (rate_pct / 100) * years"
        }
    },
    {
        "question": {
            "task": "finance_compute",
            "formula": "profit_margin",
            "inputs": {"revenue": 15000, "cost": 9000}
        },
        "ground_truth": {
            "answer": 40.0,
            "formula": "(revenue - cost) / revenue * 100"
        }
    },
]


# ============================================================================
# Dataset 3: CODE_AGENT (List Operations)
# ============================================================================

CODE_AGENT_TRAIN = [
    {
        "question": {
            "task": "list_aggregate",
            "input": [3, 7, 7, 10],
            "op": "mode"
        },
        "ground_truth": {
            "answer": 7,
            "explanation": "Most frequent element"
        }
    },
    {
        "question": {
            "task": "list_aggregate",
            "input": [1, 2, 3, 4, 5],
            "op": "median"
        },
        "ground_truth": {
            "answer": 3,
            "explanation": "Middle element in sorted list"
        }
    },
    {
        "question": {
            "task": "list_aggregate",
            "input": [10, 20, 30],
            "op": "sum"
        },
        "ground_truth": {
            "answer": 60,
            "explanation": "Sum of all elements"
        }
    },
    {
        "question": {
            "task": "list_aggregate",
            "input": [5, 2, 9, 1, 7],
            "op": "max"
        },
        "ground_truth": {
            "answer": 9,
            "explanation": "Maximum element"
        }
    },
]

CODE_AGENT_TEST = [
    {
        "question": {
            "task": "list_aggregate",
            "input": [4, 1, 2, 2, 3],
            "op": "mode"
        },
        "ground_truth": {
            "answer": 2,
            "explanation": "Most frequent element"
        }
    },
    {
        "question": {
            "task": "list_aggregate",
            "input": [100, 200, 300, 400],
            "op": "sum"
        },
        "ground_truth": {
            "answer": 1000,
            "explanation": "Sum of all elements"
        }
    },
]


# ============================================================================
# Dataset Registry
# ============================================================================

DATASETS = {
    "labeling": {
        "train": LABELING_TRAIN,
        "test": LABELING_TEST,
        "description": "Named entity recognition (FiNER-like)"
    },
    "numeric": {
        "train": NUMERIC_TRAIN,
        "test": NUMERIC_TEST,
        "description": "Formula-based calculations (finance-like)"
    },
    "code_agent": {
        "train": CODE_AGENT_TRAIN,
        "test": CODE_AGENT_TEST,
        "description": "List operations (AppWorld-like)"
    },
}


def get_dataset(name: str, split: str = "train") -> List[Dict[str, Any]]:
    """
    Get dataset by name and split.

    Args:
        name: Dataset name ("labeling", "numeric", "code_agent")
        split: "train" or "test"

    Returns:
        List of samples with "question" and "ground_truth"
    """
    if name not in DATASETS:
        raise ValueError(f"Unknown dataset: {name}. Available: {list(DATASETS.keys())}")

    if split not in ["train", "test"]:
        raise ValueError(f"Unknown split: {split}. Available: ['train', 'test']")

    return DATASETS[name][split]


def get_all_datasets(split: str = "train") -> Dict[str, List[Dict[str, Any]]]:
    """Get all datasets for a given split."""
    return {
        name: get_dataset(name, split)
        for name in DATASETS.keys()
    }


def get_dataset_info() -> Dict[str, Dict[str, Any]]:
    """Get information about all available datasets."""
    return {
        name: {
            "description": data["description"],
            "train_size": len(data["train"]),
            "test_size": len(data["test"]),
            "total_size": len(data["train"]) + len(data["test"])
        }
        for name, data in DATASETS.items()
    }


def add_sample_ids(samples: List[Dict[str, Any]], dataset_name: str, split: str) -> List[Dict[str, Any]]:
    """Add deterministic sample_id to each sample."""
    result = []
    for i, sample in enumerate(samples):
        sample_with_id = sample.copy()
        sample_with_id["sample_id"] = f"{dataset_name}_{split}_{i:03d}"
        result.append(sample_with_id)
    return result
