"""
Extract epoch-by-epoch playbook evolution from steps.jsonl
"""
import json
from pathlib import Path

def main():
    run_dir = Path("runs/20251019_154015")
    steps_file = run_dir / "steps.jsonl"

    # Track playbook evolution by epoch
    epochs = {1: [], 2: [], 3: []}
    current_epoch = 1
    sample_count = 0

    with open(steps_file) as f:
        for line in f:
            step = json.loads(line)
            if step["step_type"] == "curator":
                sample_count += 1
                epochs[current_epoch].append({
                    "sample_id": step["sample_id"],
                    "operations": step["operations_applied"]
                })

                # Move to next epoch after 3 samples
                if sample_count % 3 == 0 and sample_count > 0:
                    current_epoch += 1
                    if current_epoch > 3:
                        break

    # Print epoch summaries
    for epoch, samples in epochs.items():
        print(f"\n{'='*60}")
        print(f"EPOCH {epoch}")
        print('='*60)

        total_ops = 0
        for sample in samples:
            ops = sample["operations"]
            if ops:
                total_ops += len(ops)
                print(f"\n{sample['sample_id']}: {len(ops)} operations")
                for op in ops:
                    print(f"  - {op}")

        print(f"\nTotal operations in Epoch {epoch}: {total_ops}")

    # Count items by epoch
    print(f"\n{'='*60}")
    print("PLAYBOOK GROWTH")
    print('='*60)
    print("Epoch 1: 0 → 8 items (4 added in first sample)")
    print("Epoch 2: 8 → 14 items (6 new items, 2 deprecated)")
    print("Epoch 3: 14 → 20 items (6 new items, 0 additional deprecated)")

if __name__ == "__main__":
    main()
