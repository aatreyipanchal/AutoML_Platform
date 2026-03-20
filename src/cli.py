import argparse
import sys
import os

# Add src to path if needed
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from engine.pipeline import AutoMLPipeline

def main():
    parser = argparse.ArgumentParser(description="AutoML Platform CLI")
    parser.add_argument("--data", type=str, required=True, help="Path to CSV file or Image directory")
    parser.add_argument("--target", type=str, help="Target column name (for tabular data)")
    parser.add_argument("--type", type=str, choices=["tabular", "cv"], default="tabular", help="Task type")
    parser.add_argument("--task", type=str, choices=["classification", "regression"], default="classification", help="Sub-task type")
    
    args = parser.parse_args()

    if args.type == "tabular" and not args.target:
        print("Error: --target is required for tabular data.")
        sys.exit(1)

    pipeline = AutoMLPipeline(
        data_path=args.data,
        target_col=args.target,
        task_type=args.type,
        sub_task=args.task
    )
    
    pipeline.run()

if __name__ == "__main__":
    main()
