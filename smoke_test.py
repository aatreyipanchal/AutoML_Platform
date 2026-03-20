import pandas as pd
import numpy as np
import os
import subprocess

# 1. Create dummy dataset
data = {
    'age': [25, 30, 35, 40, 45, 50, 55, 60],
    'salary': [50000, 60000, 70000, 80000, 90000, 100000, 110000, 120000],
    'city': ['NY', 'SF', 'NY', 'SF', 'NY', 'SF', 'NY', 'SF'],
    'bought': [0, 1, 0, 1, 0, 1, 0, 1]
}
df = pd.DataFrame(data)
df.to_csv('dummy_data.csv', index=False)

print("Created dummy_data.csv")

# 2. Run CLI
print("Running AutoML CLI...")
cmd = "python src/cli.py --data dummy_data.csv --target bought --type tabular --task classification"
result = subprocess.run(cmd, shell=True, capture_output=True, text=True)

print("--- CLI Output ---")
print(result.stdout)
print("--- CLI Errors ---")
print(result.stderr)

# 3. Check for artifacts
if os.path.exists("src/models/best_tabular_model.pkl"):
    print("[SUCCESS] Model saved successfully!")
else:
    print("[FAILURE] Model not found.")

if os.path.exists("src/reports/summary.csv"):
    print("[SUCCESS] EDA reports generated successfully!")
else:
    print("[FAILURE] EDA reports not found.")

# Cleanup
# os.remove('dummy_data.csv')
