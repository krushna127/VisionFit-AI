
from pathlib import Path
import pandas as pd

csv_path = Path(r"C:\Users\Krushna\Desktop\VisionFit-AI\dataset\mmfit\w02\w02_labels.csv")

df = pd.read_csv(
    csv_path,
    header=None,
    names=["start", "end", "reps", "exercise"]
)

print("First 10 labels:")
print(df.head(10).to_string(index=False))

print("\nExercise counts:")
print(df["exercise"].value_counts().to_string())

print("\nTotal labeled intervals:", len(df))
