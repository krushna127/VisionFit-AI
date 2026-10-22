
from pathlib import Path
import pandas as pd
import numpy as np

folder = Path(r"C:\Users\Krushna\Desktop\VisionFit-AI\dataset\mmfit\w02")

labels = pd.read_csv(
    folder / "w02_labels.csv",
    header=None,
    names=["start", "end", "reps", "exercise"]
)

pose = np.load(folder / "w02_pose_3d.npy", allow_pickle=False)

# Inspect the first pose stream.
frames = pose[0, :, 0]

print("Pose frame range:", frames.min(), "to", frames.max())
print("Number of pose frames:", len(frames))

print("\nLabel coverage:")
for _, row in labels.iterrows():
    available = np.sum(
        (frames >= row["start"]) & (frames <= row["end"])
    )
    print(
        f'{row["exercise"]:25s} '
        f'{int(row["start"])}–{int(row["end"])} '
        f'expected_reps={int(row["reps"])} '
        f'available_pose_frames={available}'
    )

print("\nRepetition counts by exercise:")
print(labels.groupby("exercise")["reps"].agg(["count", "min", "max"]))
