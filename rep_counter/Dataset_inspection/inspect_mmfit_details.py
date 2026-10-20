
from pathlib import Path
import numpy as np

DATA_DIR = Path(r"C:\Users\Krushna\Desktop\VisionFit-AI\dataset\mmfit\w02")

for filename in ("w02_pose_2d.npy", "w02_pose_3d.npy"):
    path = DATA_DIR / filename
    data = np.load(path, allow_pickle=False)

    print(f"\n{filename}")
    print("Full shape:", data.shape)

    # Inspect the first stream only.
    stream = data[0]

    print("Stream shape:", stream.shape)
    print("First row:", stream[0])
    print("Second row:", stream[1])

    # Check whether the first column increments by one.
    time_diffs = np.diff(stream[:100, 0])
    print("First-column increments:", np.unique(time_diffs))

    # Check whether coordinate columns are constant.
    print("Coordinate-column standard deviations:")
    print(np.std(stream[:, 1:], axis=0))
