
from pathlib import Path
import numpy as np

folder = Path(r"C:\Users\Krushna\Desktop\VisionFit-AI\dataset\mmfit\w02")
pose = np.load(folder / "w02_pose_3d.npy", allow_pickle=False)

# Inspect the three streams without assuming their meanings.
for stream_index in range(pose.shape[0]):
    stream = pose[stream_index]

    print(f"\n--- Stream {stream_index} ---")
    print("Shape:", stream.shape)
    print("First row:", stream[0])

    # Exclude column 0, which holds the frame index.
    for col in range(1, stream.shape[1]):
        values = stream[:, col]
        print(
            f"Column {col:02d}: "
            f"min={values.min():.2f}, "
            f"max={values.max():.2f}, "
            f"std={values.std():.2f}"
        )
