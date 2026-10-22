
from pathlib import Path
import numpy as np

folder = Path(r"C:\Users\Krushna\Desktop\VisionFit-AI\dataset\mmfit\w02")

for filename in ["w02_pose_2d.npy", "w02_pose_3d.npy"]:
    data = np.load(folder / filename, allow_pickle=False)

    print("\nFILE:", filename)
    print("Full shape:", data.shape)

    for i in range(data.shape[0]):
        stream = data[i]
        print(f"\nStream {i}: shape = {stream.shape}")
        print("First row:", stream[0])
        print("Last row:", stream[-1])
        print("First 5 column means:",
              np.mean(stream[:, :5], axis=0))
