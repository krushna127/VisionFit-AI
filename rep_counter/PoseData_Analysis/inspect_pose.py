
from pathlib import Path
import numpy as np

DATA_DIR = Path(r"C:\Users\Krushna\Desktop\VisionFit-AI\dataset\mmfit\w02")

for filename in ["w02_pose_2d.npy", "w02_pose_3d.npy"]:
    path = DATA_DIR / filename

    print("\n" + "=" * 50)
    print("FILE:", filename)

    if not path.exists():
        print("File not found:", path)
        continue

    data = np.load(path, allow_pickle=False)

    print("Full shape:", data.shape)
    print("Number of dimensions:", data.ndim)
    print("Data type:", data.dtype)

    # Inspect the first element at each dimension.
    for axis in range(data.ndim):
        print(f"Size of dimension {axis}: {data.shape[axis]}")

    print("\nFirst element:")
    print(data[0])

    if data.ndim >= 2:
        print("\nFirst 3 rows of the first element:")
        print(data[0, :3])

