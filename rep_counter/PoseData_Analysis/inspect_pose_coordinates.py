
from pathlib import Path
import numpy as np

DATA_DIR = Path(r"C:\Users\Krushna\Desktop\VisionFit-AI\dataset\mmfit\w02")

pose = np.load(DATA_DIR / "w02_pose_3d.npy")

print("Full shape:", pose.shape)

# Examine the three streams at the first squat frame.
# The pose data's first column appears to contain frame numbers.
target_frame = 2495
frame_numbers = pose[0, :, 0]
index = np.argmin(np.abs(frame_numbers - target_frame))

print("\nTarget frame:", target_frame)
print("Closest frame number:", frame_numbers[index])
print("Array index:", index)

for stream in range(pose.shape[0]):
    print(f"\nStream {stream}:")
    print(pose[stream, index, :])

print("\nJoint mapping from your screenshot:")
joint_names = [
    "Hip", "Left Hip", "Left Knee", "Left Foot",
    "Right Hip", "Right Knee", "Right Foot", "Spine",
    "Thorax", "Neck/Nose", "Head", "Right Shoulder",
    "Right Elbow", "Right Wrist", "Left Shoulder",
    "Left Elbow", "Left Wrist"
]

for i, name in enumerate(joint_names):
    print(i, name)
