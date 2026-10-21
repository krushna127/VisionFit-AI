
from pathlib import Path

import numpy as np
import pandas as pd


# 1. LOAD MM-FIT DATA

DATA_DIR = Path(r"C:\Users\Krushna\Desktop\VisionFit-AI\dataset\mmfit\w02")

pose = np.load(DATA_DIR / "w02_pose_3d.npy", allow_pickle=False)

# The CSV has no header row.
labels = pd.read_csv(DATA_DIR / "w02_labels.csv", header=None)
labels.columns = ["start", "end", "reps", "exercise"]

# Selecting all squat intervals.
squat_labels = labels[
    labels["exercise"].astype(str).str.strip().str.lower() == "squats"
]

# Frame numbers are stored in column 0.
frame_numbers = pose[0, :, 0]


# 2. CALCULATE THE ANGLE AT A JOINT

def calculate_angle(a, b, c):
    """Calculate angle ABC in degrees."""

    ba = a - b
    bc = c - b

    denominator = np.linalg.norm(ba) * np.linalg.norm(bc)

    if denominator == 0:
        return np.nan

    cosine = np.dot(ba, bc) / denominator
    cosine = np.clip(cosine, -1.0, 1.0)

    return np.degrees(np.arccos(cosine))


# 3. COUNT REPETITIONS FOR ONE SQUAT SET

def count_squats(frame_indices):
    left_angles = []
    right_angles = []

    for frame_index in frame_indices:

        # Each joint has three coordinate components.
        # Column 0 is the frame number, so joint columns
        # start at index 1.

        def joint_position(joint_index):
            return pose[:, frame_index, joint_index + 1]

        # Joint mapping from our current MM-Fit notebook
        left_hip = joint_position(1)
        left_knee = joint_position(2)
        left_foot = joint_position(3)

        right_hip = joint_position(4)
        right_knee = joint_position(5)
        right_foot = joint_position(6)

        left_angles.append(
            calculate_angle(left_hip, left_knee, left_foot)
        )

        right_angles.append(
            calculate_angle(right_hip, right_knee, right_foot)
        )

    left_angles = np.array(left_angles)
    right_angles = np.array(right_angles)

    # Average the two knees.
    average_angles = (left_angles + right_angles) / 2

    # Initial thresholds from our first successful test.
    squat_threshold = 120
    standing_threshold = 155

    repetitions = 0
    in_squat = False

    for angle in average_angles:

        if np.isnan(angle):
            continue

        # Start a squat when the knees bend sufficiently.
        if not in_squat and angle < squat_threshold:
            in_squat = True

        # Count one repetition after returning to standing.
        elif in_squat and angle > standing_threshold:
            repetitions += 1
            in_squat = False

    return repetitions, len(frame_indices)

# 4. TEST ALL SQUAT SETS

results = []

for set_number, (_, squat) in enumerate(
    squat_labels.iterrows(), start=1
):

    start = int(squat["start"])
    end = int(squat["end"])
    expected = int(squat["reps"])

    # Find available pose rows within this interval.
    mask = (frame_numbers >= start) & (frame_numbers <= end)
    frame_indices = np.flatnonzero(mask)

    if len(frame_indices) == 0:
        detected = 0
        status = "NO POSE FRAMES"
    else:
        detected, frame_count = count_squats(frame_indices)

        status = "MATCH" if detected == expected else "CHECK"

    results.append({
        "Set": set_number,
        "Start": start,
        "End": end,
        "Frames": len(frame_indices),
        "Expected": expected,
        "Detected": detected,
        "Status": status
    })


# 5. DISPLAY COMPARISON RESULTS

results_df = pd.DataFrame(results)

print("\n========== MM-FIT SQUAT TEST RESULTS ==========")
print(results_df.to_string(index=False))

if len(results_df) > 0:
    matched_sets = (results_df["Expected"] == results_df["Detected"]).sum()
    total_sets = len(results_df)

    expected_total = results_df["Expected"].sum()
    detected_total = results_df["Detected"].sum()

    print("\n============== SUMMARY ==============")
    print(f"Sets tested: {total_sets}")
    print(f"Sets with matching counts: {matched_sets}/{total_sets}")
    print(f"Expected total repetitions: {expected_total}")
    print(f"Detected total repetitions: {detected_total}")

    if expected_total > 0:
        count_match_percentage = (
            matched_sets / total_sets * 100
        )

        print(
            f"Exact set-match rate: {count_match_percentage:.1f}%"
        )
