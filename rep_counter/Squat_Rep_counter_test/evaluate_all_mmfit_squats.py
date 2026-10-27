
from pathlib import Path

import numpy as np
import pandas as pd

# 1. CONFIGURATION

# Replace with the parent folder containing w00, w01, w02, etc.
DATASET_DIR = Path(r"C:\Users\Krushna\Desktop\VisionFit-AI\dataset\mmfit")

SQUAT_THRESHOLD = 120
STANDING_THRESHOLD = 155

# Output reports will be saved in DATASET_DIR.
SET_REPORT_FILE = DATASET_DIR / "squat_set_evaluation.csv"
SESSION_REPORT_FILE = DATASET_DIR / "squat_session_summary.csv"


# 2. ANGLE CALCULATION

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


# 3. COUNT SQUATS IN ONE LABELED INTERVAL

def count_squats(pose, frame_indices):
    left_angles = []
    right_angles = []

    for frame_index in frame_indices:

        # Each joint has 3 coordinate components.
        # Column 0 contains the frame number.
        def joint_position(joint_index):
            return pose[:, frame_index, joint_index + 1]

        # Joint mapping used in the existing prototype.
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

    left_angles = np.asarray(left_angles, dtype=float)
    right_angles = np.asarray(right_angles, dtype=float)

    average_angles = (left_angles + right_angles) / 2

    repetitions = 0
    in_squat = False

    for angle in average_angles:

        if not np.isfinite(angle):
            continue

        # Enter squat state.
        if not in_squat and angle < SQUAT_THRESHOLD:
            in_squat = True

        # Count after returning to standing.
        elif in_squat and angle > STANDING_THRESHOLD:
            repetitions += 1
            in_squat = False

    return repetitions


# 4. FIND ALL VALID SESSION FOLDERS

if not DATASET_DIR.is_dir():
    raise FileNotFoundError(
        f"Dataset folder does not exist: {DATASET_DIR}"
    )

session_folders = sorted(
    [
        folder
        for folder in DATASET_DIR.iterdir()
        if folder.is_dir()
        and (folder / f"{folder.name}_pose_3d.npy").is_file()
        and (folder / f"{folder.name}_labels.csv").is_file()
    ],
    key=lambda folder: folder.name
)

print("MM-Fit squat evaluation")
print("Dataset:", DATASET_DIR)
print("Sessions found:", len(session_folders))
print("Squat threshold:", SQUAT_THRESHOLD)
print("Standing threshold:", STANDING_THRESHOLD)


# ==================================================
# 5. TEST ALL SQUAT SETS IN ALL SESSIONS
# ==================================================

set_results = []
session_results = []

for folder in session_folders:

    session = folder.name
    pose_file = folder / f"{session}_pose_3d.npy"
    labels_file = folder / f"{session}_labels.csv"

    print(f"\nProcessing {session}...")

    try:
        pose = np.load(pose_file, allow_pickle=False)

        if pose.ndim != 3 or pose.shape[0] != 3 or pose.shape[2] < 7:
            raise ValueError(
                f"Unexpected pose array shape: {pose.shape}"
            )

        labels = pd.read_csv(labels_file, header=None)

        if labels.shape[1] < 4:
            raise ValueError(
                f"Expected at least 4 label columns; "
                f"found {labels.shape[1]}"
            )

        # The first four columns are:
        # start frame, end frame, expected reps, exercise.
        labels = labels.iloc[:, :4].copy()
        labels.columns = ["start", "end", "reps", "exercise"]

        labels["exercise"] = (
            labels["exercise"].astype(str).str.strip().str.lower()
        )

        labels["start"] = pd.to_numeric(
            labels["start"], errors="coerce"
        )
        labels["end"] = pd.to_numeric(
            labels["end"], errors="coerce"
        )
        labels["reps"] = pd.to_numeric(
            labels["reps"], errors="coerce"
        )

        squat_labels = labels[
            labels["exercise"] == "squats"
        ].dropna(subset=["start", "end", "reps"])

        frame_numbers = pose[0, :, 0]

        session_set_results = []

        for set_number, (_, squat) in enumerate(
            squat_labels.iterrows(), start=1
        ):
            start = int(squat["start"])
            end = int(squat["end"])
            expected = int(squat["reps"])

            mask = (
                (frame_numbers >= start)
                & (frame_numbers <= end)
            )

            frame_indices = np.flatnonzero(mask)

            if len(frame_indices) == 0:
                detected = None
                status = "NO POSE FRAMES"
            else:
                detected = count_squats(pose, frame_indices)

                status = (
                    "MATCH"
                    if detected == expected
                    else "CHECK"
                )

            result = {
                "Session": session,
                "Set": set_number,
                "Start": start,
                "End": end,
                "Frames": len(frame_indices),
                "Expected": expected,
                "Detected": detected,
                "Status": status
            }

            set_results.append(result)
            session_set_results.append(result)

        # Per-session summary includes only sets with pose data.
        usable = [
            row for row in session_set_results
            if row["Detected"] is not None
        ]

        matched = sum(
            row["Expected"] == row["Detected"]
            for row in usable
        )

        expected_total = sum(row["Expected"] for row in usable)
        detected_total = sum(row["Detected"] for row in usable)

        session_results.append({
            "Session": session,
            "Squat Sets": len(session_set_results),
            "Usable Sets": len(usable),
            "Matched Sets": matched,
            "Expected Reps": expected_total,
            "Detected Reps": detected_total,
            "Exact Match Rate (%)": (
                round(100 * matched / len(usable), 2)
                if usable else np.nan
            ),
            "Sets Without Pose": (
                len(session_set_results) - len(usable)
            ),
            "Status": (
                "OK" if usable else "NO USABLE SQUAT SETS"
            )
        })

    except Exception as error:
        print(f"WARNING: Could not evaluate {session}: {error}")

        session_results.append({
            "Session": session,
            "Squat Sets": 0,
            "Usable Sets": 0,
            "Matched Sets": 0,
            "Expected Reps": 0,
            "Detected Reps": 0,
            "Exact Match Rate (%)": np.nan,
            "Sets Without Pose": 0,
            "Status": f"ERROR: {error}"
        })


# 6. DISPLAY AND SAVE THE REPORTS

set_columns = [
    "Session", "Set", "Start", "End", "Frames",
    "Expected", "Detected", "Status"
]

session_columns = [
    "Session", "Squat Sets", "Usable Sets", "Matched Sets",
    "Expected Reps", "Detected Reps", "Exact Match Rate (%)",
    "Sets Without Pose", "Status"
]

set_df = pd.DataFrame(set_results, columns=set_columns)
session_df = pd.DataFrame(session_results, columns=session_columns)

print("\n========== PER-SESSION SUMMARY ==========")

if not session_df.empty:
    print(session_df.to_string(index=False))
else:
    print("No usable session folders were found.")

print("\n========== PER-SET RESULTS ==========")

if not set_df.empty:
    print(set_df.to_string(index=False))
else:
    print("No squat sets were found.")

# Overall metrics use only sets with available pose frames.
if not set_df.empty:
    usable_df = set_df.dropna(subset=["Detected"]).copy()

    if not usable_df.empty:
        matched_sets = (
            usable_df["Expected"] == usable_df["Detected"]
        ).sum()

        total_sets = len(usable_df)
        expected_total = usable_df["Expected"].sum()
        detected_total = usable_df["Detected"].sum()

        total_absolute_error = (
            usable_df["Expected"] - usable_df["Detected"]
        ).abs().sum()

        print("\n========== OVERALL EVALUATION ==========")
        print("Sessions discovered:", len(session_folders))
        print("Sessions evaluated:", len(session_df[
            session_df["Usable Sets"] > 0
        ]))
        print("Labeled squat sets found:", len(set_df))
        print("Sets with pose data:", total_sets)
        print("Exact set matches:", f"{matched_sets}/{total_sets}")
        print(
            "Exact set-match rate:",
            f"{100 * matched_sets / total_sets:.2f}%"
        )
        print("Expected repetitions:", expected_total)
        print("Detected repetitions:", detected_total)
        print("Total absolute count error:", total_absolute_error)

        if expected_total > 0:
            print(
                "Aggregate count agreement:",
                f"{100 * (1 - total_absolute_error / expected_total):.2f}%"
            )

    else:
        print("\nNo squat sets had available pose frames.")

# Save CSV files for later analysis and project reporting.
set_df.to_csv(SET_REPORT_FILE, index=False)
session_df.to_csv(SESSION_REPORT_FILE, index=False)

print("\nReports saved:")
print(SET_REPORT_FILE)
print(SESSION_REPORT_FILE)