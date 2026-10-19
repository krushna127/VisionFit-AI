
import csv
from pathlib import Path

from cv.features.joint_angles import (
    calculate_angle,
    calculate_curl_angles,
    calculate_squat_angles,
)

CONFIDENCE_THRESHOLD = 0.25


def get_landmark(row, name):
    """Return a landmark coordinate only when its confidence is sufficient."""
    try:
        x = float(row[f"{name}_x"])
        y = float(row[f"{name}_y"])
        confidence = float(row[f"{name}_confidence"])
    except (KeyError, TypeError, ValueError):
        return None

    if confidence < CONFIDENCE_THRESHOLD:
        return None

    return (x, y)


def main():
    project_root = Path(__file__).resolve().parents[2]
    input_path = project_root / "runs" / "pose" / "squats_keypoints.csv"
    output_path = project_root / "runs" / "pose" / "squats_joint_angles.csv"

    if not input_path.exists():
        raise FileNotFoundError(f"Keypoint CSV not found: {input_path}")

    output_columns = [
        "frame_index",
        "timestamp_ms",
        "left_knee_angle_deg",
        "right_knee_angle_deg",
        "left_hip_angle_deg",
        "right_hip_angle_deg",
        "left_elbow_angle_deg",
        "right_elbow_angle_deg",
    ]

    processed_frames = 0

    with input_path.open("r", newline="", encoding="utf-8") as source:
        reader = csv.DictReader(source)

        with output_path.open("w", newline="", encoding="utf-8") as target:
            writer = csv.DictWriter(target, fieldnames=output_columns)
            writer.writeheader()

            for row in reader:
                landmarks = {
                    name: get_landmark(row, name)
                    for name in (
                        "left_shoulder",
                        "right_shoulder",
                        "left_elbow",
                        "right_elbow",
                        "left_wrist",
                        "right_wrist",
                        "left_hip",
                        "right_hip",
                        "left_knee",
                        "right_knee",
                        "left_ankle",
                        "right_ankle",
                    )
                }

                # Missing or low-confidence points become None.
                # The angle function returns None for invalid geometry.
                squat_angles = calculate_squat_angles_safe(landmarks)
                curl_angles = calculate_curl_angles_safe(landmarks)

                writer.writerow({
                    "frame_index": row["frame_index"],
                    "timestamp_ms": row["timestamp_ms"],
                    **squat_angles,
                    **curl_angles,
                })
                processed_frames += 1

    print("Joint-angle export completed.")
    print(f"Frames processed: {processed_frames}")
    print(f"Output CSV: {output_path}")


def calculate_squat_angles_safe(landmarks):
    """Calculate squat angles only when all required points are available."""
    def safe_angle(a, b, c):
        if landmarks.get(a) is None or landmarks.get(b) is None or landmarks.get(c) is None:
            return None
        return calculate_angle(landmarks[a], landmarks[b], landmarks[c])

    return {
        "left_knee_angle_deg": safe_angle("left_hip", "left_knee", "left_ankle"),
        "right_knee_angle_deg": safe_angle("right_hip", "right_knee", "right_ankle"),
        "left_hip_angle_deg": safe_angle("left_shoulder", "left_hip", "left_knee"),
        "right_hip_angle_deg": safe_angle("right_shoulder", "right_hip", "right_knee"),
    }


def calculate_curl_angles_safe(landmarks):
    """Calculate elbow angles only when all required points are available."""
    def safe_angle(a, b, c):
        if landmarks.get(a) is None or landmarks.get(b) is None or landmarks.get(c) is None:
            return None
        return calculate_angle(landmarks[a], landmarks[b], landmarks[c])

    return {
        "left_elbow_angle_deg": safe_angle("left_shoulder", "left_elbow", "left_wrist"),
        "right_elbow_angle_deg": safe_angle("right_shoulder", "right_elbow", "right_wrist"),
    }


if __name__ == "__main__":
    main()
