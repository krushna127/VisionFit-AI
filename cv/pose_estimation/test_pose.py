
from pathlib import Path

import cv2
from ultralytics import YOLO


def main():
    # Locate the project root from this script's location.
    project_root = Path(__file__).resolve().parents[2]

    # Input video and output video paths.
    video_path = project_root / "samples" / "squats.mp4"
    output_path = project_root / "samples" / "squats_pose.mp4"

    # Stop early if the input video is missing.
    if not video_path.exists():
        raise FileNotFoundError(f"Input video not found: {video_path}")

    # Load the pretrained YOLO11 nano pose model.
    model = YOLO("yolo11n-pose.pt")

    # Run pose estimation and save the annotated video.
    results = model.predict(
        source=str(video_path),
        save=True,
        project=str(project_root / "runs"),
        name="pose",
        exist_ok=True,
        imgsz=640,
        conf=0.25,
        verbose=False,
    )

    print("Pose estimation completed.")
    print(f"Processed frames returned: {len(results)}")
    print(f"Annotated output directory: {project_root / 'runs' / 'pose'}")
    print("Open the generated video to inspect the detected skeletons.")


if __name__ == "__main__":
    main()
