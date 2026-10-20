# VisionFit-AI — Computer Vision Module

## 1. Overview

The computer vision module processes workout videos using a pretrained YOLO11-Pose model. It detects body landmarks, records their coordinates and confidence scores, and calculates joint angles for further exercise analysis.

The output data is designed to support the other VisionFit-AI modules, including exercise classification, repetition counting, and form assessment.

## 2. Features

- Detects 17 body landmarks using YOLO11-Pose.
- Saves landmark coordinates and confidence scores to a CSV file.
- Generates a video with the detected pose overlay.
- Calculates left and right knee, hip, and elbow angles.
- Handles missing, invalid, and low-confidence landmarks during angle calculation.
- Includes unit tests for joint-angle calculations and invalid input handling.

## 3. Folder Structure

```text
cv/
├── pose_estimation/
├── preprocessing/
├── features/
│   ├── joint_angles.py
│   └── export_joint_angles.py
├── tests/
│   └── test_joint_angles.py
└── README.md
```

## 4. Requirements

- Python
- Ultralytics
- OpenCV
- NumPy

The required Python packages are listed in the repository's `requirements.txt` file. The pipeline uses the pretrained `yolo11n-pose.pt` model weights.

## 5. Setup

From the repository root, activate the project's virtual environment:

```bash
source .venv/bin/activate
```

Install the dependencies:

```bash
pip install -r requirements.txt
```

Make sure the model weights and sample video are available at:

- Model: `yolo11n-pose.pt`
- Sample video: `samples/squats.mp4`

## 6. Run Pose Estimation

Run the pose-estimation script from the repository root:

```bash
python -m cv.pose_estimation.test_pose
```

The script processes the sample video and generates the pose outputs under `runs/pose/`.

The generated outputs include:

- `squats_keypoints.csv` — frame metadata, body landmark coordinates, and confidence scores.
- `squats_pose.mp4` — video with the detected pose overlay.

## 7. Export Joint Angles

After generating the keypoint CSV, run:

```bash
python -m cv.features.export_joint_angles
```

This reads the landmark data and generates:

`runs/pose/squats_joint_angles.csv`

The file contains the frame index, timestamp, and six joint-angle features for the left and right knees, hips, and elbows.

The current confidence threshold for angle calculations is `0.25`. If a required landmark is missing or unreliable, the corresponding angle is left blank rather than treated as zero.

## 8. Run Tests

From the repository root, run:

```bash
python -m unittest cv.tests.test_joint_angles -v
```

These tests check angle calculations, invalid coordinates, missing landmarks, and confidence validation.

## 9. Feature Schema

Refer to [`docs/FEATURE_SCHEMA.md`](../docs/FEATURE_SCHEMA.md) for the documented CSV fields, landmark names, angle features, timestamps, confidence handling, and integration notes.

## 10. Current Limitations

- The current implementation has been tested with a squat sample video.
- The pipeline assumes one primary person in the frame.
- Landmark and angle accuracy can be affected by occlusion, camera angle, body orientation, and perspective.
- Exercise classification, repetition counting, and form assessment are handled by separate modules and are not implemented by this CV module alone.