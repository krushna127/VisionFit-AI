
# MM-Fit Squat Repetition Counter

## 1. Dataset Inspection

### `inspect_mmfit.py`
- Inspected the MM-Fit dataset structure.
- Explored available dataset files and components.
- Established an initial understanding of the dataset.

### `inspect_labels.py`
- Examined exercise annotation labels.
- Identified the columns: `start`, `end`, `reps`, and `exercise`.
- Found three labeled squat sets.

### `inspect_mmfit_details.py`
- Investigated dataset and recording details.
- Supported the identification of suitable samples for testing.

### `inspect_mmfit_layout.py`
- Examined the arrangement of pose data.
- Helped clarify the structure of pose streams.

## 2. Pose Data Analysis

### `inspect_pose.py`
- Inspected the pose array dimensions.
- Identified the pose shape as `(3, 73546, 18)`.

### `inspect_pose_columns.py`
- Investigated the ordering of pose columns.
- Identified joints such as hips, knees, feet, spine, shoulders, elbows, and wrists.

### `inspect_pose_coordinates.py`
- Examined numerical joint coordinates.
- Inspected coordinate ranges across three pose streams.
- Prepared the data for joint-angle calculations.