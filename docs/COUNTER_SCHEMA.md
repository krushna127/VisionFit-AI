
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

## 3. Frame Alignment

### `check_pose_alignment.py`
- Compared pose frames with labeled exercise intervals.
- First squat set: frames `2495–3144`.
- Available pose frames: `2505–3144`.
- Total available pose frames: **640**.
- Confirmed the usable pose-frame range for the first squat set.

## 4. Knee-Angle Analysis

- Calculated left and right knee angles across the first squat set.

| Measurement | Angle |
|---|---:|
| Left knee minimum | 104.41° |
| Left knee maximum | 169.88° |
| Right knee minimum | 100.65° |
| Right knee maximum | 172.76° |

- Experimental squat threshold: `120°`.
- Standing threshold: `155°`.
- These thresholds were used to distinguish the lowered squat position from the standing position.

## 5. Squat Repetition-Counter Testing

**Test file:** `test_mmfit_squat_counter.py`

- Tested the counter against three labeled squat sets.
- Compared expected repetitions with detected repetitions.

| Set | Expected Reps | Detected Reps | Result |
|---:|---:|---:|---|
| 1 | 10 | 10 | Match |
| 2 | 10 | 10 | Match |
| 3 | 10 | 10 | Match |
| **Total** | **30** | **30** | **Match** |

**Test Result:** All three tested sets matched their annotated repetition counts, with 30 expected repetitions and 30 detected repetitions. This result applies to the tested sets and does not, by itself, establish generalization to other participants or sessions.

## 6.Work Update — MM-Fit Squat Counter Validation

### a. `check_mmfit_sessions.py`
- **Purpose:** Check available MM-Fit sessions and identify suitable data for testing.
- **Working:** Inspect session data before running repetition-counter tests.

### b. `test_mmfit_squat_counter.py`
- **Purpose:** Test the squat repetition counter against expected repetition labels.
- **Working:** Validate the counter using the existing knee-angle thresholds: 120° for squat detection and 155° for standing detection.
- **Result:** Three sets matched exactly, with 30 expected and 30 detected repetitions.

### c. `test_mmfit_squat_generalization.py`
- **Purpose:** Test whether the squat counter generalizes to additional MM-Fit sessions.
- **Working:** Validate performance across sessions beyond the initial test without changing the thresholds.
- **Previously reported result:** Sessions `w02` and `w03` achieved 6/6 exact set matches and 60/60 repetitions combined.

## 7. Insights from Full-Dataset Testing

After testing the fixed-threshold squat counter across 21 MM-Fit sessions, the following insights were identified.

### a. Consistent Repetition Counting
- All 64 labeled squat sets matched their expected repetition counts.
- The counter detected 639 repetitions against 639 expected repetitions.
- No counting errors were reported in the evaluated sets.

### b. Performance Across Sessions
- The evaluation covered 21 sessions (`w00`–`w20`).
- Most sessions contained three labeled squat sets.
- Session `w17` contained four sets, including sets with different annotated repetition counts.
- All evaluated sets matched their expected counts.

### c. Fixed-Threshold Performance
- The counter used a squat threshold of `120°` and a standing threshold of `155°`.
- The same thresholds were used across all evaluated sessions.
- These results indicate consistent performance on the evaluated dataset.
- Independent testing is still required to establish generalization to unseen participants and recording conditions.

### d. Pose Data Availability
- All 64 labeled squat sets had usable pose data.
- No sets were reported as missing pose data.
- All identified squat sets were included in the evaluation.

### e. Evaluation Metrics

| Metric | Result |
|---|---:|
| Sessions Evaluated | 21 |
| Labeled Squat Sets | 64 |
| Exact Set Matches | 64/64 |
| Expected Repetitions | 639 |
| Detected Repetitions | 639 |
| Exact Set-Match Rate | 100% |
| Total Absolute Count Error | 0 |
| Mean Absolute Error (MAE) | 0 repetitions per set |

These metrics measure agreement between detected repetition counts and MM-Fit annotations. They do not measure individual repetition timing or exercise-form accuracy.
