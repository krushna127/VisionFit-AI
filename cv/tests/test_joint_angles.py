from cv.features.export_joint_angles import get_landmark
import unittest

from cv.features.joint_angles import (
    calculate_angle,
    calculate_curl_angles,
    calculate_squat_angles,
)


class TestJointAngles(unittest.TestCase):
    def test_straight_angle(self):
        self.assertEqual(
            calculate_angle((0, 0), (1, 0), (2, 0)),
            180.0,
        )

    def test_right_angle(self):
        self.assertEqual(
            calculate_angle((1, 0), (0, 0), (0, 1)),
            90.0,
        )

    def test_zero_length_vector_returns_none(self):
        self.assertIsNone(
            calculate_angle((0, 0), (0, 0), (1, 1))
        )

    def test_invalid_coordinates_return_none(self):
        self.assertIsNone(
            calculate_angle((0, 0), (float("nan"), 1), (2, 2))
        )

    def test_squat_angle_mapping(self):
        landmarks = {
            "left_shoulder": (0, 0),
            "left_hip": (0, 1),
            "left_knee": (1, 1),
            "left_ankle": (1, 2),
            "right_shoulder": (3, 0),
            "right_hip": (3, 1),
            "right_knee": (4, 1),
            "right_ankle": (4, 2),
        }

        angles = calculate_squat_angles(landmarks)

        self.assertEqual(angles["left_knee_angle_deg"], 90.0)
        self.assertEqual(angles["right_knee_angle_deg"], 90.0)
        self.assertEqual(angles["left_hip_angle_deg"], 90.0)
        self.assertEqual(angles["right_hip_angle_deg"], 90.0)

    def test_missing_landmarks_return_none(self):
        angles = calculate_curl_angles({})

        self.assertIsNone(angles["left_elbow_angle_deg"])
        self.assertIsNone(angles["right_elbow_angle_deg"])

    def test_low_confidence_landmark_is_rejected(self):
        row = {
            "hip_x": "10",
            "hip_y": "20",
            "hip_confidence": "0.1",
        }
        self.assertIsNone(get_landmark(row, "hip"))

    def test_invalid_landmark_values_are_rejected(self):
        base = {
            "hip_x": "10",
            "hip_y": "20",
            "hip_confidence": "0.9",
        }

        self.assertEqual(get_landmark(base, "hip"), (10.0, 20.0))

        self.assertIsNone(
            get_landmark({**base, "hip_x": "nan"}, "hip")
        )
        self.assertIsNone(
            get_landmark({**base, "hip_confidence": "nan"}, "hip")
        )
        self.assertIsNone(
            get_landmark({**base, "hip_confidence": "1.2"}, "hip")
        )
        self.assertIsNone(get_landmark({}, "hip"))


if __name__ == "__main__":
    unittest.main()
