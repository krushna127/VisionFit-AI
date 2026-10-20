
from math import acos, degrees
from typing import Optional, Sequence

import numpy as np


def calculate_angle(
    point_a: Sequence[float],
    point_b: Sequence[float],
    point_c: Sequence[float],
) -> Optional[float]:
    """Calculate the angle ABC in degrees, with the vertex at point B.

    Return None if the points are invalid or either vector has zero length.
    """
    a = np.asarray(point_a, dtype=float)
    b = np.asarray(point_b, dtype=float)
    c = np.asarray(point_c, dtype=float)

    if a.shape != (2,) or b.shape != (2,) or c.shape != (2,):
        return None

    if not (
        np.isfinite(a).all()
        and np.isfinite(b).all()
        and np.isfinite(c).all()
    ):
        return None

    vector_ba = a - b
    vector_bc = c - b

    length_ba = np.linalg.norm(vector_ba)
    length_bc = np.linalg.norm(vector_bc)

    if length_ba == 0 or length_bc == 0:
        return None

    cosine = np.dot(vector_ba, vector_bc) / (length_ba * length_bc)
    cosine = np.clip(cosine, -1.0, 1.0)

    return round(degrees(acos(float(cosine))), 2)


def calculate_squat_angles(
    landmarks: dict[str, Sequence[float]],
) -> dict[str, Optional[float]]:
    """Calculate left and right knee and hip angles.

    Expected landmark values are (x, y) pixel coordinates.
    Missing or invalid points produce None for the affected angle.
    """
    return {
        "left_knee_angle_deg": calculate_angle(
            landmarks.get("left_hip", ()),
            landmarks.get("left_knee", ()),
            landmarks.get("left_ankle", ()),
        ),
        "right_knee_angle_deg": calculate_angle(
            landmarks.get("right_hip", ()),
            landmarks.get("right_knee", ()),
            landmarks.get("right_ankle", ()),
        ),
        "left_hip_angle_deg": calculate_angle(
            landmarks.get("left_shoulder", ()),
            landmarks.get("left_hip", ()),
            landmarks.get("left_knee", ()),
        ),
        "right_hip_angle_deg": calculate_angle(
            landmarks.get("right_shoulder", ()),
            landmarks.get("right_hip", ()),
            landmarks.get("right_knee", ()),
        ),
    }


def calculate_curl_angles(
    landmarks: dict[str, Sequence[float]],
) -> dict[str, Optional[float]]:
    """Calculate left and right elbow angles."""
    return {
        "left_elbow_angle_deg": calculate_angle(
            landmarks.get("left_shoulder", ()),
            landmarks.get("left_elbow", ()),
            landmarks.get("left_wrist", ()),
        ),
        "right_elbow_angle_deg": calculate_angle(
            landmarks.get("right_shoulder", ()),
            landmarks.get("right_elbow", ()),
            landmarks.get("right_wrist", ()),
        ),
    }
