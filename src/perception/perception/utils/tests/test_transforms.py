import numpy as np
from perception.utils.transforms import apply_dock_pose


def test_identity_pose_leaves_offsets_unchanged():
    offsets = np.array(
        [
            [-0.35, -0.31, 0.455],
            [0.35, -0.31, 0.455],
            [-0.35, -0.31, -0.075],
            [0.35, -0.31, -0.075],
        ]
    )
    result = apply_dock_pose([0.0, 0.0, 0.0], [0.0, 0.0, 0.0], offsets)
    np.testing.assert_allclose(result, offsets, atol=1e-9)


def test_pure_translation():
    offsets = np.array([[1.0, 0.0, 0.0]])
    result = apply_dock_pose([2.0, 3.0, 4.0], [0.0, 0.0, 0.0], offsets)
    np.testing.assert_allclose(result, [[3.0, 3.0, 4.0]], atol=1e-9)


def test_yaw_90_degrees():
    offsets = np.array([[1.0, 0.0, 0.0]])
    result = apply_dock_pose([0.0, 0.0, 0.0], [0.0, 0.0, np.pi / 2], offsets)
    np.testing.assert_allclose(result, [[0.0, 1.0, 0.0]], atol=1e-9)


def test_output_shape():
    offsets = np.array(
        [
            [-0.35, -0.31, 0.455],
            [0.35, -0.31, 0.455],
            [-0.35, -0.31, -0.075],
            [0.35, -0.31, -0.075],
        ]
    )
    result = apply_dock_pose([5.0, 0.0, -1.5], [0.0, 0.0, 1.5707963], offsets)
    assert result.shape == (4, 3)
