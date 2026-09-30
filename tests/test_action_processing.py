import numpy as np
import pytest

from cobotmagic_deployment.common.action_processing import (
    adaptive_bridge_to_first_action,
    adaptive_delta_upsample_chunks,
    exponential_temporal_ensemble,
    lowpass_action_chunk_zero_phase,
    parse_arm_selection,
    project_action_chunk_monotonic_to_endpoint,
    quat_xyzw_to_euler_xyz,
    split_joint_threshold,
    threshold_gripper_targets,
    validate_policy_action_mode,
)


@pytest.mark.parametrize("value, arms, invalid", [
    ("left", {"left"}, set()),
    (" Right ", {"right"}, set()),
    ("both", {"left", "right"}, set()),
    (["left", "bogus"], {"left"}, {"bogus"}),
    ([], set(), set()),
])
def test_parse_arm_selection(value, arms, invalid):
    assert parse_arm_selection(value) == (arms, invalid)


def test_quat_to_euler_matches_scipy():
    rotation = pytest.importorskip("scipy.spatial.transform").Rotation
    rng = np.random.default_rng(0)
    for quat in rng.normal(size=(50, 4)):
        expected = rotation.from_quat(quat).as_euler("xyz")
        np.testing.assert_allclose(quat_xyzw_to_euler_xyz(quat), expected, atol=1e-5)


def test_split_joint_threshold_accepts_7_or_14_values():
    left, right = split_joint_threshold([0.1] * 7, 7, 7)
    np.testing.assert_array_equal(left, right)
    left, right = split_joint_threshold(list(range(14)), 7, 7)
    assert left[0] == pytest.approx(1e-6) and right[0] == 7
    with pytest.raises(ValueError):
        split_joint_threshold([0.1] * 5, 7, 7)


def test_adaptive_upsample_maps_source_indices():
    left = np.zeros((3, 7), dtype=np.float32)
    left[1, 0] = 0.25  # 2.5 thresholds -> factor 3 into and out of index 1
    left_out, right_out, source_to_expanded, factors = adaptive_delta_upsample_chunks(
        left, left.copy(), 1, 4, [0.1] * 7)
    assert factors == [3, 3]
    assert source_to_expanded == [0, 3, 6]
    np.testing.assert_allclose(left_out[source_to_expanded], left)
    assert right_out.shape == left_out.shape


def test_initial_bridge_keeps_gripper_target_immediate():
    left = np.full((2, 7), 1.0, dtype=np.float32)
    command = np.zeros(7, dtype=np.float32)
    left_out, _, mapping, factor = adaptive_bridge_to_first_action(
        left, left.copy(), command, command, 1, 4, [0.5] * 7, [0, 1])
    assert factor == 2
    assert mapping == [1, 2]
    np.testing.assert_allclose(left_out[0, :6], 0.5)
    assert left_out[0, 6] == 1.0


def test_lowpass_preserves_endpoints_and_gripper():
    t = np.arange(24, dtype=np.float32)[:, None]
    chunk = np.hstack([np.sin(t) * 0.1 + t * 0.01] * 6 + [np.where(t > 10, 1.0, 0.0)]).astype(np.float32)
    out = lowpass_action_chunk_zero_phase(chunk, 15.0, 1.2, order=6, preserve_endpoints=True)
    np.testing.assert_allclose(out[[0, -1], :6], chunk[[0, -1], :6], atol=1e-5)
    np.testing.assert_array_equal(out[:, 6], chunk[:, 6])
    assert np.abs(np.diff(out[:, 0], 2)).max() < np.abs(np.diff(chunk[:, 0], 2)).max()
    with pytest.raises(ValueError):
        lowpass_action_chunk_zero_phase(chunk, 15.0, 8.0)


def test_monotonic_projection_removes_reversals():
    start = np.zeros(7, dtype=np.float32)
    chunk = np.zeros((5, 7), dtype=np.float32)
    chunk[:, 0] = [0.2, 0.1, 0.3, 0.25, 0.4]
    out = project_action_chunk_monotonic_to_endpoint(chunk, start, strength=1.0)
    assert np.all(np.diff(np.r_[0.0, out[:, 0]]) >= -1e-7)
    assert out[-1, 0] == pytest.approx(0.4)


def test_temporal_ensemble_prefers_newer_chunks():
    old = {"start_step": 0, "request_id": 1, "left": np.zeros((4, 7)), "right": np.zeros((4, 7))}
    new = {"start_step": 2, "request_id": 2, "left": np.ones((4, 7)), "right": np.ones((4, 7))}
    left, _, count, weights = exponential_temporal_ensemble([old, new], 2, 2, decay=1.0, max_candidate_age=None)
    assert count == 2
    assert weights[1] > weights[0]
    assert left[0] == pytest.approx(weights[1])


def test_gripper_threshold_and_action_mode_validation():
    left, right = threshold_gripper_targets(
        np.array([0, 0.9]), np.array([0, 0.1]),
        np.array([0.2, 0.2]), np.array([0.8, 0.8]), np.array([-1, -1]), np.array([2, 2]))
    assert left[-1] == 2 and right[-1] == -1
    validate_policy_action_mode({}, "absolute")
    validate_policy_action_mode({"action_mode": "ABSOLUTE"}, "absolute")
    with pytest.raises(ValueError):
        validate_policy_action_mode({"action_mode": "eef_absolute"}, "absolute")
