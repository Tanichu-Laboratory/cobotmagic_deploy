from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from cobotmagic_deployment.common.piper_ik import PiperNumericalIK, rotation_error_vector

URDF = str(Path(__file__).parent / 'data/piper_ik_chain.urdf')
SEED = np.array([0.2, 0.7, -0.65, 0.2, 0.5, 0.1])


def solver(**kw):
    cfg = dict(urdf_path=URDF, solver='least_squares', joint_regularization_weight=0.,
               max_position_error_m=.005, max_orientation_error_rad=.05,
               max_joint_delta_rad=[.18, .18, .18, .24, .24, .24])
    cfg.update(kw)
    return PiperNumericalIK(cfg)


def pose(ik, q):
    t = ik.fk(q)
    return np.r_[t[:3, 3], Rotation.from_matrix(t[:3, :3]).as_euler('xyz'), .03]


def test_reachable_pose_is_solved_without_seed_bias():
    ik = solver()
    target = pose(ik, SEED + [.08, .07, -.06, .08, -.06, -.09])
    result = ik.solve('left', target, SEED)
    assert result['acceptable']
    assert result['position_error_m'] < 1e-6
    assert result['orientation_error_rad'] < 1e-6
    assert result['joints'][6] == np.float32(.03)
    legacy = solver(solver='damped_least_squares', joint_regularization_weight=.08)
    assert legacy.solve('left', target, SEED)['position_error_m'] > 1e-4


def test_converged_solution_does_not_bypass_post_limit_residual_check():
    ik = solver(max_joint_delta_rad=.001, max_position_error_m=1e-4,
                max_orientation_error_rad=1e-4)
    target = pose(ik, SEED + [.12, .1, -.1, .1, -.1, .1])
    result = ik.solve('left', target, SEED)
    assert result['solution_position_error_m'] < 1e-6
    assert result['solution_orientation_error_rad'] < 1e-6
    assert result['joint_delta_limited']
    assert not result['converged']
    assert not result['acceptable']
    assert np.all(np.abs(result['joints'][:6] - SEED) <= .001 + 1e-7)


def test_bounded_refinement_never_worsens_weighted_clip_error():
    ik = solver(max_joint_delta_rad=[.02, .03, 0., .04, .04, .04])
    target = pose(ik, SEED + [.15, .12, -.1, .1, -.1, .1])
    result = ik.solve('left', target, SEED)
    clipped = np.clip(result['solution_joints'], SEED - ik.max_joint_delta,
                      SEED + ik.max_joint_delta)
    model_target = ik._model_target('left', target)
    before = np.linalg.norm(ik._error(model_target, clipped)[0])
    after = np.linalg.norm(ik._error(model_target, result['joints'])[0])
    assert result['joint_delta_limited']
    assert after <= before + 1e-7
    assert np.all(np.abs(result['joints'][:6] - SEED) <= ik.max_joint_delta + 1e-7)
    assert np.all(result['joints'][:6] >= ik.lower - 1e-7)
    assert np.all(result['joints'][:6] <= ik.upper + 1e-7)


def test_unreachable_target_is_rejected():
    ik = solver()
    target = pose(ik, SEED)
    target[0] += 2.
    result = ik.solve('left', target, SEED)
    assert not result['solution_acceptable']
    assert not result['acceptable']


@pytest.mark.parametrize('bad', [np.nan, np.inf, -np.inf])
def test_invalid_inputs_do_not_produce_commands(bad):
    ik = solver()
    target = pose(ik, SEED)
    target[0] = bad
    with pytest.raises(ValueError, match='finite'):
        ik.solve('left', target, SEED)
    seed = SEED.copy()
    seed[0] = bad
    with pytest.raises(ValueError, match='finite'):
        ik.solve('left', pose(ik, SEED), seed)


@pytest.mark.parametrize('angle', [1e-10, .4, np.pi - 1e-9, np.pi])
def test_rotation_residual_near_zero_and_pi(angle):
    axis = np.array([1., -2., 3.]); axis /= np.linalg.norm(axis)
    target = Rotation.from_rotvec(axis * angle).as_matrix()
    error = rotation_error_vector(target, np.eye(3))
    np.testing.assert_allclose(Rotation.from_rotvec(error).as_matrix(), target, atol=1e-10)
    assert np.linalg.norm(error) == pytest.approx(angle, abs=1e-10)


def test_diagnostics_use_transmitted_angles_and_calibrated_frame():
    ik = solver()
    measured = pose(ik, SEED)
    measured[:3] += [.1, -.2, .05]
    measured[3:6] += [.1, -.1, .2]
    ik.calibrate('left', SEED, measured)
    goal_q = SEED + [.05, -.03, -.04, .02, .02, -.03]
    goal = ik.calibration['left'] @ ik.fk(goal_q)
    target = np.r_[goal[:3, 3], Rotation.from_matrix(goal[:3, :3]).as_euler('xyz'), .01]
    seed_before = SEED.copy()
    result = ik.solve('left', target, SEED)
    command = ik.calibration['left'] @ ik.fk(result['joints'])
    np.testing.assert_allclose(result['command_fk_xyz'], command[:3, 3], atol=1e-12)
    assert result['position_error_m'] == pytest.approx(np.linalg.norm(command[:3, 3] - target[:3]), abs=1e-12)
    np.testing.assert_array_equal(SEED, seed_before)


def test_pose_solver_rejects_accidental_regularization():
    with pytest.raises(ValueError, match='joint_regularization_weight=0'):
        solver(joint_regularization_weight=.08)


def test_all_zero_delta_limits_hold_all_joints():
    ik = solver(max_joint_delta_rad=0.)
    result = ik.solve('left', pose(ik, SEED + .05), SEED)
    assert result['joint_delta_limited']
    np.testing.assert_array_equal(result['joints'][:6], SEED.astype(np.float32))


@pytest.mark.parametrize('side,max_command_mm', [('left', 21.), ('right', 7.5)])
def test_saved_62_step_chunk_regression(side, max_command_mm):
    import json
    fixture = json.loads((Path(__file__).parent / 'data/openwam_ik_084939.json').read_text())
    ik = solver(max_position_error_m=.04, max_orientation_error_rad=.8,
                solution_max_position_error_m=.005, solution_max_orientation_error_rad=.05)
    calibration = fixture['calibration'][side]
    ik.calibrate(side, calibration['joints'], calibration['eef'])
    q = np.array(calibration['joints'])
    errors = []
    for row in fixture['commands']:
        saved = row[side]
        # Check that the replay reconstructs the original runtime calibration.
        _, pe, _ = ik._error(ik._model_target(side, saved['target']), saved['published_joints'])
        assert np.linalg.norm(pe) == pytest.approx(saved['recorded_position_error_m'], abs=5e-7)
        result = ik.solve(side, saved['target'], q)
        assert result['acceptable']
        assert result['solution_position_error_m'] <= .005
        assert result['solution_orientation_error_rad'] <= .05
        assert np.all(np.abs(result['joints'][:6] - q) <= ik.max_joint_delta + 1e-7)
        assert np.all(result['joints'][:6] >= ik.lower - 1e-7)
        assert np.all(result['joints'][:6] <= ik.upper + 1e-7)
        errors.append(result['position_error_m'] * 1000)
        q = result['joints'][:6].copy()  # Explicit ideal-tracking assumption.
    assert np.median(errors) < .001
    assert max(errors) < max_command_mm


def test_tool_calibration_preserves_base_and_rotates_with_tip():
    from cobotmagic_deployment.common.piper_ik import transform_from_xyz_rpy
    ik = solver(calibration_mode='tool', joint_signs=[1, 1, 1, -1, 1, -1])
    tool = transform_from_xyz_rpy([.005, .01, -.002], [.1, -.09, -1.57])
    measured = ik.fk(SEED) @ tool
    initial = np.r_[measured[:3, 3], Rotation.from_matrix(measured[:3, :3]).as_euler('xyz'), .01]
    ik.calibrate('left', SEED, initial)
    np.testing.assert_allclose(ik.calibration['left'], tool, atol=1e-12)
    q_goal = SEED + [.04, .05, -.03, .04, -.05, .06]
    desired = ik.fk(q_goal) @ tool
    target = np.r_[desired[:3, 3], Rotation.from_matrix(desired[:3, :3]).as_euler('xyz'), .01]
    result = ik.solve('left', target, SEED)
    actual = ik.eef_fk('left', result['joints'])
    assert result['acceptable']
    assert result['position_error_m'] < 1e-6
    np.testing.assert_allclose(actual, desired, atol=1e-6)
    np.testing.assert_allclose(result['command_fk_xyz'], actual[:3, 3], atol=1e-12)


def test_joint_sign_mapping_also_maps_asymmetric_limits():
    original = solver()
    mapped = solver(joint_signs=[1, -1, -1, 1, 1, 1])
    signs = mapped.joint_signs
    np.testing.assert_allclose(mapped.lower, np.minimum(original.lower * signs, original.upper * signs))
    np.testing.assert_allclose(mapped.upper, np.maximum(original.lower * signs, original.upper * signs))
    np.testing.assert_allclose(mapped.fk(SEED * signs), original.fk(SEED))


def test_tool_calibration_is_required_before_solve():
    ik = solver(calibration_mode='tool')
    with pytest.raises(ValueError, match='requires calibration'):
        ik.solve('left', pose(ik, SEED), SEED)


@pytest.mark.parametrize('side,max_mm,max_deg', [('left', 3.8, 2.4), ('right', 5.7, 3.5)])
def test_corrected_fk_matches_independent_measured_poses(side, max_mm, max_deg):
    import json
    fixture = json.loads((Path(__file__).parent / 'data/piper_measured_frames_20260917.json').read_text())
    ik = solver(calibration_mode='tool', joint_signs=[1, 1, 1, -1, 1, -1])
    initial = fixture['samples'][fixture['calibration_index']][side]
    ik.calibrate(side, initial['joints'], initial['eef'])
    errors = []
    for i, sample in enumerate(fixture['samples']):
        if i == fixture['calibration_index']:
            continue
        measured = sample[side]
        prediction = ik.eef_fk(side, measured['joints'])
        errors.append(np.linalg.norm(prediction[:3, 3] - measured['eef'][:3]) * 1000)
        actual_rot = Rotation.from_euler('xyz', measured['eef'][3:6]).as_matrix()
        rotation_error = np.rad2deg(np.linalg.norm(rotation_error_vector(actual_rot, prediction[:3, :3])))
        assert errors[-1] < max_mm, sample['source']
        assert rotation_error < max_deg, sample['source']
    assert np.median(errors) < 1.2
