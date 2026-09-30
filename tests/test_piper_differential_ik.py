import numpy as np
import pytest
from test_piper_ik import solver, pose, SEED


def differential(**kw):
    return solver(solver='differential', **kw)


def test_reachable_target_converges_with_bounded_velocity_and_acceleration():
    ik = differential()
    q = SEED.copy()
    target = pose(ik, q + [.2,.15,-.15,.15,-.1,-.15])
    velocity = np.zeros(6)
    for _ in range(35):
        result = ik.solve('left', target, q, q, dt=.2)
        assert result['acceptable']
        new_velocity = (result['joints'][:6]-q)/.2
        assert np.all(abs(new_velocity) <= ik.differential.vmax+1e-6)
        assert np.all(abs(new_velocity-velocity) <= ik.differential.amax*.2+1e-6)
        ik.commit('left', result)
        q = result['joints'][:6].copy()
        velocity = new_velocity
    assert result['position_error_m'] < .001
    assert result['orientation_error_rad'] < .01


def test_unpublished_solve_does_not_change_history_and_pause_resets_velocity():
    ik = differential()
    target = pose(ik, SEED + [.2,.1,-.1,.1,.1,.1])
    a = ik.solve('left', target, SEED, dt=.2)
    assert ik.differential.history == {}
    b = ik.solve('left', target, SEED, dt=.2)
    np.testing.assert_array_equal(a['joints'], b['joints'])
    ik.commit('left', a)
    b = ik.solve('left', target, a['joints'], a['joints'], dt=10.)
    assert b['differential_ik']['dt_sec'] == .2
    assert np.all(abs(np.array(b['differential_ik']['velocity_rad_s'])) <= ik.differential.amax*.2+1e-6)


def test_singular_target_remains_finite_without_condition_hold():
    ik = differential()
    q = SEED.copy()
    q[4] = 0
    target = pose(ik, q)
    target[0] += .004
    a = ik.solve('right', target, q)
    assert a['acceptable']
    assert np.isfinite(a['joints']).all()
    assert a['wrist_singularity_avoidance']['mode'] != 'hold_last_safe'
    assert np.all(abs(a['joints'][:6]-q) <= ik.differential.vmax*.2+1e-7)


def test_incompatible_feedback_and_previous_command_is_rejected():
    ik = differential()
    ref = SEED.copy()
    ref[0] += 1
    a = ik.solve('left', pose(ik, SEED), SEED, ref)
    assert not a['acceptable']
    assert a['differential_ik']['mode'] == 'rejected_joint_bounds'


@pytest.mark.parametrize('dt', [0, -1, float('nan'), float('inf')])
def test_invalid_time_rejected(dt):
    ik = differential()
    with pytest.raises(ValueError):
        ik.solve('left', pose(ik, SEED), SEED, dt=dt)


def test_residual_is_not_misreported_as_target_reached():
    ik = differential()
    target = pose(ik, SEED)
    target[0] += 2
    a = ik.solve('left', target, SEED)
    assert a['acceptable']  # bounded partial step; not an exact IK solution
    assert not a['target_reached']
    assert a['position_error_m'] > 1
    assert np.all(a['joints'][:6] >= ik.lower-1e-7)
    assert np.all(a['joints'][:6] <= ik.upper+1e-7)


def test_exact_target_has_no_pull_towards_historic_posture():
    ik = differential()
    q = SEED.copy()
    q[4] = .02
    ik.differential.reset('left', SEED + [.1, .1, -.1, .2, .1, -.2])
    a = ik.solve('left', pose(ik, q), q)
    np.testing.assert_allclose(a['joints'][:6], q, atol=1e-7, rtol=0)
    assert a['position_error_m'] < 1e-7
    assert a['orientation_error_rad'] < 1e-7


def test_physical_joint_bound_is_not_reported_as_delta_limit():
    ik = differential()
    q = SEED.copy()
    q[0] = ik.upper[0]
    a = ik.solve('left', pose(ik, q), q)
    assert 0 in a['differential_ik']['active_limits']['joint_position']
    assert not a['joint_delta_limited']
    assert a['target_reached']


def test_measured_tracking_limit_is_identified():
    ik = differential(max_joint_delta_rad=.001)
    a = ik.solve('left', pose(ik, SEED + [.1,.1,-.1,.1,.1,.1]), SEED)
    assert a['differential_ik']['active_limits']['measured_tracking']
    assert not a['differential_ik']['active_limits']['velocity']
    assert a['joint_delta_limited']
    assert np.max(abs(a['joints'][:6]-SEED)) <= .001+1e-7
