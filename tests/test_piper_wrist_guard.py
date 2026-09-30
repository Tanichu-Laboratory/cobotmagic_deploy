from pathlib import Path
import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from cobotmagic_deployment.common.piper_ik import PiperNumericalIK

URDF=str(Path(__file__).resolve().parents[1]/'cobotmagic_deployment/assets/piper_description_cobotmagic_joint5_pitch0.urdf')

def make(enabled=True, **guard):
    return PiperNumericalIK(dict(urdf_path=URDF, solver='least_squares',
        joint_regularization_weight=0, joint_signs=[1,1,1,-1,1,-1],
        max_joint_delta_rad=[.18,.18,.18,.24,.24,.24],
        wrist_singularity_avoidance=dict(enabled=enabled,**guard)))

def pose(ik,q):
    t=ik.fk(q)
    return np.r_[t[:3,3],Rotation.from_matrix(t[:3,:3]).as_euler('xyz'),.03]

@pytest.mark.parametrize('side',['left','right'])
@pytest.mark.parametrize('sign',[1,-1])
def test_near_singularity_preserves_branch_and_position(side,sign):
    ik=make();seed=np.array([.2,.7,-.65,.2,sign*.14,.1]);goal=seed.copy();goal[4]=sign*.01
    a=ik.solve(side,pose(ik,goal),seed,seed)
    assert a['acceptable'] and a['wrist_singularity_avoidance']['active']
    assert sign*a['joints'][4]>=np.deg2rad(6)-1e-7
    assert a['position_error_m']<.003001
    assert a['orientation_error_rad']<=np.deg2rad(8)+1e-6
    assert max(abs(a['joints'][[3,5]]-seed[[3,5]]))<=.100001
    assert a['joints'][6]==np.float32(.03)
    assert np.all(abs(a['joints'][:6]-seed)<=ik.max_joint_delta+1e-7)

def test_ordinary_pose_is_unchanged():
    seed=np.array([-.395,.833,-.67,-.076,.494,.237]);goal=seed+np.array([.02,.02,-.02,.01,.01,-.01])
    guarded=make();original=make(False);cmd=pose(original,goal)
    a=guarded.solve('left',cmd,seed);b=original.solve('left',cmd,seed)
    assert not a['wrist_singularity_avoidance']['active']
    np.testing.assert_array_equal(a['joints'],b['joints'])

def test_unreachable_goal_is_rejected_without_singular_fallback():
    ik=make();seed=np.array([.2,.7,-.65,.2,.05,.1]);cmd=pose(ik,seed);cmd[0]+=2
    a=ik.solve('left',cmd,seed,seed)
    assert a['wrist_singularity_avoidance']['active'] and not a['acceptable']

def test_previous_published_bound_conflict_rejects():
    ik=make();seed=np.array([.2,.7,-.65,.2,.05,.1]);reference=seed.copy();reference[3]+=1
    a=ik.solve('left',pose(ik,seed),seed,reference)
    assert not a['acceptable']
    assert a['wrist_singularity_avoidance']['reason']=='incompatible_joint_bounds'

def test_solve_does_not_commit_reference_on_failure_or_other_arm():
    ik=make();seed=np.array([.2,.7,-.65,.2,.14,.1]);cmd=pose(ik,seed)
    a=ik.solve('left',cmd,seed,seed);bad=cmd.copy();bad[0]+=2
    ik.solve('right',bad,seed,seed)
    b=ik.solve('left',cmd,seed,seed)
    np.testing.assert_array_equal(a['joints'],b['joints'])

@pytest.mark.parametrize('options',[{'min_bend_rad':.5},{'max_position_error_m':float('nan')},{'max_wrist_command_delta_rad':0}])
def test_invalid_limits_fail_closed(options):
    with pytest.raises(ValueError):make(**options)


def test_whole_arm_trigger_with_bent_wrist_and_safe_hold():
    ik=make()
    seed=np.array([-.395,.833,-.67,-.076,.494,.237,0.02])
    goal=np.array([-.4,.4,-.2,1.2,.5,-1.3])
    assert abs(goal[4]) > ik.wrist_guard['trigger_rad']
    assert ik.condition_number('left',goal) > ik.wrist_guard['trigger_condition']
    a=ik.solve('left',pose(ik,goal),seed,seed)
    assert a['wrist_singularity_avoidance']['active']
    assert a['acceptable']
    assert a['wrist_singularity_avoidance']['command_condition'] <= 40.0004
    if a['wrist_singularity_avoidance']['mode']=='hold_last_safe':
        assert not a['target_reached']
        np.testing.assert_allclose(a['joints'],seed,atol=1e-7)


def test_geometric_condition_matches_finite_difference():
    ik=make();q=np.array([-.2,.7,-.4,.4,.5,-.3]);t=ik.fk(q);columns=[]
    for j in range(6):
        v=q.copy();v[j]+=1e-6;u=ik.fk(v)
        columns.append(np.r_[(u[:3,3]-t[:3,3])/1e-6,
            .25*Rotation.from_matrix(u[:3,:3]@t[:3,:3].T).as_rotvec()/1e-6])
    sv=np.linalg.svd(np.array(columns).T,compute_uv=False)
    assert ik.condition_number('left',q)==pytest.approx(sv[0]/sv[-1],rel=1e-4)


@pytest.mark.parametrize('condition,can_hold',[(40.0000062,True),(40.001,False)])
def test_float32_boundary_uses_same_tolerance_for_hold(monkeypatch,condition,can_hold):
    ik=make();seed=np.array([-.395,.833,-.67,-.076,.494,.237,.02])
    cmd=pose(ik,seed);cmd[0]+=2
    monkeypatch.setattr(ik,'condition_number',lambda side,q:condition)
    result=ik.solve('left',cmd,seed,seed)
    assert result['acceptable'] == can_hold
    assert not result['target_reached']
    if can_hold:
        assert result['wrist_singularity_avoidance']['mode']=='hold_last_safe'
        np.testing.assert_allclose(result['joints'],seed,atol=1e-7)
