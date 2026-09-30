import numpy as np
import pytest
from cobotmagic_deployment.common.gripper_hysteresis import GripperHysteresis

CFG = dict(closed=[0.,0.], open=[.06558,.06558], close_threshold_normalized=[.55,.55],
           open_threshold_normalized=[.7,.7], confirm_steps=2)

def proposal(c, values, measured=(0.,0.)):
    return c.propose(np.array(values)*.06558, measured)

def test_confirm_deadband_and_independent_arms():
    c=GripperHysteresis(CFG)
    v,c,_=proposal(c,[.9,.1]);assert list(v)==[0,0]
    v,c,_=proposal(c,[.9,.1]);assert v[0]==.06558 and v[1]==0
    for _ in range(10):
        v,c,_=proposal(c,[.65,.65]);assert v[0]==.06558 and v[1]==0
    v,c,_=proposal(c,[.495,.9]);assert v[0]==.06558 and v[1]==0
    v,c,_=proposal(c,[.495,.9]);assert v[0]==0 and v[1]==.06558

def test_skipped_publish_does_not_advance_state():
    c=GripperHysteresis(CFG)
    _,discarded,_=proposal(c,[.9,.9])
    assert c.is_open is None
    v,c,_=proposal(c,[.9,.9]);assert np.all(v==0)
    v,c,_=proposal(c,[.9,.9]);assert np.all(v==.06558)

def test_single_spike_and_interrupted_confirmation():
    c=GripperHysteresis(CFG)
    for values in ([.9,.9],[.6,.6],[.9,.9],[.6,.6]):
        v,c,_=proposal(c,values);assert np.all(v==0)

def test_initial_measured_open_and_clipping():
    c=GripperHysteresis(CFG)
    v,c,_=proposal(c,[.65,2.],(.06558,-.0005));assert list(v)==[.06558,0]
    v,c,_=proposal(c,[-.01,2.]);assert list(v)==[.06558,.06558]
    v,c,_=proposal(c,[-.01,.65]);assert list(v)==[0,.06558]

@pytest.mark.parametrize('change',[{'confirm_steps':0},{'confirm_steps':1.5},{'open':[0,0]},
    {'open_threshold_normalized':[.5,.5]},{'closed':[float('nan'),0]}])
def test_invalid_configuration(change):
    with pytest.raises(ValueError):GripperHysteresis(dict(CFG,**change))

def test_invalid_input():
    c=GripperHysteresis(CFG)
    with pytest.raises(ValueError):proposal(c,[float('nan'),.9])

REL = dict(CFG, close_threshold_normalized=[.45,.45],
           open_threshold_normalized=[.55,.55], request_relative={'enabled':True})


def relative(c, command, measured, request):
    return c.propose(np.array(command)*.06558, np.array(measured)*.06558,
                     request_opening=np.array(request)*.06558)


def test_relative_closed_start_opens_without_immediate_feedback_reversal():
    c=GripperHysteresis(REL)
    _,c,_=relative(c,[.60,.02],[0,0],[0,0])
    v,c,_=relative(c,[.60,.02],[0,0],[0,0])
    assert list(v)==[.06558,0]
    for _ in range(3):
        v,c,d=relative(c,[.60,.02],[1,0],[0,0])
        assert list(v)==[.06558,0]
        assert d['close_threshold_normalized']==[.45,.45]


def test_relative_open_request_closes_and_does_not_reopen_at_same_input():
    c=GripperHysteresis(REL)
    _,c,_=relative(c,[.79,.02],[1,0],[1,0])
    v,c,d=relative(c,[.79,.02],[1,0],[1,0])
    assert np.all(v==0)
    assert d['close_threshold_normalized'][0]==pytest.approx(.82)
    for _ in range(3):
        v,c,_=relative(c,[.79,.02],[0,0],[1,0])
        assert np.all(v==0)
    # A new inference observing closed gripper can request opening again.
    _,c,_=relative(c,[.60,.02],[0,0],[0,0])
    v,c,_=relative(c,[.60,.02],[0,0],[0,0])
    assert list(v)==[.06558,0]


def test_relative_discarded_proposal_does_not_close():
    c=GripperHysteresis(REL)
    _,discarded,_=relative(c,[.79,.02],[1,0],[1,0])
    assert c.is_open is None
    v,c,_=relative(c,[.79,.02],[1,0],[1,0])
    assert v[0]==.06558


def test_relative_requires_request_reference():
    c=GripperHysteresis(REL)
    with pytest.raises(ValueError,match='request_opening'):
        c.propose([0,0],[0,0])


def test_hold_resyncs_state_to_published_value():
    h = GripperHysteresis(CFG)
    _, candidate, diag = h.propose([0.0, 0.0], [0.07, 0.07])
    _, candidate, diag = candidate.propose([0.0, 0.0], [0.07, 0.07])
    assert diag['switched'] == [True, True]
    candidate.hold(0, 0.07, diag)
    assert candidate.is_open.tolist() == [True, False]
    assert diag['output_open'] == [True, False]
    assert diag['switched'] == [False, True]
    assert candidate.count.tolist() == [0, 0]
