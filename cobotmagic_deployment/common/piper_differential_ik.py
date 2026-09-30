"""Bounded resolved-rate IK, using the calibrated Piper geometric Jacobian.

Independent implementation of damped differential IK / box-constrained least
squares. No ROS, Pinocchio, firmware mode changes, or global branch search.
"""
import time
import numpy as np
from scipy.optimize import lsq_linear
from scipy.spatial.transform import Rotation


class DifferentialIK:
    def __init__(self, model, cfg):
        self.model = model
        self.period = float(cfg.get('control_period_sec', .2))
        self.vmax = self._vector(cfg.get('max_velocity_rad_s', [.6,.6,.6,.8,.8,.8]))
        self.amax = self._vector(cfg.get('max_acceleration_rad_s2', [3.,3.,3.,4.,4.,4.]))
        self.iterations = int(cfg.get('iterations', 8))
        self.sigma_soft = float(cfg.get('singular_value_soft', .025))
        self.damping = float(cfg.get('singular_damping', .02))
        self.rotation_scale = float(cfg.get('rotation_scale_m', .15))
        self.position_tolerance = float(cfg.get('position_tolerance_m', .003))
        self.orientation_tolerance = float(cfg.get('orientation_tolerance_rad', .14))
        if (not all(np.isfinite(x) and x > 0 for x in [self.period,self.sigma_soft,self.damping,
                self.rotation_scale,self.position_tolerance,self.orientation_tolerance])
                or not 1 <= self.iterations <= 30):
            raise ValueError('invalid differential IK settings')
        self.history = {}
        self.rest = {}

    @staticmethod
    def _vector(value):
        a = np.asarray(value, dtype=float)
        if a.shape != (6,) or not np.isfinite(a).all() or np.any(a <= 0):
            raise ValueError('velocity/acceleration limits must be six positive finite values')
        return a

    def reset(self, side, joints):
        self.history.pop(side, None)
        self.rest[side] = np.asarray(joints,dtype=float)[:6].copy()

    def commit(self, side, result):
        if result.get('acceptable'):
            self.history[side] = (np.asarray(result['joints'][:6],dtype=float).copy(),
                                  np.asarray(result['differential_ik']['velocity_rad_s'],dtype=float))

    def solve(self, side, command, measured, reference=None, dt=None):
        started = time.perf_counter()
        ik = self.model
        measured = np.asarray(measured,dtype=float)
        command = np.asarray(command,dtype=float)
        previous = measured[:6].copy() if reference is None else np.asarray(reference,dtype=float)[:6].copy()
        if previous.shape != (6,) or not np.isfinite(previous).all():
            raise ValueError('reference must have six finite angles')
        elapsed = self.period if dt is None else float(dt)
        if not np.isfinite(elapsed) or elapsed <= 0:
            raise ValueError('dt must be finite and positive')
        step_time = min(elapsed,self.period)  # no catch-up jump after policy wait
        old = self.history.get(side)
        velocity = np.zeros(6)
        if old is not None and elapsed <= 2*self.period and np.allclose(previous,old[0],atol=1e-6,rtol=0):
            velocity = old[1].copy()
        target = ik._model_target(side,command)
        tool = ik.calibration.get(side) if ik.calibration_mode == 'tool' else None
        # URDF and measured tracking bounds always remain hard.
        hard_lo = np.maximum(ik.lower, measured[:6]-ik.max_joint_delta)
        hard_hi = np.minimum(ik.upper, measured[:6]+ik.max_joint_delta)
        lo = np.maximum(hard_lo, previous-self.vmax*step_time)
        hi = np.minimum(hard_hi, previous+self.vmax*step_time)
        accel_lo = previous + (velocity-self.amax*step_time)*step_time
        accel_hi = previous + (velocity+self.amax*step_time)*step_time
        lower = np.maximum(lo,accel_lo)
        upper = np.minimum(hi,accel_hi)
        acceleration_override = bool(np.any(lower > upper))
        if acceleration_override:
            # Position/measurement limits outrank smoothness. Explicitly report
            # emergency deceleration rather than silently violating a limit.
            lower,upper = lo,hi
        feasible = bool(np.all(lower <= upper))
        q = np.clip(previous,lower,upper) if feasible else previous.copy()
        last_sigma = 0.0
        iterations = 0
        if feasible:
            for iterations in range(1,self.iterations+1):
                _,pos,rot = ik._error(target,q,tool)
                jac = ik.geometric_jacobian(side,q)
                weighted_jac = np.vstack([jac[:3],self.rotation_scale*jac[3:]])
                u,s,vh = np.linalg.svd(weighted_jac,full_matrices=False)
                last_sigma = float(s[-1])
                # Dampen only ill-conditioned singular directions, preserving
                # motion in the well-conditioned directions.
                lam = .0005 + self.damping*np.clip(1-(s/self.sigma_soft)**2,0,1)
                reg = np.diag(lam) @ vh
                # Damping regularizes each incremental correction, not the
                # absolute distance from a historic pose. Re-linearization can
                # therefore reduce residual without a permanent posture bias.
                a = np.vstack([weighted_jac,reg])
                b = np.r_[pos,self.rotation_scale*rot,np.zeros(6)]
                # Local trust region prevents a rotation-log linearization
                # from traversing a different wrist branch in one iteration.
                dlo=np.maximum(lower-q,-.06);dhi=np.minimum(upper-q,.06)
                free=dhi-dlo>1e-10
                delta=np.clip(np.zeros(6),dlo,dhi)
                if not np.any(free):
                    break
                result=lsq_linear(a[:,free],b-a[:,~free]@delta[~free],
                                  bounds=(dlo[free],dhi[free]),method='bvls',tol=1e-10,max_iter=40)
                if not np.isfinite(result.x).all():
                    feasible=False
                    break
                delta[free]=result.x
                q=np.clip(q+delta,lower,upper)
                if np.linalg.norm(delta)<1e-6:
                    break
        output=np.r_[q,command[6]].astype(np.float32)
        # Check transmitted float32 values, not just optimizer doubles.
        safe=bool(feasible and np.isfinite(output).all()
                  and np.all(output[:6]>=lower-1e-7) and np.all(output[:6]<=upper+1e-7))
        _,pos,rot=ik._error(target,output[:6],tool)
        pn,rn=float(np.linalg.norm(pos)),float(np.linalg.norm(rot))
        reached=bool(pn<=self.position_tolerance and rn<=self.orientation_tolerance)
        movement=float(np.linalg.norm(output[:6]-previous))
        mode='tracking' if reached else ('progressing' if movement>1e-5 else 'stalled')
        if not safe:mode='rejected_joint_bounds'
        pose=ik.eef_fk(side,output[:6])
        # Name the actual active constraint instead of calling every box
        # boundary (including the URDF limit) a post-IK delta clip.
        def active_indices(bound_lo, bound_hi):
            active = (np.abs(output[:6]-bound_lo) < 1e-6) | (np.abs(output[:6]-bound_hi) < 1e-6)
            return np.flatnonzero(active).tolist()
        active_limits = {
            'joint_position': active_indices(ik.lower, ik.upper),
            'measured_tracking': active_indices(measured[:6]-ik.max_joint_delta,
                                                measured[:6]+ik.max_joint_delta),
            'velocity': active_indices(previous-self.vmax*step_time, previous+self.vmax*step_time),
            'acceleration': ([] if acceleration_override else active_indices(accel_lo, accel_hi)),
        }
        motion_limited = any(active_limits[k] for k in ('measured_tracking','velocity','acceleration'))
        diagnostic={'mode':mode,'dt_sec':step_time,'elapsed_sec':elapsed,
                    'active_limits': active_limits,
                    'measured_command_delta_rad': (output[:6]-measured[:6]).astype(float).tolist(),

                    'velocity_rad_s':((output[:6]-previous)/step_time).tolist(),
                    'acceleration_override':acceleration_override,'sigma_min':last_sigma,
                    'command_condition':ik.condition_number(side,output[:6]),
                    'measured_condition':ik.condition_number(side,measured[:6]),
                    'position_error_m':pn,'orientation_error_rad':rn}
        return {'joints':output,'solver':'differential','calibration_mode':ik.calibration_mode,
                'joint_signs':ik.joint_signs.tolist(),'acceptable':safe,'target_reached':reached,
                'converged':reached,'solution_acceptable':safe,
                'solution_joints':output[:6].astype(float).tolist(),
                'solution_position_error_m':pn,'solution_orientation_error_rad':rn,
                'position_error_m':pn,'orientation_error_rad':rn,
                'command_fk_xyz':pose[:3,3].tolist(),
                'command_fk_rpy':Rotation.from_matrix(pose[:3,:3]).as_euler('xyz').tolist(),
                'joint_delta_limited':bool(motion_limited),
                'joint_delta_polished':False,'joint_delta_norm':float(np.linalg.norm(output[:6]-measured[:6])),
                'unclipped_joint_delta_norm':movement,'iterations':iterations,
                'solve_time_sec':time.perf_counter()-started,'differential_ik':diagnostic,
                'wrist_singularity_avoidance':{'active':False,'mode':'replaced_by_differential_ik'}}
