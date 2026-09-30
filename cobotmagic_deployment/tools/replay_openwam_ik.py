"""Offline IK replay. No ROS imports, network calls, or robot commands.

Recorded final joint targets are available, but per-command measured seeds were
not logged in the source run. Sequential replay therefore assumes ideal tracking
of the previous NEW command. It is not a reconstruction of hardware execution.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import yaml

from cobotmagic_deployment.common.piper_ik import PiperNumericalIK


def stats(values):
    return {'median': float(np.median(values)), 'max': float(np.max(values))}


def replay(cfg, fixture):
    ik = PiperNumericalIK(cfg)
    # These historical fixtures were published with the legacy frame mapping.
    recorded_urdf = Path(__file__).resolve().parents[2] / 'tests/data/piper_ik_chain.urdf'
    recorded_ik = PiperNumericalIK(dict(cfg, urdf_path=str(recorded_urdf),
                                      calibration_mode='base', joint_signs=[1] * 6))
    full_cfg = dict(cfg, clip_to_max_joint_delta=False)
    full = PiperNumericalIK(full_cfg)
    for side, calibration in fixture['calibration'].items():
        for instance in (ik, full, recorded_ik):
            instance.calibrate(side, calibration['joints'], calibration['eef'])
    result = {
        'source': fixture['source'],
        'assumption': 'Sequential replay uses previous new command as measured seed (ideal tracking). No robot I/O.',
        'config': cfg, 'arms': {},
    }
    for side, calibration in fixture['calibration'].items():
        q = np.array(calibration['joints'])
        records = []
        for row in fixture['commands']:
            saved = row[side]
            _, pos, _ = recorded_ik._error(recorded_ik._model_target(side, saved['target']), saved['published_joints'])
            recorded_error = float(np.linalg.norm(pos))
            # Demonstrate whether the old published pose can be refined, subject
            # to URDF limits but without per-command delta limits.
            refinement = full.solve(side, saved['target'], saved['published_joints'])
            seed = q.copy()
            solved = ik.solve(side, saved['target'], seed)
            q = solved['joints'][:6].astype(float) if solved['acceptable'] else seed
            records.append({
                'request_id': row['request_id'], 'action_index': row['action_index'],
                'target': saved['target'], 'seed': seed.tolist(),
                'recorded_error_m': recorded_error,
                'recorded_residual_reproduction_error_m': abs(recorded_error - saved['recorded_position_error_m']),
                'refinement_error_m': refinement['position_error_m'],
                'refinement_rotation_error_rad': refinement['orientation_error_rad'],
                **{k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in solved.items()},
            })
        result['arms'][side] = {
            'recorded_position_mm': stats([r['recorded_error_m'] * 1000 for r in records]),
            'recorded_residual_reproduction_max_mm': max(r['recorded_residual_reproduction_error_m'] * 1000 for r in records),
            'refinement_position_mm': stats([r['refinement_error_m'] * 1000 for r in records]),
            'sequential_solution_position_mm': stats([r['solution_position_error_m'] * 1000 for r in records]),
            'sequential_command_position_mm': stats([r['position_error_m'] * 1000 for r in records]),
            'sequential_command_orientation_deg': stats([np.rad2deg(r['orientation_error_rad']) for r in records]),
            'sequential_solve_ms': stats([r['solve_time_sec'] * 1000 for r in records]),
            'joint_delta_limited_count': sum(r['joint_delta_limited'] for r in records),
            'rejected_count': sum(not r['acceptable'] for r in records),
            'commands': records,
        }
    return result


def main():
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=root / 'cobotmagic_deployment/configs/config_openwam_piper.yaml')
    parser.add_argument('--fixture', type=Path, default=root / 'tests/data/openwam_ik_084939.json')
    parser.add_argument('--output', type=Path, default=root / 'logs/openwam_ik_diagnosis/solver_fix_084939.json')
    args = parser.parse_args()
    cfg = yaml.safe_load(args.config.read_text())['ros']['eef_ik']
    report = replay(cfg, json.loads(args.fixture.read_text()))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({s: {k: v for k, v in arm.items() if k != 'commands'} for s, arm in report['arms'].items()}, indent=2))
    print('Report:', args.output)


if __name__ == '__main__':
    main()
