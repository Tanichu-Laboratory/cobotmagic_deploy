"""One-step counterfactual using recorded feedback, previous command and velocity."""
import csv
import json
import sys
from pathlib import Path
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from cobotmagic_deployment.common.piper_ik import PiperNumericalIK

cfg = yaml.safe_load((ROOT/'cobotmagic_deployment/configs/config_openwam_piper.yaml').read_text())['ros']['eef_ik']
reports = {}
for stamp in (sys.argv[1:] or ['20260924_044441', '20260924_044524']):
    rows = list(csv.DictReader((ROOT/f'logs/action_commands/action_commands_{stamp}.csv').open()))
    report = {}
    for side in ['left', 'right']:
        ik = PiperNumericalIK(cfg)
        ik.calibrate(side, json.loads(rows[0]['ik_seed_joint_'+side]),
                     json.loads(rows[0]['command_'+side+'_before']))
        records = []
        for i, row in enumerate(rows):
            old = json.loads(row['ik_diagnostics_'+side])
            ref = json.loads(rows[i-1]['ik_joint_'+side]) if i else None
            if i:
                previous = json.loads(rows[i-1]['ik_diagnostics_'+side])
                previous['joints'] = ref
                ik.commit(side, previous)
            result = ik.solve(side, json.loads(row['target_'+side]),
                              json.loads(row['ik_seed_joint_'+side]), ref,
                              dt=old['differential_ik']['elapsed_sec'])
            records.append(dict(step=i, old_mm=old['position_error_m']*1000,
                new_mm=result['position_error_m']*1000,
                old_deg=float(np.degrees(old['orientation_error_rad'])),
                new_deg=float(np.degrees(result['orientation_error_rad'])),
                acceptable=result['acceptable'], active_limits=result['differential_ik']['active_limits']))
        summary = {k: np.percentile([r[k] for r in records], [50,95,100]).tolist()
                   for k in ['old_mm','new_mm','old_deg','new_deg']}
        summary['rejected'] = sum(not r['acceptable'] for r in records)
        report[side] = dict(summary=summary, records=records)
        print(stamp, side, json.dumps(summary))
    reports[stamp] = report
out = ROOT/'logs/openwam_differential_ik/delta_limit_replay.json'
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(reports, indent=2))
