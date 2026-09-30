"""Replay old /0.75-gain commands through the new gripper controller offline."""
import csv,json,sys
from pathlib import Path
import numpy as np,yaml
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from cobotmagic_deployment.common.gripper_hysteresis import GripperHysteresis
cfg=yaml.safe_load((ROOT/'cobotmagic_deployment/configs/config_openwam_piper.yaml').read_text())['ros']['gripper_hysteresis']
cfg['request_relative'] = {'enabled': False}  # Historical fixed-threshold replay.
# Saturated old commands only establish normalized>=.75. New thresholds must
# be below that bound; then all discrete decisions remain unambiguous.
assert max(cfg['open_threshold_normalized'])<.75
report={}
for date in ['20260924_033603','20260924_033828','20260917_110643','20260917_111658']:
 rows=list(csv.DictReader((ROOT/f'logs/action_commands/action_commands_{date}.csv').open()))
 c=GripperHysteresis(cfg);events=[];result=[]
 for row in rows:
  physical=[json.loads(row['raw_'+s])[-1]*.75 for s in ['left','right']]
  measured=[json.loads(row['ik_seed_joint_'+s])[-1] for s in ['left','right']]
  v,c,d=c.propose(physical,measured);result.append(v.tolist())
  if any(d['switched']):events.append({'step':int(row['global_action_step']),'request':int(row['request_id']),**d})
 report[date]={'samples':len(rows),'events':events,'closed_counts':np.sum(np.array(result)==0,axis=0).tolist()}
 print(date,json.dumps(report[date]))
out=ROOT/'logs/openwam_tracking_diagnosis/gripper_hysteresis_replay.json';out.write_text(json.dumps(report,indent=2))
