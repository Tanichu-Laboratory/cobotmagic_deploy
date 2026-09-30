"""Replay GM100 openings; fixed recorded inputs, not a closed-loop grasp test."""
import csv,json,sys
from pathlib import Path
import numpy as np,yaml
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from cobotmagic_deployment.common.gripper_hysteresis import GripperHysteresis
cfg=yaml.safe_load((ROOT/'cobotmagic_deployment/configs/config_openwam_piper.yaml').read_text())['ros']['gripper_hysteresis']
cfg['request_relative'] = {'enabled': False}  # Historical fixed-threshold replay.
rows=list(csv.DictReader((ROOT/'logs/action_commands/action_commands_20260924_062340.csv').open()))
report={}
for name,settings in [('old',dict(cfg,close_threshold_normalized=[.75,.75],open_threshold_normalized=[.90,.90])),
                      ('gm100',dict(cfg,close_threshold_normalized=[.80,.75],open_threshold_normalized=[.86,.82]))]:
 for init in ['recorded','closed']:
  ctrl=GripperHysteresis(settings);events=[]
  for i,row in enumerate(rows):
   measured=[json.loads(row['ik_seed_joint_'+s])[-1] for s in ['left','right']]
   if init=='closed' and i==0:measured=[0.,0.]
   commands=[json.loads(row['raw_'+s])[-1] for s in ['left','right']]
   values,ctrl,diag=ctrl.propose(commands,measured)
   if any(diag['switched']):events.append(dict(row=i,**diag))
  report[name+'_'+init]=events
print(json.dumps(report,indent=2))
out=ROOT/'logs/openwam_tracking_diagnosis/gm100_gripper_replay.json'
out.write_text(json.dumps(report,indent=2))
