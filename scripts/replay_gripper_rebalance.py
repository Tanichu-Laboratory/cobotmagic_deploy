"""Replay unamplified September 24 logs; run from repository root with PYTHONPATH=."""
import csv,json,sys
from pathlib import Path
import yaml,numpy as np
from cobotmagic_deployment.common.gripper_hysteresis import GripperHysteresis
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
cfg=yaml.safe_load(Path('cobotmagic_deployment/configs/config_openwam_piper.yaml').read_text())['ros']['gripper_hysteresis']
cfg['request_relative'] = {'enabled': False}  # Historical fixed-threshold replay.
new=cfg
cfg=dict(cfg,close_threshold_normalized=[.55,.65],open_threshold_normalized=[.70,.70])
report={}
for p in sorted(Path('logs/action_commands').glob('*.csv'))[-4:]:
 rows=list(csv.DictReader(p.open()));result={}
 for label,c in [('old',cfg),('new',new)]:
  ctrl=GripperHysteresis(c);events=[];closed=np.zeros(2,int)
  for i,row in enumerate(rows):
   physical=[json.loads(row['raw_'+s])[-1] for s in ['left','right']]
   measured=[json.loads(row['ik_seed_joint_'+s])[-1] for s in ['left','right']]
   v,ctrl,d=ctrl.propose(physical,measured);closed+=v==0
   if any(d['switched']):events.append(dict(row=i,**d))
  result[label]=dict(closed=closed.tolist(),events=events)
 print(p.name,len(rows),{k:dict(closed=v['closed'],events=[(e['row'],e['output_open']) for e in v['events']]) for k,v in result.items()})
 report[p.name]=result
Path('logs/openwam_tracking_diagnosis/gripper_rebalance_20260924.json').write_text(json.dumps(report,indent=2))
