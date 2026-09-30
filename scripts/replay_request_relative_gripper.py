"""Compare fixed and request-relative thresholds with recorded policy inputs."""
import csv,json,sys
from pathlib import Path
import yaml
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from cobotmagic_deployment.common.gripper_hysteresis import GripperHysteresis
cfg=yaml.safe_load((ROOT/'cobotmagic_deployment/configs/config_openwam_piper.yaml').read_text())['ros']['gripper_hysteresis']
reports={}
for stamp in ['20260924_063643','20260924_063943','20260924_064529']:
 rows=list(csv.DictReader((ROOT/f'logs/action_commands/action_commands_{stamp}.csv').open()))
 report={}
 for name,options in [('fixed',dict(cfg,request_relative={'enabled':False})),('relative',cfg)]:
  c=GripperHysteresis(options);events=[]
  for i,r in enumerate(rows):
   v,c,d=c.propose([json.loads(r['raw_'+s])[-1] for s in ['left','right']],
       [json.loads(r['ik_seed_joint_'+s])[-1] for s in ['left','right']],
       request_opening=[float(r[s+'_gripper_request_policy']) for s in ['left','right']])
   if any(d['switched']):events.append(dict(row=i,request=r['request_id'],**d))
  report[name]=events
  print(stamp,name,[(e['row'],e['output_open']) for e in events])
 reports[stamp]=report
(ROOT/'logs/openwam_tracking_diagnosis/request_relative_gripper_replay.json').write_text(json.dumps(reports,indent=2))
