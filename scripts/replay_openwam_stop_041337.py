"""Replay the unexecuted tail after step36, assuming perfect tracking."""
import sys,csv,json
from pathlib import Path
import numpy as np,yaml
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from cobotmagic_deployment.common.piper_ik import PiperNumericalIK
r=list(csv.DictReader((ROOT/'logs/action_commands/action_commands_20260924_041337.csv').open()))
ik=PiperNumericalIK(yaml.safe_load((ROOT/'cobotmagic_deployment/configs/config_openwam_piper.yaml').read_text())['ros']['eef_ik'])
z=np.load(ROOT/r[-1]['action_chunk_path']);refs={}
for side in ['left','right']:
 ik.calibrate(side,json.loads(r[0]['ik_seed_joint_'+side]),json.loads(r[0]['command_'+side+'_before']))
 refs[side]=np.array(json.loads(r[-1]['ik_joint_'+side]))
records=[]
for index in range(5,32):
 pair={}
 for side in ['left','right']:
  goal=z['received_'+side][index].copy();goal[-1]=refs[side][-1]
  a=ik.solve(side,goal,refs[side],refs[side]);pair[side]=a
  records.append({'step':32+index,'side':side,'acceptable':a['acceptable'],'target_reached':a['target_reached'],'position_mm':a['position_error_m']*1000,'guard':a['wrist_singularity_avoidance']})
 if all(a['acceptable'] for a in pair.values()):refs={s:a['joints'].copy() for s,a in pair.items()}
summary={s:{'samples':27,'rejected':sum(not a['acceptable'] for a in records if a['side']==s),'held':sum(a['guard'].get('mode')=='hold_last_safe' for a in records if a['side']==s)} for s in refs}
print(json.dumps(summary));print('first formerly failing',json.dumps(records[:2]))
(ROOT/'logs/openwam_tracking_diagnosis/stop_041337_recovery.json').write_text(json.dumps({'assumption':'perfect tracking, saved target chunk fixed','summary':summary,'records':records},indent=2))
assert all(a['acceptable'] for a in records)
