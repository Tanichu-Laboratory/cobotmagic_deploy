"""Offline replay: recorded observations, plus ideal-tracking counterfactual."""
import csv,json,sys,time,copy
from pathlib import Path
import numpy as np,yaml
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from cobotmagic_deployment.common.piper_ik import PiperNumericalIK
cfg=yaml.safe_load((ROOT/'cobotmagic_deployment/configs/config_openwam_piper.yaml').read_text())['ros']['eef_ik']
reports={}
for name in (sys.argv[1:] or ['20260924_040452','20260924_035331','20260924_035437','20260917_111658']):
 rows=list(csv.DictReader((ROOT/f'logs/action_commands/action_commands_{name}.csv').open()))
 for mode in ['recorded','ideal_tracking']:
  ik=PiperNumericalIK(cfg);refs={};records=[]
  for side in ['left','right']:
   ik.calibrate(side,json.loads(rows[0]['ik_seed_joint_'+side]),json.loads(rows[0]['command_'+side+'_before']))
  for i,row in enumerate(rows):
   pair={}
   for side in ['left','right']:
    seed=json.loads(row['ik_seed_joint_'+side])
    if mode=='ideal_tracking' and side in refs:seed=refs[side]
    ref=refs.get(side)
    if mode=='recorded':ref=json.loads(rows[i-1]['ik_joint_'+side]) if i else None
    a=ik.solve(side,json.loads(row['target_'+side]),seed,ref)
    pair[side]=a
    records.append({'step':i,'side':side,'active':a['wrist_singularity_avoidance']['active'],'acceptable':a['acceptable'],'old_bend_deg':float(np.rad2deg(abs(json.loads(row['ik_joint_'+side])[4]))),'bend_deg':float(np.rad2deg(abs(a['joints'][4]))),'position_mm':a['position_error_m']*1000,'orientation_deg':float(np.rad2deg(a['orientation_error_rad'])),'wrist_delta_deg':float(np.rad2deg(np.max(abs(a['joints'][[3,5]]-np.asarray(seed if ref is None else ref)[[3,5]])))),'guard_mode':a['wrist_singularity_avoidance'].get('mode'), 'condition':a['wrist_singularity_avoidance'].get('command_condition'), 'guard_reason':a['wrist_singularity_avoidance'].get('reason'), 'seconds':a['solve_time_sec']})
   if all(a['acceptable'] for a in pair.values()):
    refs={side:a['joints'].copy() for side,a in pair.items()}
  summary={}
  for side in ['left','right']:
   sel=[a for a in records if a['side']==side];active=[a for a in sel if a['active']];good=[a for a in active if a['acceptable'] and a['guard_mode']!='hold_last_safe']
   summary[side]={'samples':len(sel),'active':len(active),'held':sum(a['guard_mode']=='hold_last_safe' for a in sel),'rejected':sum(not a['acceptable'] for a in sel),'accepted_guard_bend_min_deg':min([a['bend_deg'] for a in good],default=None),'accepted_guard_pos_max_mm':max([a['position_mm'] for a in good],default=None),'accepted_guard_rot_max_deg':max([a['orientation_deg'] for a in good],default=None),'accepted_guard_wrist_delta_max_deg':max([a['wrist_delta_deg'] for a in good],default=None),'solve_p95_ms':float(np.percentile([a['seconds']*1000 for a in sel],95))}
  reports[name+'_'+mode]={'summary':summary,'records':records};print(name,mode,json.dumps(summary),flush=True)
  out=ROOT/'logs/openwam_tracking_diagnosis/whole_arm_guard_replay.json';out.write_text(json.dumps(reports,indent=2))
