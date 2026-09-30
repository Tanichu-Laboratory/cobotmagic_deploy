"""Offline counterfactual; ideal tracking is not a physical robot measurement."""
import csv,json,sys
from pathlib import Path
import numpy as np,yaml
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from cobotmagic_deployment.common.piper_ik import PiperNumericalIK
cfg=yaml.safe_load((ROOT/'cobotmagic_deployment/configs/config_openwam_piper.yaml').read_text())['ros']['eef_ik']
reports={}
for name in (sys.argv[1:] or ['20260924_041814','20260924_040452','20260924_035331','20260924_035437','20260917_111658']):
 rows=list(csv.DictReader((ROOT/f'logs/action_commands/action_commands_{name}.csv').open()))
 for alpha in [1.,.7]:
  ik=PiperNumericalIK(cfg);refs={};states={};records=[]
  for side in ['left','right']:
   states[side]=np.asarray(json.loads(rows[0]['ik_seed_joint_'+side]))
   ik.calibrate(side,states[side],json.loads(rows[0]['command_'+side+'_before']))
  for i,row in enumerate(rows):
   pair={}
   for side in ['left','right']:
    a=ik.solve(side,json.loads(row['target_'+side]),states[side],refs.get(side),dt=.2)
    pair[side]=a
    records.append(dict(step=i,side=side,acceptable=a['acceptable'],position_mm=a['position_error_m']*1000,
        orientation_deg=float(np.rad2deg(a['orientation_error_rad'])),seconds=a['solve_time_sec'],**a['differential_ik']))
   if all(a['acceptable'] for a in pair.values()):
    for side,a in pair.items():
     ik.commit(side,a);refs[side]=a['joints'].copy()
     states[side][:6]+=alpha*(refs[side][:6]-states[side][:6])
  summary={}
  for side in ['left','right']:
   sel=[a for a in records if a['side']==side]
   summary[side]=dict(samples=len(sel),rejected=sum(not a['acceptable'] for a in sel),
       stalled=sum(a['mode']=='stalled' for a in sel),acceleration_override=sum(a['acceleration_override'] for a in sel))
   for key in ['position_mm','orientation_deg','seconds','command_condition']:
    summary[side][key]=dict(p50=float(np.percentile([a[key] for a in sel],50)),p95=float(np.percentile([a[key] for a in sel],95)),max=max(a[key] for a in sel))
  reports[name+f'_alpha_{alpha}']=dict(summary=summary,records=records)
  print(name,alpha,json.dumps(summary),flush=True)
out=ROOT/'logs/openwam_differential_ik/replay.json';out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(reports,indent=2))
