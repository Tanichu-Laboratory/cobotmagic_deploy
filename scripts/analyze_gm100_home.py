"""Select an observed GM100 episode start nearest the task-balanced median."""
import json,sys
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq
import yaml
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from cobotmagic_deployment.common.piper_ik import PiperNumericalIK
cfg=yaml.safe_load((ROOT/'cobotmagic_deployment/configs/config_openwam_piper.yaml').read_text())
ik=PiperNumericalIK(cfg['ros']['eef_ik'])
records=[];medians=[]
for task in sorted(Path('/workspace/dataset/GM100').glob('task_*')):
 group=[]
 for path in sorted(task.glob('data/*/*.parquet')):
  d=next(pq.ParquetFile(path).iter_batches(batch_size=1,columns=['observation.state.arm.position','observation.state.effector.position','frame_index'])).to_pydict()
  q=np.asarray(d['observation.state.arm.position'][0],float);g=np.asarray(d['observation.state.effector.position'][0],float)
  if q.shape!=(12,) or g.shape!=(2,) or not np.isfinite(np.r_[q,g]).all() or d['frame_index'][0]!=0:raise ValueError(str(path))
  group.append(np.r_[q,g])
  records.append(dict(path=str(path),q=q.tolist(),gripper=g.tolist()))
 if group:medians.append(np.median(group,axis=0))
center=np.median(medians,axis=0)
candidates=[]
for r in records:
 q=np.array(r['q']);g=np.array(r['gripper'])
 valid=all(np.all(q[o:o+6]>=ik.lower) and np.all(q[o:o+6]<=ik.upper) for o in [0,6]) and np.all(g>=0) and np.all(g<=.06558)
 if valid:candidates.append((float(np.linalg.norm(q-center[:12])),r))
distance,selected=min(candidates,key=lambda x:x[0])
report=dict(tasks=len(medians),episodes=len(records),selection='Observed start minimizing 12-joint Euclidean distance to median of per-task medians; valid URDF and physical gripper bounds',center=center.tolist(),distance_rad=distance,selected=selected,
            left=selected['q'][:6]+[selected['gripper'][0]],right=selected['q'][6:]+[selected['gripper'][1]])
out=ROOT/'logs/openwam_tracking_diagnosis/gm100_initial_pose.json';out.write_text(json.dumps(report,indent=2))
print(json.dumps(report,indent=2))
