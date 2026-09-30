"""Replay saved observations offline; no sockets or robot commands."""
from pathlib import Path
import sys,json,time,hashlib,shutil
import numpy as np,yaml
from PIL import Image,ImageDraw
import imageio.v2 as imageio
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from cobotmagic_deployment.servers.policy_server_openwam_piper import OpenWAMPiperPolicy
from cobotmagic_deployment.common.openwam_piper import request_state
OUT=ROOT/'logs/openwam_video_replay_20260919';OUT.mkdir(exist_ok=True)
cfg=yaml.safe_load((ROOT/'cobotmagic_deployment/configs/config_openwam_piper.yaml').read_text())
policy=OpenWAMPiperPolicy(cfg)
# Same generation path as deployment; enable only the final VAE decode.
policy.server.engine._decode_video=True
policy.server.cfg.optimization.decode_video=True
from omegaconf import OmegaConf
from openwam.dataloader.transforms.multiview import format_prompt_for_inference
(OUT/'resolved_model_config.yaml').write_text(OmegaConf.to_yaml(policy.server.cfg))
(OUT/'deployment_config.yaml').write_text(yaml.safe_dump(cfg))
manifest=[]
for episode in ['20260917_110643','20260917_111025','20260917_111658']:
 snaps=sorted((ROOT/f'logs/action_commands/request_snapshots_{episode}').glob('*/metadata.json'))
 for label,idx in [('start',0),('middle',len(snaps)//2),('late',len(snaps)-1)]:
  src=snaps[idx];meta=json.loads(src.read_text());h=meta['header'];out=OUT/f'{episode}_{label}_{src.parent.name}';out.mkdir(exist_ok=True)
  state=request_state(h,cfg['openwam']['gripper'])
  obs=policy.preprocess.preprocess({'images':{name:Image.open(src.parent/f'{key}.jpg').convert('RGB') for name,key in [('head_camera','front'),('left_wrist_camera','left'),('right_wrist_camera','right')]},'prompt':format_prompt_for_inference(h['task_prompt'].strip()),'state':state})
  obs['image'].save(out/'input_composite.png');shutil.copy2(src,out/'source_metadata.json')
  for key in ['front','left','right']:shutil.copy2(src.parent/f'{key}.jpg',out/f'input_{key}.jpg')
  print('GENERATE',episode,label,src.parent.name,flush=True);t=time.monotonic()
  result=policy.server.engine.generate({'observation':obs,'first_frame_image':[obs['image']],'prompt':obs['prompt'],'proprio':state,'seed':42})
  video=result['video'];frames=[np.asarray(f.convert('RGB') if isinstance(f,Image.Image) else f,dtype=np.uint8) for f in video]
  assert frames and all(f.shape==frames[0].shape for f in frames)
  actions=result['actions'];actions=actions.detach().float().cpu().numpy() if hasattr(actions,'detach') else np.asarray(actions)
  np.save(out/'predicted_actions_eef20.npy',actions);np.save(out/'input_state_eef20.npy',state)
  # Match action duration to the logged 5 Hz control clock, with video stride 4.
  stride=int(policy.server.cfg.dataloader.video_stride);fps=float(h['control_hz'])/stride
  imageio.mimsave(out/'prediction.mp4',frames,fps=fps,codec='libx264',macro_block_size=1,ffmpeg_log_level='error')
  Image.fromarray(frames[0]).save(out/'prediction.gif',save_all=True,append_images=[Image.fromarray(f) for f in frames[1:]],duration=round(1000/fps),loop=0)
  w,hh=Image.fromarray(frames[0]).size;sheet=Image.new('RGB',(w*3,(hh+24)*((len(frames)+2)//3)),(245,245,245));draw=ImageDraw.Draw(sheet)
  for j,f in enumerate(frames):
   x=(j%3)*w;y=(j//3)*(hh+24);sheet.paste(Image.fromarray(f),(x,y+24));draw.text((x+5,y+4),f'frame {j} / action step {j*stride}',fill='black')
  sheet.save(out/'contact_sheet.jpg',quality=92)
  record={'episode':episode,'phase':label,'source':str(src),'output':out.name,'task_prompt':h['task_prompt'],'formatted_prompt':obs['prompt'],'seed':42,'denoise_steps':cfg['openwam']['denoise_steps'],'decode_video':True,'num_video_frames':len(frames),'frame_shape':list(frames[0].shape),'action_shape':list(actions.shape),'playback_fps':fps,'playback_clock':'logged control_hz / checkpoint video_stride; not a measured forecast time calibration','seconds':time.monotonic()-t,'image_sha256':{k:hashlib.sha256((src.parent/f'{k}.jpg').read_bytes()).hexdigest() for k in ['front','left','right']},'state_eef20':state.tolist()}
  (out/'replay.json').write_text(json.dumps(record,ensure_ascii=False,indent=2));manifest.append(record);(OUT/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2));print('DONE',record['output'],record['num_video_frames'],record['action_shape'],round(record['seconds'],2),flush=True)
html=['<!doctype html><meta charset="utf-8"><title>OpenWAM 予測動画の再推論</title><style>body{font-family:sans-serif;max-width:1100px;margin:30px auto}video{width:320px}article{display:inline-block;width:350px;vertical-align:top;margin:8px}img{max-width:100%}</style><h1>OpenWAM 予測動画の再推論</h1><p>実機録画ではなく、保存済みの観測から再生成した予測です。各クリップは独立したリクエスト。再生速度は当時の5 Hz制御と動画stride=4から1.25 fpsに設定。最初のフレームは入力画像を条件とする再構成フレームです。</p>']
for r in manifest:
 name=r['output'];html.append(f'<article><h3>{r["episode"]} / {r["phase"]}</h3><p>{r["task_prompt"]}</p><video controls loop preload="metadata" src="{name}/prediction.mp4"></video><p><a href="{name}/contact_sheet.jpg">コマ一覧</a> · <a href="{name}/input_composite.png">入力画像</a> · <a href="{name}/replay.json">入力条件</a></p></article>')
(OUT/'index.html').write_text('\n'.join(html))
print('COMPLETE',OUT,flush=True)
