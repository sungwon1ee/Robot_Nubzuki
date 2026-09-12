import csv,json
from pathlib import Path
import numpy as np,torch
import mjlab.tasks,mjlab_nubzuki.tasks
from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_env_cfg
p=Path('../response_20260909_223002.csv');rows=list(csv.DictReader(p.open()));meta=json.load(p.with_suffix('.json').open());names0=meta['joint_order'];d={k:np.array([float(r[k]) for r in rows]) for k in rows[0] if k not in ('phase','active_joint')}
cfg=load_env_cfg('Mjlab-Velocity-Flat-Backlash-BAM-Nubzuki',play=True);cfg.scene.num_envs=1;cfg.seed=42
for ac in cfg.scene.entities['robot'].articulation.actuators:ac.vin_range=(7.4,7.4)
env=ManagerBasedRlEnv(cfg=cfg,device='cpu');env.reset();robot=env.scene['robot'];term=env.action_manager.get_term('joint_pos');ids=list(term.target_ids);names=[robot.joint_names[i] for i in ids];pids=[robot.joint_names.index('passive_'+n+'_backlash') for n in names]
q=robot.data.joint_pos.clone()
for j,n in enumerate(names):q[:,ids[j]]=d[n+'_actual_rad'][0];q[:,pids[j]]=0
robot.write_joint_state_to_sim(q,torch.zeros_like(q));root=torch.cat([robot.data.root_link_pose_w,robot.data.root_link_vel_w],1).clone();root[:,2]=.5;root[:,7:]=0;robot.write_root_state_to_sim(root);env.scene.write_data_to_sim();env.sim.forward()
centre=torch.tensor([[meta['centre_rad'][names0.index(n)] for n in names]],dtype=torch.float32);env.action_manager.process_action((centre-term.offset)/term.scale)
reads=(d['read_start_s']+d['position_read_end_s'])/2;writes=(d['write_start_s']+d['write_end_s'])/2;events=sorted([(float(t),0,i) for i,t in enumerate(reads)]+[(float(t),1,i) for i,t in enumerate(writes)])
result=np.zeros((len(rows),len(names)));simtime=0;maxerr=0
with torch.inference_mode():
 for number,(t,kind,i) in enumerate(events):
  while simtime+env.physics_dt/2<t:
   robot.write_root_state_to_sim(root);env.action_manager.apply_action();env.scene.write_data_to_sim();env.sim.step();env.scene.update(dt=env.physics_dt);simtime+=env.physics_dt
  if kind:
   target=torch.tensor([[d[n+'_target_rad'][i] for n in names]],dtype=torch.float32);env.action_manager.process_action((target-term.offset)/term.scale);maxerr=max(maxerr,float(torch.max(abs(term._processed_actions-target))))
  else:result[i]=(robot.data.joint_pos[:,ids]+robot.data.joint_pos[:,pids]).cpu().numpy()[0]
  if number%600==0:print('PROGRESS',number,len(events),flush=True)
np.savez('../analysis/response_223002/replay.npz',q=result,names=names,read_time=reads,write_time=writes);print('DONE',maxerr,flush=True);env.close()
