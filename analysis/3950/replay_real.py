"""Replay logged sent targets through BAM. Controlled support sensitivity, not a reconstruction of unlogged hand forces."""
import csv,json
from pathlib import Path
import numpy as np,torch
import mjlab.tasks,mjlab_nubzuki.tasks
from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_env_cfg
p=Path('../walk_3950_20260909_221555.csv');rows=list(csv.DictReader(p.open()));d={k:np.array([float(r[k]) for r in rows]) for k in rows[0]};tt=d['time_s'];sel=np.where((tt>=74)&(tt<=101))[0]
cfg=load_env_cfg('Mjlab-Velocity-Flat-Backlash-BAM-Nubzuki',play=True);cfg.scene.num_envs=3;cfg.seed=42
# Same BAM coefficients/gains as deployment model; fixed nominal voltage removes random confound.
for ac in cfg.scene.entities['robot'].articulation.actuators:
 ac.vin_range=(7.4,7.4)
env=ManagerBasedRlEnv(cfg=cfg,device='cpu');env.reset();robot=env.scene['robot'];term=env.action_manager.get_term('joint_pos');ids=list(term.target_ids);names=[robot.joint_names[i] for i in ids];pids=[robot.joint_names.index('passive_'+n+'_backlash') for n in names]
q=robot.data.joint_pos.clone();v=torch.zeros_like(q)
for j,n in enumerate(names):q[:,ids[j]]=float(d[n+'_actual_rad'][sel[0]]);q[:,pids[j]]=0
robot.write_joint_state_to_sim(q,v);root=torch.cat([robot.data.root_link_pose_w,robot.data.root_link_vel_w],dim=1).clone();root[:,7:]=0;root[2,2]=.5;robot.write_root_state_to_sim(root);env.scene.write_data_to_sim();env.sim.forward()
records=[];pos=[];errors=[];time_remain=0
with torch.inference_mode():
 for count,k in enumerate(sel):
  actual=(robot.data.joint_pos[:,ids]+robot.data.joint_pos[:,pids]).clone().cpu().numpy();records.append(actual);pos.append(torch.cat([robot.data.root_link_pose_w,robot.data.root_link_vel_w],dim=1).clone().cpu().numpy())
  target=torch.tensor([d[n+'_target_rad'][k] for n in names],dtype=torch.float32).repeat(3,1)
  action=(target-term.offset)/term.scale;env.action_manager.process_action(action)
  errors.append(float(torch.max(abs(term._processed_actions-target))))
  interval=(tt[k+1]-tt[k]) if k+1<len(tt) else .02;time_remain+=interval;nsteps=int(time_remain/env.physics_dt);time_remain-=nsteps*env.physics_dt
  for _ in range(nsteps):
   # env 0 free; env 1 trunk held at nominal ground height; env 2 held above ground.
   robot.write_root_state_to_sim(root[1:],env_ids=torch.tensor([1,2]))
   env.action_manager.apply_action();env.scene.write_data_to_sim();env.sim.step();env.scene.update(dt=env.physics_dt)
  env.sim.forward()
  if count%200==0:print('PROGRESS',count,'/',len(sel),flush=True)
out=Path('../analysis/3950');np.savez(out/'replay_real.npz',q=np.array(records),root=np.array(pos),time=tt[sel],indices=sel,names=names,modes=['free','held_ground','held_air']);print('DONE target mapping max error',max(errors),flush=True);env.close()
