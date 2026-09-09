import json
from pathlib import Path
from dataclasses import asdict
import numpy as np,torch
import mjlab.tasks,mjlab_nubzuki.tasks
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg,load_rl_cfg,load_runner_cls
from mjlab_microduck.tasks.mdp import VelocityCommandCommandOnly,UniformPoseCommand
speeds=[0.,.1,.2,.25]
def twist(self):
 self.vel_command_b[:]=0;self.vel_command_b[:,0]=torch.tensor(speeds,device=self.device);self.vel_command_w[:]=self.vel_command_b;self.is_standing_env[:]=False
VelocityCommandCommandOnly._update_command=twist
UniformPoseCommand._update_command=lambda self:self._command.zero_()
task='Mjlab-Velocity-Flat-Backlash-BAM-Nubzuki';cfg=load_env_cfg(task,play=True);cfg.scene.num_envs=4;cfg.seed=42
acfg=load_rl_cfg(task);env=ManagerBasedRlEnv(cfg=cfg,device='cpu');w=RslRlVecEnvWrapper(env,clip_actions=acfg.clip_actions)
runner=load_runner_cls(task)(w,asdict(acfg),device='cpu');runner.load('checkpoints/walking_v2/model_3950.pt',load_cfg={'actor':True},strict=True,map_location='cpu');policy=runner.get_inference_policy(device='cpu')
obs,_=w.reset();robot=env.scene['robot'];term=env.action_manager.get_term('joint_pos');ids=list(term.target_ids);names=[robot.joint_names[i] for i in ids];pids=[robot.joint_names.index('passive_'+n+'_backlash') for n in names];limits=robot.data.joint_pos_limits[0,ids].cpu().numpy();records=[];dones=np.zeros(4,int)
with torch.inference_mode():
 for step in range(600):
  action=policy(obs);obs,rew,done,extra=w.step(action);dones+=done.cpu().numpy().astype(int)
  raw=term.raw_action*term.scale+term.offset;q=robot.data.joint_pos[:,ids];enc=q+robot.data.joint_pos[:,pids]
  records.append(np.stack([raw.cpu().numpy(),q.cpu().numpy(),enc.cpu().numpy()]))
  if step%100==0:print('PROGRESS',step,flush=True)
a=np.array(records);out=Path('../analysis/3950');np.savez(out/'rollout.npz',data=a,names=names,limits=limits,speeds=speeds)
summary={'checkpoint':'model_3950.pt','steps':600,'dt':env.step_dt,'dones':dones.tolist(),'scenarios':[]}
for e,v in enumerate(speeds):
 rows=[]
 for j,n in enumerate(names):
  t=a[100:,0,e,j];q=a[100:,1,e,j];enc=a[100:,2,e,j];lo,hi=limits[j]
  rows.append({'joint':n,'limits_deg':np.rad2deg([lo,hi]).tolist(),'target_deg':np.rad2deg([t.min(),t.max()]).tolist(),'encoder_deg':np.rad2deg([enc.min(),enc.max()]).tolist(),'target_outside_pct':float(np.mean((t<lo)|(t>hi))*100),'position_outside_1deg_pct':float(np.mean((q<lo-np.pi/180)|(q>hi+np.pi/180))*100),'upper_near_1deg_pct':float(np.mean(q>hi-np.pi/180)*100),'mae_deg':float(np.rad2deg(abs(enc-t).mean()))})
 summary['scenarios'].append({'speed':v,'joints':rows})
(out/'summary.json').write_text(json.dumps(summary,indent=2));print('DONE',dones,flush=True);env.close()
