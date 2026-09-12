import csv,json,numpy as np
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
p=Path(__file__).parent;z=np.load(p/'replay_real.npz');r=list(csv.DictReader((p.parents[1]/'walk_3950_20260909_221555.csv').open()));d={k:np.array([float(x[k]) for x in r]) for k in r[0]};ix=z['indices'];t=z['time'];names=list(z['names']);m=(t>=78)&(t<100);sel=np.where(m)[0];out={}
for j in ['left_hip_pitch','right_hip_pitch','left_knee','right_knee','left_ankle','right_ankle']:
 target=d[j+'_target_rad'][ix];out[j]={}
 for label,a in [('real',d[j+'_actual_rad'][ix])]+[(str(mode),z['q'][:,e,names.index(j)]) for e,mode in enumerate(z['modes'])]:
  scores=[]
  for lag in range(21):
   er=a[sel]-target[sel-lag];scores.append(np.mean((er-er.mean())**2))
  lag=int(np.argmin(scores));er=a[sel]-target[sel-lag];out[j][label]={'range_deg':np.rad2deg([a[m].min(),a[m].max()]).tolist(),'ptp_deg':float(np.rad2deg(np.ptp(a[m]))),'lag_ms':lag*float(np.median(np.diff(t)))*1000,'aligned_mae_deg':float(np.rad2deg(abs(er).mean())),'aligned_bias_deg':float(np.rad2deg(er.mean())),'mae_deg':float(np.rad2deg(abs(a[m]-target[m]).mean()))}
print(json.dumps(out,indent=2));(p/'replay_comparison.json').write_text(json.dumps(out,indent=2))
print('free z min/max',z['root'][:,0,2].min(),z['root'][:,0,2].max());bad=np.where(z['root'][:,0,2]<.14)[0];print('free below .14 at',t[bad[0]] if len(bad) else None)
fig,axs=plt.subplots(3,1,figsize=(12,8),sharex=True);m=(t>=84)&(t<89)
for ax,j in zip(axs,['left_hip_pitch','right_hip_pitch','left_knee']):
 ax.plot(t[m],np.rad2deg(d[j+'_target_rad'][ix][m]),label='Sent target',color='gray',ls='--');ax.plot(t[m],np.rad2deg(d[j+'_actual_rad'][ix][m]),label='Real',color='#e06b27');
 for e,label in [(1,'BAM: trunk held, ground'),(2,'BAM: trunk held, feet clear')]:ax.plot(t[m],np.rad2deg(z['q'][m,e,names.index(j)]),label=label,lw=1,color=('#2674b5' if e==1 else '#8b5caa'))
 ax.set_ylabel(j+' (deg)');ax.grid(alpha=.2)
axs[0].legend(ncol=2,fontsize=8);axs[-1].set_xlabel('Real log time (s)');fig.suptitle('Identical sent-target replay | BAM 7.4 V, existing gains and backlash');fig.tight_layout();fig.savefig(p/'replay_comparison.png',dpi=150)
