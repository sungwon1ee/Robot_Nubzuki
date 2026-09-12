import csv,json,numpy as np
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
out=Path(__file__).parent;p=out.parents[1]/'response_20260909_223002.csv';rows=list(csv.DictReader(p.open()));d={k:np.array([float(x[k]) for x in rows]) for k in rows[0] if k not in ['phase','active_joint']};tr=(d['read_start_s']+d['position_read_end_s'])/2;tw=(d['write_start_s']+d['write_end_s'])/2;z=np.load(out/'replay.npz');names=list(z['names']);omega=2*np.pi*.7
summary={};fig,axs=plt.subplots(2,2,figsize=(12,7))
for ji,j in enumerate(['left_hip_pitch','right_hip_pitch']):
 summary[j]={}
 for ph in ['sine','step']:
  mask=np.array([r['phase']==ph and r['active_joint']==j for r in rows]);ix=np.where(mask)[0];ax=axs[ji,0 if ph=='sine' else 1]
  target=d[j+'_target_rad'];ax.step(tw[ix]-tw[ix[0]],np.rad2deg(target[ix]),where='post',color='gray',ls='--',label='Sent target')
  for label,a,color in [('Real',d[j+'_actual_rad'],'#de6e29'),('BAM',z['q'][:,names.index(j)],'#2376b5')]:
   ax.plot(tr[ix]-tw[ix[0]],np.rad2deg(a[ix]),color=color,label=label)
   if ph=='sine':
    idx=ix[tr[ix]>tr[ix[0]]+1.4]
    def fit(y,t):
     co=np.linalg.lstsq(np.column_stack([np.sin(omega*t),np.cos(omega*t),np.ones(len(t))]),y,rcond=None)[0];return np.hypot(co[0],co[1]),np.arctan2(co[1],co[0]),co[2]
    amp,phase,offset=fit(a[idx],tr[idx]);ta,tp,to=fit(target[idx],tw[idx]);lag=((tp-phase+np.pi)%(2*np.pi)-np.pi)/omega
    summary[j][label]={'sine_amplitude_deg':float(np.rad2deg(amp)),'sine_target_amplitude_deg':float(np.rad2deg(ta)),'sine_lag_ms':float(lag*1000),'offset_deg':float(np.rad2deg(offset-to))}
   else:
    changes=[k for k in ix[1:] if abs(target[k]-target[k-1])>.05];result=[]
    for k in changes:
     t0=tw[k];pre=(tr>t0-.2)&(tr<t0);post=(tr>=t0)&(tr<t0+1.4);q0=np.mean(a[pre]);delta=target[k]-q0;ids=np.where(post)[0];fraction=(a[ids]-q0)/delta
     times={}
     for fraction_target in [.1,.5,.9]:
      hit=np.where(fraction>=fraction_target)[0];times[str(fraction_target)]=float((tr[ids[hit[0]]]-t0)*1000) if len(hit) else None
     result.append({'time':float(t0),'crossings_ms':times,'settled_error_deg':float(np.rad2deg(np.mean(a[(tr>t0+1.1)&(tr<t0+1.4)])-target[k]))})
    summary[j][label]['steps']=result
  ax.set_title(j+' / '+ph);ax.set_xlabel('Seconds in phase');ax.set_ylabel('Degrees');ax.legend(fontsize=8);ax.grid(alpha=.2)
fig.suptitle('Suspended test: identical logged commands replayed through BAM (7.4 V)');fig.tight_layout();fig.savefig(out/'comparison.png',dpi=150);(out/'summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2))
