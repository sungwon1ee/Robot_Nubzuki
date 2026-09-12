"""Suspended-body servo response test; no walking policy is loaded."""
from __future__ import annotations
import argparse,csv,json,math,sys,time
from datetime import datetime
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from playground.nubzuki.calibration import NubzukiCalibration,HEAD_JOINTS
from playground.nubzuki.hardware import ServoHardware
from playground.nubzuki.robot_runtime import _slew_to_pose

def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--port',default='/dev/ttyACM0');p.add_argument('--calibration')
 p.add_argument('--joints',nargs='+',default=['left_hip_pitch','right_hip_pitch'])
 p.add_argument('--amplitude-deg',type=float,default=3.0)
 p.add_argument('--frequency-hz',type=float,default=.7);p.add_argument('--cycles',type=int,default=5)
 p.add_argument('--out');p.add_argument('--dry-run',action='store_true');a=p.parse_args()
 c=NubzukiCalibration(a.calibration);order=list(c.joint_order)
 if any(n not in order or n in HEAD_JOINTS for n in a.joints) or len(set(a.joints))!=len(a.joints):p.error('Select distinct leg joints only')
 if not (0<a.amplitude_deg<=5 and .1<=a.frequency_hz<=1 and 2<=a.cycles<=10):p.error('Amplitude (0,5] deg, frequency [0.1,1] Hz, cycles [2,10]')
 centre=np.array([c.park_rad(n) for n in order]);amp=math.radians(a.amplitude_deg)
 # Keep knees clear of the extension stop, including when not being tested.
 for n in ['left_knee','right_knee']:centre[order.index(n)]=math.radians(-8)
 for n in order:
  lo,hi=c.limits_rad(n);span=amp if n in a.joints else 0
  if centre[order.index(n)]-span<lo or centre[order.index(n)]+span>hi:p.error(f'{n} waveform exceeds calibration limits')
 out=Path(a.out or f'logs/response_{datetime.now():%Y%m%d_%H%M%S}.csv')
 if out.exists() or out.with_suffix('.json').exists():p.error('Output already exists')
 print('Joints:',a.joints,'amplitude +/-',a.amplitude_deg,'deg; frequency',a.frequency_hz,'Hz; cycles',a.cycles)
 print('CSV:',out.resolve())
 if a.dry_run:return
 input('Stop the walking process. Secure the trunk with BOTH FEET CLEAR of the floor. Press Enter to start: ')
 out.parent.mkdir(parents=True,exist_ok=True)
 out.with_suffix('.json').write_text(json.dumps({'schema':'response_test_v1','calibration':c.data,'arguments':vars(a),'centre_rad':centre.tolist(),'joint_order':order,'timing':'Read positions, read velocities, write next target. CSV times use monotonic clock relative to start. actual_rad precedes this row target_rad.','support':'User instructed to secure trunk with feet clear'},indent=2))
 h=ServoHardware(c,a.port);dt=1/c.control_frequency_hz;last=None;started=False
 with out.open('x',newline='') as f:
  w=csv.writer(f);w.writerow(['sample','phase','active_joint','read_start_s','position_read_end_s','velocity_read_end_s','write_start_s','write_end_s']+[f'{n}_target_rad' for n in order]+[f'{n}_actual_rad' for n in order]+[f'{n}_velocity_rad_s' for n in order])
  try:
   h.preflight();last=h.read_positions();h.set_positions(dict(zip(order,last)));started=True
   h.set_kps([int(c.data['runtime']['head_kp' if n in HEAD_JOINTS else 'leg_kp']) for n in order]);h.enable_torque()
   last=_slew_to_pose(h,c,last,centre,dt)
   origin=time.monotonic();sample=0
   def run(phase,n,duration):
    nonlocal sample,last
    begin=time.monotonic();deadline=begin
    while time.monotonic()-begin<duration:
     rs=time.monotonic();q=h.read_positions();pe=time.monotonic();v=h.read_velocities();ve=time.monotonic();target=centre.copy()
     if phase=='sine':target[order.index(n)]+=amp*math.sin(2*math.pi*a.frequency_hz*(rs-begin))
     elif phase=='step':target[order.index(n)]+=amp*(1 if int((rs-begin)/1.5)%2==0 else -1)
     ws=time.monotonic();h.set_positions(dict(zip(order,target)));we=time.monotonic();last=target
     w.writerow([sample,phase,n,rs-origin,pe-origin,ve-origin,ws-origin,we-origin,*target,*q,*v]);sample+=1
     if sample%50==0:f.flush()
     deadline+=dt;time.sleep(max(0,deadline-time.monotonic()))
     if time.monotonic()-deadline>dt:deadline=time.monotonic()
   run('baseline','',2)
   for n in a.joints:
    print('Testing',n,flush=True);run('sine',n,a.cycles/a.frequency_hz)
    run('baseline',n,2)
    run('step',n,6)
    run('baseline',n,2)
  except KeyboardInterrupt:print('Interrupted; returning to test centre.')
  finally:
   if started and last is not None:
    try:_slew_to_pose(h,c,last,centre,dt);print('Holding test centre; torque remains enabled.')
    except Exception as e:print('Could not return to centre:',e)
 print('Saved:',out.resolve())
if __name__=='__main__':main()
