"""Run existing policy with serialized, read-only STS telemetry on the same bus.
Register map: https://github.com/ftservo/FTServo_Python/blob/main/scservo_sdk/sms_sts.py
Current/load are retained as raw words: no assumption of torque or current units.
"""
from __future__ import annotations
import argparse,csv,json,sys,time
from datetime import datetime
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from playground.nubzuki.hardware import ServoHardware
from playground.nubzuki.calibration import NubzukiCalibration

class RegisterReader:
 def __init__(self,port):
  import os,rustypot
  # Configure using the SAME driver as the control loop. Do not let pyserial
  # change shared tty settings (VMIN/VTIME) underneath rustypot's handle.
  def descriptors():
   found=[]
   for entry in os.listdir('/proc/self/fd'):
    try:
     if os.readlink('/proc/self/fd/'+entry)==os.path.realpath(port):found.append(int(entry))
    except OSError:pass
   return found
  matches=descriptors();self.config=None
  if not matches:
   self.config=rustypot.feetech(port,1000000);matches=descriptors()
  if len(matches)!=1:raise RuntimeError('Cannot resolve unique servo bus file descriptor')
  self.fd=os.dup(matches[0])

 def read(self,servo_id,address,length):
  import os,select,termios
  body=bytes([servo_id,4,2,address,length]);packet=b'\xff\xff'+body+bytes([(~sum(body))&255])
  termios.tcflush(self.fd,termios.TCIFLUSH)
  if os.write(self.fd,packet)!=len(packet):raise OSError('Incomplete telemetry read request')
  buf=bytearray();deadline=time.monotonic()+.008
  while time.monotonic()<deadline:
   if not select.select([self.fd],[],[],max(0,deadline-time.monotonic()))[0]:break
   try:buf.extend(os.read(self.fd,256))
   except BlockingIOError:continue
   while len(buf)>=4:
    start=buf.find(b'\xff\xff')
    if start<0:buf[:]=buf[-1:];break
    if start:del buf[:start]
    if len(buf)<4:break
    size=buf[3]+4
    if size<6 or size>80:del buf[0];continue
    if len(buf)<size:break
    frame=bytes(buf[:size]);del buf[:size]
    if frame[2]!=servo_id or size!=length+6 or (sum(frame[2:])&255)!=255:continue
    return frame[4],frame[5:-1]
  raise TimeoutError(f'No valid telemetry from servo {servo_id}')
 def close(self):
  import os
  os.close(self.fd);self.config=None

def word(b,i):return b[i]|b[i+1]<<8

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--port',default='/dev/ttyACM0');p.add_argument('--policy',default='policies/v2_3950/policy.onnx');p.add_argument('--calibration');p.add_argument('--probe',action='store_true');p.add_argument('--web-port',type=int,default=8766);a=p.parse_args()
 c=NubzukiCalibration(a.calibration);reader=RegisterReader(a.port)
 monitored=['left_hip_pitch','right_hip_pitch','left_knee','right_knee','left_ankle','right_ankle']
 snapshots={}
 try:
  for n in monitored:
   start=time.monotonic();err,b=reader.read(c.servo_id(n),3,68)
   snapshots[n]={'status':err,'registers_3_70_hex':b.hex(),'kp_raw':b[21-3],'cw_dead_raw':b[26-3],'ccw_dead_raw':b[27-3],'acceleration_raw':b[41-3],'goal_speed_raw':word(b,46-3),'voltage_V':b[62-3]/10,'temperature_raw':b[63-3],'load_word_raw':word(b,60-3),'current_word_raw':word(b,69-3),'read_ms':(time.monotonic()-start)*1000}
   print(n,snapshots[n],flush=True)
 finally:reader.close()
 if a.probe:return
 input('Stop other robot processes. Support robot against falling, feet on floor. Press Enter to start logging and open the controller: ')
 out=Path('logs')/('load_3950_'+datetime.now().strftime('%Y%m%d_%H%M%S'));out.parent.mkdir(exist_ok=True)
 out.with_suffix('.json').write_text(json.dumps({'register_source':'https://github.com/ftservo/FTServo_Python/blob/main/scservo_sdk/sms_sts.py','calibration':c.data,'policy':json.loads(Path(a.policy).with_suffix('.json').read_text()),'snapshots':snapshots,'telemetry':'One joint read per control call, round-robin across six leg joints. current_word_raw/load_word_raw are NOT amperes or torque. host timestamps do not establish sensor internal freshness.'},indent=2))
 import playground.nubzuki.robot_runtime as runtime
 file=out.with_suffix('.telemetry.csv').open('x',newline='');w=csv.writer(file)
 order=list(c.joint_order)
 w.writerow(['sample','position_read_start_s','position_read_end_s','velocity_read_end_s','write_start_s','write_end_s','telemetry_start_s','telemetry_end_s','joint','servo_id','valid','status','voltage_V','temperature_raw','load_word_raw','current_word_raw','moving_raw','feedback_position_word_raw','feedback_speed_word_raw']+[n+'_target_rad' for n in order]+[n+'_actual_rad' for n in order]+[n+'_velocity_rad_s' for n in order])
 origin=time.monotonic();instances=[]
 class LoggingHardware(ServoHardware):
  def __init__(self,*args,**kwargs):
   super().__init__(*args,**kwargs);self.tele=RegisterReader(a.port);self.ready=False;self.sample=0;self.failures=0;instances.append(self)
  def read_positions(self):
   self.rs=time.monotonic()-origin;self.q=super().read_positions();self.re=time.monotonic()-origin;return self.q
  def read_velocities(self):
   self.v=super().read_velocities();self.ve=time.monotonic()-origin;self.ready=True;return self.v
  def set_positions(self,positions):
   ws=time.monotonic()-origin;super().set_positions(positions);we=time.monotonic()-origin
   if not self.ready:return
   self.ready=False;n=monitored[self.sample%len(monitored)];sid=c.servo_id(n);ts=time.monotonic()-origin
   try:
    status,b=self.tele.read(sid,56,15);values=[1,status,b[6]/10,b[7],word(b,4),word(b,13),b[10],word(b,0),word(b,2)];self.failures=0
   except (OSError,TimeoutError) as e:
    values=[0,'','','','','','','',''];self.failures+=1
    if self.failures==1:print('Telemetry read failed:',e,flush=True)
   te=time.monotonic()-origin;w.writerow([self.sample,self.rs,self.re,self.ve,ws,we,ts,te,n,sid,*values,*[positions.get(j,float('nan')) for j in order],*self.q,*self.v]);self.sample+=1
   if self.sample%50==0:file.flush()
   if self.failures>=3:raise RuntimeError('Telemetry failed three times; stopping test to avoid repeated bus delays')
 runtime.ServoHardware=LoggingHardware
 print('Logs:',out,flush=True)
 try:runtime.run_robot(a.policy,a.port,a.calibration,'config/head_dynamics.json',control='phone',web_port=a.web_port,debug_log_path=str(out.with_suffix('.csv')))
 finally:
  file.close()
  for h in instances:h.tele.close()
if __name__=='__main__':main()
