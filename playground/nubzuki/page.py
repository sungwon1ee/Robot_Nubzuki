"""The touch control page served by `PhoneController`.

Kept as one self-contained string with no external requests: the Mac running
the simulation is not necessarily on a network that can reach a CDN, and a
control surface that fails to load is worse than one that looks plain.
"""

CONTROL_PAGE = """<!doctype html>
<html lang="ko">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1,maximum-scale=1,user-scalable=no,viewport-fit=cover">
<title>Nubzuki Standing</title>
<style>
:root{color-scheme:dark;--bg:#11151c;--panel:#1b2230;--edge:#2c3647;--text:#e7edf7;--dim:#8b98ad;--live:#4ade80;--stop:#f87171}
*{box-sizing:border-box;-webkit-tap-highlight-color:transparent}
html,body{margin:0;height:100%;overflow:hidden;background:var(--bg);color:var(--text);
  font:15px/1.45 -apple-system,BlinkMacSystemFont,"Segoe UI",system-ui,sans-serif;touch-action:none;user-select:none}
body{display:flex;flex-direction:column;padding:max(10px,env(safe-area-inset-top)) 10px max(10px,env(safe-area-inset-bottom))}
header{display:flex;align-items:center;gap:10px;padding:2px 4px 10px}
#dot{width:9px;height:9px;border-radius:50%;background:var(--stop);flex:none}
#dot.live{background:var(--live)}
#status{color:var(--dim);font-size:13px;flex:1;min-width:0;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
#pads{flex:1;display:flex;gap:10px;min-height:0}
.pad{flex:1;position:relative;background:var(--panel);border:1px solid var(--edge);border-radius:16px;overflow:hidden}
.pad .name{position:absolute;top:9px;left:0;right:0;text-align:center;font-size:11px;letter-spacing:.09em;color:var(--dim)}
.pad .val{position:absolute;bottom:9px;left:0;right:0;text-align:center;font-size:11px;color:var(--dim);font-variant-numeric:tabular-nums}
.pad .cross{position:absolute;inset:22px;border-radius:12px;
  background:linear-gradient(var(--edge),var(--edge)) center/1px 100% no-repeat,
             linear-gradient(var(--edge),var(--edge)) center/100% 1px no-repeat}
.knob{position:absolute;width:64px;height:64px;margin:-32px 0 0 -32px;border-radius:50%;
  background:#3b82f6;border:2px solid #93c5fd;pointer-events:none;transition:background .12s}
.pad.held .knob{background:#60a5fa}
#row{display:flex;gap:10px;padding-top:10px}
button{flex:1;padding:15px 0;font-size:16px;font-weight:600;color:var(--text);
  background:var(--panel);border:1px solid var(--edge);border-radius:14px}
button:active{background:#27334a}
#b{border-color:#7f1d1d;color:#fecaca}
#opts{display:flex;flex-wrap:wrap;gap:8px 14px;padding:10px 4px 0;font-size:12px;color:var(--dim)}
label{display:flex;align-items:center;gap:5px}
</style>
</head>
<body>
<header><span id="dot"></span><span id="status">연결 대기 중</span></header>
<div id="pads">
  <div class="pad" id="left"><div class="cross"></div><div class="name">HEAD YAW / PITCH</div>
    <div class="knob"></div><div class="val">0.00, 0.00</div></div>
  <div class="pad" id="right"><div class="cross"></div><div class="name">HEAD ROLL / NECK PITCH</div>
    <div class="knob"></div><div class="val">0.00, 0.00</div></div>
</div>
<div id="row"><button id="a">A · 시작</button><button id="b">B · 정지</button></div>
<div id="opts">
  <label><input type="checkbox" data-flip="left_x">yaw 반전</label>
  <label><input type="checkbox" data-flip="left_y">pitch 반전</label>
  <label><input type="checkbox" data-flip="right_x">roll 반전</label>
  <label><input type="checkbox" data-flip="right_y">neck 반전</label>
</div>
<script>
const state={left_x:0,left_y:0,right_x:0,right_y:0,a:false,b:false};
const flip={left_x:1,left_y:1,right_x:1,right_y:1};
document.querySelectorAll('[data-flip]').forEach(box=>{
  box.addEventListener('change',()=>{flip[box.dataset.flip]=box.checked?-1:1});
});

function stick(id,xKey,yKey){
  const pad=document.getElementById(id),knob=pad.querySelector('.knob'),val=pad.querySelector('.val');
  let touchId=null;
  const centre=()=>{knob.style.left='50%';knob.style.top='50%';
    state[xKey]=0;state[yKey]=0;val.textContent='0.00, 0.00';pad.classList.remove('held')};
  centre();
  addEventListener('resize',centre);
  function move(event){
    const box=pad.getBoundingClientRect(),radius=Math.min(box.width,box.height)/2-32;
    let dx=event.clientX-box.left-box.width/2, dy=event.clientY-box.top-box.height/2;
    const distance=Math.hypot(dx,dy);
    if(distance>radius&&distance>0){dx*=radius/distance;dy*=radius/distance}
    knob.style.left=(box.width/2+dx)+'px';knob.style.top=(box.height/2+dy)+'px';
    // Screen y grows downward; the joints treat up as positive.
    const x=radius?dx/radius:0, y=radius?-dy/radius:0;
    state[xKey]=x*flip[xKey];state[yKey]=y*flip[yKey];
    val.textContent=x.toFixed(2)+', '+y.toFixed(2);
  }
  pad.addEventListener('pointerdown',e=>{if(touchId!==null)return;
    touchId=e.pointerId;pad.setPointerCapture(e.pointerId);pad.classList.add('held');move(e);e.preventDefault()});
  pad.addEventListener('pointermove',e=>{if(e.pointerId===touchId){move(e);e.preventDefault()}});
  for(const type of ['pointerup','pointercancel','lostpointercapture'])
    pad.addEventListener(type,e=>{if(e.pointerId===touchId){touchId=null;centre()}});
}
stick('left','left_x','left_y');
stick('right','right_x','right_y');

for(const [id,key] of [['a','a'],['b','b']]){
  const button=document.getElementById(id);
  button.addEventListener('pointerdown',e=>{state[key]=true;e.preventDefault()});
  for(const type of ['pointerup','pointercancel','pointerleave'])
    button.addEventListener(type,()=>{state[key]=false});
}

const dot=document.getElementById('dot'),status=document.getElementById('status');
let failures=0;
async function send(){
  try{
    const response=await fetch('/input',{method:'POST',headers:{'Content-Type':'application/json'},
      body:JSON.stringify(state)});
    if(!response.ok)throw new Error('HTTP '+response.status);
    failures=0;dot.classList.add('live');
    status.textContent='연결됨 · 스틱을 놓으면 정면으로 복귀';
  }catch(error){
    failures++;
    if(failures>2){dot.classList.remove('live');status.textContent='연결 끊김 — 시뮬레이터가 실행 중인지 확인'}
  }
}
setInterval(send,50);
send();
</script>
</body>
</html>
"""
