"""The touch control page served by `PhoneController`.

Kept as one self-contained string with no external requests: the Mac running
the simulation is not necessarily on a network that can reach a CDN, and a
control surface that fails to load is worse than one that looks plain.
"""

CONTROL_PAGE = """<!doctype html>
<html lang="ko">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1,minimum-scale=.5,maximum-scale=5,user-scalable=yes,viewport-fit=cover">
<title>Nubzuki Control</title>
<style>
:root{color-scheme:dark;--bg:#090d14;--panel:#121925;--panel2:#182231;--edge:#2b3a50;
  --text:#f4f7fb;--dim:#8e9caf;--blue:#60a5fa;--cyan:#67e8f9;--live:#34d399;--stop:#fb7185}
*{box-sizing:border-box;-webkit-tap-highlight-color:transparent;-webkit-user-drag:none;user-select:none}
html,body{margin:0;height:100%;overflow:hidden;background:var(--bg);color:var(--text);
  font:15px/1.45 -apple-system,BlinkMacSystemFont,"Segoe UI",system-ui,sans-serif;
  touch-action:pinch-zoom;overscroll-behavior:none}
body{display:flex;flex-direction:column;padding:max(12px,env(safe-area-inset-top)) 14px max(12px,env(safe-area-inset-bottom));
  background:radial-gradient(circle at 50% -20%,#172b46 0,#0d1521 32%,var(--bg) 68%)}
header{display:flex;align-items:center;gap:10px;padding:0 4px 12px}
.brand{font-size:13px;font-weight:750;letter-spacing:.16em;color:#dbeafe}
.connection{margin-left:auto;display:flex;align-items:center;gap:8px;padding:6px 10px;border:1px solid #253247;
  border-radius:999px;background:#0b111bbb;box-shadow:0 8px 24px #0004}
#dot{width:8px;height:8px;border-radius:50%;background:var(--stop);flex:none;box-shadow:0 0 12px currentColor}
#dot.live{background:var(--live)}
#status{color:var(--dim);font-size:12px;min-width:0;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
#pads{flex:1;display:flex;align-items:center;justify-content:space-evenly;gap:18px;min-height:0}
.pad{position:relative;width:min(42vw,42vh);height:min(42vw,42vh);max-width:360px;max-height:360px;
  background:radial-gradient(circle at 38% 32%,#24364f,#111a28 68%,#0b111a);border:1px solid #344762;
  border-radius:50%;overflow:hidden;box-shadow:inset 0 0 0 9px #080d1466,inset 0 0 42px #60a5fa12,0 18px 42px #000a;
  touch-action:none}
.pad::after{content:"";position:absolute;inset:7%;border:1px solid #8ec5ff18;border-radius:50%;pointer-events:none}
.pad .name{position:absolute;z-index:2;top:12%;left:8%;right:8%;text-align:center;font-size:10px;font-weight:700;
  letter-spacing:.12em;color:#a9bdd5}
.pad .val{position:absolute;z-index:2;bottom:11%;left:8%;right:8%;text-align:center;font-size:11px;color:#718198;font-variant-numeric:tabular-nums}
.pad .cross{position:absolute;inset:16%;border-radius:50%;
  background:linear-gradient(#3b506d99,#3b506d99) center/1px 100% no-repeat,
             linear-gradient(#3b506d99,#3b506d99) center/100% 1px no-repeat}
.knob{position:absolute;width:64px;height:64px;margin:-32px 0 0 -32px;border-radius:50%;
  background:radial-gradient(circle at 35% 28%,#dbeafe,#60a5fa 42%,#2563eb 74%);border:1px solid #dbeafe;
  box-shadow:0 10px 24px #000b,0 0 24px #3b82f655,inset 0 1px 4px #fff;pointer-events:none;transition:filter .12s,transform .12s}
.pad.held .knob{filter:brightness(1.18);transform:scale(1.06)}
.row{display:flex;gap:10px;padding-top:10px}
button{flex:1;padding:14px 0;font-size:14px;font-weight:700;letter-spacing:.04em;color:var(--text);
  background:linear-gradient(180deg,var(--panel2),var(--panel));border:1px solid var(--edge);border-radius:14px;
  box-shadow:0 8px 20px #0005;touch-action:manipulation}
button:active{background:#24344b;transform:translateY(1px)}
#mode{border-color:#2563eb;color:#bfdbfe;background:linear-gradient(180deg,#172b48,#111d30)}
#b{border-color:#7f1d1d;color:#fecdd3;background:linear-gradient(180deg,#311820,#211218)}
#opts{display:flex;flex-wrap:wrap;gap:8px 14px;padding:10px 4px 0;font-size:12px;color:var(--dim)}
#opts[hidden]{display:none}
label{display:flex;align-items:center;gap:5px}
@media(max-width:700px) and (orientation:portrait){#pads{gap:12px}.pad{width:44vw;height:44vw}.brand{font-size:11px}}
</style>
</head>
<body>
<header><span class="brand">NUBZUKI CONTROL</span><span class="connection"><span id="dot"></span><span id="status">연결 대기 중</span></span></header>
<div id="pads">
  <div class="pad" id="left"><div class="cross"></div><div class="name">TURN / FORWARD·BACK</div>
    <div class="knob"></div><div class="val">0.00, 0.00</div></div>
  <div class="pad" id="right"><div class="cross"></div><div class="name">HEAD YAW / PITCH</div>
    <div class="knob"></div><div class="val">0.00, 0.00</div></div>
</div>
<div class="row"><button id="mode">WALK + HEAD CONTROL</button></div>
<div class="row"><button id="a">ARM · 시작</button><button id="b">PARK · 정지</button></div>
<div id="opts">
  <label><input type="checkbox" data-flip="left_x">yaw 반전</label>
  <label><input type="checkbox" data-flip="left_y">pitch 반전</label>
  <label><input type="checkbox" data-flip="right_x">roll 반전</label>
  <label><input type="checkbox" data-flip="right_y">neck 반전</label>
</div>
<script>
const state={left_x:0,left_y:0,right_x:0,right_y:0,a:false,b:false,mode:'walk'};
const flip={left_x:1,left_y:1,right_x:1,right_y:1};
addEventListener('dragstart',event=>event.preventDefault());
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
  return centre;
}
const centreLeft=stick('left','left_x','left_y');
const centreRight=stick('right','right_x','right_y');

const modeButton=document.getElementById('mode');
function renderMode(){
  const head=state.mode==='head';
  modeButton.textContent=head?'HEAD CONTROL':'WALK + HEAD CONTROL';
  document.querySelector('#left .name').textContent=head?'HEAD YAW / PITCH':'TURN / FORWARD·BACK';
  document.querySelector('#right .name').textContent=head?'HEAD ROLL / NECK PITCH':'HEAD YAW / PITCH';
  document.getElementById('opts').hidden=!head;
  centreLeft();centreRight();
}
modeButton.addEventListener('click',()=>{state.mode=state.mode==='head'?'walk':'head';renderMode()});
renderMode();

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
    status.textContent=state.mode==='head'?'연결됨 · HEAD':'연결됨 · WALK + HEAD';
  }catch(error){
    failures++;
    if(failures>2){dot.classList.remove('live');status.textContent='연결 끊김 — __TARGET_LABEL__가 실행 중인지 확인'}
  }
}
setInterval(send,50);
send();
</script>
</body>
</html>
"""
