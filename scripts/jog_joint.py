"""Jog one joint from a phone and commit the offset it should have been at.

Re-zeroing by hand means holding a limb steady while reading a terminal. This
holds the joint with torque instead and moves it from a slider, so you can
stand back, look at the robot, line the joint up against its opposite number,
and commit -- without touching it.

The commit does not trust the slider. It reads the servo's raw angle at that
moment and solves servo = direction * logical + offset for the offset that
makes the pose you are looking at read as --at-deg. Only that one joint's
offset_deg changes; the file is backed up first.

    ./.venv/bin/python -u scripts/jog_joint.py left_hip_pitch
    ./.venv/bin/python -u scripts/jog_joint.py left_hip_pitch --at-deg 0.48
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
import threading
import time
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from playground.nubzuki.calibration import HEAD_JOINTS, NubzukiCalibration  # noqa: E402
from playground.nubzuki.hardware import ServoHardware, _bus  # noqa: E402
from playground.nubzuki.phone_controller import local_address  # noqa: E402
from playground.nubzuki.robot_runtime import park  # noqa: E402

MIRROR = {"left": "right", "right": "left"}

PAGE = """<!doctype html><html lang="ko"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1,maximum-scale=1">
<title>__JOINT__ 영점</title>
<style>
:root{--bg:#0b1220;--card:#111d30;--line:#24344b;--text:#e6edf7;--dim:#8fa3bf;--ok:#34d399}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--text);font:15px/1.5 -apple-system,system-ui,sans-serif;
padding:max(14px,env(safe-area-inset-top)) 16px max(14px,env(safe-area-inset-bottom))}
h1{font-size:17px;margin:0 0 2px}
.sub{color:var(--dim);font-size:13px;margin-bottom:16px}
.card{background:var(--card);border:1px solid var(--line);border-radius:14px;padding:16px;margin-bottom:12px}
.big{font-size:40px;font-weight:700;font-variant-numeric:tabular-nums;letter-spacing:-.02em}
.unit{font-size:18px;color:var(--dim);font-weight:400}
.row{display:flex;gap:8px;margin-top:12px}
button{flex:1;padding:14px 0;font-size:15px;font-weight:700;color:var(--text);background:#182842;
border:1px solid var(--line);border-radius:11px;-webkit-tap-highlight-color:transparent}
button:active{background:#24344b;transform:translateY(1px)}
button.go{background:#14532d;border-color:#166534}
input[type=range]{width:100%;margin:18px 0 4px;accent-color:#3b82f6}
.meta{display:flex;justify-content:space-between;color:var(--dim);font-size:13px}
.line{display:flex;justify-content:space-between;padding:5px 0;font-variant-numeric:tabular-nums}
.line span:last-child{font-weight:600}
#msg{margin-top:10px;font-size:14px;min-height:20px;color:var(--ok)}
</style>
<h1>__JOINT__ 영점 맞추기</h1>
<div class="sub">__TWIN_TEXT__</div>

<div class="card">
  <div class="big"><span id="cmd">0.00</span><span class="unit"> deg 명령</span></div>
  <input type="range" id="slider" min="__MIN__" max="__MAX__" step="0.05" value="0">
  <div class="meta"><span>__MIN__</span><span>__MAX__</span></div>
  <div class="row">
    <button data-step="-1">−1.0</button><button data-step="-0.1">−0.1</button>
    <button data-step="0.1">+0.1</button><button data-step="1">+1.0</button>
  </div>
</div>

<div class="card">
  <div class="line"><span>실측 (현재 보정 기준)</span><span id="actual">—</span></div>
  <div class="line"><span>__TWIN__</span><span id="twin">—</span></div>
  <div class="line"><span>적용하면 오프셋</span><span id="offset">—</span></div>
</div>

<div class="card">
  <div class="sub" style="margin:0 0 10px">눈으로 맞춘 뒤 누르면, 지금 자세가 __AT__ deg로 읽히도록 오프셋을 저장합니다.</div>
  <button class="go" id="apply">이 자세를 __AT__ deg로 저장</button>
  <div id="msg"></div>
</div>

<script>
const slider=document.getElementById('slider'),cmd=document.getElementById('cmd');
let target=Number(slider.value);
function show(){cmd.textContent=target.toFixed(2);slider.value=target}
function push(){fetch('/set',{method:'POST',headers:{'Content-Type':'application/json'},
  body:JSON.stringify({angle:target})})}
slider.addEventListener('input',()=>{target=Number(slider.value);show();push()});
document.querySelectorAll('[data-step]').forEach(b=>b.addEventListener('click',()=>{
  target=Math.min(__MAX__,Math.max(__MIN__,target+Number(b.dataset.step)));show();push()}));
document.getElementById('apply').addEventListener('click',async()=>{
  const msg=document.getElementById('msg');msg.textContent='저장 중...';
  const r=await fetch('/apply',{method:'POST'});msg.textContent=await r.text();});
async function poll(){
  try{
    const s=await (await fetch('/state')).json();
    document.getElementById('actual').textContent=s.actual.toFixed(2)+' deg';
    document.getElementById('twin').textContent=s.twin===null?'—':s.twin.toFixed(2)+' deg';
    document.getElementById('offset').textContent=
      s.current_offset.toFixed(2)+' → '+s.new_offset.toFixed(2)+' deg';
  }catch(e){}
}
setInterval(poll,200);poll();
</script></html>
"""


def opposite(name: str) -> str | None:
    side = name.split("_")[0]
    return name.replace(side, MIRROR[side], 1) if side in MIRROR else None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("joint")
    parser.add_argument("--port", default="/dev/ttyACM0")
    parser.add_argument("--calibration", default="config/nubzuki_calibration.json")
    parser.add_argument("--at-deg", type=float, default=None,
                        help="angle the aligned pose should read as; defaults to the joint's park angle")
    parser.add_argument("--web-port", type=int, default=8767)
    args = parser.parse_args()

    path = Path(args.calibration)
    calibration = NubzukiCalibration(str(path))
    joint = args.joint
    if joint not in calibration.joint_order:
        raise SystemExit(f"Unknown joint: {joint}")
    at_deg = (
        args.at_deg if args.at_deg is not None
        else float(calibration.joints[joint]["park_deg"])
    )
    twin = opposite(joint)
    order = list(calibration.joint_order)
    index = order.index(joint)
    servo_id = calibration.servo_id(joint)
    direction = calibration.direction(joint)
    lower, upper = calibration.limits_rad(joint)
    dt = 1.0 / calibration.control_frequency_hz

    hardware = ServoHardware(calibration, args.port)
    lock = threading.Lock()
    state = {"target": math.radians(at_deg), "pose": None}

    print(f"Joint {joint} (servo {servo_id}, direction {direction:+.0f})")
    print(f"Current offset {float(calibration.joints[joint]['offset_deg']):+.2f} deg; "
          f"aligning to {at_deg:+.2f} deg")
    print("\nParking...")
    hardware.disable_torque()
    hardware.preflight()
    start = hardware.read_positions()
    hardware.set_positions(dict(zip(order, start)))
    hardware.set_kps([int(calibration.data["runtime"]["low_kp"])] * 14)
    hardware.enable_torque()
    park(hardware, calibration, start, dt)
    pose = [calibration.park_rad(name) for name in order]
    state["pose"] = pose
    state["target"] = pose[index]
    # The jogged joint needs a real gain to actually go where the slider says;
    # the rest stay soft so nothing fights the pose you are referencing.
    hardware.set_joint_kps([joint], [int(calibration.data["runtime"]["leg_kp"])])

    def snapshot() -> dict:
        measured = hardware.read_positions()
        raw = float(_bus(hardware.io.read_present_position, [servo_id])[0])
        return {
            "actual": math.degrees(float(measured[index])),
            "twin": math.degrees(float(measured[order.index(twin)])) if twin else None,
            "current_offset": float(calibration.joints[joint]["offset_deg"]),
            "new_offset": math.degrees(raw - direction * math.radians(at_deg)),
        }

    def apply_offset() -> str:
        raw = float(_bus(hardware.io.read_present_position, [servo_id])[0])
        new_offset = math.degrees(raw - direction * math.radians(at_deg))
        old_offset = float(calibration.joints[joint]["offset_deg"])
        if abs(new_offset - old_offset) > 45.0:
            return (f"거부: {new_offset - old_offset:+.1f} deg 변화는 너무 큽니다. "
                    f"기준각이나 관절이 잘못됐습니다.")
        data = json.loads(path.read_text(encoding="utf-8"))
        backup = path.with_name(
            f"{path.stem}.{datetime.now().strftime('%Y%m%d_%H%M%S')}.bak{path.suffix}"
        )
        shutil.copyfile(path, backup)
        data["joints"][joint]["offset_deg"] = round(new_offset, 3)
        path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n",
                        encoding="utf-8")
        message = (f"저장됨: {old_offset:+.2f} → {new_offset:+.2f} deg "
                   f"(백업 {backup.name})")
        print(f"\n{message}")
        print("Restart any running policy so it picks up the new calibration.")
        return message

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_a):
            pass

        def _send(self, code, body=b"", kind="text/plain; charset=utf-8"):
            self.send_response(code)
            self.send_header("Content-Type", kind)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            if body:
                self.wfile.write(body)

        def do_GET(self):
            route = self.path.split("?")[0]
            if route in ("/", "/index.html"):
                page = (PAGE.replace("__JOINT__", joint)
                        .replace("__TWIN_TEXT__",
                                 f"{twin} 와 대칭이 되도록 맞추세요" if twin
                                 else "기준 자세에 맞추세요")
                        .replace("__TWIN__", twin or "-")
                        .replace("__MIN__", f"{math.degrees(lower):.1f}")
                        .replace("__MAX__", f"{math.degrees(upper):.1f}")
                        .replace("__AT__", f"{at_deg:.2f}"))
                self._send(200, page.encode("utf-8"), "text/html; charset=utf-8")
                return
            if route == "/state":
                with lock:
                    body = json.dumps(snapshot()).encode("utf-8")
                self._send(200, body, "application/json")
                return
            self._send(404)

        def do_POST(self):
            route = self.path.split("?")[0]
            if route == "/set":
                length = int(self.headers.get("Content-Length", "0"))
                payload = json.loads(self.rfile.read(length) or b"{}")
                angle = math.radians(float(payload.get("angle", 0.0)))
                with lock:
                    state["target"] = max(lower, min(upper, angle))
                self._send(204)
                return
            if route == "/apply":
                with lock:
                    message = apply_offset()
                self._send(200, message.encode("utf-8"))
                return
            self._send(404)

    server = ThreadingHTTPServer(("0.0.0.0", args.web_port), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    print(f"\nOpen this on your phone:\n    http://{local_address()}:{args.web_port}\n")
    print("Ctrl-C when you are done.")

    try:
        while True:
            with lock:
                pose[index] = state["target"]
                hardware.set_positions(dict(zip(order, pose)))
            time.sleep(dt)
    except KeyboardInterrupt:
        print("\nStopping.")
    finally:
        server.shutdown()
        try:
            park(hardware, calibration, hardware.read_positions(), dt)
            hardware.set_kps([
                int(calibration.data["runtime"]["head_kp"]) if name in HEAD_JOINTS
                else int(calibration.data["runtime"]["leg_kp"]) for name in order
            ])
            print("Parked and holding.")
        except Exception as error:
            print(f"Park failed: {error}")


if __name__ == "__main__":
    main()
