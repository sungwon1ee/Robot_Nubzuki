#!/usr/bin/env bash
# Prepare this Raspberry Pi to run the standing policy on the physical robot.
#
# Idempotent: safe to re-run after every `git pull`. It only creates what is
# missing and verifies everything the hardware loop refuses to start without.
#
#   ./scripts/pi_setup.sh                 # pull, install, verify policy "test_2"
#   ./scripts/pi_setup.sh --policy test_3
#   ./scripts/pi_setup.sh --no-pull
#
# Run this ON THE PI. It never touches the Mac.

set -euo pipefail

POLICY_NAME="test_2"
DO_PULL=1
BRANCH="standing"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --policy)  POLICY_NAME="$2"; shift 2 ;;
    --branch)  BRANCH="$2"; shift 2 ;;
    --no-pull) DO_PULL=0; shift ;;
    -h|--help) sed -n '2,12p' "$0" | sed 's/^# \?//'; exit 0 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

VENV="$REPO/.venv"
PY="$VENV/bin/python"
PIP="$VENV/bin/pip"
POLICY_DIR="$REPO/policies/$POLICY_NAME"
POLICY_ONNX="$POLICY_DIR/policy.onnx"
POLICY_JSON="$POLICY_DIR/policy.json"
HEAD_PROFILE="$REPO/config/head_dynamics.json"

step() { printf '\n\033[1m==> %s\033[0m\n' "$1"; }
ok()   { printf '    \033[32mok\033[0m   %s\n' "$1"; }
warn() { printf '    \033[33mwarn\033[0m %s\n' "$1"; }
die()  { printf '    \033[31mfail\033[0m %s\n' "$1" >&2; exit 1; }

# ---------------------------------------------------------------- source ----
if [[ "$DO_PULL" == "1" ]]; then
  step "Updating source"
  git rev-parse --abbrev-ref HEAD | grep -qx "$BRANCH" || git checkout "$BRANCH"
  git pull --ff-only origin "$BRANCH"
  ok "on $BRANCH at $(git rev-parse --short HEAD)"
fi

# ------------------------------------------------------------ interpreter ----
# pyproject pins requires-python = ">=3.11,<3.12".
step "Locating Python 3.11"
PY311=""
for candidate in python3.11 python3; do
  if command -v "$candidate" >/dev/null 2>&1; then
    if "$candidate" -c 'import sys; sys.exit(0 if sys.version_info[:2] == (3, 11) else 1)'; then
      PY311="$(command -v "$candidate")"
      break
    fi
  fi
done
[[ -n "$PY311" ]] || die "no Python 3.11 found. Raspberry Pi OS bookworm ships it as python3; on another release install python3.11 and python3.11-venv."
ok "$PY311 ($("$PY311" -V 2>&1))"

# ------------------------------------------------------------------ venv ----
# .venv/ is gitignored, so a fresh clone or `git pull` never brings one. This
# is the step people miss.
step "Virtual environment"
if [[ ! -x "$PY" ]]; then
  "$PY311" -m venv "$VENV" || die "venv creation failed. Try: sudo apt install python3.11-venv"
  ok "created .venv"
else
  ok "reusing .venv"
fi
"$PIP" install --quiet --upgrade pip

# --------------------------------------------------------- dependencies ----
step "Installing the robot dependencies"
"$PIP" install --quiet -e '.[robot]'
ok "installed '.[robot]'"

# The base dependency set pins numpy>=2.1 (brax reassembles env_steps with
# `(hi << 32) | lo`, which needs NEP 50 during training). The robot extra pins
# onnxruntime==1.18.1, whose wheels are built against numpy 1.x. Nothing on
# this host trains, so onnxruntime wins the tie-break here.
step "Checking the numpy / onnxruntime ABI"
if "$PY" -c 'import numpy, onnxruntime' >/dev/null 2>&1; then
  ok "numpy $("$PY" -c 'import numpy; print(numpy.__version__)') + onnxruntime $("$PY" -c 'import onnxruntime; print(onnxruntime.__version__)')"
else
  warn "import failed; onnxruntime 1.18.1 predates numpy 2 support. Upgrading onnxruntime."
  "$PIP" install --quiet 'onnxruntime>=1.19'
  if "$PY" -c 'import numpy, onnxruntime' >/dev/null 2>&1; then
    ok "resolved with onnxruntime $("$PY" -c 'import onnxruntime; print(onnxruntime.__version__)')"
  else
    warn "still failing; pinning numpy<2 instead (this host only does inference)"
    "$PIP" install --quiet 'numpy<2'
    "$PY" -c 'import numpy, onnxruntime' >/dev/null 2>&1 \
      || die "numpy/onnxruntime still will not import together. Run: $PY -c 'import onnxruntime'"
    ok "resolved with numpy $("$PY" -c 'import numpy; print(numpy.__version__)')"
  fi
fi

# --------------------------------------------------------- calibration ----
step "Calibration"
CALIB_SHA="$("$PY" -c 'from playground.nubzuki.calibration import NubzukiCalibration; print(NubzukiCalibration().sha256)')"
ok "config/nubzuki_calibration.json sha256 ${CALIB_SHA:0:16}..."

# ------------------------------------------------------------- artifacts ----
# policy.onnx, policy.json and config/head_dynamics.json are all gitignored.
# They arrive by scp/rsync, not by git. Check them before the servos are live.
step "Deployment artifacts"

[[ -f "$POLICY_ONNX" ]] || die "missing $POLICY_ONNX  (gitignored; copy it from the Mac, e.g. ./scripts/deploy_to_pi.sh)"
ok "$(realpath --relative-to="$REPO" "$POLICY_ONNX")"

[[ -f "$POLICY_JSON" ]] || die "missing $POLICY_JSON  (StandingPolicy reads the sidecar metadata next to the .onnx)"
ok "$(realpath --relative-to="$REPO" "$POLICY_JSON")"

[[ -f "$HEAD_PROFILE" ]] || die "missing config/head_dynamics.json  (gitignored and robot-specific). Measure it here with the feet unloaded:
             $VENV/bin/nubzuki-standing identify-head --port /dev/ttyACM0"
ok "config/head_dynamics.json"

# Reproduce the contract StandingPolicy enforces, without opening the serial
# port, so a mismatch is a message here instead of a crash with torque on.
step "Policy contract"
"$PY" - "$POLICY_JSON" "$HEAD_PROFILE" "$CALIB_SHA" <<'PYEOF'
import json, sys
from pathlib import Path

meta = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
head = json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))
calib_sha = sys.argv[3]

problems = []
if meta.get("deployable") is not True:
    problems.append("deployable is not true (a smoke or unmarked policy is refused)")
for key, want in (
    ("observation_size", 87),
    ("action_size", 14),
    ("control_frequency_hz", 50),
    ("model_semantics_version", 4),
    ("calibration_sha256", calib_sha),
):
    got = meta.get(key)
    if got != want:
        problems.append(f"{key}: {got!r} != {want!r}")

recorded = meta.get("head_dynamics_sha256")
if head.get("measured") is not True:
    problems.append("head_dynamics.json is marked unmeasured; the physical loop refuses it")
if head.get("calibration_sha256") not in (None, calib_sha):
    problems.append("head_dynamics.json was measured against a different calibration; re-run identify-head")

if problems:
    print("    \033[31mfail\033[0m policy contract:")
    for p in problems:
        print(f"           - {p}")
    if any("model_semantics_version" in p or "calibration_sha256" in p for p in problems):
        print("\n    This policy predates commit da28f91 (v3 hardware semantics) or was")
        print("    trained against a different calibration file. It has to be retrained;")
        print("    the hip roll axes changed in eb09d87 and the sign convention moved with them.")
    sys.exit(1)

print("    \033[32mok\033[0m   87/14 @ 50 Hz, semantics v4, calibration and head profile match")
if recorded is None:
    print("    \033[33mwarn\033[0m policy.json has no head_dynamics_sha256; the profile is not cross-checked")
PYEOF

# ------------------------------------------------------------------ port ----
step "Serial port"
if [[ -e /dev/ttyACM0 ]]; then
  ok "/dev/ttyACM0 present"
  if [[ ! -r /dev/ttyACM0 || ! -w /dev/ttyACM0 ]]; then
    warn "no read/write access. Fix once with:  sudo usermod -aG dialout $USER   then log out and back in."
  fi
else
  warn "/dev/ttyACM0 not present. Is the servo bus plugged in? Check: ls /dev/ttyACM* /dev/ttyUSB*"
fi

# ------------------------------------------------------------------ done ----
IP="$(hostname -I 2>/dev/null | awk '{print $1}')"
step "Ready"
cat <<EOF
Support the robot so its feet cannot bear weight, then:

  cd $REPO
  ./.venv/bin/python -u -m playground.nubzuki.cli robot \\
    --policy policies/$POLICY_NAME/policy.onnx \\
    --port /dev/ttyACM0 \\
    --control phone \\
    --web-port 8766

Torque is off during preflight. Press A to ramp to logical home and arm the
policy, B for immediate torque-off.
EOF
if [[ -n "$IP" ]]; then echo "Phone controller: http://$IP:8766"; fi
