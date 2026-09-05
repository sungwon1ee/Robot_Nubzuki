#!/usr/bin/env bash
# Export an MJLab checkpoint and run it on the physical Nubzuki.
#
# Replaces the manual export/scp/ssh sequence for MJLab checkpoints: the two
# files the robot needs (policy.onnx + policy.json) are produced from the .pt
# by the training environment itself, so the observation layout, joint order,
# action scale and default pose on the robot are the ones the policy was
# trained with rather than something typed in by hand.
#
#   scripts/deploy_robot.sh mjlab_nubzuki/checkpoints/model_570.pt
#   scripts/deploy_robot.sh <ckpt> --name test_6 --no-run
#
set -euo pipefail

ROBOT_HOST="${ROBOT_HOST:-sungwon@192.168.45.18}"
ROBOT_REPO="${ROBOT_REPO:-~/Robot_Nubzuki}"
ROBOT_BRANCH="${ROBOT_BRANCH:-walking}"
SERIAL_PORT="${SERIAL_PORT:-/dev/ttyACM0}"
WEB_PORT="${WEB_PORT:-8766}"
TASK="${TASK:-Mjlab-Velocity-Flat-BAM-Nubzuki}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# One authentication for the whole run. Without this the script asks for the
# password four separate times (mkdir, scp, git pull, run); ControlMaster opens
# a single connection up front and every later ssh/scp rides on it.
# %C is a short hash of user/host/port. macOS TMPDIR is long enough that a
# readable name blows past the 104-character unix socket limit.
CONTROL_PATH="/tmp/nz-%C"
SSH_OPTS=(-o ControlMaster=auto -o "ControlPath=$CONTROL_PATH" -o ControlPersist=600)

close_master() {
  ssh "${SSH_OPTS[@]}" -O exit "$ROBOT_HOST" 2>/dev/null || true
}
CHECKPOINT=""
NAME=""
RUN=1

while [[ $# -gt 0 ]]; do
  case "$1" in
    --name) NAME="$2"; shift 2 ;;
    --message|-m) COMMIT_MSG="$2"; shift 2 ;;
    --task) TASK="$2"; shift 2 ;;
    --host) ROBOT_HOST="$2"; shift 2 ;;
    --no-run) RUN=0; shift ;;
    -h|--help) sed -n '2,12p' "${BASH_SOURCE[0]}"; exit 0 ;;
    *) CHECKPOINT="$1"; shift ;;
  esac
done

if [[ -z "$CHECKPOINT" ]]; then
  echo "usage: $(basename "$0") <checkpoint.pt> [--name test_6] [--no-run]" >&2
  exit 2
fi
CHECKPOINT="$(cd "$(dirname "$CHECKPOINT")" && pwd)/$(basename "$CHECKPOINT")"
[[ -f "$CHECKPOINT" ]] || { echo "No such checkpoint: $CHECKPOINT" >&2; exit 2; }
NAME="${NAME:-$(basename "${CHECKPOINT%.pt}")}"

echo "== 0/4 connect =="
echo "Authenticating to $ROBOT_HOST once; the rest of the run reuses it."
ssh "${SSH_OPTS[@]}" -o ConnectTimeout=10 "$ROBOT_HOST" true
trap close_master EXIT

echo
echo "== 1/6 calibration =="
# The robot is where joints get re-zeroed, so its calibration file is the
# authority, not this checkout. Bring any change back before anything else:
# the robot's git pull further down would otherwise collide with it, and the
# export needs the same calibration the robot will run.
CAL="config/nubzuki_calibration.json"
ROBOT_CAL="$(mktemp)"
scp -q "${SSH_OPTS[@]}" "$ROBOT_HOST:$ROBOT_REPO/$CAL" "$ROBOT_CAL"
if cmp -s "$ROBOT_CAL" "$REPO_ROOT/$CAL"; then
  echo "Calibration matches the robot."
else
  echo "The robot's calibration differs; taking it as the source of truth:"
  diff <(python3 -m json.tool "$REPO_ROOT/$CAL") <(python3 -m json.tool "$ROBOT_CAL") \
    | grep -E '^[<>]' | head -20 || true
  cp "$ROBOT_CAL" "$REPO_ROOT/$CAL"
fi
rm -f "$ROBOT_CAL"

echo
echo "== 2/6 commit and push =="
cd "$REPO_ROOT"
if [[ -n "$(git status --porcelain)" ]]; then
  git add -A
  git commit -q -m "${COMMIT_MSG:-Deploy $NAME}"
  echo "Committed: $(git log --oneline -1)"
else
  echo "Nothing to commit."
fi
git push -q origin "$ROBOT_BRANCH"
echo "Pushed to origin/$ROBOT_BRANCH."

echo
echo "== 3/6 export =="
OUT_DIR="$REPO_ROOT/mjlab_nubzuki/deploy/$NAME"
( cd "$REPO_ROOT/mjlab_nubzuki" \
  && uv run python scripts/export_deploy.py "$CHECKPOINT" --task "$TASK" --out "$OUT_DIR" )
[[ -f "$OUT_DIR/policy.onnx" && -f "$OUT_DIR/policy.json" ]] \
  || { echo "Export did not produce both files" >&2; exit 1; }

echo
echo "== 4/6 copy to $ROBOT_HOST =="
ssh "${SSH_OPTS[@]}" "$ROBOT_HOST" "mkdir -p $ROBOT_REPO/policies/$NAME"
scp "${SSH_OPTS[@]}" "$OUT_DIR/policy.onnx" "$OUT_DIR/policy.json" "$ROBOT_HOST:$ROBOT_REPO/policies/$NAME/"

echo
echo "== 5/6 update the robot's checkout =="
# The MJLab observation builder lives in the repo, so the robot needs the same
# commit that exported the policy.
# The robot's own calibration edit is now committed here, so discarding its
# working copy loses nothing and lets the pull run clean.
ssh "${SSH_OPTS[@]}" "$ROBOT_HOST" \
  "cd $ROBOT_REPO && git checkout -- $CAL 2>/dev/null; \
   git checkout $ROBOT_BRANCH && git pull origin $ROBOT_BRANCH"

if [[ "$RUN" -eq 0 ]]; then
  echo
  echo "Copied. To run it yourself:"
  echo "  ssh -t $ROBOT_HOST 'cd $ROBOT_REPO && ./.venv/bin/python -u -m playground.nubzuki.cli robot --policy policies/$NAME/policy.onnx --port $SERIAL_PORT --control phone --web-port $WEB_PORT'"
  exit 0
fi

echo
echo "== 6/6 run =="
echo "Support the robot BEFORE pressing ARM. Phone: http://${ROBOT_HOST#*@}:$WEB_PORT"
echo
ssh "${SSH_OPTS[@]}" -t "$ROBOT_HOST" \
  "cd $ROBOT_REPO && ./.venv/bin/python -u -m playground.nubzuki.cli robot \
     --policy policies/$NAME/policy.onnx \
     --port $SERIAL_PORT \
     --control phone \
     --web-port $WEB_PORT"
