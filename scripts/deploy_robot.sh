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
CHECKPOINT=""
NAME=""
RUN=1

while [[ $# -gt 0 ]]; do
  case "$1" in
    --name) NAME="$2"; shift 2 ;;
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

echo "== 1/4 export =="
OUT_DIR="$REPO_ROOT/mjlab_nubzuki/deploy/$NAME"
( cd "$REPO_ROOT/mjlab_nubzuki" \
  && uv run python scripts/export_deploy.py "$CHECKPOINT" --task "$TASK" --out "$OUT_DIR" )
[[ -f "$OUT_DIR/policy.onnx" && -f "$OUT_DIR/policy.json" ]] \
  || { echo "Export did not produce both files" >&2; exit 1; }

echo
echo "== 2/4 copy to $ROBOT_HOST =="
ssh "$ROBOT_HOST" "mkdir -p $ROBOT_REPO/policies/$NAME"
scp "$OUT_DIR/policy.onnx" "$OUT_DIR/policy.json" "$ROBOT_HOST:$ROBOT_REPO/policies/$NAME/"

echo
echo "== 3/4 update the robot's checkout =="
# The MJLab observation builder lives in the repo, so the robot needs the same
# commit that exported the policy.
ssh "$ROBOT_HOST" "cd $ROBOT_REPO && git checkout $ROBOT_BRANCH && git pull origin $ROBOT_BRANCH"

if [[ "$RUN" -eq 0 ]]; then
  echo
  echo "Copied. To run it yourself:"
  echo "  ssh -t $ROBOT_HOST 'cd $ROBOT_REPO && ./.venv/bin/python -u -m playground.nubzuki.cli robot --policy policies/$NAME/policy.onnx --port $SERIAL_PORT --control phone --web-port $WEB_PORT'"
  exit 0
fi

echo
echo "== 4/4 run =="
echo "Support the robot BEFORE pressing ARM. Phone: http://${ROBOT_HOST#*@}:$WEB_PORT"
echo
ssh -t "$ROBOT_HOST" \
  "cd $ROBOT_REPO && ./.venv/bin/python -u -m playground.nubzuki.cli robot \
     --policy policies/$NAME/policy.onnx \
     --port $SERIAL_PORT \
     --control phone \
     --web-port $WEB_PORT"
