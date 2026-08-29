#!/usr/bin/env bash
# Push a trained policy from the Mac to the robot and set the Pi up to run it.
#
#   ./scripts/deploy_to_pi.sh                          # policy "test_2" -> sungwon@nubzuki
#   ./scripts/deploy_to_pi.sh --policy test_3
#   ./scripts/deploy_to_pi.sh --host 192.168.0.42
#   ./scripts/deploy_to_pi.sh --push-head-profile      # also send config/head_dynamics.json
#
# Requires ssh key access to the Pi (ssh-copy-id sungwon@nubzuki once).
#
# Run this ON THE MAC.

set -euo pipefail

POLICY_NAME="test_2"
PI_HOST="nubzuki"
PI_USER="sungwon"
PI_DIR="Robot_Nubzuki"
BRANCH="standing"
PUSH_HEAD_PROFILE=0
RUN_SETUP=1

while [[ $# -gt 0 ]]; do
  case "$1" in
    --policy)            POLICY_NAME="$2"; shift 2 ;;
    --host)              PI_HOST="$2"; shift 2 ;;
    --user)              PI_USER="$2"; shift 2 ;;
    --remote-dir)        PI_DIR="$2"; shift 2 ;;
    --branch)            BRANCH="$2"; shift 2 ;;
    --push-head-profile) PUSH_HEAD_PROFILE=1; shift ;;
    --no-setup)          RUN_SETUP=0; shift ;;
    -h|--help) sed -n '2,12p' "$0" | sed 's/^# \?//'; exit 0 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

TARGET="$PI_USER@$PI_HOST"
POLICY_DIR="policies/$POLICY_NAME"

step() { printf '\n\033[1m==> %s\033[0m\n' "$1"; }
ok()   { printf '    \033[32mok\033[0m   %s\n' "$1"; }
warn() { printf '    \033[33mwarn\033[0m %s\n' "$1"; }
die()  { printf '    \033[31mfail\033[0m %s\n' "$1" >&2; exit 1; }

# ------------------------------------------------------------ local files ----
step "Local artifacts"
[[ -f "$POLICY_DIR/policy.onnx" ]] || die "missing $POLICY_DIR/policy.onnx"
[[ -f "$POLICY_DIR/policy.json" ]] || die "missing $POLICY_DIR/policy.json  (the sidecar metadata StandingPolicy requires)"
ok "$POLICY_DIR/policy.onnx  ($(du -h "$POLICY_DIR/policy.onnx" | cut -f1))"
ok "$POLICY_DIR/policy.json"

if [[ "$PUSH_HEAD_PROFILE" == "1" ]]; then
  [[ -f config/head_dynamics.json ]] || die "--push-head-profile given but config/head_dynamics.json does not exist here"
  ok "config/head_dynamics.json (will overwrite the Pi's copy)"
fi

# ------------------------------------------------------------------- ssh ----
step "Reaching $TARGET"
ssh -o BatchMode=yes -o ConnectTimeout=8 "$TARGET" true \
  || die "cannot ssh to $TARGET without a password. Run:  ssh-copy-id $TARGET"
ok "ssh ok"

ssh "$TARGET" "test -d ~/$PI_DIR/.git" \
  || die "~/$PI_DIR is not a git checkout on the Pi. Clone it there first."

# ----------------------------------------------------------------- source ----
# scripts/pi_setup.sh itself arrives over git, so pull before invoking it.
step "Updating the Pi checkout"
ssh "$TARGET" "cd ~/$PI_DIR && git checkout $BRANCH >/dev/null 2>&1; git pull --ff-only origin $BRANCH"

# ------------------------------------------------------------------ files ----
# policy.onnx, policy.json and head_dynamics.json are all gitignored, so this
# is the only way they reach the robot.
step "Copying artifacts"
ssh "$TARGET" "mkdir -p ~/$PI_DIR/$POLICY_DIR"
if command -v rsync >/dev/null 2>&1; then
  rsync -avh --progress "$POLICY_DIR/policy.onnx" "$POLICY_DIR/policy.json" "$TARGET:$PI_DIR/$POLICY_DIR/"
  if [[ "$PUSH_HEAD_PROFILE" == "1" ]]; then
    rsync -avh config/head_dynamics.json "$TARGET:$PI_DIR/config/"
  fi
else
  warn "rsync not found, falling back to scp"
  scp "$POLICY_DIR/policy.onnx" "$POLICY_DIR/policy.json" "$TARGET:$PI_DIR/$POLICY_DIR/"
  if [[ "$PUSH_HEAD_PROFILE" == "1" ]]; then
    scp config/head_dynamics.json "$TARGET:$PI_DIR/config/"
  fi
fi
ok "copied"

# ------------------------------------------------------------------ setup ----
if [[ "$RUN_SETUP" == "1" ]]; then
  step "Running scripts/pi_setup.sh on the Pi"
  # Source is already current, so skip the redundant pull over there.
  ssh -t "$TARGET" "cd ~/$PI_DIR && ./scripts/pi_setup.sh --policy $POLICY_NAME --no-pull"
else
  step "Skipped remote setup (--no-setup)"
  echo "    Run it yourself:  ssh $TARGET 'cd ~/$PI_DIR && ./scripts/pi_setup.sh --policy $POLICY_NAME'"
fi
