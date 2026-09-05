"""Re-stamp a head dynamics profile after a NON-head calibration change.

Re-zeroing a leg changes the calibration file's hash, and the head profile
pins that hash, so a perfectly valid head measurement starts being refused.
Re-running identify-head would work but measures nothing new.

This does not rubber-stamp. It takes the calibration backup that the zeroing
tool wrote, proves two things, and only then updates the profile:

  1. the backup is the calibration the profile was actually measured against
     (its hash matches the one recorded in the profile), and
  2. every head joint is byte-identical between that backup and the current
     calibration, so nothing the profile measured has moved.

If a head joint did change, it refuses and tells you to measure again.

    ./.venv/bin/python -u scripts/restamp_head_profile.py \\
        --backup config/nubzuki_calibration.20260905_211530.bak.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from playground.nubzuki.calibration import HEAD_JOINTS, NubzukiCalibration  # noqa: E402


def whole_sha(data: dict) -> str:
    canonical = json.dumps(data, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", default="config/head_dynamics.json")
    parser.add_argument("--calibration", default="config/nubzuki_calibration.json")
    parser.add_argument("--backup", required=True,
                        help="the calibration as it was when the profile was measured")
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()

    profile_path = Path(args.profile)
    backup_path = Path(args.backup)
    profile = json.loads(profile_path.read_text(encoding="utf-8"))
    old_cal = json.loads(backup_path.read_text(encoding="utf-8"))
    calibration = NubzukiCalibration(args.calibration)

    recorded = profile.get("calibration_sha256")
    if recorded != whole_sha(old_cal):
        raise SystemExit(
            f"{backup_path.name} is not the calibration this profile was measured "
            f"against (hash mismatch). Point --backup at the right one, or re-run "
            f"identify-head."
        )

    changed = [
        name for name in HEAD_JOINTS
        if old_cal["joints"][name] != calibration.data["joints"][name]
    ]
    if old_cal["control_frequency_hz"] != calibration.data["control_frequency_hz"]:
        changed.append("control_frequency_hz")
    if changed:
        raise SystemExit(
            f"These head-side entries changed: {changed}. The profile measured "
            f"the old ones, so it no longer applies. Run "
            f"`nubzuki-standing identify-head`."
        )

    moved = [
        name for name in calibration.joint_order
        if name not in HEAD_JOINTS
        and old_cal["joints"][name] != calibration.data["joints"][name]
    ]
    print(f"Head-side calibration is unchanged; only {moved or 'nothing'} moved.")
    print(f"  calibration_sha256      {recorded[:12]} -> {calibration.sha256[:12]}")
    print(f"  head_calibration_sha256 (new) {calibration.head_sha256[:12]}")

    if not args.apply:
        print("\nReport only. Re-run with --apply to write it.")
        return

    backup = profile_path.with_name(
        f"{profile_path.stem}.{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        f".bak{profile_path.suffix}"
    )
    shutil.copyfile(profile_path, backup)
    profile["calibration_sha256"] = calibration.sha256
    profile["head_calibration_sha256"] = calibration.head_sha256
    profile_path.write_text(
        json.dumps(profile, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"\nWrote {profile_path} (backup at {backup.name})")


if __name__ == "__main__":
    main()
