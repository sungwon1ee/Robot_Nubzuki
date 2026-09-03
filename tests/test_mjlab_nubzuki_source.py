from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_mjlab_package_pins_real_bam_stack():
    pyproject = (ROOT / "mjlab_nubzuki/pyproject.toml").read_text()
    robot = (ROOT / "mjlab_nubzuki/src/mjlab_nubzuki/robot.py").read_text()
    assert "mjlab==1.3.0" in pyproject
    assert 'branch = "mjlab_frictionloss"' in pyproject
    assert "feetech_sts3215_7_4V_m6.json" in robot
    assert "delay_min_lag=3" in robot
    assert "delay_max_lag=6" in robot


def test_bam_stage_keeps_head_commands_small_but_alive():
    tasks = (ROOT / "mjlab_nubzuki/src/mjlab_nubzuki/tasks.py").read_text()
    assert "(-0.03, 0.03)" in tasks
    assert "(-0.05, 0.05)" in tasks
    assert "(-0.01, 0.01)" in tasks
    assert "head.zero_command_prob = 0.25" in tasks
    assert 'twist.ranges.lin_vel_x = (0.04, 0.18)' in tasks
    assert 'twist.ranges.ang_vel_z = (-0.70, 0.70)' in tasks
    assert 'cfg.curriculum.pop("standing_envs"' not in tasks


def test_bam_home_uses_root_height_and_calibrated_park_pose():
    robot = (ROOT / "mjlab_nubzuki/src/mjlab_nubzuki/robot.py").read_text()
    assert "pos=(0.0, 0.0, 0.209)" in robot
    assert "HOME_JOINT_POS = _load_park_pose()" in robot
    assert 'calibration["joints"][name]["park_deg"]' in robot
    assert "target.copy_(cmd.pos)" in robot
