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


def test_bam_stage_does_not_enable_head_commands():
    tasks = (ROOT / "mjlab_nubzuki/src/mjlab_nubzuki/tasks.py").read_text()
    assert 'head.ranges = ((0.0, 0.0),) * 4' in tasks
    assert 'twist.ranges.lin_vel_x = (0.04, 0.18)' in tasks
    assert 'twist.ranges.ang_vel_z = (-0.70, 0.70)' in tasks
