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
    assert "pos=(0.0, 0.0, 0.212)" in robot
    assert "HOME_JOINT_POS = _load_park_pose()" in robot
    assert 'calibration["joints"][name]["park_deg"]' in robot
    assert 'pose["left_knee"] = math.radians(-2.25)' in robot
    assert 'pose["right_knee"] = math.radians(-2.25)' in robot
    assert 'pose["left_ankle"] = math.radians(-3.2)' in robot
    assert 'pose["right_ankle"] = math.radians(-3.2)' in robot
    assert "target.copy_(cmd.pos)" in robot


def test_bam_preserves_nubzuki_collision_masks_and_clean_playback():
    robot = (ROOT / "mjlab_nubzuki/src/mjlab_nubzuki/robot.py").read_text()
    tasks = (ROOT / "mjlab_nubzuki/src/mjlab_nubzuki/tasks.py").read_text()
    assert "CollisionCfg" not in robot
    assert "collisions=(COLLISIONS,)" not in robot
    assert '"push_robot"' in tasks
    assert "cfg.events.pop(event_name, None)" in tasks
    assert "cfg.sim.contact_sensor_maxmatch = 128" in tasks
    assert 'cfg.events["reset_base"].params["pose_range"]' in tasks
    assert "NUBZUKI_BAM_DETAILED_ROBOT_CFG if play" in tasks
    assert 'NUBZUKI_DETAILED_XML = REPOSITORY_ROOT.parent / "Nubzuki/mjcf/nubzuki_v1.xml"' in robot
    assert 'pos=(0.0, 0.0, 0.20945)' in robot
    assert 'trunk_proxy.name = "trunk_head_collision_proxy"' in robot
    assert 'head_proxy.name = "head_trunk_collision_proxy"' in robot
    assert "for joint_name in JOINT_ACTUATOR_ORDER:" in robot
    assert "spec.delete(actuator)" in robot


def test_bam_training_starts_with_balance_curriculum():
    tasks = (ROOT / "mjlab_nubzuki/src/mjlab_nubzuki/tasks.py").read_text()
    assert '{"step": 0, "rel_standing_envs": 1.00}' in tasks
    assert '{"step": 50 * 24, "rel_standing_envs": 0.60}' in tasks
    assert '{"step": 220 * 24, "rel_standing_envs": 0.10}' in tasks
    assert "twist.rel_standing_envs = 1.0" in tasks
