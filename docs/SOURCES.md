# Source audit

The training task is derived from the public Open Duck Playground
`standing_policy` branch at commit
`ba59de88ab76163f2e0c2c95b4cd45fea5745106` (2025-03-07).

Directly retained behavior:

- 50 Hz control, 500 Hz simulation and 20 second episodes.
- 87 actor observations (including gait phase sine/cosine), 155 privileged
  observations and 14 actions.
- Seven commands with zero `vx`, `vy` and `yaw_rate` and four head targets.
- 20 percent all-zero commands and command resampling every 10 seconds.
- Residual position actions with scale 0.25 rad and 0-2 step action delay.
- Reward scales: orientation -0.5, torque -0.001, action-rate -0.375,
  leg stand-still -0.3, alive +20 and head position -5.0.
- Random pushes every 4-8 seconds: 5-15 N on the torso or 2-5 N on the head,
  sustained for 0.12-0.30 seconds.

Nubzuki adaptations:

- Model indices, joint order, masses and limits come from the Nubzuki MJCF
  and `config/nubzuki_calibration.json`.
- Open Duck head ranges are replaced by the measured Nubzuki limits.
- The zero home pose makes Open Duck's multiplicative reset perturbation a
  no-op; Nubzuki uses bounded additive reset perturbations instead.
- Mac training uses a CPU-sized environment count selected by a no-update
  compilation benchmark. The upstream 8192-environment value is preserved
  only as provenance.
- Joystick command shaping is based on a local head-response identification
  file. No Disney hardware filter constants are copied.

The Disney papers in the workspace use artist-authored reference imitation
and an animation engine. They are background reading, not the implementation
source for this no-Placo task.
