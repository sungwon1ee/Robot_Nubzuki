# Robot Nubzuki - standing

This branch contains one task: keep Nubzuki standing while tracking four head
commands. It is a no-Placo, no-imitation adaptation of Open Duck Playground's
public `standing_policy` branch at commit
`ba59de88ab76163f2e0c2c95b4cd45fea5745106`.

The policy interface is fixed at 85 actor observations, 153 privileged
observations and 14 residual joint-position actions. The first three command
values (`vx`, `vy`, `yaw_rate`) are always zero. See `docs/SOURCES.md` for the
exact upstream/adaptation boundary.

## MacBook setup

The supported local training path is Apple Silicon CPU. `jax-metal` is not
installed or used.

```bash
cd /Users/sungwon/Desktop/Robot/Robot_Nubzuki
python3.11 -m venv .venv
.venv/bin/pip install --upgrade pip
.venv/bin/pip install -e '.[mac-train,test]'
```

Validate the model and observation/action contract. This does not train:

```bash
.venv/bin/nubzuki-standing validate
```

Select the largest of 256, 512, 1024 and 2048 environments whose isolated
compile/rollout/gradient check remains finite and below 16 GiB peak RSS. This
does not apply an optimizer update:

```bash
.venv/bin/nubzuki-standing benchmark-mac
```

Only the following commands start PPO. `smoke` creates a non-deployable policy
and exists only for a user-requested end-to-end pipeline check.

```bash
.venv/bin/nubzuki-standing train --preset smoke

caffeinate -dimsu .venv/bin/nubzuki-standing train \
  --preset macbook \
  --num-timesteps 150000000 \
  --output runs/standing
```

### Stopping and resuming

Press `Ctrl+C` at any time. Re-running the same command continues from the
newest checkpoint in `--output`, because `--restore` defaults to `auto`:

```bash
caffeinate -dimsu .venv/bin/nubzuki-standing train \
  --preset macbook \
  --num-timesteps 150000000 \
  --output runs/standing
```

`--num-timesteps` is the total for the whole schedule, not per run: resuming at
step 15,000,000 trains 135,000,000 more steps. Checkpoint directories and
TensorBoard steps are absolute, so a resumed run never overwrites the run it
continues. The JAX compilation cache in `.tmp/jax_cache` survives a restart, so
only the first run pays for compilation.

Work since the last checkpoint is lost, so the checkpoint interval sets how
long a session has to be. It defaults to 5,000,000 steps; `--checkpoint-every`
changes it:

```bash
caffeinate -dimsu .venv/bin/nubzuki-standing train \
  --preset macbook --num-timesteps 150000000 --output runs/standing \
  --checkpoint-every 1000000 --num-eval-envs 32
```

Every checkpoint costs one evaluation rollout plus an ONNX export. The rollout
is `--num-eval-envs` environments for one full episode, so at the Brax default
of 128 that is 128,000 environment steps each time. Checkpointing five times as
often multiplies that overhead by five, which is why `--num-eval-envs` is worth
lowering alongside it. Divide the interval by the throughput reported by
`benchmark-mac` to size a session before starting one.

Checkpoint frequency does not change the learning schedule: the gradient-step
count depends on `num_timesteps`, `num_updates_per_batch`, `batch_size` and
`unroll_length`, none of which move here.

Brax restores the policy, the value network and the observation normalizer, but
not the Adam optimizer state, which restarts on every resume. Expect a short
dip after each restart and prefer few long runs over many short ones.

To continue from a specific checkpoint, or to start over in a directory that
already holds a run:

```bash
.venv/bin/nubzuki-standing train --preset macbook --output runs/standing \
  --restore runs/standing/checkpoints/step_000015000000/params

.venv/bin/nubzuki-standing train --preset macbook --output runs/standing --fresh
```

TensorBoard logs are under `runs/standing/tensorboard`. Checkpoints, ONNX and
metadata are stored every approximately five million environment steps. The
Mac benchmark reports measured throughput; this project does not promise a
completion time for 150 million CPU steps.

## Head identification and joystick

Control is absolute-position. Left stick X/Y controls head yaw/pitch; right
stick X/Y controls head roll/neck pitch. Releasing the sticks returns all four
commands to logical zero.

A phone is the default control surface for `sim`, so no gamepad is needed. The
simulation serves a touch page and prints its address; open that on a phone on
the same network. The page is self-contained and loads nothing from the
internet, and it carries per-axis invert switches so a stick that moves the
head the wrong way is fixed on the phone rather than in the source. Samples are
timestamped: if the phone drops off the network the head returns to centre
instead of holding its last deflection.

```bash
.venv/bin/nubzuki-standing sim --policy runs/standing/latest/policy.onnx
.venv/bin/nubzuki-standing sim --policy POLICY.onnx --control joystick
```

No guessed deadzone or motion speed is committed. First inspect the procedure:

```bash
.venv/bin/nubzuki-standing identify-head --dry-run
```

On the robot host, support the robot so its feet cannot bear weight, install
the hardware dependencies, then measure the controller and head response:

```bash
.venv/bin/pip install -e '.[robot]'
.venv/bin/nubzuki-standing identify-head --port /dev/ttyACM0
```

The resulting `config/head_dynamics.json` is robot-specific and ignored by Git.
Copy it to the Mac and `sim` picks it up, so the simulated head moves the way
the real one does:

```bash
.venv/bin/nubzuki-standing sim \
  --policy runs/standing/checkpoints/step_000015000000/policy.onnx \
  --head-profile config/head_dynamics.json
```

Without that file `sim` still runs, on unmeasured defaults it announces at
startup. Those defaults are marked as unmeasured and the physical robot refuses
them.

## Physical robot

The physical loop refuses policies that are not 85/14 at 50 Hz, are marked
non-deployable, use a different joint order, or have a different calibration
hash. It also requires a matching head-dynamics profile.

```bash
.venv/bin/nubzuki-standing robot \
  --policy policy.onnx \
  --port /dev/ttyACM0 \
  --head-profile config/head_dynamics.json
```

Servo torque is off during preflight. Press A to ramp to logical home and arm
the policy. Press B for immediate torque-off. Stale IMU data, invalid policy
values, sensor failure or process exit also disables torque.

