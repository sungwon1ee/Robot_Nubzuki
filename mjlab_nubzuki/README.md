# Nubzuki MJLab + BAM training

This is a separate Python 3.12 training environment. It leaves the existing
Python 3.11 JAX/MJX trainer and real-robot runtime untouched.

The first task uses MicroDuck's velocity-task design with the Nubzuki model and
the official `feetech_sts3215_7_4V` BAM M6 actuator. Motor commands are delayed
3--6 simulation steps inside BAM. Episodes start at the calibrated hardware
park pose (with a 2.25-degree soft-limit-safe knee bend and symmetric
-3.2-degree ankles) and a 0.212 m root height. Small head-pose commands remain active from
the beginning so head-command inputs can be expanded in a later curriculum.

## Colab setup

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
uv python install 3.12
cd /content/Robot_Nubzuki/mjlab_nubzuki
uv sync --python 3.12
```

## Smoke test

```bash
uv run train Mjlab-Velocity-Flat-BAM-Nubzuki \
  --env.scene.num-envs 64 \
  --agent.max-iterations 2 \
  --agent.logger tensorboard
```

## Full training

```bash
uv run train Mjlab-Velocity-Flat-BAM-Nubzuki \
  --env.scene.num-envs 4096 \
  --agent.max-iterations 50000 \
  --agent.logger tensorboard
```

## Film a robot crowd

The crowd player composites parallel MJLab environments into one native
MuJoCo view. Each robot runs the same policy with its own randomly resampled
walking command. The robots do not physically collide with each other.

```bash
uv run mjpython src/mjlab_nubzuki/play_crowd.py \
  --checkpoint checkpoints/walking_v2/model_3950.pt \
  --robots 36
```

The crowd player keeps the detailed CAD exterior but replaces its expensive
SDF contacts with the lightweight training collision shapes. It also disables
shadows/reflections and spaces robots 1 m apart. Use `--spacing 1.5` for a
wider formation, `--command-arrows` to show target directions, or
`--distance 8` to override the automatic camera framing.
