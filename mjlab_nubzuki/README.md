# Nubzuki MJLab + BAM training

This is a separate Python 3.12 training environment. It leaves the existing
Python 3.11 JAX/MJX trainer and real-robot runtime untouched.

The first task uses MicroDuck's velocity-task design with the Nubzuki model and
the official `feetech_sts3215_7_4V` BAM M6 actuator. Motor commands are delayed
3--6 simulation steps inside BAM. Episodes start at the calibrated hardware
park pose and a 0.209 m root height. Small head-pose commands remain active from
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
