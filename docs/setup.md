# Environment setup

## This machine (verified working)

| Component | Value |
|---|---|
| OS | Windows 11 |
| Python | 3.11.9 (`.venv`) |
| GPU | NVIDIA RTX 5070 Ti Laptop, 12 GB VRAM, driver 595.79 |
| Compute capability | sm_120 (Blackwell) |
| CPU / RAM | Intel Core Ultra 9 275HX, 24 cores / 31 GB |
| PyTorch | 2.11.0+cu128, CUDA available |

Blackwell (sm_120) needs a CUDA 12.8 PyTorch build. The default `pip install
torch` wheel may not cover it — if `torch.cuda.is_available()` is `False` or a
matmul raises a kernel error, install explicitly:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu128
```

Verify:

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_capability(0))"
```

## Install

```bash
python -m venv .venv
.venv/Scripts/activate          # Linux/macOS: source .venv/bin/activate
pip install -e ".[dev,viz]"
pytest
```

Or with conda, which pins the interpreter as well:

```bash
conda env create -f environment.yml
conda activate vision-nav
pytest
```

That covers everything except the Nav2 baseline, which needs its own
environment — see below.

Optional extras:

- `pip install -e ".[tracking]"` — Weights & Biases
- `pip install tensorboard` — TensorBoard logging (training degrades to CSV
  gracefully without it)

## A note on CPU vs GPU for this phase

Training deliberately defaults to **CPU** (`train.device: auto` resolves to
`cpu`). The policy is a small MLP and the simulator is fast NumPy on the host;
per-batch host-to-device transfer costs more than the matmuls save. Measured
throughput is ~2,000 environment steps/s on CPU.

The GPU becomes the right choice at Phase 3, when observations become images
and the encoder becomes a CNN. Set `train.device=cuda` then.

## ROS 2 + Nav2 (for the Nav2 baseline)

Report §4.1 scores the real Nav2 stack on the same worlds. It lives in a
**separate** environment from the training venv, because ROS 2 pins its own
Python and NumPy and has no need of torch:

```bash
bash scripts/install_ros2_nav2.sh      # ~937 packages, all under $HOME
bash ros2_bridge/run_nav2.sh --condition narrow --episodes 100
```

The installer uses [RoboStack](https://robostack.github.io/) rather than apt,
because the apt route needs root and this WSL image has no passwordless sudo.
Nothing is installed system-wide; `rm -rf ~/mamba ~/bin/micromamba` removes the
entire stack. Versions are pinned (ROS 2 Jazzy, Nav2 1.3.12) so the numbers in
the report are reproducible rather than "whatever RoboStack ships today".

What Nav2 is given, and the failure modes the harness guards against:
[`ros2_bridge/README.md`](../ros2_bridge/README.md).

## Roadmap dependencies not yet installed

These belong to later phases and are intentionally absent:

| Tool | Phase | Note |
|---|---|---|
| Gazebo | Sim-to-real | Not needed yet: the Nav2 comparison drives `ProceduralNavEnv` directly through the ROS 2 bridge, so there is no second physics engine to reconcile. |
| Isaac Sim / Isaac Lab | Final runs | Needs 16 GB VRAM; this GPU has 12 GB. Plan on rented cloud GPU (RTX 4090/A6000 class) for the two or three heavy runs. |
| Habitat-Sim | Alternate track | Much lighter than Isaac Sim; viable locally if the project leans embodied-AI rather than ground-robot control. |
| Depth Anything v2 / YOLOv11 / SAM2 | Phase 3+ | Only needed once observations become images. |

## Reproducibility

Every training run writes its fully resolved config to
`runs/<run_name>/config.yaml` alongside its checkpoints, so any result can be
traced back to the settings that produced it. `runs/` is gitignored; curated
outputs in `results/` are committed.
