<div align="center">

# Lightning-Hydra-Template

<a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white"></a>
<a href="https://pytorchlightning.ai/"><img alt="Lightning" src="https://img.shields.io/badge/-Lightning-792ee5?logo=pytorchlightning&logoColor=white"></a>
<a href="https://hydra.cc/"><img alt="Config: Hydra" src="https://img.shields.io/badge/Config-Hydra-89b8cd"></a>
<a href="https://docs.astral.sh/uv/"><img alt="uv" src="https://img.shields.io/badge/uv-package%20manager-blueviolet"></a>
<a href="https://docs.astral.sh/ruff/"><img alt="Ruff" src="https://img.shields.io/badge/Ruff-linter%20%2B%20formatter-orange"></a>

A clean, modern template for deep learning projects.<br>
Click on [<kbd>Use this template</kbd>](https://github.com/your-org/lightning-hydra-template/generate) to initialize a new repository.

_Adapted from [ashleve/lightning-hydra-template](https://github.com/ashleve/lightning-hydra-template), modernized with **uv**, **ruff**, and **Python 3.13**._

</div>

<br>

## Why This Template?

**Rapid experimentation** -- swap models, datasets, loggers, callbacks, and trainer configs from the command line without changing code.

**Reproducible experiments** -- every hyperparameter is version-controlled in YAML configs. Share an experiment config and anyone can reproduce your results.

**Modern Python tooling** -- uses [uv](https://docs.astral.sh/uv/) for fast dependency management, [ruff](https://docs.astral.sh/ruff/) for linting + formatting, and Python 3.13 with modern type hints throughout.

**Minimal boilerplate** -- start training with `uv run python src/train.py`. Add your model and dataset, write a config, and go.

<br>

## Main Technologies

[PyTorch Lightning](https://github.com/PyTorchLightning/pytorch-lightning) - A lightweight PyTorch wrapper for high-performance AI research. Think of it as a framework for organizing your PyTorch code. (v2.6+)

[Hydra](https://github.com/facebookresearch/hydra) - A framework for elegantly configuring complex applications. The key feature is the ability to dynamically create a hierarchical configuration by composition and override it through config files and the command line. (v1.3+)

[uv](https://docs.astral.sh/uv/) - An extremely fast Python package manager, replacing pip, conda, requirements.txt, and setup.py with a single `pyproject.toml`. (Latest)

[Ruff](https://docs.astral.sh/ruff/) - An extremely fast Python linter and formatter, replacing black, isort, flake8, bandit, and pyupgrade. (v0.15+)

<br>

## Project Structure

The directory structure of the new project looks like this:

```
├── .github/                    <- GitHub Actions workflows
│   ├── workflows/
│   │   ├── ci.yml                  <- CI: tests on Python 3.12/3.13, ubuntu/macos
│   │   └── code-quality.yml        <- Ruff lint + format checks
│   ├── dependabot.yml              <- Automated dependency updates
│   └── PULL_REQUEST_TEMPLATE.md    <- PR template
│
├── configs/                    <- Hydra configuration files
│   ├── callbacks/                  <- Callback configs
│   │   ├── default.yaml                <- ModelCheckpoint, EarlyStopping, RichProgressBar, LRMonitor
│   │   ├── early_stopping.yaml
│   │   ├── model_checkpoint.yaml
│   │   ├── model_summary.yaml
│   │   ├── rich_progress_bar.yaml
│   │   └── none.yaml
│   ├── data/                       <- Data configs
│   │   └── mnist.yaml
│   ├── debug/                      <- Debug configs
│   │   ├── default.yaml                <- 1 epoch, CPU, debug logging
│   │   ├── fdr.yaml                    <- Fast dev run (1 batch)
│   │   ├── limit.yaml                  <- Limit data to 1%/5%
│   │   ├── overfit.yaml                <- Overfit to 3 batches
│   │   └── profiler.yaml              <- Profile execution time
│   ├── experiment/                 <- Experiment configs
│   │   └── example.yaml
│   ├── extras/                     <- Extra utilities configs
│   │   └── default.yaml
│   ├── hparams_search/             <- Hyperparameter search configs
│   │   └── mnist_optuna.yaml
│   ├── hydra/                      <- Hydra framework configs
│   │   └── default.yaml
│   ├── local/                      <- Local machine-specific configs (gitignored)
│   ├── logger/                     <- Logger configs
│   │   ├── csv.yaml
│   │   └── wandb.yaml
│   ├── model/                      <- Model configs
│   │   └── mnist.yaml
│   ├── paths/                      <- Path configs
│   │   └── default.yaml
│   ├── trainer/                    <- Trainer configs
│   │   ├── default.yaml                <- CPU, 10 epochs, mixed precision
│   │   ├── cpu.yaml
│   │   ├── gpu.yaml
│   │   ├── mps.yaml                    <- Apple Silicon
│   │   ├── ddp.yaml                    <- Distributed Data Parallel
│   │   └── ddp_sim.yaml               <- DDP simulation on CPU
│   ├── train.yaml                  <- Main training config
│   └── eval.yaml                   <- Main evaluation config
│
├── data/                       <- Project data (MNIST auto-downloaded here)
├── logs/                       <- Logs generated by Hydra and Lightning loggers
├── notebooks/                  <- Jupyter notebooks
│
├── src/
│   ├── data/                       <- LightningDataModules
│   │   └── mnist_datamodule.py
│   ├── models/                     <- LightningModules
│   │   ├── components/                 <- Model sub-components (networks, layers)
│   │   │   └── simple_dense_net.py
│   │   └── mnist_module.py
│   ├── callbacks/                  <- Custom Lightning callbacks
│   ├── utils/                      <- Utility functions
│   │   ├── instantiators.py            <- Instantiate callbacks and loggers from config
│   │   ├── logging_utils.py            <- Log hyperparameters to loggers
│   │   ├── pylogger.py                 <- Multi-GPU-friendly logger (RankedLogger)
│   │   ├── rich_utils.py               <- Rich config tree printing, tag enforcement
│   │   └── utils.py                    <- Task wrapper, extras, metric retrieval
│   ├── train.py                    <- Training entry point
│   └── eval.py                     <- Evaluation entry point
│
├── tests/                      <- Tests
│   ├── helpers/                    <- Test utilities (RunIf, package checks)
│   ├── conftest.py                 <- Test fixtures
│   ├── test_configs.py             <- Config instantiation tests
│   ├── test_datamodules.py         <- DataModule tests
│   ├── test_eval.py                <- Evaluation tests
│   ├── test_train.py               <- Training tests (fast_dev_run, DDP, resume)
│   └── test_sweeps.py              <- Sweep/multirun tests
│
├── pyproject.toml              <- Project config, dependencies, and all tool settings
├── uv.lock                     <- Locked dependency versions
├── .python-version             <- Python version (3.13)
├── .pre-commit-config.yaml     <- Pre-commit hooks (ruff + basic checks)
├── .env.example                <- Example environment variables
├── .gitignore                  <- Files ignored by git
├── .project-root               <- Project root indicator for rootutils
├── Makefile                    <- Common development commands
└── LICENSE                     <- MIT License
```

<br>

## Quick Start

### Install

```bash
# Install uv (if not already installed)
# https://docs.astral.sh/uv/getting-started/installation/
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone this template
git clone <your-repo-url>
cd <your-project>

# Install all dependencies (creates .venv automatically)
uv sync --all-extras

# [Optional] Install pre-commit hooks
uv run pre-commit install
```

### Train

```bash
# Train with default configuration (MNIST on CPU)
uv run python src/train.py

# Train on different hardware
uv run python src/train.py trainer=cpu
uv run python src/train.py trainer=gpu
uv run python src/train.py trainer=mps       # Apple Silicon

# Train with experiment config
uv run python src/train.py experiment=example

# Train with logger
uv run python src/train.py logger=wandb

# Evaluate a checkpoint
uv run python src/eval.py ckpt_path="/path/to/checkpoint.ckpt"
```

### Debug

```bash
uv run python src/train.py debug=fdr          # 1 train + 1 val + 1 test step
uv run python src/train.py debug=default       # 1 epoch with debug logging
uv run python src/train.py debug=limit         # 1% train data, 5% val/test
uv run python src/train.py debug=overfit       # Overfit on 3 batches
uv run python src/train.py debug=profiler      # Profile execution time
```

<br>

## How It Works

All PyTorch Lightning modules are instantiated dynamically from Hydra configs using the `_target_` key:

```yaml
# configs/model/mnist.yaml
_target_: src.models.mnist_module.MNISTLitModule
optimizer:
  _target_: torch.optim.Adam
  _partial_: true
  lr: 0.001
net:
  _target_: src.models.components.simple_dense_net.SimpleDenseNet
  input_size: 784
  output_size: 10
```

This means you can swap any component (model, dataset, logger, callbacks) just by changing a config file -- no code changes needed.

<br>

## Features

### Override Any Parameter From the Command Line

```bash
uv run python src/train.py trainer.max_epochs=20 model.optimizer.lr=1e-4
```

You can override any parameter from the config tree. This is powered by Hydra.

### Experiment Configs

Store complete experiment configurations as YAML files for reproducibility:

```yaml
# configs/experiment/example.yaml
# @package _global_

defaults:
  - override /data: mnist
  - override /model: mnist
  - override /logger: wandb

tags: ["mnist", "simple_dense_net"]
seed: 12345

trainer:
  min_epochs: 10
  max_epochs: 50

model:
  optimizer:
    lr: 0.002
  net:
    lin1_size: 128
    lin2_size: 256
    lin3_size: 64
```

```bash
uv run python src/train.py experiment=example
```

### Hyperparameter Search with Optuna

```bash
# Run Optuna sweep (20 trials by default)
uv run python src/train.py -m hparams_search=mnist_optuna experiment=example

# Override number of trials
uv run python src/train.py -m hparams_search=mnist_optuna hydra.sweeper.n_trials=50
```

The Optuna config (`configs/hparams_search/mnist_optuna.yaml`) defines the search space:

```yaml
params:
  model.optimizer.lr: interval(0.0001, 0.1)
  data.batch_size: choice(32, 64, 128, 256)
  model.net.lin1_size: choice(64, 128, 256)
  model.net.lin2_size: choice(64, 128, 256)
  model.net.lin3_size: choice(32, 64, 128, 256)
```

### Multi-Run Parameter Sweeps

```bash
# Sweep over multiple values
uv run python src/train.py -m data.batch_size=32,64,128 model.optimizer.lr=0.001,0.0005
```

### Logging

Choose a logger by overriding from the command line:

```bash
# CSV logger (default, lightweight)
uv run python src/train.py logger=csv

# Weights & Biases
uv run python src/train.py logger=wandb
```

You can easily add more loggers (TensorBoard, Neptune, MLflow, etc.) by creating new configs in `configs/logger/`.

### Callbacks

The default callback configuration includes:
- **ModelCheckpoint** -- saves best and last checkpoints
- **EarlyStopping** -- stops training when validation loss stops improving
- **RichModelSummary** -- prints model architecture summary
- **RichProgressBar** -- pretty training progress bar
- **LearningRateMonitor** -- logs learning rate to the logger

Override from the command line:

```bash
uv run python src/train.py callbacks=none      # Disable all callbacks
```

### Multi-GPU Training

```bash
# DDP on 4 GPUs
uv run python src/train.py trainer=ddp

# Simulate DDP on CPU (for debugging)
uv run python src/train.py trainer=ddp_sim
```

### Mixed Precision Training

Enabled by default (16-mixed). Override if needed:

```bash
uv run python src/train.py trainer.precision=32     # Full precision
uv run python src/train.py trainer.precision=bf16    # bfloat16
```

### Resume Training From a Checkpoint

```bash
uv run python src/train.py ckpt_path="/path/to/checkpoint.ckpt"
```

### Tag-Based Experiment Organization

Tags help organize experiments in loggers like Weights & Biases:

```bash
uv run python src/train.py tags=["experiment_v1","baseline"]
```

If `extras.enforce_tags=True` (default), you'll be prompted to enter tags if none are provided.

### Local Config Files

Create a `configs/local/default.yaml` for machine-specific settings (gitignored):

```yaml
# configs/local/default.yaml

# set paths specific to this machine
paths:
  data_dir: /fast-storage/datasets/

# set hardware
trainer:
  accelerator: gpu
  devices: 4
```

<br>

## How To Add New Components

### Add a New Model

1. Create a LightningModule in `src/models/`:

```python
class MyModel(LightningModule):
    def __init__(self, net, optimizer, scheduler, compile):
        super().__init__()
        self.save_hyperparameters(logger=False)
        self.net = net
        ...
```

2. Create a config in `configs/model/`:

```yaml
# configs/model/my_model.yaml
_target_: src.models.my_model.MyModel
optimizer:
  _target_: torch.optim.Adam
  _partial_: true
  lr: 0.001
net:
  _target_: src.models.components.my_net.MyNet
  hidden_size: 256
compile: false
```

3. Train: `uv run python src/train.py model=my_model`

### Add a New Dataset

1. Create a LightningDataModule in `src/data/`:

```python
class MyDataModule(LightningDataModule):
    def __init__(self, data_dir, batch_size, num_workers, pin_memory):
        super().__init__()
        self.save_hyperparameters(logger=False)
        ...
```

2. Create a config in `configs/data/`:

```yaml
# configs/data/my_data.yaml
_target_: src.data.my_datamodule.MyDataModule
data_dir: ${paths.data_dir}
batch_size: 64
num_workers: 0
pin_memory: false
```

3. Train: `uv run python src/train.py data=my_data`

### Add a New Experiment

1. Create a config in `configs/experiment/`:

```yaml
# @package _global_

defaults:
  - override /data: my_data
  - override /model: my_model
  - override /logger: wandb

tags: ["my_experiment"]
seed: 42

trainer:
  max_epochs: 100
  accelerator: gpu

data:
  batch_size: 256

model:
  optimizer:
    lr: 0.0005
```

2. Train: `uv run python src/train.py experiment=my_experiment`

### Add a New Logger

1. Create a config in `configs/logger/` (example for TensorBoard):

```yaml
# configs/logger/tensorboard.yaml
tensorboard:
  _target_: lightning.pytorch.loggers.tensorboard.TensorBoardLogger
  save_dir: "${paths.output_dir}"
  name: "tensorboard/"
  log_graph: false
```

2. Train: `uv run python src/train.py logger=tensorboard`

<br>

## Development

### Makefile Commands

```bash
make help           # Show all available commands
make sync           # Install dependencies with uv
make train          # Train with default config
make test           # Run fast tests
make test-full      # Run all tests (including slow)
make format         # Run pre-commit hooks
make lint           # Run ruff linter
make lint-fix       # Run ruff linter with auto-fix
make clean          # Remove autogenerated files
make clean-logs     # Remove training logs
```

### Running Tests

```bash
uv run pytest                       # All tests
uv run pytest -k "not slow"        # Fast tests only
uv run pytest -v                    # Verbose output
```

The test suite includes:
- **Config tests** -- verify all configs instantiate correctly
- **DataModule tests** -- verify data loading and batching
- **Training tests** -- fast_dev_run, GPU, mixed precision, DDP
- **Evaluation tests** -- train then evaluate pipeline
- **Sweep tests** -- Optuna and Hydra multirun

### Code Quality

```bash
# Lint
uv run ruff check .
uv run ruff check --fix .

# Format
uv run ruff format .

# Pre-commit (runs both)
uv run pre-commit run -a
```

### CI/CD

Two GitHub Actions workflows are included:

- **CI** (`ci.yml`) -- runs tests on Python 3.12/3.13 across Ubuntu and macOS
- **Code Quality** (`code-quality.yml`) -- runs `ruff check` and `ruff format --check`

Both use [astral-sh/setup-uv](https://github.com/astral-sh/setup-uv) for fast dependency installation.

<br>

## Configuration Hierarchy

The main training config (`configs/train.yaml`) composes from these defaults:

```yaml
defaults:
  - data: mnist             # Dataset
  - model: mnist            # Model architecture
  - callbacks: default      # Training callbacks
  - logger: csv             # Logging backend
  - trainer: default        # Hardware and training settings
  - paths: default          # Directory paths
  - extras: default         # Extra utilities
  - hydra: default          # Hydra framework settings

  # Optional overrides (null = disabled)
  - experiment: null        # Experiment-specific overrides
  - debug: null             # Debug mode overrides
  - hparams_search: null    # Hyperparameter search
  - local: default          # Machine-specific settings
```

The `debug` config, when activated, overrides `callbacks: null` and `logger: null` to simplify debugging.

<br>

## Differences From the Original Template

This template is based on [ashleve/lightning-hydra-template](https://github.com/ashleve/lightning-hydra-template) with the following modernizations:

| Feature | Original | This Template |
|---|---|---|
| **Package manager** | pip / conda | **uv** |
| **Linter/Formatter** | black, isort, flake8, bandit, pyupgrade | **ruff** (all-in-one) |
| **Python version** | 3.8 - 3.11 | **3.12+** (3.13 default) |
| **Type hints** | `Dict`, `List`, `Optional[X]` | `dict`, `list`, `X \| None` |
| **Pre-commit hooks** | 14 repos | **2 repos** (pre-commit-hooks + ruff) |
| **Build system** | setup.py + setup.cfg + requirements.txt | **pyproject.toml** (single file) |
| **CI/CD** | pip-based | **uv-based** with matrix testing |
| **Loggers** | csv, wandb, tensorboard, neptune, comet, mlflow | **csv, wandb** (easy to add more) |

<br>

## Resources

- [PyTorch Lightning docs](https://lightning.ai/docs/pytorch/latest/)
- [Hydra docs](https://hydra.cc/docs/intro/)
- [uv docs](https://docs.astral.sh/uv/)
- [Ruff docs](https://docs.astral.sh/ruff/)
- [Original template](https://github.com/ashleve/lightning-hydra-template)
- [Lightning-Hydra-Template tutorial](https://github.com/ashleve/lightning-hydra-template/tree/main#your-superpowers)

<br>

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
