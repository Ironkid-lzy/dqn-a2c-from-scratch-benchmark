# Playing Atari 2600 with Reinforcement Learning

Language: English | [中文](readme.md)

This project reproduced and systematically compared **Deep Q-Network (DQN)** and **Advantage Actor-Critic (A2C)** algorithms in a controlled reinforcement learning environment, analyzing convergence behavior under identical experimental settings.

The experiments are conducted on Atari `Assault`, including training, visualization, and policy playback.

## Project Structure

```text
.
├── readme.md
├── readme_en.md
├── requirements.txt          # Install dependencies
├── models/
│   ├── dqn/                  # DQN checkpoint output
│   └── a2c/                  # A2C checkpoint output
├── scripts/
│   ├── train_DQN.ipynb       # Train DQN
│   ├── train_A2C.ipynb       # Train A2C
│   ├── play_DQN.ipynb        # Play DQN results
│   └── play_A2C.ipynb        # Play A2C results
├── report/                   # Experiment report
│   ├── main.pdf
│   └── images/
│       ├── reward_dqn.png
│       └── reward_a2c.png
└── resource/                 # Reference papers
    ├── A3C.pdf
    └── DQN.pdf
```

## Environment Setup

Using a virtual environment is recommended.

### 1) Create and activate a virtual environment

Windows PowerShell:

```powershell
python -m venv deeplearning
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
.\deeplearning\Scripts\Activate.ps1
```

Linux/macOS:

```bash
python -m venv .venv
source .venv/bin/activate
```

### 2) Install dependencies

Install PyTorch first (CUDA 12.8 example; check the official site for your platform):

```powershell
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

Then install the remaining dependencies:

```powershell
pip install -r requirements.txt
```

## Quick Start

Launch Jupyter Notebook from the project root:

```powershell
jupyter notebook
```

Recommended execution order:

1. `scripts/train_DQN.ipynb`
2. `scripts/train_A2C.ipynb`
3. `scripts/play_DQN.ipynb`
4. `scripts/play_A2C.ipynb`

## Checkpoint Paths

- DQN training saves to: `models/dqn`
- A2C training saves to: `models/a2c`
- DQN playback loads: `models/dqn/timesteps_10000000.pt`
- A2C playback automatically loads the latest `timesteps_*.pt` from `models/a2c`

## Implementation Summary

### DQN

- CNN-based Q-network
- Replay Buffer
- Target Network synchronization
- Epsilon-greedy exploration
- Periodic checkpoint saving

### A2C

- Shared CNN backbone + Actor/Critic heads
- Parallel environment sampling (`num_envs=16`)
- n-step return (`n_steps=5`)
- Advantage normalization
- Entropy regularization + gradient clipping

## Results (Brief)

- A2C improves faster in early training due to higher sampling throughput.
- DQN reaches higher average reward in later stages with replay and target-network stability.

### A2C Reward Curve

![A2C Reward Curve](report/images/reward_a2c.png)

### DQN Reward Curve

![DQN Reward Curve](report/images/reward_dqn.png)

For full analysis, see `report/main.pdf`.


## References

- Mnih et al., *Playing Atari with Deep Reinforcement Learning*, 2013.
- Mnih et al., *Asynchronous Methods for Deep Reinforcement Learning*, 2016.
