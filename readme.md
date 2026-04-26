# Playing Atari 2600 with Reinforcement Learning

本项目用于复现并对比两种经典强化学习算法：

- DQN (Deep Q-Network)
- A2C (Advantage Actor-Critic)

任务环境为 Atari `Assault`。项目包含训练、可视化和模型播放推理流程。

## 项目结构

```text
.
├── readme.md
├── readme_en.md
├── requirements.txt          # 安装依赖
├── models/
│   ├── dqn/                  # DQN checkpoint 输出目录
│   └── a2c/                  # A2C checkpoint 输出目录
├── scripts/
│   ├── train_DQN.ipynb       # 训练 DQN
│   ├── train_A2C.ipynb       # 训练 A2C
│   ├── play_DQN.ipynb        # 播放 DQN 效果
│   └── play_A2C.ipynb        # 播放 A2C 效果
├── play_demonstration/       # 演示视频
│   ├── demonstration_dqn.mp4
│   └── demonstration_a2c.mp4
├── report/                   # 实验报告
│   ├── main.pdf
│   └── images/
│       ├── reward_dqn.png
│       └── reward_a2c.png
└── resource/                 # 算法论文
    ├── A3C.pdf
    └── DQN.pdf
```

## 环境依赖

建议使用 `venv`。

### 1) 创建并激活虚拟环境 

```
python -m venv .venv
source .venv/bin/activate
```

### 2) 安装依赖

先安装 PyTorch（CUDA 12.8 示例，实际请以官网命令为准）：

```powershell
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

再安装其余依赖：

```powershell
pip install -r requirements.txt
```


## 快速开始

在项目根目录启动 Notebook：

```powershell
jupyter notebook
```

建议执行顺序：

1. `scripts/train_DQN.ipynb`
2. `scripts/train_A2C.ipynb`
3. `scripts/play_DQN.ipynb`
4. `scripts/play_A2C.ipynb`

## 模型保存与加载路径

- DQN 训练保存到：`models/dqn`
- A2C 训练保存到：`models/a2C`
- DQN 播放默认加载：`models/dqn/timesteps_10000000.pt`
- A2C 播放会在 `models/a2C` 中自动查找最新 `timesteps_*.pt`

## 算法实现摘要

### DQN

- CNN Q-network
- Replay Buffer
- Target Network
- epsilon-greedy 探索
- 定期同步目标网络与保存 checkpoint

### A2C

- 共享 CNN 主干 + Actor/Critic 双头
- 并行环境采样 (`num_envs=16`)
- n-step return (`n_steps=5`)
- Advantage 标准化
- 熵正则与梯度裁剪

## 训练结果（简要）

- A2C 前期提升更快（并行采样吞吐更高）
- DQN 后期平均 reward 更高（回放与目标网络提升稳定性）

### A2C Reward 曲线

![A2C Reward Curve](report/images/reward_a2c.png)

### DQN Reward 曲线

![DQN Reward Curve](report/images/reward_dqn.png)

完整分析可见：`report/main.pdf`


## 参考文献

- Mnih et al., *Playing Atari with Deep Reinforcement Learning*, 2013.
- Mnih et al., *Asynchronous Methods for Deep Reinforcement Learning*, 2016.

