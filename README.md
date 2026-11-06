# Computational Intelligence - Projects

This repository contains project implementations developed as part of the **Computational Intelligence (2024/2025)** course.
Each project focuses on a different aspect of artificial intelligence-ranging from classic game-tree search to reinforcement learning and multi-agent environments.

---

## Projects

### `board_game_ai/` - Board Game AI (Negamax)
Implementation of **TicTacDoh**, a Tic-Tac-Toe variant with an element of randomness: in the probabilistic variant, each chosen move has a 20% chance of failing and the turn passes to the opponent.
A comparison of the **Negamax** and **Negamax with alpha-beta pruning** algorithms across both deterministic and probabilistic versions of the game, over 1000 self-play games per configuration (win rates and average decision time).

**Libraries:** Python standard library only (`random`, `time`) - no external dependencies.

---

### `rl_fundamentals/` - Basics of Gymnasium (Reinforcement Learning)
Initial experiments with **Gymnasium** using the `FrozenLake-v1` environment (8x8 map).
Implementation of a tabular **Q-learning** agent with epsilon decay, along with learning curve visualization (episode rewards and training error with moving averages).

**Libraries:** `gymnasium`, `numpy`, `matplotlib`, `tqdm`

---

### `crossy_road_env/` - Custom Gymnasium Environment
Creation of a custom **Crossy Road** environment compliant with the `gymnasium.Env` API (grid world with cars, logs, and water obstacles), with a graphical mode rendered via **pygame**.
Training of RL agents (**A2C**, **DQN** from Stable-Baselines3) and results analysis.

**Libraries:** `gymnasium`, `pygame`, `stable-baselines3`, `numpy`

---

### `custom_env_experiments/` - Custom Environment Experiments
Follow-up experiments with custom environments: refined **CrossyRoad** training runs (PPO/DQN with a flattened observation wrapper and training plots), plus a **Minecraft** environment built on `minerl`.

**Libraries:** `gymnasium`, `pygame`, `stable-baselines3`, `matplotlib`, `minerl`

---

### `multi_agent_rl/` - Multi-Agent Environments
Experiments in multi-agent environments using **PettingZoo** (Connect Four) and **PPO** from **Stable-Baselines3** in a self-play setup: a single agent trains against a copy of itself, with reward-logging callbacks and self-play training curves.

**Libraries:** `pettingzoo`, `stable-baselines3`, `gymnasium`, `numpy`, `matplotlib`
