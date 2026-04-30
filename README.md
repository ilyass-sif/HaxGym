# HaxGym 🏟️⚡

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![JAX](https://img.shields.io/badge/JAX-Accelerated-blue)](https://github.com/google/jax)

**HaxGym** is a fast, fully hardware-accelerated Reinforcement Learning environment for [Haxball](https://www.haxball.com/), written entirely in **JAX**.

By leveraging JAX's `vmap` and `jit` capabilities, HaxGym can simulate **over 1.35 Million frames per second** on a single **NVIDIA RTX 4050 Laptop GPU**, making it possible to train high-level multi-agent policies via self-play in a matter of hours instead of weeks.

## ⚡ Performance

HaxGym is built for extreme scale. Here are the benchmark results on an **NVIDIA RTX 4050 Laptop GPU**:

| Metric | Result |
| :--- | :--- |
| **Parallel Environments** | 6,144 |
| **Player Frames Per Second** | **1,356,718 FPS** ⚡ |
| **Physics Ticks Per Second** | **6,783,589 TPS** 🔥 |
| **AI Decisions Per Second** | 226,120 Hz |

*Benchmarked using `examples/benchmark.py` with Frame Skip 5.*

## 🚀 Key Features

*   **Pure JAX Physics Engine**: A complete recreation of the Haxball physics engine (collisions, rigid bodies, restitution, damping) written in pure JAX, allowing massive batched parallelization.
*   **Insane Throughput**: Simulate 6,000+ environments simultaneously on a single GPU.
*   **Multi-Agent Ready**: Supports 3v3 (6 agents) simultaneously.
*   **OpenAI Gym-like API**: Familiar `make`, `reset`, and `step` semantics.

## 📦 Installation

To install HaxGym from source:

```bash
git clone https://github.com/YOUR-USERNAME/HaxGym.git
cd HaxGym
pip install -e .
```

*Note: Ensure you have installed JAX with GPU support if you want maximum performance.*

## 🕹️ Usage

```python
import jax
import haxgym

# 1. Initialize Environment
env = haxgym.make("Haxball-Classic-v0")

# 2. Reset
rng = jax.random.PRNGKey(42)
state = env.reset(rng)

# 3. Step
# HaxGym expects an array of actions for all 6 players (3 Red, 3 Blue)
# Actions are integers 0-9 (0: None, 1: Up, 2: Down, ... 9: Kick)
actions = jax.numpy.array([1, 0, 9, 2, 5, 0])
next_state, obs, rewards, done = env.step(rng, state, actions)

print(f"Rewards: {rewards}")
```


## 🏆 Available Environments

| ID | Description |
| :--- | :--- |
| `Haxball-Classic-v0` | The standard 3v3 Haxball game. First team to score in 1 minute. Sparse + dense curriculum rewards. |

## 🧠 What's in the Observation Space?
- Self absolute position and velocity.
- Ball relative position and velocity.
- 2 Teammates relative positions and velocities.
- 3 Opponents relative positions and velocities.
- 2 Raycast variables predicting if a passing lane to teammates is blocked by opponents.
- Distance to the ball.

## 🤝 Contributing

Pull requests are welcome! If you want to add new stadiums, better observation spaces, or optimize the physics engine further, feel free to open an issue or PR.

## 📝 License

[MIT License](LICENSE)
