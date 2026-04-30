# HaxGym Performance Benchmarks ⚡

HaxGym is designed for massive-scale parallelization using JAX. By moving the entire physics engine and environment logic into JIT-compiled XLA kernels, we achieve throughput that is orders of magnitude faster than traditional CPU-based environments.

## Hardware Configuration
- **GPU**: NVIDIA RTX 4050 Laptop GPU (6GB)
- **CPU**: Intel(R) Core(TM) i5/i7 (Laptop)
- **Framework**: JAX + XLA

## Benchmark Results (3v3 Match)

| Metric | CPU (1,000 Envs) | GPU (6,144 Envs) |
| :--- | :--- | :--- |
| **AI Decisions/sec** | 30,470 Hz | **226,120 Hz** |
| **Player Frames/sec (FPS)** | 182,825 | **1,356,718** 🚀 |
| **Physics Ticks/sec (TPS)** | 914,125 | **6,783,589** 🔥 |

### Why is it so fast?
1. **Zero Python Overhead**: The entire 1000-step benchmark loop runs inside a `jax.lax.scan` kernel, meaning the CPU never has to wait for the GPU.
2. **Massive Vectorization**: Using `jax.vmap`, we process 6,144 independent football matches in a single SIMD operation.
3. **Hardware Frame Skip**: We perform 5 physics sub-steps for every AI decision directly in the compiled kernel.

---
*To reproduce these results, run:*
`python3 examples/benchmark.py`
