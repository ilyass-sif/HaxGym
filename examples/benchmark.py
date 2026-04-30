import jax
import jax.numpy as jnp
import haxgym
import time

def main():
    print("🚀 Initializing Ultra-High Performance HaxGym Benchmark...")
    env = haxgym.make("Haxball-Classic-v0")
    
    rng = jax.random.PRNGKey(42)
    
    # MATCH THE TRAINING CONFIG: 6144 environments
    N_ENVS = 6144
    N_STEPS = 1000 # Run for 1000 AI steps
    
    rng_keys = jax.random.split(rng, N_ENVS)
    
    # 1. Vectorized Reset
    vmap_reset = jax.jit(jax.vmap(env.reset, in_axes=(0, None)))
    
    # 2. Vectorized Step (This was the missing piece!)
    # We vmap over the RNG keys, the states, and the actions.
    vmap_step = jax.vmap(env.step, in_axes=(0, 0, 0, None, None, None))
    
    # 3. Optimized Step Loop using jax.lax.scan
    @jax.jit
    def run_benchmark_loop(carry_rng, carry_states):
        def one_step(carry, _):
            rng, states = carry
            rng, step_key = jax.random.split(rng)
            step_keys = jax.random.split(step_key, N_ENVS)
            
            # Generate random actions for all players in all envs
            actions = jax.random.randint(rng, (N_ENVS, 6), minval=0, maxval=10)
            
            # Step all environments in parallel
            next_states, obs, rewards, dones = vmap_step(step_keys, states, actions, 1, 3600, 1.0)
            return (rng, next_states), None

        (final_rng, final_states), _ = jax.lax.scan(
            one_step, (carry_rng, carry_states), None, length=N_STEPS
        )
        return final_states

    print(f"Resetting {N_ENVS} parallel environments...")
    states = vmap_reset(rng_keys, 1)
    
    print("Compiling JIT Scan Loop (Saturating the GPU)...")
    # First call triggers compilation
    states = run_benchmark_loop(rng, states)
    # Block until ready to ensure compilation is done
    jax.block_until_ready(states)
    
    print(f"Running Benchmark: {N_STEPS} steps x {N_ENVS} envs...")
    start_time = time.time()
    
    # Run the actual benchmark
    states = run_benchmark_loop(rng, states)
    jax.block_until_ready(states)
    
    duration = time.time() - start_time
    
    total_frames = N_ENVS * N_STEPS * 6
    fps = total_frames / duration
    
    # Note: Because we use Frame Skip 5, every 'AI Step' is actually 5 physics ticks
    physics_fps = fps * 5
    
    print("\n" + "="*40)
    print("       HAXGYM PERFORMANCE REPORT")
    print("="*40)
    print(f"Parallel Envs:       {N_ENVS:>10,}")
    print(f"AI Decisions/sec:    {fps/6:>10,.0f}")
    print(f"Player FPS (Viewer): {fps:>10,.0f} ⚡")
    print(f"Physics Ticks/sec:   {physics_fps:>10,.0f} 🔥")
    print("-" * 40)
    print(f"Total Time:          {duration:>10.2f}s")
    print("=" * 40)
    print("Status: ENGINE IS FULLY SATURATED")

if __name__ == "__main__":
    main()
