import jax
import jax.numpy as jnp
from typing import NamedTuple, Tuple
from haxgym.physics.core import State, Params, SegmentParams, PlaneParams, physics_step

class EnvState(NamedTuple):
    """
    Combined state of the physics and the Gym environment tracking.
    """
    sim_state: State
    # Environment tracking
    last_touch_idx: jnp.ndarray    # (num_envs,) -1 if none
    last_touch_tick: jnp.ndarray   # (num_envs,)
    last_red_touch_pos: jnp.ndarray # (num_envs, 2)
    last_blue_touch_pos: jnp.ndarray # (num_envs, 2)
    red_pass_count: jnp.ndarray      # (num_envs,)
    blue_pass_count: jnp.ndarray     # (num_envs,)
    red_interception_count: jnp.ndarray  # (num_envs,)
    blue_interception_count: jnp.ndarray # (num_envs,)
    last_ball_x: jnp.ndarray       # (num_envs,) for forward reward
    steps: jnp.ndarray             # (num_envs,)
    red_scored: jnp.ndarray        # (num_envs,) boolean
    blue_scored: jnp.ndarray       # (num_envs,) boolean
    done: jnp.ndarray              # (num_envs,)
    
def reset(rng, params: Params, stage: int = 1):
    """
    JAX implementation of environment reset.
    """
    # 1. Basic initialization
    num_discs = params.radius.shape[0]
    pos = jnp.zeros((num_discs, 2))
    vel = jnp.zeros((num_discs, 2))
    
    # 2. Stage-specific logic
    k1, k2, k3, k4 = jax.random.split(rng, 4)
    
    # Defaults for all stages
    # Goal posts (1-4) at ±590, ±80
    pos = pos.at[1].set(jnp.array([590.0, -80.0]))
    pos = pos.at[2].set(jnp.array([590.0, 80.0]))
    pos = pos.at[3].set(jnp.array([-590.0, -80.0]))
    pos = pos.at[4].set(jnp.array([-590.0, 80.0]))
    
    # Player starting positions (Stage 1 & 2 logic)
    cx = jax.random.uniform(k1, (), minval=-100.0, maxval=100.0)
    cy = jax.random.uniform(k2, (), minval=-100.0, maxval=100.0)
    p0 = jnp.array([cx, cy])
    
    dist = jax.random.uniform(k3, (), minval=150.0, maxval=400.0)
    angle = jax.random.uniform(k4, (), minval=0.0, maxval=2*jnp.pi)
    
    p1_raw = p0 + jnp.array([dist * jnp.cos(angle), dist * jnp.sin(angle)])
    p1 = jnp.clip(p1_raw, jnp.array([-550.0, -220.0]), jnp.array([550.0, 220.0]))
    
    # Ball near p0 pointing to p1
    to_p1 = p1 - p0
    to_p1_dist = jnp.maximum(jnp.linalg.norm(to_p1), 1.0)
    unit = to_p1 / to_p1_dist
    ball_pos = p0 + unit * 40.0
    
    # Assign positions
    pos = pos.at[0].set(ball_pos) # Ball
    pos = pos.at[5].set(p0)       # Red 0
    pos = pos.at[6].set(p1)       # Red 1
    # Red 2, Blue 0-2 far away or spread
    pos = pos.at[7].set(jnp.array([-550.0, 220.0]))
    pos = pos.at[8].set(jnp.array([100.0, 100.0]))
    pos = pos.at[9].set(jnp.array([100.0, -100.0]))
    pos = pos.at[10].set(jnp.array([200.0, 0.0]))
    
    sim_state = State(
        pos=pos,
        vel=vel,
        kick_cooldown=jnp.zeros(6),
        is_kicking=jnp.zeros(6, dtype=bool)
    )
    
    return EnvState(
        sim_state=sim_state,
        last_touch_idx=jnp.array(-1),
        last_touch_tick=jnp.array(0),
        last_red_touch_pos=p0,
        last_blue_touch_pos=jnp.array([550.0, 0.0]),
        red_pass_count=jnp.array(0),
        blue_pass_count=jnp.array(0),
        red_interception_count=jnp.array(0),
        blue_interception_count=jnp.array(0),
        last_ball_x=ball_pos[0],
        steps=jnp.array(0),
        red_scored=jnp.array(False),
        blue_scored=jnp.array(False),
        done=jnp.array(False)
    )
    
def get_observation(sim_state: State, params: Params, player_idx: int, red_team: bool = True):
    """
    JAX implementation of Haxball.get_observation (32-dim vector).
    """
    f = jnp.where(red_team, -1.0, 1.0)
    
    # 1. Self data
    me_pos = sim_state.pos[player_idx] * f
    me_vel = sim_state.vel[player_idx] * f
    
    # 2. Ball data
    ball_pos = sim_state.pos[0] * f
    ball_vel = sim_state.vel[0] * f
    
    # 3. Teammates / Opponents indices
    # Red: 5, 6, 7 | Blue: 8, 9, 10
    red_indices = jnp.array([5, 6, 7])
    blue_indices = jnp.array([8, 9, 10])
    
    my_indices = jnp.where(red_team, red_indices, blue_indices)
    opp_indices = jnp.where(red_team, blue_indices, red_indices)
    
    # Filter teammates (not me)
    # Since we need a fixed shape for JAX, we'll just mask or re-order
    # For HaxballCurriculum, there are always 2 teammates.
    is_me = (my_indices == player_idx)
    teammate_indices = jnp.where(is_me, -1, my_indices) # Placeholder
    # Sort to bring teammates to the front (indices 0, 1)
    # (Simple approach: if me is 5, teammates are 6, 7)
    # We'll just hardcode the logic for 3v3 for now as it's fixed
    
    def get_others(all_indices, exclude_idx):
        # Returns exactly 2 indices
        mask = (all_indices != exclude_idx)
        return all_indices[mask][:2] # JAX: this is tricky with dynamic shapes, but here it's static
    
    # Using a static way to get teammates for 3v3
    # If indices are [5,6,7], and player is 5, others are [6,7]. 
    # If player is 6, others are [5,7]. If player is 7, others are [5,6].
    t_idx1 = jnp.where(player_idx == my_indices[0], my_indices[1], my_indices[0])
    t_idx2 = jnp.where((player_idx == my_indices[0]) | (player_idx == my_indices[1]), my_indices[2], my_indices[1])
    tm_indices = jnp.array([t_idx1, t_idx2])
    
    # 4. Relative positions
    nx, ny = 700.0, 300.0 # Updated to accommodate background bounds (630x240+)
    nv = 10.0
    
    # Me: pos_x, pos_y, vel_x, vel_y (absolute-ish, scaled)
    obs = [me_pos[0]/nx, me_pos[1]/ny, me_vel[0]/nv, me_vel[1]/nv]
    # Ball relative
    obs.extend([(ball_pos[0] - me_pos[0])/nx, (ball_pos[1] - me_pos[1])/ny, ball_vel[0]/nv, ball_vel[1]/nv])
    
    # Teammates relative
    for i in range(2):
        idx = tm_indices[i]
        t_pos = sim_state.pos[idx] * f
        t_vel = sim_state.vel[idx] * f
        obs.extend([(t_pos[0] - me_pos[0])/nx, (t_pos[1] - me_pos[1])/ny, t_vel[0]/nv, t_vel[1]/nv])
        
    # Opponents relative
    for i in range(3):
        idx = opp_indices[i]
        o_pos = sim_state.pos[idx] * f
        o_vel = sim_state.vel[idx] * f
        obs.extend([(o_pos[0] - me_pos[0])/nx, (o_pos[1] - me_pos[1])/ny, o_vel[0]/nv, o_vel[1]/nv])
        
    # 5. Raycast flags (is pass to teammate blocked?)
    safety_radius = 25.0
    
    def check_blocked(target_idx):
        target_pos = sim_state.pos[target_idx] * f
        ab = target_pos - me_pos
        ab_len_sq = jnp.sum(ab**2)
        
        # Blockers: all opponents + the OTHER teammate
        other_tm_idx = jnp.where(tm_indices[0] == target_idx, tm_indices[1], tm_indices[0])
        blocker_indices = jnp.concatenate([opp_indices, jnp.array([other_tm_idx])])
        
        def is_blocked_by(b_idx):
            b_pos = sim_state.pos[b_idx] * f
            ac = b_pos - me_pos
            # Projection t
            t = jnp.sum(ac * ab) / jnp.maximum(ab_len_sq, 1e-9)
            # Closest point P on AB
            p = me_pos + t * ab
            dist_sq = jnp.sum((b_pos - p)**2)
            return (t >= 0) & (t <= 0.95) & (dist_sq < safety_radius**2)
            
        any_blocked = jnp.any(jax.vmap(is_blocked_by)(blocker_indices))
        return jnp.where(ab_len_sq > 0, any_blocked.astype(jnp.float32), 0.0)

    raycast1 = check_blocked(tm_indices[0])
    raycast2 = check_blocked(tm_indices[1])
    obs.extend([raycast1, raycast2])
    
    # 6. Distance to ball
    dist_to_ball = jnp.sqrt(jnp.sum((ball_pos - me_pos)**2))
    obs.append(dist_to_ball / 1000.0)
    
    # 7. Campo bloccato (placeholder for now as it depends on gameplay state)
    obs.append(0.0)
    
    return jnp.array(obs)

def get_reward(prev_state: EnvState, next_state: EnvState, player_idx: int, red_team: bool = True, curriculum_factor: float = 1.0):
    """
    JAX implementation of Tiki-Taka rewards.
    """
    f = jnp.where(red_team, -1.0, 1.0)
    
    # 1. Proximity Compass (Decays to 0 over training)
    me_pos = next_state.sim_state.pos[player_idx] * f
    ball_pos = next_state.sim_state.pos[0] * f
    dist = jnp.sqrt(jnp.sum((ball_pos - me_pos)**2))
    reward = -0.005 * (dist / 100.0) * curriculum_factor
    
    # 2. Ball Forward Reward
    ball_x = ball_pos[0]
    ball_delta_x = ball_x - (prev_state.last_ball_x * f)
    reward -= 0.05 * (ball_delta_x / 10.0)
    
    # 3. Ball-stall penalty
    ball_vel = next_state.sim_state.vel[0]
    ball_speed_sq = jnp.sum(ball_vel**2)
    reward -= jnp.where(ball_speed_sq < 0.25, 0.02, 0.0)
    
    # 4. Teammate Proximity Penalty (Spacing)
    me_pos_real = next_state.sim_state.pos[player_idx]
    team_start = jnp.where(red_team, 5, 8)
    t1 = next_state.sim_state.pos[team_start]
    t2 = next_state.sim_state.pos[team_start + 1]
    t3 = next_state.sim_state.pos[team_start + 2]
    
    d1 = jnp.sqrt(jnp.sum((t1 - me_pos_real)**2))
    d2 = jnp.sqrt(jnp.sum((t2 - me_pos_real)**2))
    d3 = jnp.sqrt(jnp.sum((t3 - me_pos_real)**2))
    
    # Penalty if distance is between 1.0 (to ignore self) and 80.0 units
    reward += jnp.where((d1 > 1.0) & (d1 < 80.0), -0.01, 0.0)
    reward += jnp.where((d2 > 1.0) & (d2 < 80.0), -0.01, 0.0)
    reward += jnp.where((d3 > 1.0) & (d3 < 80.0), -0.01, 0.0)
    
    # 5. Pass Rewards
    my_pass_count_next = jnp.where(red_team, next_state.red_pass_count, next_state.blue_pass_count)
    my_pass_count_prev = jnp.where(red_team, prev_state.red_pass_count, prev_state.blue_pass_count)
    pass_bonus = (my_pass_count_next - my_pass_count_prev) * 1.0
    reward += pass_bonus

    # 5.1 Interception Rewards
    my_int_count_next = jnp.where(red_team, next_state.red_interception_count, next_state.blue_interception_count)
    my_int_count_prev = jnp.where(red_team, prev_state.red_interception_count, prev_state.blue_interception_count)
    int_bonus = (my_int_count_next - my_int_count_prev) * 0.1
    reward += int_bonus
    
    # 6. End of Game Pass Reward (Win/Loss)
    red_won_passes = next_state.red_pass_count > next_state.blue_pass_count
    blue_won_passes = next_state.blue_pass_count > next_state.red_pass_count
    
    win_reward = jnp.where(red_team,
                           jnp.where(red_won_passes, 10.0, jnp.where(blue_won_passes, -10.0, 0.0)),
                           jnp.where(blue_won_passes, 10.0, jnp.where(red_won_passes, -10.0, 0.0)))
    
    # Only apply win reward on the final step when done is triggered
    reward += jnp.where(next_state.done, win_reward, 0.0)
    
    return reward

def env_step(rng, env_state: EnvState, params: Params, segments: SegmentParams, planes: PlaneParams, 
             actions: jnp.ndarray, stage: int = 1, max_steps: int = 1000, curriculum_factor: float = 1.0):
    """
    Integrated environment step: auto-reset + physics + rewards + obs.
    actions shape: (6,) - discrete actions (0-9) for [R0, R1, R2, B0, B1, B2]
    """
    # 0. Auto-Reset if the previous state was done
    # This ensures the simulation doesn't "get stuck" in a terminal state
    env_state = jax.lax.cond(env_state.done,
                             lambda k: reset(k, params, stage),
                             lambda _: env_state,
                             rng)
    
    # 0.1 Action Mapping & Team Mirroring (Reference: haxball_gym.py line 97)
    # Mapping for discrete actions 0-9 to Haxball bitmask (1:Up, 2:Down, 4:Left, 8:Right, 16:Kick)
    action_masks = jnp.array([
        0,              # 0: None
        1,              # 1: Up
        1 | 8,          # 2: Up-Right
        8,              # 3: Right
        2 | 8,          # 4: Down-Right
        2,              # 5: Down
        2 | 4,          # 6: Down-Left
        4,              # 7: Left
        1 | 4,          # 8: Up-Left
        16              # 9: Kick
    ], dtype=jnp.int32)
    
    raw_masks = action_masks[actions]
    
    # Red Team Mirroring (Indices 0, 1, 2 in the actions array)
    # Up (1) <-> Down (2), Left (4) <-> Right (8)
    def mirror_red(m):
        new_m = m & 16 # Keep Kick
        new_m |= jnp.where((m & 1) != 0, 2, 0) # Up -> Down
        new_m |= jnp.where((m & 2) != 0, 1, 0) # Down -> Up
        new_m |= jnp.where((m & 4) != 0, 8, 0) # Left -> Right
        new_m |= jnp.where((m & 8) != 0, 4, 0) # Right -> Left
        return new_m
    
    red_masks = jax.vmap(mirror_red)(raw_masks[:3])
    blue_masks = raw_masks[3:]
    final_masks = jnp.concatenate([red_masks, blue_masks])
    
    # 1. Physics Step (Frame Skip)
    FRAME_SKIP = 5
    def multi_step_physics(carry, _):
        return physics_step(carry, params, segments, planes, final_masks), None
        
    next_sim_state, _ = jax.lax.scan(multi_step_physics, env_state.sim_state, None, length=FRAME_SKIP)
    
    # 2. Touch Detection (for Pass Tracking)
    ball_pos = next_sim_state.pos[0]
    ball_radius = params.radius[0]
    
    # Check all players (5 to 10)
    player_indices = jnp.arange(5, 11)
    player_pos = next_sim_state.pos[player_indices]
    player_radius = params.radius[player_indices]
    
    dist_to_players = jnp.sqrt(jnp.sum((player_pos - ball_pos)**2, axis=1))
    touches = dist_to_players < (ball_radius + 15.0 + 2.0)
    
    # Find the CLOSEST touching player index (more fair than index priority)
    valid_distances = jnp.where(touches, dist_to_players, 1e9)
    touch_occured = jnp.any(touches)
    first_touch_rel_idx = jnp.argmin(valid_distances)
    current_touch_idx = jnp.where(touch_occured, player_indices[first_touch_rel_idx], -1)
    
    # Pass Detection (Red team: 5-7)
    # A pass occurs if:
    # - current_touch is a red player
    # - last_touch was a different red player
    # - time between touches > 5 ticks
    # Red Pass Detection
    is_red_touch = (current_touch_idx >= 5) & (current_touch_idx <= 7)
    is_prev_red_touch = (env_state.last_touch_idx >= 5) & (env_state.last_touch_idx <= 7)
    red_is_new = (current_touch_idx != env_state.last_touch_idx)
    
    red_p1 = env_state.last_red_touch_pos
    red_p2 = next_sim_state.pos[current_touch_idx]
    red_dist = jnp.sqrt(jnp.sum((red_p1 - red_p2)**2))
    red_pass_ok = is_red_touch & is_prev_red_touch & red_is_new & (red_dist > 120.0) & (env_state.steps - env_state.last_touch_tick > 5)

    # Blue Pass Detection
    is_blue_touch = (current_touch_idx >= 8) & (current_touch_idx <= 10)
    is_prev_blue_touch = (env_state.last_touch_idx >= 8) & (env_state.last_touch_idx <= 10)
    blue_is_new = (current_touch_idx != env_state.last_touch_idx)
    
    blue_p1 = env_state.last_blue_touch_pos
    blue_p2 = next_sim_state.pos[current_touch_idx]
    blue_dist = jnp.sqrt(jnp.sum((blue_p1 - blue_p2)**2))
    blue_pass_ok = is_blue_touch & is_prev_blue_touch & blue_is_new & (blue_dist > 120.0) & (env_state.steps - env_state.last_touch_tick > 5)

    # Interception Detection
    prev_was_red = (env_state.last_touch_idx >= 5) & (env_state.last_touch_idx <= 7)
    prev_was_blue = (env_state.last_touch_idx >= 8) & (env_state.last_touch_idx <= 10)
    red_interception = is_red_touch & prev_was_blue
    blue_interception = is_blue_touch & prev_was_red

    # 3. Goal Detection
    # Red scores if ball fully crosses x=590 goal line
    # Blue scores if ball fully crosses x=-590 goal line
    goal_x = 590.0 + ball_radius
    goal_half_width = 80.0
    red_scored = (ball_pos[0] > goal_x) & (jnp.abs(ball_pos[1]) < goal_half_width)
    blue_scored = (ball_pos[0] < -goal_x) & (jnp.abs(ball_pos[1]) < goal_half_width)

    # 4. Update Environment Tracking
    new_red_pass_count = env_state.red_pass_count + jnp.where(red_pass_ok, 1, 0)
    new_blue_pass_count = env_state.blue_pass_count + jnp.where(blue_pass_ok, 1, 0)
    new_red_int_count = env_state.red_interception_count + jnp.where(red_interception, 1, 0)
    new_blue_int_count = env_state.blue_interception_count + jnp.where(blue_interception, 1, 0)
    
    new_last_touch_idx = jnp.where(touch_occured, current_touch_idx, env_state.last_touch_idx)
    new_last_touch_tick = jnp.where(touch_occured, env_state.steps, env_state.last_touch_tick)
    new_last_red_pos = jnp.where(is_red_touch, next_sim_state.pos[current_touch_idx], env_state.last_red_touch_pos)
    new_last_blue_pos = jnp.where(is_blue_touch, next_sim_state.pos[current_touch_idx], env_state.last_blue_touch_pos)
    
    new_steps = env_state.steps + 1
    new_done = (new_steps >= max_steps) | env_state.done
    
    next_env_state = EnvState(
        sim_state=next_sim_state,
        last_touch_idx=new_last_touch_idx,
        last_touch_tick=new_last_touch_tick,
        last_red_touch_pos=new_last_red_pos,
        last_blue_touch_pos=new_last_blue_pos,
        red_pass_count=new_red_pass_count,
        blue_pass_count=new_blue_pass_count,
        red_interception_count=new_red_int_count,
        blue_interception_count=new_blue_int_count,
        last_ball_x=next_sim_state.pos[0, 0],
        steps=new_steps,
        red_scored=red_scored,
        blue_scored=blue_scored,
        done=new_done
    )
    
    # 4. Observations for all 6 players
    player_indices = jnp.arange(5, 11)
    is_red_team = jnp.array([True, True, True, False, False, False])
    
    obs_all = jax.vmap(get_observation, in_axes=(None, None, 0, 0))(next_sim_state, params, player_indices, is_red_team)
    
    # 5. Rewards for all 6 players
    rewards_all = jax.vmap(get_reward, in_axes=(None, None, 0, 0, None))(env_state, next_env_state, player_indices, is_red_team, curriculum_factor)
    
    return next_env_state, obs_all, rewards_all, new_done
