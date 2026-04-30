import jax
import jax.numpy as jnp
from typing import NamedTuple, Dict, Any

class State(NamedTuple):
    """
    Represents the dynamic state of the Haxball field.
    """
    pos: jnp.ndarray   # Shape (N_discs, 2)
    vel: jnp.ndarray   # Shape (N_discs, 2)
    kick_cooldown: jnp.ndarray # (N_players,) - for kick activation
    is_kicking: jnp.ndarray    # (N_players,) - boolean
    
class Params(NamedTuple):
    """
    Static parameters for discs and field objects.
    """
    radius: jnp.ndarray     # (N_discs,)
    inv_mass: jnp.ndarray   # (N_discs,)
    damping: jnp.ndarray    # (N_discs,)
    b_coef: jnp.ndarray     # (N_discs,)
    collision_mask: jnp.ndarray  # (N_discs,) - bitmask
    collision_group: jnp.ndarray # (N_discs,) - bitmask
    
    # Player specific (acceleration, etc.)
    accel: float
    kicking_accel: float
    kicking_damping: float
    kick_strength: float

class SegmentParams(NamedTuple):
    """
    Parameters for field segments (walls, goals).
    """
    pos_a: jnp.ndarray # (N_segments, 2)
    pos_b: jnp.ndarray # (N_segments, 2)
    normal: jnp.ndarray # (N_segments, 2)
    b_coef: jnp.ndarray # (N_segments,)
    collision_mask: jnp.ndarray # (N_segments,)
    collision_group: jnp.ndarray # (N_segments,)
    bias: jnp.ndarray # (N_segments,) - xc in Haxball

class PlaneParams(NamedTuple):
    """
    Parameters for field planes (infinite boundaries).
    """
    normal: jnp.ndarray # (N_planes, 2)
    dist: jnp.ndarray   # (N_planes,) distance from origin
    b_coef: jnp.ndarray # (N_planes,)
    collision_mask: jnp.ndarray # (N_planes,)
    collision_group: jnp.ndarray # (N_planes,)

def resolve_disc_collision(pos_a, vel_a, p_a: Params, idx_a, 
                           pos_b, vel_b, p_b: Params, idx_b):
    """
    JAX implementation of 'dollar_m' disc-to-disc collision.
    Returns (new_pos_a, new_vel_a, new_pos_b, new_vel_b).
    """
    delta_pos = pos_a - pos_b
    dist_sq = jnp.sum(delta_pos**2)
    sum_radius = p_a.radius[idx_a] + p_b.radius[idx_b]
    
    is_colliding = (dist_sq > 0) & (dist_sq <= sum_radius**2)
    
    # Pre-calculate values even if not colliding (JAX style)
    dist = jnp.sqrt(jnp.maximum(dist_sq, 1e-9))
    unit_direction = delta_pos / dist
    
    # Position correction
    mass_ratio = p_a.inv_mass[idx_a] / (p_a.inv_mass[idx_a] + p_b.inv_mass[idx_b])
    overlap = sum_radius - dist
    
    # Correct positions based on overlap and mass
    pos_a_new = pos_a + unit_direction * (overlap * mass_ratio)
    pos_b_new = pos_b - unit_direction * (overlap * (1.0 - mass_ratio))
    
    # Velocity correction (Impulse)
    relative_vel = vel_a - vel_b
    projection = jnp.sum(unit_direction * relative_vel)
    
    should_bounce = projection < 0
    
    # Calculate bounce impulse
    # e *= self.l * a.l + 1
    bounce_factor = p_a.b_coef[idx_a] * p_b.b_coef[idx_b] + 1.0
    impulse_mag = projection * bounce_factor
    
    vel_a_new = vel_a - unit_direction * (impulse_mag * mass_ratio)
    vel_b_new = vel_b + unit_direction * (impulse_mag * (1.0 - mass_ratio))
    
    # Only apply if colliding AND should bounce
    # (Haxball logic only applies velocity change if relative velocity is inward)
    pos_a = jnp.where(is_colliding, pos_a_new, pos_a)
    pos_b = jnp.where(is_colliding, pos_b_new, pos_b)
    
    vel_a = jnp.where(is_colliding & should_bounce, vel_a_new, vel_a)
    vel_b = jnp.where(is_colliding & should_bounce, vel_b_new, vel_b)
    
    return pos_a, vel_a, pos_b, vel_b

def resolve_segment_collision(pos_disc, vel_disc, p_disc: Params, idx_disc,
                             p_seg: SegmentParams, idx_seg):
    """
    JAX implementation of 'an' segment collision (linear case).
    """
    # For now, only implementing linear segments (tb == Infinity case)
    # R = p_seg.pos_a[idx_seg], V = p_seg.pos_b[idx_seg]
    A = p_seg.pos_a[idx_seg]
    B = p_seg.pos_b[idx_seg]
    normal = p_seg.normal[idx_seg]
    
    # Distance from line: (pos - A) dot normal
    rel_pos = pos_disc - B
    dist = jnp.sum(normal * rel_pos)
    
    # xc logic from Haxball:
    # if e == 0: if 0 > d: d = -d, b = -b, c = -c
    # else: if 0 > e: ... if d < -e: return
    
    bias = p_seg.bias[idx_seg]
    
    # Case bias == 0 (Two-sided collision with absolute distance)
    # We use a trick to stay branchless: if bias == 0 and dist < 0, flip everything
    should_flip = (bias == 0) & (dist < 0)
    dist = jnp.where(should_flip, -dist, dist)
    normal = jnp.where(should_flip, -normal, normal)
    
    # Case bias != 0 (One-sided collision)
    # If bias > 0, dist is already correct.
    # If bias < 0, we flip and check.
    should_flip_bias = (bias < 0)
    dist = jnp.where(should_flip_bias, -dist, dist)
    normal = jnp.where(should_flip_bias, -normal, normal)
    abs_bias = jnp.abs(bias)
    
    # Check if disc is within segment bounds
    seg_vec = B - A
    disc_to_A = pos_disc - A
    disc_to_B = pos_disc - B
    in_bounds = (jnp.sum(disc_to_A * seg_vec) > 0) & (jnp.sum(disc_to_B * seg_vec) < 0)
    
    # Final collision check
    is_colliding = in_bounds & (dist < p_disc.radius[idx_disc])
    
    # For one-sided with bias, we also check if it's past the bias
    is_colliding = jnp.where(bias != 0, is_colliding & (dist >= -abs_bias), is_colliding)
    
    # Position correction
    overlap = p_disc.radius[idx_disc] - dist
    pos_new = pos_disc + normal * overlap
    
    # Velocity correction
    proj_vel = jnp.sum(normal * vel_disc)
    should_bounce = proj_vel < 0
    
    bounce_factor = p_disc.b_coef[idx_disc] * p_seg.b_coef[idx_seg] + 1.0
    impulse_mag = proj_vel * bounce_factor
    vel_new = vel_disc - normal * impulse_mag
    
    pos_disc = jnp.where(is_colliding, pos_new, pos_disc)
    vel_disc = jnp.where(is_colliding & should_bounce, vel_new, vel_disc)
    
    return pos_disc, vel_disc

def resolve_plane_collision(pos_disc, vel_disc, p_disc: Params, d_idx,
                            p_plane: PlaneParams, pl_idx):
    """
    JAX implementation of 'I' plane collision.
    """
    normal = p_plane.normal[pl_idx]
    # dist_from_origin = p_plane.dist[pl_idx]
    
    # In Haxball: g = f.Oa - (g.x * h.x + g.y * h.y) + c.la
    # where Oa is the distance, sa is the normal, la is the radius.
    
    dist_to_plane = p_plane.dist[pl_idx] - jnp.sum(pos_disc * normal) + p_disc.radius[d_idx]
    
    is_colliding = dist_to_plane > 0
    
    pos_new = pos_disc + normal * dist_to_plane
    
    proj_vel = jnp.sum(vel_disc * normal)
    should_bounce = proj_vel < 0
    
    bounce_factor = p_disc.b_coef[d_idx] * p_plane.b_coef[pl_idx] + 1.0
    impulse_mag = proj_vel * bounce_factor
    vel_new = vel_disc - normal * impulse_mag
    
    pos_disc = jnp.where(is_colliding, pos_new, pos_disc)
    vel_disc = jnp.where(is_colliding & should_bounce, vel_new, vel_disc)
    
    return pos_disc, vel_disc

def physics_step(state: State, params: Params, segments: SegmentParams, planes: PlaneParams, 
                 actions: jnp.ndarray, ball_idx: int = 0):
    """
    Single physics step (1/60s).
    - apply actions (input acceleration)
    - apply kick impulse
    - update positions
    - apply damping
    - resolve collisions
    """
    num_discs = params.radius.shape[0]
    num_players = actions.shape[0]
    
    # 1. Process Input (Actions)
    # actions is a bitmask for each player (1: Up, 2: Down, 4: Left, 8: Right, 16: Kick)
    
    # Extract directions
    up = (actions & 1) != 0
    down = (actions & 2) != 0
    left = (actions & 4) != 0
    right = (actions & 8) != 0
    kick_pressed = (actions & 16) != 0
    
    dx = jnp.where(right, 1.0, 0.0) - jnp.where(left, 1.0, 0.0)
    dy = jnp.where(down, 1.0, 0.0) - jnp.where(up, 1.0, 0.0)
    
    # Normalize input
    input_mag = jnp.sqrt(dx**2 + dy**2)
    dx = jnp.where(input_mag > 0, dx / input_mag, 0.0)
    dy = jnp.where(input_mag > 0, dy / input_mag, 0.0)
    
    # Player Damping and Acceleration changes if kicking
    is_kicking = kick_pressed # Simplified for now
    
    # Player indices (usually start after ball and posts)
    # In this codebase: Ball is 0, Red is 5-7, Blue is 8-10.
    # We'll assume player_indices is passed or fixed.
    player_indices = jnp.arange(5, 5 + num_players, dtype=jnp.int32) 
    
    # Apply acceleration to player velocities
    # Note: This happens BEFORE position update and damping
    accel_val = jnp.where(is_kicking, params.kicking_accel, params.accel)
    
    # Explicitly cast to float32 to avoid type issues
    dx = dx.astype(jnp.float32)
    dy = dy.astype(jnp.float32)
    accel_val = accel_val.astype(jnp.float32)

    new_vel = state.vel
    new_vel = new_vel.at[player_indices, 0].add(dx * accel_val)
    new_vel = new_vel.at[player_indices, 1].add(dy * accel_val)
    
    # Update damping based on kicking state
    curr_damping = params.damping
    player_damping = jnp.where(is_kicking, params.kicking_damping, params.damping[player_indices])
    curr_damping = curr_damping.at[player_indices].set(player_damping.astype(jnp.float32))
    
    # 2. Kick Impulse (Ball)
    ball_pos = state.pos[ball_idx]
    
    def apply_kick(carry, i):
        p_idx = player_indices[i]
        p_pos = state.pos[p_idx]
        is_kick = is_kicking[i]
        
        dp = ball_pos - p_pos
        dist = jnp.sum(dp**2) # Should be squared for dist check? No, Haxball uses sqrt
        dist = jnp.sqrt(jnp.maximum(dist, 1e-9))
        
        # Distance to ball surfaces
        dist_surf = dist - params.radius[ball_idx] - params.radius[p_idx]
        
        # If kicking and close enough
        can_kick = (is_kick) & (dist_surf < 4.0)
        
        kick_impulse = (dp / dist) * params.kick_strength * params.inv_mass[ball_idx]
        
        ball_vel_update = jnp.where(can_kick, kick_impulse, 0.0)
        # Haxball resets bc to 0 if kick connects
        new_is_kicking = jnp.where(can_kick, False, is_kick)
        
        return carry + ball_vel_update, new_is_kicking

    kick_total_impulse, final_is_kicking = jax.lax.scan(apply_kick, jnp.zeros(2), jnp.arange(num_players))
    new_vel = new_vel.at[ball_idx].add(kick_total_impulse)
    
    # RE-APPLY acceleration and damping with the updated kicking state
    # This is because in Haxball, if you kick, your accel/damping for that tick 
    # depends on whether the kick connected.
    
    # Re-calculate accel/damping for players
    accel_val = jnp.where(final_is_kicking, params.kicking_accel, params.accel)
    
    # We need to subtract the previous accel and add the new one, 
    # or just redo the whole velocity update.
    # Actually, simpler: just calculate the correct accel from the start.
    # But wait, the kick logic needs the INITIAL is_kicking.
    
    # Let's redo the velocity update for players
    new_vel = new_vel.at[player_indices, 0].set(state.vel[player_indices, 0] + dx * accel_val)
    new_vel = new_vel.at[player_indices, 1].set(state.vel[player_indices, 1] + dy * accel_val)
    
    curr_damping = params.damping
    player_damping = jnp.where(final_is_kicking, params.kicking_damping, params.damping[player_indices])
    curr_damping = curr_damping.at[player_indices].set(player_damping.astype(jnp.float32))

    # 3. Update positions (pos += vel)
    new_pos = state.pos + new_vel
    
    # 4. Apply damping (vel *= damping)
    new_vel = new_vel * curr_damping[:, None]
    
    # 5. Resolve collisions (Disc-Disc)
    num_discs = params.radius.shape[0]
    curr_pos = new_pos
    curr_vel = new_vel
    
    # Disc-Disc collisions
    for i in range(num_discs):
        for j in range(i + 1, num_discs):
            mask_a = params.collision_mask[i]
            group_a = params.collision_group[i]
            mask_b = params.collision_mask[j]
            group_b = params.collision_group[j]
            
            can_collide = (mask_a & group_b != 0) & (mask_b & group_a != 0)
            
            res_pos_a, res_vel_a, res_pos_b, res_vel_b = resolve_disc_collision(
                curr_pos[i], curr_vel[i], params, i,
                curr_pos[j], curr_vel[j], params, j
            )
            
            curr_pos = curr_pos.at[i].set(jnp.where(can_collide, res_pos_a, curr_pos[i]))
            curr_vel = curr_vel.at[i].set(jnp.where(can_collide, res_vel_a, curr_vel[i]))
            curr_pos = curr_pos.at[j].set(jnp.where(can_collide, res_pos_b, curr_pos[j]))
            curr_vel = curr_vel.at[j].set(jnp.where(can_collide, res_vel_b, curr_vel[j]))
            
    # Disc-Plane collisions
    num_planes = planes.normal.shape[0]
    for pl_idx in range(num_planes):
        for d_idx in range(num_discs):
            # Mask check
            can_collide = (params.collision_mask[d_idx] & planes.collision_group[pl_idx] != 0) & \
                          (planes.collision_mask[pl_idx] & params.collision_group[d_idx] != 0)
            
            res_pos, res_vel = resolve_plane_collision(curr_pos[d_idx], curr_vel[d_idx], 
                                                       params, d_idx, planes, pl_idx)
            
            curr_pos = curr_pos.at[d_idx].set(jnp.where(can_collide, res_pos, curr_pos[d_idx]))
            curr_vel = curr_vel.at[d_idx].set(jnp.where(can_collide, res_vel, curr_vel[d_idx]))

    # Disc-Segment collisions
    num_segments = segments.pos_a.shape[0]
    for seg_idx in range(num_segments):
        for d_idx in range(num_discs):
            # Mask check
            can_collide = (params.collision_mask[d_idx] & segments.collision_group[seg_idx] != 0) & \
                          (segments.collision_mask[seg_idx] & params.collision_group[d_idx] != 0)
            
            res_pos, res_vel = resolve_segment_collision(curr_pos[d_idx], curr_vel[d_idx], 
                                                         params, d_idx, segments, seg_idx)
            
            curr_pos = curr_pos.at[d_idx].set(jnp.where(can_collide, res_pos, curr_pos[d_idx]))
            curr_vel = curr_vel.at[d_idx].set(jnp.where(can_collide, res_vel, curr_vel[d_idx]))
            
    return State(pos=curr_pos, vel=curr_vel, 
                 kick_cooldown=state.kick_cooldown, is_kicking=is_kicking)

def get_classic_params(ball_radius=6.0, player_radius=15.0):
    """Returns static physics parameters for the Classic map.
    
    Args:
        ball_radius: Radius of the ball (default 6.0 as requested).
        player_radius: Radius of the players (default 15.0).
    """
    # N_discs = 11 (0:ball, 1-4:posts, 5-7:red, 8-10:blue)
    radius = jnp.array([ball_radius, 8.0, 8.0, 8.0, 8.0, player_radius, player_radius, player_radius, player_radius, player_radius, player_radius])
    inv_mass = jnp.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5])
    damping = jnp.array([0.99, 0.99, 0.99, 0.99, 0.99, 0.96, 0.96, 0.96, 0.96, 0.96, 0.96])
    b_coef = jnp.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5])
    mask = jnp.array([103, -1, -1, -1, -1, 39, 39, 39, 39, 39, 39], dtype=jnp.int32)
    group = jnp.array([1, -1, -1, -1, -1, 2, 2, 2, 4, 4, 4], dtype=jnp.int32)
    
    return Params(
        radius=radius, inv_mass=inv_mass, damping=damping, b_coef=b_coef,
        collision_mask=mask, collision_group=group,
        accel=0.1, kicking_accel=0.07, kicking_damping=0.96, kick_strength=5.0
    )

def get_classic_segments(half_pitch_width=590, half_pitch_height=240, half_goal_width=80):
    """Returns static segments for the Classic map (inner rectangle + goals).
    
    Args:
        half_pitch_width: Half-width of the pitch (default 590).
        half_pitch_height: Half-height of the pitch (default 240).
        half_goal_width: Half-width of the goal opening (default 80).
    """
    # pos_a and pos_b define the 4 corner segments behind the goals
    pos_a = jnp.array([
        [half_pitch_width, half_goal_width], [half_pitch_width, -half_pitch_height], [-half_pitch_width, half_pitch_height], [-half_pitch_width, -half_goal_width]
    ])
    pos_b = jnp.array([
        [half_pitch_width, half_pitch_height], [half_pitch_width, -half_goal_width], [-half_pitch_width, half_goal_width], [-half_pitch_width, -half_pitch_height]
    ])
    
    # Normal vectors pointing inward
    normal = jnp.array([
        [-1, 0], [-1, 0], [1, 0], [1, 0]
    ], dtype=jnp.float32)
    
    return SegmentParams(
        pos_a=pos_a, pos_b=pos_b, normal=normal,
        b_coef=jnp.ones(4)*1.0, 
        collision_mask=jnp.ones(4, dtype=jnp.int32)*1, # Only blocks ball
        collision_group=jnp.ones(4, dtype=jnp.int32)*64,
        bias=jnp.zeros(4)
    )

def get_classic_planes(half_pitch_height=240, half_outer_width=640, half_outer_height=264):
    """Returns static planes for the Classic map boundaries.
    
    Args:
        half_pitch_height: Half-height of the pitch lines (ball only, default 240).
        half_outer_width: Half-width of the stadium wall (everything, default 640).
        half_outer_height: Half-height of the stadium wall (everything, default 264).
    """
    # 0, 1: Inner pitch lines (y = +/- 240). Group 64 (ball-only collision logic).
    # 2, 3: Outer vertical stadium walls (x = +/- 640). Group 32 (blocks everything).
    # 4, 5: Outer horizontal stadium walls (y = +/- 264). Group 32 (blocks everything).
    
    normal = jnp.array([
        [0, 1], [0, -1], [1, 0], [-1, 0], [0, 1], [0, -1]
    ], dtype=jnp.float32)
    
    dist = jnp.array([
        -half_pitch_height, -half_pitch_height, 
        -half_outer_width, -half_outer_width, 
        -half_outer_height, -half_outer_height
    ], dtype=jnp.float32)
    
    return PlaneParams(
        normal=normal,
        dist=dist,
        b_coef=jnp.ones(6)*1.0,
        collision_mask=jnp.ones(6, dtype=jnp.int32) * -1, # Planes block everything matched by their group
        collision_group=jnp.array([64, 64, 32, 32, 32, 32], dtype=jnp.int32)
    )

