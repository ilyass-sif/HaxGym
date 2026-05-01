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
    """
    delta_pos = pos_a - pos_b
    dist_sq = jnp.sum(delta_pos**2)
    sum_radius = p_a.radius[idx_a] + p_b.radius[idx_b]
    
    is_colliding = (dist_sq > 0) & (dist_sq <= sum_radius**2)
    
    dist = jnp.sqrt(jnp.maximum(dist_sq, 1e-9))
    unit_direction = delta_pos / dist
    
    # Position correction
    mass_ratio = p_a.inv_mass[idx_a] / (p_a.inv_mass[idx_a] + p_b.inv_mass[idx_b])
    overlap = sum_radius - dist
    
    pos_a_new = pos_a + unit_direction * (overlap * mass_ratio)
    pos_b_new = pos_b - unit_direction * (overlap * (1.0 - mass_ratio))
    
    # Velocity correction (Impulse)
    relative_vel = vel_a - vel_b
    projection = jnp.sum(unit_direction * relative_vel)
    should_bounce = projection < 0
    
    bounce_factor = p_a.b_coef[idx_a] * p_b.b_coef[idx_b] + 1.0
    impulse_mag = projection * bounce_factor
    
    vel_a_new = vel_a - unit_direction * (impulse_mag * mass_ratio)
    vel_b_new = vel_b + unit_direction * (impulse_mag * (1.0 - mass_ratio))
    
    pos_a = jnp.where(is_colliding, pos_a_new, pos_a)
    pos_b = jnp.where(is_colliding, pos_b_new, pos_b)
    vel_a = jnp.where(is_colliding & should_bounce, vel_a_new, vel_a)
    vel_b = jnp.where(is_colliding & should_bounce, vel_b_new, vel_b)
    
    return pos_a, vel_a, pos_b, vel_b

def resolve_segment_collision(pos_disc, vel_disc, p_disc: Params, idx_disc,
                             p_seg: SegmentParams, idx_seg):
    """
    JAX implementation of segment collision.
    """
    A = p_seg.pos_a[idx_seg]
    B = p_seg.pos_b[idx_seg]
    normal = p_seg.normal[idx_seg]
    
    rel_pos = pos_disc - B
    dist = jnp.sum(normal * rel_pos)
    
    bias = p_seg.bias[idx_seg]
    should_flip = (bias == 0) & (dist < 0)
    dist = jnp.where(should_flip, -dist, dist)
    normal = jnp.where(should_flip, -normal, normal)
    
    should_flip_bias = (bias < 0)
    dist = jnp.where(should_flip_bias, -dist, dist)
    normal = jnp.where(should_flip_bias, -normal, normal)
    
    seg_vec = B - A
    disc_to_A = pos_disc - A
    disc_to_B = pos_disc - B
    in_bounds = (jnp.sum(disc_to_A * seg_vec) > 0) & (jnp.sum(disc_to_B * seg_vec) < 0)
    
    is_colliding = in_bounds & (dist < p_disc.radius[idx_disc])
    is_colliding = jnp.where(bias != 0, is_colliding & (dist >= -jnp.abs(bias)), is_colliding)
    
    overlap = p_disc.radius[idx_disc] - dist
    pos_new = pos_disc + normal * overlap
    
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
    JAX implementation of infinite plane collision.
    """
    normal = p_plane.normal[pl_idx]
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
    num_discs = params.radius.shape[0]
    num_players = actions.shape[0]
    player_indices = jnp.arange(5, 5 + num_players, dtype=jnp.int32) 
    
    # 1. Process Input
    up = (actions & 1) != 0
    down = (actions & 2) != 0
    left = (actions & 4) != 0
    right = (actions & 8) != 0
    kick_pressed = (actions & 16) != 0
    
    dx = (jnp.where(right, 1.0, 0.0) - jnp.where(left, 1.0, 0.0)).astype(jnp.float32)
    dy = (jnp.where(down, 1.0, 0.0) - jnp.where(up, 1.0, 0.0)).astype(jnp.float32)
    input_mag = jnp.sqrt(dx**2 + dy**2)
    dx = jnp.where(input_mag > 0, dx / input_mag, 0.0)
    dy = jnp.where(input_mag > 0, dy / input_mag, 0.0)
    
    # 2. Kick Impulse
    ball_pos = state.pos[ball_idx]
    def apply_kick(carry, i):
        p_idx = player_indices[i]
        p_pos = state.pos[p_idx]
        is_kick = (actions[i] & 16) != 0
        dp = ball_pos - p_pos
        dist = jnp.sqrt(jnp.maximum(jnp.sum(dp**2), 1e-9))
        dist_surf = dist - params.radius[ball_idx] - params.radius[p_idx]
        can_kick = is_kick & (dist_surf < 4.0)
        kick_impulse = (dp / dist) * params.kick_strength * params.inv_mass[ball_idx]
        ball_vel_update = jnp.where(can_kick, kick_impulse, 0.0)
        return carry + ball_vel_update, can_kick

    kick_total_impulse, connected_kicks = jax.lax.scan(apply_kick, jnp.zeros(2), jnp.arange(num_players))
    is_kicking = connected_kicks # True if a kick is active
    
    new_vel = state.vel.at[ball_idx].add(kick_total_impulse)
    
    # Update player velocities with accel
    accel_val = jnp.where(connected_kicks, params.kicking_accel, params.accel)
    new_vel = new_vel.at[player_indices, 0].add(dx * accel_val)
    new_vel = new_vel.at[player_indices, 1].add(dy * accel_val)
    
    # 3. Sub-stepping Movement and Collision (2 steps for 120Hz stability)
    num_planes = planes.normal.shape[0]
    num_segments = segments.pos_a.shape[0]
    curr_damping = params.damping
    player_damping = jnp.where(connected_kicks, params.kicking_damping, params.damping[player_indices])
    curr_damping = curr_damping.at[player_indices].set(player_damping.astype(jnp.float32))

    def sub_step(carry, _):
        cp, cv = carry
        sub_dt = 0.5
        # Move
        np = cp + cv * sub_dt
        # Damp
        nv = cv * jnp.sqrt(curr_damping[:, None])
        
        # Resolve Disc-Disc
        for i in range(num_discs):
            for j in range(i + 1, num_discs):
                ca = (params.collision_mask[i] & params.collision_group[j] != 0) & (params.collision_mask[j] & params.collision_group[i] != 0)
                rpi, rvi, rpj, rvj = resolve_disc_collision(np[i], nv[i], params, i, np[j], nv[j], params, j)
                np = np.at[i].set(jnp.where(ca, rpi, np[i]))
                nv = nv.at[i].set(jnp.where(ca, rvi, nv[i]))
                np = np.at[j].set(jnp.where(ca, rpj, np[j]))
                nv = nv.at[j].set(jnp.where(ca, rvj, nv[j]))
        
        # Resolve Planes (Hard Walls)
        for pl_idx in range(num_planes):
            for d_idx in range(num_discs):
                ca = (params.collision_mask[d_idx] & planes.collision_group[pl_idx] != 0) & (planes.collision_mask[pl_idx] & params.collision_group[d_idx] != 0)
                rp, rv = resolve_plane_collision(np[d_idx], nv[d_idx], params, d_idx, planes, pl_idx)
                np = np.at[d_idx].set(jnp.where(ca, rp, np[d_idx]))
                nv = nv.at[d_idx].set(jnp.where(ca, rv, nv[d_idx]))
                
        # Resolve Segments (Goal Corners)
        for seg_idx in range(num_segments):
            for d_idx in range(num_discs):
                ca = (params.collision_mask[d_idx] & segments.collision_group[seg_idx] != 0) & (segments.collision_mask[seg_idx] & params.collision_group[d_idx] != 0)
                rp, rv = resolve_segment_collision(np[d_idx], nv[d_idx], params, d_idx, segments, seg_idx)
                np = np.at[d_idx].set(jnp.where(ca, rp, np[d_idx]))
                nv = nv.at[d_idx].set(jnp.where(ca, rv, nv[d_idx]))
        return (np, nv), None

    (final_p, final_v), _ = jax.lax.scan(sub_step, (state.pos, new_vel), None, length=2)
    
    return State(pos=final_p, vel=final_v, kick_cooldown=state.kick_cooldown, is_kicking=is_kicking)

def get_classic_params(ball_radius=6.0, player_radius=15.0):
    radius = jnp.array([ball_radius, 8.0, 8.0, 8.0, 8.0, player_radius, player_radius, player_radius, player_radius, player_radius, player_radius])
    inv_mass = jnp.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5])
    damping = jnp.array([0.99, 0.99, 0.99, 0.99, 0.99, 0.96, 0.96, 0.96, 0.96, 0.96, 0.96])
    b_coef = jnp.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5])
    mask = jnp.array([103, -1, -1, -1, -1, 39, 39, 39, 39, 39, 39], dtype=jnp.int32)
    group = jnp.array([1, -1, -1, -1, -1, 2, 2, 2, 4, 4, 4], dtype=jnp.int32)
    return Params(radius=radius, inv_mass=inv_mass, damping=damping, b_coef=b_coef, collision_mask=mask, collision_group=group, accel=0.1, kicking_accel=0.07, kicking_damping=0.96, kick_strength=5.0)

def get_classic_segments(half_pitch_width=590, half_pitch_height=240, half_goal_width=80):
    pos_a = jnp.array([[half_pitch_width, half_goal_width], [half_pitch_width, -half_pitch_height], [-half_pitch_width, half_pitch_height], [-half_pitch_width, -half_goal_width]])
    pos_b = jnp.array([[half_pitch_width, half_pitch_height], [half_pitch_width, -half_goal_width], [-half_pitch_width, half_goal_width], [-half_pitch_width, -half_pitch_height]])
    normal = jnp.array([[-1, 0], [-1, 0], [1, 0], [1, 0]], dtype=jnp.float32)
    return SegmentParams(pos_a=pos_a, pos_b=pos_b, normal=normal, b_coef=jnp.ones(4)*1.0, collision_mask=jnp.ones(4, dtype=jnp.int32)*1, collision_group=jnp.ones(4, dtype=jnp.int32)*64, bias=jnp.zeros(4))

def get_classic_planes(half_pitch_height=240, half_outer_width=640, half_outer_height=264):
    normal = jnp.array([[0, 1], [0, -1], [1, 0], [-1, 0], [0, 1], [0, -1]], dtype=jnp.float32)
    dist = jnp.array([-half_pitch_height, -half_pitch_height, -half_outer_width, -half_outer_width, -half_outer_height, -half_outer_height], dtype=jnp.float32)
    return PlaneParams(normal=normal, dist=dist, b_coef=jnp.ones(6)*1.0, collision_mask=jnp.ones(6, dtype=jnp.int32) * -1, collision_group=jnp.array([64, 64, 32, 32, 32, 32], dtype=jnp.int32))
