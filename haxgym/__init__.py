from typing import Tuple, Any, Dict, Callable, Optional
from .physics.core import get_classic_params, get_classic_segments, get_classic_planes
from .envs.classic import env_step as classic_env_step, reset as classic_reset, EnvState
from .envs.pass_game import env_step as pass_env_step, reset as pass_reset

__version__ = "0.2.0"

# --- Global Registry ---
# This stores the "recipe" for every environment
_REGISTRY: Dict[str, Dict[str, Any]] = {}

def register(
    env_id: str,
    step_fn: Callable,
    reset_fn: Callable,
    description: Optional[str] = None
):
    """
    Registers a new environment into HaxGym.
    
    Args:
        env_id: Unique string ID (e.g. 'Haxball-1v1-v0')
        step_fn: The environment's step function
        reset_fn: The environment's reset function
        description: Optional text describing the mode
    """
    if env_id in _REGISTRY:
        print(f"Warning: Overwriting environment '{env_id}'")
        
    _REGISTRY[env_id] = {
        "step": step_fn,
        "reset": reset_fn,
        "description": description
    }

# --- Standard HaxEnvironment Class ---
class HaxEnvironment:
    """
    The main interface for HaxGym.
    """
    def __init__(self, env_id: str, config: Dict[str, Any]):
        self.env_id = env_id
        self._step_fn = config["step"]
        self._reset_fn = config["reset"]
        self.description = config.get("description", "")
        
        # Load the physical parameters of the classic stadium by default
        # (Users can override these if they want custom field dimensions)
        self.params = get_classic_params()
        self.segments = get_classic_segments()
        self.planes = get_classic_planes()

    def reset(self, rng, stage: int = 1) -> EnvState:
        return self._reset_fn(rng, self.params, stage)
        
    def step(self, rng, state: EnvState, actions, stage: int = 1, max_steps: int = 3600, curriculum_factor: float = 1.0):
        return self._step_fn(rng, state, self.params, self.segments, self.planes, actions, stage, max_steps, curriculum_factor)

def make(env_id: str) -> HaxEnvironment:
    """
    Creates and returns a HaxGym environment instance.
    """
    if env_id not in _REGISTRY:
        available = list(_REGISTRY.keys())
        raise ValueError(f"Environment '{env_id}' not found. Registered: {available}")
    
    return HaxEnvironment(env_id, _REGISTRY[env_id])

# --- Auto-Register Standard Modes ---
register(
    env_id="Haxball-Classic-v0",
    step_fn=classic_env_step,
    reset_fn=classic_reset,
    description="Standard 3v3 Haxball with goal-based rewards."
)

register(
    env_id="Haxball-Pass-v0",
    step_fn=pass_env_step,
    reset_fn=pass_reset,
    description="Training mode: Most successful passes wins the match."
)
