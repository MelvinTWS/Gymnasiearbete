"""
Reinforcement Learning Environment Module

This module implements a Gym-compatible environment for training RL agents
to defend against UAV swarm attacks. Compatible with Stable-Baselines3.
"""

import gymnasium as gym
import numpy as np
from typing import Dict, Any, Tuple, Optional
from gymnasium import spaces

from uav_swarm import UAVSwarm
from defense_system import DefenseSystem, WeaponType


class AirDefenseEnv(gym.Env):
    """
    Gym environment for air defense against UAV swarms.
    
    The agent learns to optimize weapon selection to minimize total cost
    (defense cost + penetration penalties) while maximizing UAV interceptions.
    
    State Space (5-dimensional Box):
        0. Normalized remaining UAVs (0-1, relative to initial swarm)
        1. Normalized remaining kinetic interceptors (0-1, relative to initial)
        2. Normalized remaining directed energy shots (0-1, relative to initial)
        3. Normalized cumulative defense cost (0-1, scaled by max expected cost)
        4. Normalized nearest threat distance (0-1, relative to initial distance)
    
    Action Space (3 discrete actions):
        0: Fire kinetic interceptor
        1: Fire directed energy weapon
        2: Skip (no weapon fired)
    
    Reward Function:
        reward = -cost_of_action - (1,000,000 × UAVs_penetrated_this_step)
        
    Episode Termination:
        - All UAVs neutralized (destroyed or penetrated)
        - Maximum steps reached (safety limit)
    """
    
    # Environment metadata
    metadata = {'render_modes': ['human', 'ansi'], 'render_fps': 30}
    
    def __init__(
        self,
        swarm_size_range: Tuple[int, int] = (100, 500),
        kinetic_initial: int = 50,
        de_initial: int = 100,
        penetration_penalty: float = 1_000_000.0,
        distance_per_step: float = 5.0,
        max_steps: int = 10000,
        normalize_observations: bool = True,
        random_seed: Optional[int] = None
    ):
        """
        Initialize the air defense environment.
        """
        super().__init__()
        
        # Store configuration
        self.swarm_size_range = swarm_size_range
        self.kinetic_initial = kinetic_initial
        self.de_initial = de_initial
        self.penetration_penalty = penetration_penalty
        self.distance_per_step = distance_per_step
        self.max_steps = max_steps
        self.normalize_observations = normalize_observations
        
        # Set random seed
        if random_seed is not None:
            self.seed(random_seed)
        
        # Define action space (3 discrete actions)
        self.action_space = spaces.Discrete(3)
        
        # Define observation space (5-dimensional continuous)
        # All values normalized to [0, 1] if normalize_observations=True
        if normalize_observations:
            self.observation_space = spaces.Box(
                low=0.0,
                high=1.0,
                shape=(5,),
                dtype=np.float32
            )
        else:
            # Raw values (for debugging/analysis)
            self.observation_space = spaces.Box(
                low=np.array([0, 0, 0, 0, 0], dtype=np.float32),
                high=np.array([
                    swarm_size_range[1],  # Max UAVs
                    kinetic_initial,  # Max kinetic
                    de_initial,  # Max DE
                    1e9,  # Max cost (large number)
                    100.0  # Max distance
                ], dtype=np.float32),
                dtype=np.float32
            )
        
        # Initialize environment state
        self.swarm: Optional[UAVSwarm] = None
        self.defense: Optional[DefenseSystem] = None
        self.current_step = 0
        self.episode_reward = 0.0
        self.initial_swarm_size = 0
        
        # Normalization constants (for scaling observations)
        self.max_cost = kinetic_initial * DefenseSystem.KINETIC_COST + \
                        de_initial * DefenseSystem.DE_COST
        self.initial_distance = 100.0  # Default from UAVSwarm
        
        # Episode tracking
        self.episode_count = 0
        self.total_steps = 0
    
    def seed(self, seed: Optional[int] = None) -> list:
        """
        Set random seed for reproducibility.
        """
        if seed is not None:
            np.random.seed(seed)
        return [seed]
    
    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Reset the environment to start a new episode.
        
        Creates new UAV swarm with random size and fresh defense systems.
        """
        # Set seed if provided
        if seed is not None:
            self.seed(seed)
        
        # Randomly determine swarm size for this episode
        self.initial_swarm_size = np.random.randint(
            self.swarm_size_range[0],
            self.swarm_size_range[1] + 1
        )
        
        # Create new swarm and defense systems
        self.swarm = UAVSwarm(swarm_size=self.initial_swarm_size)
        self.defense = DefenseSystem(
            kinetic_quantity=self.kinetic_initial,
            de_quantity=self.de_initial
        )
        
        # Reset episode tracking
        self.current_step = 0
        self.episode_reward = 0.0
        self.episode_count += 1
        
        # Get initial observation
        observation = self._get_observation()
        
        # Info dictionary
        info = {
            'episode': self.episode_count,
            'swarm_size': self.initial_swarm_size,
            'initial_kinetic': self.kinetic_initial,
            'initial_de': self.de_initial
        }
        
        return observation, info
    
    def step(
        self,
        action: int
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Execute one step in the environment.
        """
        # Validate action
        if not self.action_space.contains(action):
            raise ValueError(f"Invalid action {action}, must be 0, 1, or 2")
        
        # Map action to weapon type
        action_map = {
            0: WeaponType.KINETIC,
            1: WeaponType.DIRECTED_ENERGY,
            2: WeaponType.SKIP
        }
        weapon_type = action_map[action]
        
        # Execute action (fire weapon)
        success, cost, had_ammunition = self.defense.fire_weapon(weapon_type)
        
        # If weapon fired and hit, destroy nearest UAV
        if weapon_type != WeaponType.SKIP and had_ammunition and success:
            nearest_uav = self.swarm.get_nearest_threat()
            if nearest_uav is not None:
                self.swarm.destroy_uav(nearest_uav)
        
        # Advance swarm toward target
        penetrated_this_step = self.swarm.advance_swarm(self.distance_per_step)
        
        # Calculate reward
        # Negative cost of action + large penalty for penetrations
        penetration_cost = penetrated_this_step * self.penetration_penalty
        reward = -cost - penetration_cost
        
        # Track episode reward
        self.episode_reward += reward
        self.current_step += 1
        self.total_steps += 1
        
        # Check termination conditions
        terminated = self.swarm.is_threat_eliminated()
        truncated = self.current_step >= self.max_steps
        
        # Get new observation
        observation = self._get_observation()
        
        # Info dictionary with useful metrics
        info = {
            'step': self.current_step,
            'action_taken': weapon_type.value,
            'weapon_success': success if had_ammunition else None,
            'had_ammunition': had_ammunition,
            'cost_this_step': cost,
            'penetrated_this_step': penetrated_this_step,
            'penetration_cost_this_step': penetration_cost,
            'reward_this_step': reward,
            'remaining_uavs': self.swarm.get_alive_count(),
            'remaining_kinetic': self.defense.kinetic_remaining,
            'remaining_de': self.defense.de_remaining,
            'cumulative_defense_cost': self.defense.total_cost_spent,
            'episode_reward': self.episode_reward
        }
        
        # Add final episode statistics if terminated
        if terminated or truncated:
            info.update({
                'episode_length': self.current_step,
                'uavs_destroyed': self.swarm.destroyed_count,
                'uavs_penetrated': self.swarm.penetration_count,
                'penetration_rate': self.swarm.get_penetration_rate(),
                'total_defense_cost': self.defense.total_cost_spent,
                'total_penetration_cost': self.swarm.penetration_count * self.penetration_penalty,
                'total_cost': self.defense.total_cost_spent + 
                             self.swarm.penetration_count * self.penetration_penalty,
                'cost_exchange_ratio': (self.defense.total_cost_spent + 
                                       self.swarm.penetration_count * self.penetration_penalty) / 
                                      self.swarm.get_total_attack_cost()
            })
        
        return observation, reward, terminated, truncated, info
    
    def _get_observation(self) -> np.ndarray:
        """
        Get current state observation.
        """
        # Get raw state values
        remaining_uavs = self.swarm.get_alive_count()
        remaining_kinetic = self.defense.kinetic_remaining
        remaining_de = self.defense.de_remaining
        cumulative_cost = self.defense.total_cost_spent
        nearest_distance = self.swarm.get_nearest_threat_distance()
        
        # Handle None distance (no UAVs left)
        if nearest_distance is None:
            nearest_distance = 0.0
        
        if self.normalize_observations:
            # Normalize to [0, 1]
            obs = np.array([
                remaining_uavs / self.initial_swarm_size,
                remaining_kinetic / self.kinetic_initial,
                remaining_de / self.de_initial,
                min(cumulative_cost / self.max_cost, 1.0),  # Clip at 1.0
                nearest_distance / self.initial_distance
            ], dtype=np.float32)
        else:
            # Raw values
            obs = np.array([
                remaining_uavs,
                remaining_kinetic,
                remaining_de,
                cumulative_cost,
                nearest_distance
            ], dtype=np.float32)
        
        return obs
    
    def render(self, mode: str = 'human') -> Optional[str]:
        """
        Render the environment state.
        """
        if self.swarm is None or self.defense is None:
            return "Environment not initialized. Call reset() first."
        
        output = []
        output.append(f"\n{'='*60}")
        output.append(f"AIR DEFENSE ENVIRONMENT - Step {self.current_step}")
        output.append(f"{'='*60}")
        
        # Swarm status
        output.append(f"\nUAV SWARM:")
        output.append(f"  Initial: {self.initial_swarm_size}")
        output.append(f"  Alive: {self.swarm.get_alive_count()}")
        output.append(f"  Destroyed: {self.swarm.destroyed_count}")
        output.append(f"  Penetrated: {self.swarm.penetration_count}")
        output.append(f"  Nearest distance: {self.swarm.get_nearest_threat_distance():.1f} km"
                     if self.swarm.get_nearest_threat_distance() else "  Nearest distance: N/A")
        
        # Defense status
        output.append(f"\nDEFENSE SYSTEMS:")
        output.append(f"  Kinetic: {self.defense.kinetic_remaining}/{self.kinetic_initial}")
        output.append(f"  DE: {self.defense.de_remaining}/{self.de_initial}")
        output.append(f"  Cost spent: ${self.defense.total_cost_spent:,.0f}")
        
        # Episode metrics
        output.append(f"\nEPISODE METRICS:")
        output.append(f"  Episode reward: {self.episode_reward:,.0f}")
        output.append(f"  Steps: {self.current_step}")
        
        output.append(f"{'='*60}\n")
        
        text = "\n".join(output)
        
        if mode == 'human':
            print(text)
            return None
        elif mode == 'ansi':
            return text
        else:
            raise ValueError(f"Unsupported render mode: {mode}")
    
    def close(self):
        """Clean up resources."""
        pass
    
    def get_episode_info(self) -> Dict[str, Any]:
        """
        Get comprehensive episode information.
        """
        if self.swarm is None or self.defense is None:
            return {}
        
        return {
            'episode_number': self.episode_count,
            'steps': self.current_step,
            'total_steps': self.total_steps,
            'episode_reward': self.episode_reward,
            'swarm_size': self.initial_swarm_size,
            'uavs_destroyed': self.swarm.destroyed_count,
            'uavs_penetrated': self.swarm.penetration_count,
            'penetration_rate': self.swarm.get_penetration_rate(),
            'defense_cost': self.defense.total_cost_spent,
            'kinetic_fired': self.defense.kinetic_fired,
            'de_fired': self.defense.de_fired,
            'kinetic_remaining': self.defense.kinetic_remaining,
            'de_remaining': self.defense.de_remaining
        }


if __name__ == "__main__":
    env = AirDefenseEnv(swarm_size_range=(50, 150), random_seed=42)
    obs, info = env.reset()
    print(f"Obs shape: {obs.shape}, sample: {obs}")
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    print(f"Step: reward={reward:.0f}, done={terminated or truncated}")
