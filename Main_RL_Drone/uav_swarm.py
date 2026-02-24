"""
UAV Swarm Module

This module implements the UAV swarm model for air defense simulation.
Models a Shahed-136 style drone swarm with linear approach dynamics.
"""

import numpy as np
from typing import Dict, List, Optional


class UAVSwarm:
    """
    Represents a swarm of attacking UAVs (Shahed-136 style drones).
    
    The swarm uses a simplified linear approach model where UAVs travel
    in straight lines toward the defended target. Each UAV has binary
    status (alive or destroyed) and uniform cost.
    """
    
    # Class constants
    DEFAULT_UAV_COST = 35000.0  # USD per UAV
    DEFAULT_INITIAL_DISTANCE = 1000.0  # km from target (allows 200 steps at 5km/step)
    MIN_SWARM_SIZE = 100
    MAX_SWARM_SIZE = 500
    
    def __init__(
        self,
        swarm_size: Optional[int] = None,
        uav_cost: float = DEFAULT_UAV_COST,
        initial_distance: float = DEFAULT_INITIAL_DISTANCE,
        random_seed: Optional[int] = None
    ):
        """
        Initialize a UAV swarm.
        """
        # Set random seed if provided
        if random_seed is not None:
            np.random.seed(random_seed)
        
        # Determine swarm size
        if swarm_size is None:
            self.initial_size = np.random.randint(
                self.MIN_SWARM_SIZE, 
                self.MAX_SWARM_SIZE + 1
            )
        else:
            if swarm_size < 1:
                raise ValueError(f"Swarm size must be positive, got {swarm_size}")
            self.initial_size = int(swarm_size)
        
        # Validate inputs
        if uav_cost < 0:
            raise ValueError(f"UAV cost cannot be negative, got {uav_cost}")
        if initial_distance <= 0:
            raise ValueError(f"Initial distance must be positive, got {initial_distance}")
        
        self.uav_cost = float(uav_cost)
        self.initial_distance = float(initial_distance)
        
        # Initialize swarm state
        self.alive_uavs = np.ones(self.initial_size, dtype=bool)
        self.targeted_uavs = np.zeros(self.initial_size, dtype=bool)
        self.passed_uavs = np.zeros(self.initial_size, dtype=bool)
        self.destroyed_count = 0
        self.penetration_count = 0
        self.passed_count = 0
        
        # Initialize distances (all start at same distance)
        self.distances = np.full(self.initial_size, initial_distance, dtype=float)
    
    def get_alive_count(self) -> int:
        """
        Get the number of currently alive UAVs.
        """
        return int(np.sum(self.alive_uavs))
    
    def get_alive_indices(self) -> np.ndarray:
        """
        Get indices of all alive UAVs.
        """
        return np.where(self.alive_uavs)[0]
    
    def get_nearest_threat(self) -> Optional[int]:
        """
        Find the index of the nearest alive UAV to the target.
        """
        alive_indices = self.get_alive_indices()
        
        if len(alive_indices) == 0:
            return None
        
        # Get distances of alive UAVs
        alive_distances = self.distances[alive_indices]
        
        # Find the minimum distance
        nearest_idx_in_alive = np.argmin(alive_distances)
        
        return alive_indices[nearest_idx_in_alive]
    
    def get_nearest_threat_distance(self) -> Optional[float]:
        """
        Get the distance of the nearest alive UAV.
        """
        nearest_idx = self.get_nearest_threat()
        
        if nearest_idx is None:
            return None
        
        return float(self.distances[nearest_idx])
    
    def get_nearest_untargeted(self) -> Optional[int]:
        """
        Find the index of the nearest alive AND untargeted UAV.
        Critical for one-shot-per-drone mechanic.
        """
        # Get alive UAVs that haven't been targeted yet
        untargeted_alive = self.alive_uavs & ~self.targeted_uavs
        untargeted_indices = np.where(untargeted_alive)[0]
        
        if len(untargeted_indices) == 0:
            return None
        
        # Get distances of untargeted alive UAVs
        untargeted_distances = self.distances[untargeted_indices]
        
        # Find the minimum distance
        nearest_idx_in_untargeted = np.argmin(untargeted_distances)
        
        return untargeted_indices[nearest_idx_in_untargeted]
    
    def mark_as_passed(self, uav_index: int) -> bool:
        """
        Mark a UAV as "passed" (shot at but missed, can't target again).
        """
        if uav_index < 0 or uav_index >= self.initial_size:
            return False
        
        if not self.passed_uavs[uav_index]:
            self.passed_uavs[uav_index] = True
            self.passed_count += 1
            return True
        
        return False
    
    def destroy_uav(self, uav_index: int) -> bool:
        """
        Destroy a specific UAV.
        """
        if uav_index < 0 or uav_index >= self.initial_size:
            raise IndexError(
                f"UAV index {uav_index} out of bounds [0, {self.initial_size})"
            )
        
        if self.alive_uavs[uav_index]:
            self.alive_uavs[uav_index] = False
            self.destroyed_count += 1
            return True
        
        return False
    
    def advance_swarm(self, distance_increment: float) -> int:
        """
        Advance all alive UAVs toward the target.
        
        UAVs that reach distance <= 0 are considered to have penetrated
        the defense and are counted as successful attacks.
        """
        if distance_increment < 0:
            raise ValueError(
                f"Distance increment must be non-negative, got {distance_increment}"
            )
        
        # Get alive UAVs
        alive_indices = self.get_alive_indices()
        
        if len(alive_indices) == 0:
            return 0
        
        # Advance distances (moving closer to target, so subtract)
        self.distances[alive_indices] -= distance_increment
        
        # Check for penetrations (distance <= 0)
        penetrated = (self.distances[alive_indices] <= 0)
        penetrated_indices = alive_indices[penetrated]
        
        # Mark penetrated UAVs as no longer alive
        self.alive_uavs[penetrated_indices] = False
        
        # Update counts
        num_penetrated = len(penetrated_indices)
        self.penetration_count += num_penetrated
        
        return num_penetrated
    
    def get_total_attack_cost(self) -> float:
        """
        Calculate the total cost of the attacking swarm.
        """
        return self.initial_size * self.uav_cost
    
    def get_penetration_rate(self) -> float:
        """
        Calculate the penetration rate (fraction of UAVs that got through).
        """
        if self.initial_size == 0:
            return 0.0
        
        return self.penetration_count / self.initial_size
    
    def get_destruction_rate(self) -> float:
        """
        Calculate the destruction rate (fraction of UAVs destroyed).
        """
        if self.initial_size == 0:
            return 0.0
        
        return self.destroyed_count / self.initial_size
    
    def is_threat_eliminated(self) -> bool:
        """
        Check if all UAVs have been neutralized (destroyed or penetrated).
        """
        return self.get_alive_count() == 0
    
    def get_status_summary(self) -> Dict[str, float]:
        """
        Get a comprehensive summary of swarm status.
        """
        return {
            'initial_size': self.initial_size,
            'alive': self.get_alive_count(),
            'destroyed': self.destroyed_count,
            'penetrated': self.penetration_count,
            'total_cost': self.get_total_attack_cost(),
            'penetration_rate': self.get_penetration_rate(),
            'destruction_rate': self.get_destruction_rate(),
            'nearest_distance': self.get_nearest_threat_distance()
        }
    
    def reset(self, new_swarm_size: Optional[int] = None):
        """
        Reset the swarm to initial state (useful for re-running scenarios).
        """
        if new_swarm_size is not None:
            if new_swarm_size < 1:
                raise ValueError(f"Swarm size must be positive, got {new_swarm_size}")
            self.initial_size = int(new_swarm_size)
        
        # Reset all state
        self.alive_uavs = np.ones(self.initial_size, dtype=bool)
        self.targeted_uavs = np.zeros(self.initial_size, dtype=bool)
        self.passed_uavs = np.zeros(self.initial_size, dtype=bool)
        self.destroyed_count = 0
        self.penetration_count = 0
        self.passed_count = 0
        self.distances = np.full(self.initial_size, self.initial_distance, dtype=float)
    
    def __repr__(self) -> str:
        """String representation of the swarm."""
        return (
            f"UAVSwarm(size={self.initial_size}, alive={self.get_alive_count()}, "
            f"destroyed={self.destroyed_count}, penetrated={self.penetration_count})"
        )
    
    def __len__(self) -> int:
        """Return the initial size of the swarm."""
        return self.initial_size


if __name__ == "__main__":
    swarm = UAVSwarm(swarm_size=100, random_seed=42)
    print(swarm)
    print(swarm.get_status_summary())
