"""
UAV Swarm Module

This module implements the UAV swarm model for air defense simulation.
Models a Shahed-136 style drone swarm with linear approach dynamics.

Author: Master's Thesis Project
Date: January 2026
"""

import numpy as np
from typing import Dict, List, Optional


class UAVSwarm:
    """
    Represents a swarm of attacking UAVs (Shahed-136 style drones).
    
    The swarm uses a simplified linear approach model where UAVs travel
    in straight lines toward the defended target. Each UAV has binary
    status (alive or destroyed) and uniform cost.
    
    Attributes:
        initial_size (int): Original number of UAVs in the swarm
        uav_cost (float): Cost per UAV in USD (default: $35,000)
        alive_uavs (np.ndarray): Boolean array indicating alive status
        destroyed_count (int): Number of destroyed UAVs
        penetration_count (int): Number of UAVs that reached target
        distances (np.ndarray): Current distance of each UAV from target (km)
        initial_distance (float): Starting distance from target (km)
    """
    
    # Class constants
    DEFAULT_UAV_COST = 35000.0  # USD per UAV
    DEFAULT_INITIAL_DISTANCE = 100.0  # km from target
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
        
        Args:
            swarm_size: Number of UAVs (if None, randomly sampled from 100-500)
            uav_cost: Cost per UAV in USD
            initial_distance: Starting distance from target in km
            random_seed: Random seed for reproducibility
            
        Raises:
            ValueError: If swarm_size is outside valid range or negative values provided
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
        self.destroyed_count = 0
        self.penetration_count = 0
        
        # Initialize distances (all start at same distance)
        self.distances = np.full(self.initial_size, initial_distance, dtype=float)
    
    def get_alive_count(self) -> int:
        """
        Get the number of currently alive UAVs.
        
        Returns:
            Number of UAVs still alive
        """
        return int(np.sum(self.alive_uavs))
    
    def get_alive_indices(self) -> np.ndarray:
        """
        Get indices of all alive UAVs.
        
        Returns:
            Array of indices where UAVs are still alive
        """
        return np.where(self.alive_uavs)[0]
    
    def get_nearest_threat(self) -> Optional[int]:
        """
        Find the index of the nearest alive UAV to the target.
        
        Returns:
            Index of nearest alive UAV, or None if all destroyed
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
        
        Returns:
            Distance in km, or None if no UAVs alive
        """
        nearest_idx = self.get_nearest_threat()
        
        if nearest_idx is None:
            return None
        
        return float(self.distances[nearest_idx])
    
    def destroy_uav(self, uav_index: int) -> bool:
        """
        Destroy a specific UAV.
        
        Args:
            uav_index: Index of the UAV to destroy
            
        Returns:
            True if UAV was destroyed, False if already destroyed or invalid index
            
        Raises:
            IndexError: If uav_index is out of bounds
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
        
        Args:
            distance_increment: Distance to advance in km (positive value)
            
        Returns:
            Number of UAVs that penetrated in this step
            
        Raises:
            ValueError: If distance_increment is negative
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
        
        Returns:
            Total cost in USD (swarm_size × uav_cost)
        """
        return self.initial_size * self.uav_cost
    
    def get_penetration_rate(self) -> float:
        """
        Calculate the penetration rate (fraction of UAVs that got through).
        
        Returns:
            Penetration rate as decimal (0.0 to 1.0)
        """
        if self.initial_size == 0:
            return 0.0
        
        return self.penetration_count / self.initial_size
    
    def get_destruction_rate(self) -> float:
        """
        Calculate the destruction rate (fraction of UAVs destroyed).
        
        Returns:
            Destruction rate as decimal (0.0 to 1.0)
        """
        if self.initial_size == 0:
            return 0.0
        
        return self.destroyed_count / self.initial_size
    
    def is_threat_eliminated(self) -> bool:
        """
        Check if all UAVs have been neutralized (destroyed or penetrated).
        
        Returns:
            True if no alive UAVs remain, False otherwise
        """
        return self.get_alive_count() == 0
    
    def get_status_summary(self) -> Dict[str, float]:
        """
        Get a comprehensive summary of swarm status.
        
        Returns:
            Dictionary containing:
                - initial_size: Original swarm size
                - alive: Current number of alive UAVs
                - destroyed: Number of destroyed UAVs
                - penetrated: Number of UAVs that got through
                - total_cost: Total attack cost in USD
                - penetration_rate: Fraction that penetrated
                - destruction_rate: Fraction destroyed
                - nearest_distance: Distance of nearest threat (or None)
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
        
        Args:
            new_swarm_size: New swarm size (if None, keeps current size)
        """
        if new_swarm_size is not None:
            if new_swarm_size < 1:
                raise ValueError(f"Swarm size must be positive, got {new_swarm_size}")
            self.initial_size = int(new_swarm_size)
        
        # Reset all state
        self.alive_uavs = np.ones(self.initial_size, dtype=bool)
        self.destroyed_count = 0
        self.penetration_count = 0
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
    """
    Test and demonstration code for UAVSwarm class.
    """
    print("=" * 70)
    print("UAV SWARM MODULE - DEMONSTRATION")
    print("=" * 70)
    
    # Test 1: Basic initialization with random size
    print("\n[Test 1] Random swarm initialization:")
    swarm1 = UAVSwarm(random_seed=42)
    print(f"  Created: {swarm1}")
    print(f"  Total attack cost: ${swarm1.get_total_attack_cost():,.2f}")
    
    # Test 2: Fixed size swarm
    print("\n[Test 2] Fixed size swarm (200 UAVs):")
    swarm2 = UAVSwarm(swarm_size=200, random_seed=123)
    print(f"  Created: {swarm2}")
    status = swarm2.get_status_summary()
    print(f"  Status: {status}")
    
    # Test 3: Nearest threat detection
    print("\n[Test 3] Nearest threat detection:")
    print(f"  Nearest UAV index: {swarm2.get_nearest_threat()}")
    print(f"  Nearest distance: {swarm2.get_nearest_threat_distance()} km")
    
    # Test 4: Destroying UAVs
    print("\n[Test 4] Destroying UAVs:")
    for i in range(5):
        nearest = swarm2.get_nearest_threat()
        success = swarm2.destroy_uav(nearest)
        print(f"  Destroyed UAV #{nearest}: {success}")
    print(f"  Remaining alive: {swarm2.get_alive_count()}")
    
    # Test 5: Advancing swarm
    print("\n[Test 5] Advancing swarm toward target:")
    initial_alive = swarm2.get_alive_count()
    print(f"  Initial alive: {initial_alive}")
    print(f"  Initial nearest distance: {swarm2.get_nearest_threat_distance()} km")
    
    # Advance in 10km increments
    for step in range(1, 11):
        penetrated = swarm2.advance_swarm(10.0)
        if penetrated > 0:
            print(f"  Step {step}: Advanced 10km, {penetrated} UAVs penetrated!")
    
    print(f"  Final nearest distance: {swarm2.get_nearest_threat_distance()} km")
    print(f"  Total penetrations: {swarm2.penetration_count}")
    
    # Test 6: Simulating a complete engagement
    print("\n[Test 6] Complete engagement simulation:")
    swarm3 = UAVSwarm(swarm_size=50, random_seed=999)
    print(f"  Starting with: {swarm3}")
    
    step = 0
    while not swarm3.is_threat_eliminated():
        step += 1
        
        # Destroy nearest threat (50% chance)
        if np.random.random() < 0.5:
            nearest = swarm3.get_nearest_threat()
            if nearest is not None:
                swarm3.destroy_uav(nearest)
        
        # Advance swarm
        penetrated = swarm3.advance_swarm(5.0)
        
        if step % 5 == 0 or swarm3.is_threat_eliminated():
            print(f"  Step {step}: {swarm3}")
    
    print("\n  Final statistics:")
    final_status = swarm3.get_status_summary()
    for key, value in final_status.items():
        if isinstance(value, float) and value > 100:
            print(f"    {key}: {value:,.2f}")
        else:
            print(f"    {key}: {value}")
    
    # Test 7: Reset functionality
    print("\n[Test 7] Reset functionality:")
    print(f"  Before reset: {swarm3}")
    swarm3.reset()
    print(f"  After reset: {swarm3}")
    
    # Test 8: Error handling
    print("\n[Test 8] Error handling:")
    try:
        bad_swarm = UAVSwarm(swarm_size=-10)
    except ValueError as e:
        print(f"  ✓ Caught expected error: {e}")
    
    try:
        swarm3.destroy_uav(999)
    except IndexError as e:
        print(f"  ✓ Caught expected error: {e}")
    
    try:
        swarm3.advance_swarm(-5.0)
    except ValueError as e:
        print(f"  ✓ Caught expected error: {e}")
    
    print("\n" + "=" * 70)
    print("ALL TESTS COMPLETED SUCCESSFULLY")
    print("=" * 70)
