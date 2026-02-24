"""
Defense System Module

This module implements the defense weapon systems for air defense simulation.
Includes both kinetic interceptors (expensive missiles) and directed energy weapons (cheap lasers).
"""

import numpy as np
from typing import Dict, Tuple, Optional
from enum import Enum


class WeaponType(Enum):
    """Enumeration of available weapon types."""
    KINETIC = "kinetic_interceptor"
    DIRECTED_ENERGY = "directed_energy"
    SKIP = "skip"  # No weapon fired


class DefenseSystem:
    """
    Represents the air defense weapon systems.
    
    Manages two types of weapons:
    1. Kinetic Interceptor - High cost ($1M), high success rate (90% ± 5%)
    2. Directed Energy Weapon - Low cost ($10K), moderate success rate (70% ± 10%)
    
    Both weapons have stochastic success rates modeled with normal distributions.
    Tracks ammunition, costs, and engagement statistics.
    """
    
    # Weapon specifications (class constants)
    KINETIC_COST = 1_000_000.0  # $1M per shot
    KINETIC_SUCCESS_MEAN = 0.90  # 90% base success rate
    KINETIC_SUCCESS_STD = 0.05   # 5% standard deviation
    KINETIC_INITIAL_QUANTITY = 800  # Updated per spec: ample kinetic inventory
    
    DE_COST = 10_000.0  # $10K per shot
    DE_SUCCESS_MEAN = 0.70  # 70% base success rate
    DE_SUCCESS_STD = 0.10   # 10% standard deviation
    DE_INITIAL_QUANTITY = 150   # Updated per spec: limited DE inventory
    
    def __init__(
        self,
        kinetic_quantity: int = 200,
        de_quantity: int = 150,
        random_seed: Optional[int] = None
    ):
        """
        Initialize the defense system.
        """
        if kinetic_quantity < 0:
            raise ValueError(f"Kinetic quantity cannot be negative, got {kinetic_quantity}")
        if de_quantity < 0:
            raise ValueError(f"DE quantity cannot be negative, got {de_quantity}")
        
        # Set random seed if provided
        if random_seed is not None:
            np.random.seed(random_seed)
        
        # Initial ammunition
        self.initial_kinetic = int(kinetic_quantity)
        self.initial_de = int(de_quantity)
        
        # Current ammunition
        self.kinetic_remaining = int(kinetic_quantity)
        self.de_remaining = int(de_quantity)
        
        # Cost tracking
        self.total_cost_spent = 0.0
        
        # Engagement statistics
        self.kinetic_fired = 0
        self.de_fired = 0
        self.kinetic_hits = 0
        self.de_hits = 0
        self.kinetic_misses = 0
        self.de_misses = 0
    
    def has_kinetic(self) -> bool:
        """
        Check if kinetic interceptors are available.
        """
        return self.kinetic_remaining > 0
    
    def has_directed_energy(self) -> bool:
        """
        Check if directed energy shots are available.
        """
        return self.de_remaining > 0
    
    def has_any_weapon(self) -> bool:
        """
        Check if any weapon is available.
        """
        return self.has_kinetic() or self.has_directed_energy()
    
    def _calculate_success_probability(
        self,
        weapon_type: WeaponType
    ) -> float:
        """
        Calculate stochastic success probability for a weapon.
        
        Samples from normal distribution with weapon-specific parameters.
        Clips result to [0, 1] range.
        """
        if weapon_type == WeaponType.KINETIC:
            mean = self.KINETIC_SUCCESS_MEAN
            std = self.KINETIC_SUCCESS_STD
        elif weapon_type == WeaponType.DIRECTED_ENERGY:
            mean = self.DE_SUCCESS_MEAN
            std = self.DE_SUCCESS_STD
        else:
            raise ValueError(f"Invalid weapon type for probability calculation: {weapon_type}")
        
        # Sample from normal distribution
        probability = np.random.normal(mean, std)
        
        # Clip to valid probability range [0, 1]
        probability = np.clip(probability, 0.0, 1.0)
        
        return float(probability)
    
    def fire_kinetic(self) -> Tuple[bool, float, bool]:
        """
        Fire a kinetic interceptor at a target.
        
        Stochastically determines if the shot hits or misses based on
        sampled success probability. Decrements ammunition and tracks costs.
        """
        # Check if ammunition available
        if not self.has_kinetic():
            return False, 0.0, False
        
        # Decrement ammunition
        self.kinetic_remaining -= 1
        self.kinetic_fired += 1
        
        # Calculate cost
        cost = self.KINETIC_COST
        self.total_cost_spent += cost
        
        # Determine success stochastically
        success_prob = self._calculate_success_probability(WeaponType.KINETIC)
        success = np.random.random() < success_prob
        
        # Track statistics
        if success:
            self.kinetic_hits += 1
        else:
            self.kinetic_misses += 1
        
        return success, cost, True
    
    def fire_directed_energy(self) -> Tuple[bool, float, bool]:
        """
        Fire a directed energy weapon at a target.
        
        Stochastically determines if the shot hits or misses based on
        sampled success probability. Decrements ammunition and tracks costs.
        """
        # Check if ammunition available
        if not self.has_directed_energy():
            return False, 0.0, False
        
        # Decrement ammunition
        self.de_remaining -= 1
        self.de_fired += 1
        
        # Calculate cost
        cost = self.DE_COST
        self.total_cost_spent += cost
        
        # Determine success stochastically
        success_prob = self._calculate_success_probability(WeaponType.DIRECTED_ENERGY)
        success = np.random.random() < success_prob
        
        # Track statistics
        if success:
            self.de_hits += 1
        else:
            self.de_misses += 1
        
        return success, cost, True
    
    def fire_weapon(
        self,
        weapon_type: WeaponType
    ) -> Tuple[bool, float, bool]:
        """
        Fire a weapon of specified type.
        
        Unified interface for firing either weapon type or skipping.
        """
        if weapon_type == WeaponType.KINETIC:
            return self.fire_kinetic()
        elif weapon_type == WeaponType.DIRECTED_ENERGY:
            return self.fire_directed_energy()
        elif weapon_type == WeaponType.SKIP:
            return False, 0.0, True  # No weapon fired, no cost, always "available"
        else:
            raise ValueError(f"Unknown weapon type: {weapon_type}")
    
    def get_kinetic_accuracy(self) -> Optional[float]:
        """
        Calculate accuracy of kinetic interceptors.
        """
        if self.kinetic_fired == 0:
            return None
        return self.kinetic_hits / self.kinetic_fired
    
    def get_de_accuracy(self) -> Optional[float]:
        """
        Calculate accuracy of directed energy weapons.
        """
        if self.de_fired == 0:
            return None
        return self.de_hits / self.de_fired
    
    def get_overall_accuracy(self) -> Optional[float]:
        """
        Calculate overall accuracy across all weapon types.
        """
        total_fired = self.kinetic_fired + self.de_fired
        if total_fired == 0:
            return None
        
        total_hits = self.kinetic_hits + self.de_hits
        return total_hits / total_fired
    
    def get_ammunition_status(self) -> Dict[str, int]:
        """
        Get current ammunition status for all weapon types.
        """
        return {
            'kinetic_remaining': self.kinetic_remaining,
            'de_remaining': self.de_remaining,
            'kinetic_initial': self.initial_kinetic,
            'de_initial': self.initial_de
        }
    
    def get_cost_breakdown(self) -> Dict[str, float]:
        """
        Get detailed cost breakdown by weapon type.
        """
        kinetic_cost = self.kinetic_fired * self.KINETIC_COST
        de_cost = self.de_fired * self.DE_COST
        
        kinetic_cost_per_hit = None
        if self.kinetic_hits > 0:
            kinetic_cost_per_hit = kinetic_cost / self.kinetic_hits
        
        de_cost_per_hit = None
        if self.de_hits > 0:
            de_cost_per_hit = de_cost / self.de_hits
        
        return {
            'kinetic_cost': kinetic_cost,
            'de_cost': de_cost,
            'total_cost': self.total_cost_spent,
            'kinetic_cost_per_hit': kinetic_cost_per_hit,
            'de_cost_per_hit': de_cost_per_hit
        }
    
    def get_engagement_statistics(self) -> Dict[str, any]:
        """
        Get comprehensive engagement statistics.
        """
        return {
            # Ammunition
            'kinetic_remaining': self.kinetic_remaining,
            'de_remaining': self.de_remaining,
            
            # Shots fired
            'kinetic_fired': self.kinetic_fired,
            'de_fired': self.de_fired,
            'total_fired': self.kinetic_fired + self.de_fired,
            
            # Hit/miss stats
            'kinetic_hits': self.kinetic_hits,
            'kinetic_misses': self.kinetic_misses,
            'de_hits': self.de_hits,
            'de_misses': self.de_misses,
            'total_hits': self.kinetic_hits + self.de_hits,
            'total_misses': self.kinetic_misses + self.de_misses,
            
            # Accuracy
            'kinetic_accuracy': self.get_kinetic_accuracy(),
            'de_accuracy': self.get_de_accuracy(),
            'overall_accuracy': self.get_overall_accuracy(),
            
            # Costs
            'total_cost': self.total_cost_spent,
            'kinetic_cost': self.kinetic_fired * self.KINETIC_COST,
            'de_cost': self.de_fired * self.DE_COST
        }
    
    def reset(
        self,
        kinetic_quantity: Optional[int] = None,
        de_quantity: Optional[int] = None
    ):
        """
        Reset the defense system to initial state.
        """
        if kinetic_quantity is not None:
            if kinetic_quantity < 0:
                raise ValueError(f"Kinetic quantity cannot be negative, got {kinetic_quantity}")
            self.initial_kinetic = int(kinetic_quantity)
        
        if de_quantity is not None:
            if de_quantity < 0:
                raise ValueError(f"DE quantity cannot be negative, got {de_quantity}")
            self.initial_de = int(de_quantity)
        
        # Reset all state
        self.kinetic_remaining = self.initial_kinetic
        self.de_remaining = self.initial_de
        self.total_cost_spent = 0.0
        self.kinetic_fired = 0
        self.de_fired = 0
        self.kinetic_hits = 0
        self.de_hits = 0
        self.kinetic_misses = 0
        self.de_misses = 0
    
    def __repr__(self) -> str:
        """String representation of the defense system."""
        return (
            f"DefenseSystem(kinetic={self.kinetic_remaining}/{self.initial_kinetic}, "
            f"DE={self.de_remaining}/{self.initial_de}, "
            f"cost=${self.total_cost_spent:,.0f})"
        )


if __name__ == "__main__":
    defense = DefenseSystem(random_seed=42)
    print(defense)
    print(defense.get_engagement_statistics())
