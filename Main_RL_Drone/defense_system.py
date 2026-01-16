"""
Defense System Module

This module implements the defense weapon systems for air defense simulation.
Includes both kinetic interceptors (expensive missiles) and directed energy weapons (cheap lasers).

Author: Master's Thesis Project
Date: January 2026
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
    
    Attributes:
        kinetic_remaining (int): Remaining kinetic interceptor missiles
        de_remaining (int): Remaining directed energy shots
        total_cost_spent (float): Cumulative cost of all fired weapons in USD
        kinetic_fired (int): Total kinetic interceptors fired
        de_fired (int): Total directed energy weapons fired
        kinetic_hits (int): Successful kinetic intercepts
        de_hits (int): Successful directed energy intercepts
        kinetic_misses (int): Failed kinetic intercepts
        de_misses (int): Failed directed energy intercepts
    """
    
    # Weapon specifications (class constants)
    KINETIC_COST = 1_000_000.0  # $1M per shot
    KINETIC_SUCCESS_MEAN = 0.90  # 90% base success rate
    KINETIC_SUCCESS_STD = 0.05   # 5% standard deviation
    KINETIC_INITIAL_QUANTITY = 50
    
    DE_COST = 10_000.0  # $10K per shot
    DE_SUCCESS_MEAN = 0.70  # 70% base success rate
    DE_SUCCESS_STD = 0.10   # 10% standard deviation
    DE_INITIAL_QUANTITY = 100
    
    def __init__(
        self,
        kinetic_quantity: int = KINETIC_INITIAL_QUANTITY,
        de_quantity: int = DE_INITIAL_QUANTITY,
        random_seed: Optional[int] = None
    ):
        """
        Initialize the defense system.
        
        Args:
            kinetic_quantity: Initial number of kinetic interceptors
            de_quantity: Initial number of directed energy shots
            random_seed: Random seed for reproducibility
            
        Raises:
            ValueError: If quantities are negative
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
        
        Returns:
            True if at least one kinetic interceptor remains
        """
        return self.kinetic_remaining > 0
    
    def has_directed_energy(self) -> bool:
        """
        Check if directed energy shots are available.
        
        Returns:
            True if at least one directed energy shot remains
        """
        return self.de_remaining > 0
    
    def has_any_weapon(self) -> bool:
        """
        Check if any weapon is available.
        
        Returns:
            True if either weapon type has ammunition remaining
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
        
        Args:
            weapon_type: Type of weapon to calculate probability for
            
        Returns:
            Success probability between 0.0 and 1.0
            
        Raises:
            ValueError: If weapon_type is invalid
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
        
        Returns:
            Tuple of (success, cost, had_ammunition):
                - success: True if intercept succeeded, False if missed
                - cost: Cost of the shot in USD
                - had_ammunition: True if weapon was available to fire
                
        Example:
            >>> defense = DefenseSystem()
            >>> success, cost, available = defense.fire_kinetic()
            >>> if available:
            >>>     print(f"Fired kinetic: {'HIT' if success else 'MISS'}, cost ${cost:,}")
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
        
        Returns:
            Tuple of (success, cost, had_ammunition):
                - success: True if intercept succeeded, False if missed
                - cost: Cost of the shot in USD
                - had_ammunition: True if weapon was available to fire
                
        Example:
            >>> defense = DefenseSystem()
            >>> success, cost, available = defense.fire_directed_energy()
            >>> if available:
            >>>     print(f"Fired DE: {'HIT' if success else 'MISS'}, cost ${cost:,}")
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
        
        Args:
            weapon_type: Type of weapon to fire (KINETIC, DIRECTED_ENERGY, or SKIP)
            
        Returns:
            Tuple of (success, cost, had_ammunition):
                - success: True if intercept succeeded, False if missed or skipped
                - cost: Cost of the shot in USD (0.0 for SKIP)
                - had_ammunition: True if weapon was available (always True for SKIP)
                
        Raises:
            ValueError: If weapon_type is not recognized
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
        
        Returns:
            Hit rate as decimal (0.0 to 1.0), or None if none fired
        """
        if self.kinetic_fired == 0:
            return None
        return self.kinetic_hits / self.kinetic_fired
    
    def get_de_accuracy(self) -> Optional[float]:
        """
        Calculate accuracy of directed energy weapons.
        
        Returns:
            Hit rate as decimal (0.0 to 1.0), or None if none fired
        """
        if self.de_fired == 0:
            return None
        return self.de_hits / self.de_fired
    
    def get_overall_accuracy(self) -> Optional[float]:
        """
        Calculate overall accuracy across all weapon types.
        
        Returns:
            Overall hit rate as decimal (0.0 to 1.0), or None if no weapons fired
        """
        total_fired = self.kinetic_fired + self.de_fired
        if total_fired == 0:
            return None
        
        total_hits = self.kinetic_hits + self.de_hits
        return total_hits / total_fired
    
    def get_ammunition_status(self) -> Dict[str, int]:
        """
        Get current ammunition status for all weapon types.
        
        Returns:
            Dictionary with remaining ammunition counts
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
        
        Returns:
            Dictionary containing:
                - kinetic_cost: Total spent on kinetic interceptors
                - de_cost: Total spent on directed energy
                - total_cost: Combined total cost
                - kinetic_cost_per_hit: Average cost per successful kinetic hit (or None)
                - de_cost_per_hit: Average cost per successful DE hit (or None)
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
        
        Returns:
            Dictionary containing all tracking metrics:
                - Ammunition counts
                - Shots fired by type
                - Hit/miss statistics
                - Accuracy rates
                - Cost breakdown
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
        
        Args:
            kinetic_quantity: New kinetic quantity (if None, uses initial)
            de_quantity: New DE quantity (if None, uses initial)
            
        Raises:
            ValueError: If quantities are negative
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
    """
    Test and demonstration code for DefenseSystem class.
    """
    print("=" * 70)
    print("DEFENSE SYSTEM MODULE - DEMONSTRATION")
    print("=" * 70)
    
    # Test 1: Basic initialization
    print("\n[Test 1] Basic initialization:")
    defense = DefenseSystem(random_seed=42)
    print(f"  Created: {defense}")
    ammo = defense.get_ammunition_status()
    print(f"  Ammunition: {ammo}")
    
    # Test 2: Firing kinetic interceptors
    print("\n[Test 2] Firing kinetic interceptors (10 shots):")
    defense.reset()
    np.random.seed(100)
    
    hits = 0
    misses = 0
    for i in range(10):
        success, cost, available = defense.fire_kinetic()
        if success:
            hits += 1
            result = "HIT"
        else:
            result = "MISS"
        print(f"  Shot {i+1}: {result}, cost=${cost:,.0f}")
    
    print(f"  Summary: {hits} hits, {misses} misses")
    print(f"  Accuracy: {defense.get_kinetic_accuracy():.2%}")
    
    # Test 3: Firing directed energy weapons
    print("\n[Test 3] Firing directed energy weapons (10 shots):")
    defense.reset()
    np.random.seed(200)
    
    hits = 0
    misses = 0
    for i in range(10):
        success, cost, available = defense.fire_directed_energy()
        if success:
            hits += 1
            result = "HIT"
        else:
            result = "MISS"
        print(f"  Shot {i+1}: {result}, cost=${cost:,.0f}")
    
    print(f"  Summary: {hits} hits, {misses} misses")
    print(f"  Accuracy: {defense.get_de_accuracy():.2%}")
    
    # Test 4: Mixed weapon usage
    print("\n[Test 4] Mixed weapon engagement:")
    defense.reset()
    np.random.seed(300)
    
    # Fire 5 kinetic, 5 DE
    for i in range(5):
        defense.fire_kinetic()
    for i in range(5):
        defense.fire_directed_energy()
    
    stats = defense.get_engagement_statistics()
    print(f"  Kinetic: {stats['kinetic_hits']}/{stats['kinetic_fired']} hits "
          f"({stats['kinetic_accuracy']:.2%} accuracy)")
    print(f"  DE: {stats['de_hits']}/{stats['de_fired']} hits "
          f"({stats['de_accuracy']:.2%} accuracy)")
    print(f"  Overall: {stats['total_hits']}/{stats['total_fired']} hits "
          f"({stats['overall_accuracy']:.2%} accuracy)")
    print(f"  Total cost: ${stats['total_cost']:,.0f}")
    
    # Test 5: Ammunition depletion
    print("\n[Test 5] Ammunition depletion:")
    defense_small = DefenseSystem(kinetic_quantity=2, de_quantity=3, random_seed=400)
    print(f"  Starting: {defense_small}")
    
    # Fire all kinetic
    for i in range(3):
        success, cost, available = defense_small.fire_kinetic()
        print(f"  Kinetic shot {i+1}: available={available}, success={success}")
    
    # Fire all DE
    for i in range(4):
        success, cost, available = defense_small.fire_directed_energy()
        print(f"  DE shot {i+1}: available={available}, success={success}")
    
    print(f"  Final: {defense_small}")
    print(f"  Has any weapon: {defense_small.has_any_weapon()}")
    
    # Test 6: Cost breakdown analysis
    print("\n[Test 6] Cost breakdown analysis:")
    defense.reset()
    np.random.seed(500)
    
    # Fire mix of weapons
    for i in range(10):
        defense.fire_kinetic()
    for i in range(20):
        defense.fire_directed_energy()
    
    costs = defense.get_cost_breakdown()
    print(f"  Kinetic cost: ${costs['kinetic_cost']:,.0f}")
    print(f"  DE cost: ${costs['de_cost']:,.0f}")
    print(f"  Total cost: ${costs['total_cost']:,.0f}")
    
    if costs['kinetic_cost_per_hit']:
        print(f"  Kinetic cost per hit: ${costs['kinetic_cost_per_hit']:,.0f}")
    if costs['de_cost_per_hit']:
        print(f"  DE cost per hit: ${costs['de_cost_per_hit']:,.0f}")
    
    # Test 7: WeaponType enum interface
    print("\n[Test 7] WeaponType enum interface:")
    defense.reset()
    np.random.seed(600)
    
    actions = [
        WeaponType.KINETIC,
        WeaponType.DIRECTED_ENERGY,
        WeaponType.SKIP,
        WeaponType.DIRECTED_ENERGY,
        WeaponType.KINETIC
    ]
    
    for i, action in enumerate(actions):
        success, cost, available = defense.fire_weapon(action)
        print(f"  Action {i+1}: {action.value:20s} → success={success}, cost=${cost:,.0f}")
    
    print(f"  Final state: {defense}")
    
    # Test 8: Statistical validation (success rate distribution)
    print("\n[Test 8] Statistical validation (1000 shots each):")
    defense.reset()
    np.random.seed(700)
    
    # Fire 1000 of each
    for i in range(1000):
        defense.fire_kinetic()
        defense.fire_directed_energy()
    
    stats = defense.get_engagement_statistics()
    kinetic_rate = stats['kinetic_accuracy']
    de_rate = stats['de_accuracy']
    
    print(f"  Kinetic accuracy: {kinetic_rate:.4f} (expected ~0.90)")
    print(f"  DE accuracy: {de_rate:.4f} (expected ~0.70)")
    print(f"  Kinetic within expected range: {0.85 < kinetic_rate < 0.95}")
    print(f"  DE within expected range: {0.65 < de_rate < 0.75}")
    
    # Test 9: Reset functionality
    print("\n[Test 9] Reset functionality:")
    print(f"  Before reset: {defense}")
    defense.reset()
    print(f"  After reset: {defense}")
    print(f"  Stats after reset: {defense.get_engagement_statistics()}")
    
    # Test 10: Error handling
    print("\n[Test 10] Error handling:")
    try:
        bad_defense = DefenseSystem(kinetic_quantity=-5)
    except ValueError as e:
        print(f"  ✓ Caught expected error: {e}")
    
    try:
        defense.fire_weapon("invalid_weapon")
    except ValueError as e:
        print(f"  ✓ Caught expected error: {e}")
    
    print("\n" + "=" * 70)
    print("ALL TESTS COMPLETED SUCCESSFULLY")
    print("=" * 70)
