"""
Combat Simulation Module

This module implements the core combat simulation engine that manages
engagements between UAV swarms and defense systems.

Author: Master's Thesis Project
Date: January 2026
"""

import numpy as np
from typing import Dict, Callable, Optional, Any
from uav_swarm import UAVSwarm
from defense_system import DefenseSystem, WeaponType


class CombatSimulation:
    """
    Manages a complete air defense engagement scenario.
    
    Orchestrates the interaction between an attacking UAV swarm and a 
    defense system using a provided strategy function. Runs step-by-step
    until all UAVs are neutralized (destroyed or penetrated).
    
    The simulation operates in discrete time steps where:
    1. Strategy selects a weapon to fire (or skip)
    2. Weapon is fired (if available)
    3. UAV swarm advances toward target
    4. Metrics are tracked
    
    Attributes:
        swarm (UAVSwarm): The attacking UAV swarm
        defense (DefenseSystem): The defense weapon systems
        step_count (int): Number of simulation steps executed
        distance_per_step (float): Distance UAVs advance each step (km)
        penetration_penalty (float): Cost penalty per penetrating UAV
        max_steps (int): Maximum simulation steps before forced termination
    """
    
    # Default simulation parameters
    DEFAULT_DISTANCE_PER_STEP = 5.0  # km per step
    DEFAULT_PENETRATION_PENALTY = 1_000_000.0  # $1M per UAV that gets through
    DEFAULT_MAX_STEPS = 10000  # Prevent infinite loops
    
    def __init__(
        self,
        swarm: Optional[UAVSwarm] = None,
        defense: Optional[DefenseSystem] = None,
        swarm_size: Optional[int] = None,
        distance_per_step: float = DEFAULT_DISTANCE_PER_STEP,
        penetration_penalty: float = DEFAULT_PENETRATION_PENALTY,
        max_steps: int = DEFAULT_MAX_STEPS,
        random_seed: Optional[int] = None
    ):
        """
        Initialize a combat simulation.
        
        Args:
            swarm: Pre-configured UAVSwarm (if None, creates new one)
            defense: Pre-configured DefenseSystem (if None, creates new one)
            swarm_size: Size of swarm if creating new (None = random 100-500)
            distance_per_step: Distance UAVs advance each step in km
            penetration_penalty: Cost penalty per penetrating UAV in USD
            max_steps: Maximum simulation steps before forced termination
            random_seed: Random seed for reproducibility
            
        Raises:
            ValueError: If parameters are invalid
        """
        if random_seed is not None:
            np.random.seed(random_seed)
        
        # Validate parameters
        if distance_per_step <= 0:
            raise ValueError(f"Distance per step must be positive, got {distance_per_step}")
        if penetration_penalty < 0:
            raise ValueError(f"Penetration penalty cannot be negative, got {penetration_penalty}")
        if max_steps <= 0:
            raise ValueError(f"Max steps must be positive, got {max_steps}")
        
        # Initialize or use provided swarm
        if swarm is None:
            self.swarm = UAVSwarm(swarm_size=swarm_size, random_seed=random_seed)
        else:
            self.swarm = swarm
        
        # Initialize or use provided defense
        if defense is None:
            self.defense = DefenseSystem(random_seed=random_seed)
        else:
            self.defense = defense
        
        # Simulation parameters
        self.distance_per_step = float(distance_per_step)
        self.penetration_penalty = float(penetration_penalty)
        self.max_steps = int(max_steps)
        
        # Simulation state
        self.step_count = 0
        self.total_penetration_cost = 0.0
        self.simulation_completed = False
        self.termination_reason = None
    
    def run_simulation(
        self,
        strategy: Callable[[Dict[str, Any]], WeaponType],
        verbose: bool = False
    ) -> Dict[str, Any]:
        """
        Run a complete combat simulation using the provided strategy.
        
        The strategy function receives current state and returns which weapon
        to fire. Simulation runs until all UAVs are neutralized or max steps reached.
        
        Args:
            strategy: Function that takes state dict and returns WeaponType
                      State dict contains:
                          - remaining_uavs: int
                          - remaining_kinetic: int
                          - remaining_de: int
                          - cumulative_cost: float
                          - nearest_distance: float (or None)
                          - step: int
            verbose: If True, print step-by-step progress
            
        Returns:
            Dictionary containing complete simulation results:
                - swarm_size: Initial UAV count
                - uavs_destroyed: Number destroyed by weapons
                - uavs_penetrated: Number that reached target
                - penetration_rate: Fraction that penetrated (0-1)
                - defense_cost: Total weapon cost spent
                - attack_cost: Total swarm cost
                - penetration_cost: Penalty for penetrations
                - total_cost: defense_cost + penetration_cost
                - cost_exchange_ratio: total_cost / attack_cost
                - kinetic_fired: Kinetic missiles used
                - de_fired: Directed energy shots used
                - kinetic_hits: Successful kinetic intercepts
                - de_hits: Successful DE intercepts
                - kinetic_accuracy: Hit rate for kinetic (or None)
                - de_accuracy: Hit rate for DE (or None)
                - steps: Number of simulation steps
                - termination_reason: Why simulation ended
                
        Raises:
            ValueError: If strategy returns invalid weapon type
            RuntimeError: If simulation fails to complete properly
        """
        # Reset state
        self.step_count = 0
        self.total_penetration_cost = 0.0
        self.simulation_completed = False
        self.termination_reason = None
        
        if verbose:
            print(f"\n{'='*70}")
            print(f"COMBAT SIMULATION START")
            print(f"{'='*70}")
            print(f"Swarm: {self.swarm.initial_size} UAVs @ ${self.swarm.get_total_attack_cost():,.0f}")
            print(f"Defense: {self.defense.kinetic_remaining} kinetic, {self.defense.de_remaining} DE")
            print(f"{'='*70}\n")
        
        # Main simulation loop
        while not self.swarm.is_threat_eliminated() and self.step_count < self.max_steps:
            self.step_count += 1
            
            # Get current state for strategy
            state = self._get_current_state()
            
            # Strategy decides which weapon to fire
            try:
                action = strategy(state)
            except Exception as e:
                raise RuntimeError(f"Strategy function failed at step {self.step_count}: {e}")
            
            # Validate action
            if not isinstance(action, WeaponType):
                raise ValueError(
                    f"Strategy must return WeaponType, got {type(action).__name__}"
                )
            
            # Execute the action
            success, cost, had_ammunition = self.defense.fire_weapon(action)
            
            # If weapon fired and hit, destroy the nearest UAV
            if action != WeaponType.SKIP and had_ammunition and success:
                nearest_uav = self.swarm.get_nearest_threat()
                if nearest_uav is not None:
                    self.swarm.destroy_uav(nearest_uav)
            
            # Advance the swarm
            penetrated_this_step = self.swarm.advance_swarm(self.distance_per_step)
            
            # Calculate penetration cost for this step
            step_penetration_cost = penetrated_this_step * self.penetration_penalty
            self.total_penetration_cost += step_penetration_cost
            
            # Verbose output
            if verbose and self.step_count % 100 == 0:
                alive = self.swarm.get_alive_count()
                destroyed = self.swarm.destroyed_count
                penetrated = self.swarm.penetration_count
                print(f"Step {self.step_count:4d}: Action={action.value:20s} | "
                      f"Alive={alive:3d} | Destroyed={destroyed:3d} | "
                      f"Penetrated={penetrated:3d} | Cost=${self.defense.total_cost_spent:,.0f}")
        
        # Determine termination reason
        if self.swarm.is_threat_eliminated():
            self.termination_reason = "all_uavs_neutralized"
        elif self.step_count >= self.max_steps:
            self.termination_reason = "max_steps_reached"
        else:
            self.termination_reason = "unknown"
        
        self.simulation_completed = True
        
        # Compile results
        results = self._compile_results()
        
        if verbose:
            print(f"\n{'='*70}")
            print(f"SIMULATION COMPLETE - {self.termination_reason}")
            print(f"{'='*70}")
            self._print_results(results)
            print(f"{'='*70}\n")
        
        return results
    
    def _get_current_state(self) -> Dict[str, Any]:
        """
        Get current simulation state for strategy decision-making.
        
        Returns:
            Dictionary with current state information
        """
        return {
            'remaining_uavs': self.swarm.get_alive_count(),
            'remaining_kinetic': self.defense.kinetic_remaining,
            'remaining_de': self.defense.de_remaining,
            'cumulative_cost': self.defense.total_cost_spent,
            'nearest_distance': self.swarm.get_nearest_threat_distance(),
            'step': self.step_count
        }
    
    def _compile_results(self) -> Dict[str, Any]:
        """
        Compile comprehensive results from the simulation.
        
        Returns:
            Dictionary containing all simulation metrics
        """
        # Get statistics from components
        defense_stats = self.defense.get_engagement_statistics()
        swarm_status = self.swarm.get_status_summary()
        
        # Calculate key metrics
        attack_cost = swarm_status['total_cost']
        defense_cost = self.defense.total_cost_spent
        total_cost = defense_cost + self.total_penetration_cost
        
        # Cost-exchange ratio (lower is better for defender)
        if attack_cost > 0:
            cost_exchange_ratio = total_cost / attack_cost
        else:
            cost_exchange_ratio = float('inf') if total_cost > 0 else 0.0
        
        return {
            # Swarm metrics
            'swarm_size': self.swarm.initial_size,
            'uavs_destroyed': self.swarm.destroyed_count,
            'uavs_penetrated': self.swarm.penetration_count,
            'penetration_rate': swarm_status['penetration_rate'],
            
            # Cost metrics
            'defense_cost': defense_cost,
            'attack_cost': attack_cost,
            'penetration_cost': self.total_penetration_cost,
            'total_cost': total_cost,
            'cost_exchange_ratio': cost_exchange_ratio,
            
            # Weapon usage
            'kinetic_fired': defense_stats['kinetic_fired'],
            'de_fired': defense_stats['de_fired'],
            'total_shots_fired': defense_stats['total_fired'],
            
            # Weapon effectiveness
            'kinetic_hits': defense_stats['kinetic_hits'],
            'de_hits': defense_stats['de_hits'],
            'total_hits': defense_stats['total_hits'],
            'kinetic_accuracy': defense_stats['kinetic_accuracy'],
            'de_accuracy': defense_stats['de_accuracy'],
            'overall_accuracy': defense_stats['overall_accuracy'],
            
            # Simulation metadata
            'steps': self.step_count,
            'termination_reason': self.termination_reason,
            'simulation_completed': self.simulation_completed
        }
    
    def _print_results(self, results: Dict[str, Any]):
        """
        Print formatted simulation results.
        
        Args:
            results: Results dictionary from _compile_results()
        """
        print(f"\nSWARM RESULTS:")
        print(f"  Initial size: {results['swarm_size']}")
        print(f"  Destroyed: {results['uavs_destroyed']} ({results['uavs_destroyed']/results['swarm_size']*100:.1f}%)")
        print(f"  Penetrated: {results['uavs_penetrated']} ({results['penetration_rate']*100:.1f}%)")
        
        print(f"\nWEAPON USAGE:")
        print(f"  Kinetic: {results['kinetic_fired']} fired, {results['kinetic_hits']} hits", end="")
        if results['kinetic_accuracy'] is not None:
            print(f" ({results['kinetic_accuracy']*100:.1f}%)")
        else:
            print(" (N/A)")
        
        print(f"  DE: {results['de_fired']} fired, {results['de_hits']} hits", end="")
        if results['de_accuracy'] is not None:
            print(f" ({results['de_accuracy']*100:.1f}%)")
        else:
            print(" (N/A)")
        
        print(f"\nCOST ANALYSIS:")
        print(f"  Attack cost: ${results['attack_cost']:,.0f}")
        print(f"  Defense cost: ${results['defense_cost']:,.0f}")
        print(f"  Penetration cost: ${results['penetration_cost']:,.0f}")
        print(f"  Total cost: ${results['total_cost']:,.0f}")
        print(f"  Cost-Exchange Ratio: {results['cost_exchange_ratio']:.2f}")
        
        print(f"\nSIMULATION:")
        print(f"  Steps: {results['steps']}")
        print(f"  Termination: {results['termination_reason']}")
    
    def reset(
        self,
        swarm_size: Optional[int] = None,
        kinetic_quantity: Optional[int] = None,
        de_quantity: Optional[int] = None
    ):
        """
        Reset simulation to initial state for reuse.
        
        Args:
            swarm_size: New swarm size (if None, uses current)
            kinetic_quantity: New kinetic count (if None, uses initial)
            de_quantity: New DE count (if None, uses initial)
        """
        self.swarm.reset(new_swarm_size=swarm_size)
        self.defense.reset(kinetic_quantity=kinetic_quantity, de_quantity=de_quantity)
        self.step_count = 0
        self.total_penetration_cost = 0.0
        self.simulation_completed = False
        self.termination_reason = None
    
    def __repr__(self) -> str:
        """String representation of the simulation."""
        return (
            f"CombatSimulation(swarm_size={self.swarm.initial_size}, "
            f"step={self.step_count}, completed={self.simulation_completed})"
        )


if __name__ == "__main__":
    """
    Test and demonstration code for CombatSimulation class.
    """
    print("=" * 70)
    print("COMBAT SIMULATION MODULE - DEMONSTRATION")
    print("=" * 70)
    
    # Test 1: Simple greedy strategy (always use cheapest available weapon)
    print("\n[Test 1] Greedy strategy - always use cheapest weapon:")
    
    def greedy_strategy(state: Dict[str, Any]) -> WeaponType:
        """Always fire cheapest available weapon."""
        if state['remaining_de'] > 0:
            return WeaponType.DIRECTED_ENERGY
        elif state['remaining_kinetic'] > 0:
            return WeaponType.KINETIC
        else:
            return WeaponType.SKIP
    
    sim1 = CombatSimulation(swarm_size=50, random_seed=42)
    results1 = sim1.run_simulation(greedy_strategy, verbose=True)
    
    # Test 2: Conservative strategy (use kinetic first, save DE)
    print("\n[Test 2] Conservative strategy - prioritize kinetic:")
    
    def conservative_strategy(state: Dict[str, Any]) -> WeaponType:
        """Use expensive kinetic first, save cheap DE for backup."""
        if state['remaining_kinetic'] > 0:
            return WeaponType.KINETIC
        elif state['remaining_de'] > 0:
            return WeaponType.DIRECTED_ENERGY
        else:
            return WeaponType.SKIP
    
    sim2 = CombatSimulation(swarm_size=50, random_seed=42)
    results2 = sim2.run_simulation(conservative_strategy, verbose=False)
    
    print(f"\nConservative Results:")
    print(f"  Penetration rate: {results2['penetration_rate']*100:.1f}%")
    print(f"  Cost-exchange ratio: {results2['cost_exchange_ratio']:.2f}")
    print(f"  Total cost: ${results2['total_cost']:,.0f}")
    
    # Test 3: Threshold strategy (skip weak threats, engage close ones)
    print("\n[Test 3] Threshold strategy - engage only close threats:")
    
    def threshold_strategy(state: Dict[str, Any]) -> WeaponType:
        """Only engage UAVs within 20km of target."""
        if state['nearest_distance'] is None:
            return WeaponType.SKIP
        
        if state['nearest_distance'] <= 20.0:
            # Close threat - use any available weapon
            if state['remaining_de'] > 0:
                return WeaponType.DIRECTED_ENERGY
            elif state['remaining_kinetic'] > 0:
                return WeaponType.KINETIC
        
        return WeaponType.SKIP
    
    sim3 = CombatSimulation(swarm_size=50, random_seed=42)
    results3 = sim3.run_simulation(threshold_strategy, verbose=False)
    
    print(f"\nThreshold Results:")
    print(f"  Penetration rate: {results3['penetration_rate']*100:.1f}%")
    print(f"  Cost-exchange ratio: {results3['cost_exchange_ratio']:.2f}")
    print(f"  Total cost: ${results3['total_cost']:,.0f}")
    
    # Test 4: Adaptive strategy (switch based on inventory)
    print("\n[Test 4] Adaptive strategy - switch based on ammo:")
    
    def adaptive_strategy(state: Dict[str, Any]) -> WeaponType:
        """Use DE until low, then switch to kinetic."""
        de_ratio = state['remaining_de'] / 100  # Assuming initial 100
        
        if de_ratio > 0.3:  # More than 30% DE remaining
            if state['remaining_de'] > 0:
                return WeaponType.DIRECTED_ENERGY
        
        if state['remaining_kinetic'] > 0:
            return WeaponType.KINETIC
        elif state['remaining_de'] > 0:
            return WeaponType.DIRECTED_ENERGY
        
        return WeaponType.SKIP
    
    sim4 = CombatSimulation(swarm_size=50, random_seed=42)
    results4 = sim4.run_simulation(adaptive_strategy, verbose=False)
    
    print(f"\nAdaptive Results:")
    print(f"  Penetration rate: {results4['penetration_rate']*100:.1f}%")
    print(f"  Cost-exchange ratio: {results4['cost_exchange_ratio']:.2f}")
    print(f"  Total cost: ${results4['total_cost']:,.0f}")
    
    # Test 5: Comparing strategies on larger swarm
    print("\n[Test 5] Strategy comparison on large swarm (200 UAVs):")
    
    strategies = {
        'Greedy (DE first)': greedy_strategy,
        'Conservative (Kinetic first)': conservative_strategy,
        'Threshold (Close only)': threshold_strategy,
        'Adaptive (Switch)': adaptive_strategy
    }
    
    comparison_results = []
    for name, strategy in strategies.items():
        sim = CombatSimulation(swarm_size=200, random_seed=999)
        results = sim.run_simulation(strategy, verbose=False)
        comparison_results.append({
            'strategy': name,
            'penetration_rate': results['penetration_rate'],
            'cost_exchange_ratio': results['cost_exchange_ratio'],
            'total_cost': results['total_cost']
        })
    
    print(f"\n{'Strategy':<30} {'Penetration':>12} {'Cost-Exch':>12} {'Total Cost':>15}")
    print(f"{'-'*30} {'-'*12} {'-'*12} {'-'*15}")
    for r in comparison_results:
        print(f"{r['strategy']:<30} {r['penetration_rate']*100:>11.1f}% "
              f"{r['cost_exchange_ratio']:>12.2f} ${r['total_cost']:>14,.0f}")
    
    # Test 6: Reset functionality
    print("\n[Test 6] Reset and rerun simulation:")
    sim5 = CombatSimulation(swarm_size=30, random_seed=100)
    results_first = sim5.run_simulation(greedy_strategy, verbose=False)
    
    print(f"  First run: {results_first['uavs_penetrated']} penetrations, "
          f"${results_first['total_cost']:,.0f} cost")
    
    sim5.reset()
    results_second = sim5.run_simulation(greedy_strategy, verbose=False)
    
    print(f"  Second run: {results_second['uavs_penetrated']} penetrations, "
          f"${results_second['total_cost']:,.0f} cost")
    
    # Test 7: Edge case - no weapons
    print("\n[Test 7] Edge case - defense with no weapons:")
    
    sim6 = CombatSimulation(swarm_size=10, random_seed=200)
    sim6.defense.kinetic_remaining = 0
    sim6.defense.de_remaining = 0
    
    results6 = sim6.run_simulation(greedy_strategy, verbose=False)
    print(f"  All {results6['swarm_size']} UAVs penetrated: {results6['uavs_penetrated']}")
    print(f"  Penetration rate: {results6['penetration_rate']*100:.0f}%")
    
    # Test 8: Edge case - overwhelming weapons
    print("\n[Test 8] Edge case - overwhelming defense:")
    
    sim7 = CombatSimulation(swarm_size=10, random_seed=300)
    sim7.defense.kinetic_remaining = 1000
    sim7.defense.de_remaining = 1000
    
    results7 = sim7.run_simulation(conservative_strategy, verbose=False)
    print(f"  UAVs destroyed: {results7['uavs_destroyed']}")
    print(f"  UAVs penetrated: {results7['uavs_penetrated']}")
    print(f"  Penetration rate: {results7['penetration_rate']*100:.1f}%")
    
    # Test 9: Error handling
    print("\n[Test 9] Error handling:")
    
    def bad_strategy(state):
        return "invalid_weapon"  # Should raise error
    
    try:
        sim8 = CombatSimulation(swarm_size=10, random_seed=400)
        sim8.run_simulation(bad_strategy, verbose=False)
    except ValueError as e:
        print(f"  ✓ Caught expected error: {e}")
    
    try:
        bad_sim = CombatSimulation(distance_per_step=-5)
    except ValueError as e:
        print(f"  ✓ Caught expected error: {e}")
    
    print("\n" + "=" * 70)
    print("ALL TESTS COMPLETED SUCCESSFULLY")
    print("=" * 70)
