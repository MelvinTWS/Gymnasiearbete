"""
Combat Simulation Module

This module implements the core combat simulation engine that manages
engagements between UAV swarms and defense systems.
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
    """
    
    # Default simulation parameters
    DEFAULT_DISTANCE_PER_STEP = 5.0  # km per step
    DEFAULT_PENETRATION_PENALTY = 1_000_000.0  # Base penalty (not used with sampling)
    DEFAULT_MAX_STEPS = 200  # Enough for aggressive baseline strategy
    
    def __init__(
        self,
        swarm: Optional[UAVSwarm] = None,
        defense: Optional[DefenseSystem] = None,
        swarm_size: Optional[int] = None,
        distance_per_step: float = DEFAULT_DISTANCE_PER_STEP,
        penetration_penalty: float = DEFAULT_PENETRATION_PENALTY,
        penalty_random_range: tuple = (1_000_000.0, 6_000_000.0),
        critical_penalty: float = 10_000_000.0,
        critical_probability: float = 0.2,
        max_steps: int = DEFAULT_MAX_STEPS,
        random_seed: Optional[int] = None
    ):
        """
        Initialize a combat simulation.
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
        self.penalty_random_range = penalty_random_range
        self.critical_penalty = critical_penalty
        self.critical_probability = critical_probability
        self.max_steps = int(max_steps)
        
        # Simulation state
        self.step_count = 0
        self.total_penetration_cost = 0.0
        self.total_miss_penalty_cost = 0.0
        self.simulation_completed = False
        self.termination_reason = None
    
    def _sample_penetration_penalty(self) -> float:
        """Sample penalty for a single penetrating or missed drone.
        
        Uses mix of critical (10M) and random infrastructure (1-6M).
        """
        if np.random.random() < self.critical_probability:
            return float(self.critical_penalty)
        low, high = self.penalty_random_range
        return float(np.random.uniform(low, high))
    
    def run_simulation(
        self,
        strategy: Callable[[Dict[str, Any]], WeaponType],
        verbose: bool = False
    ) -> Dict[str, Any]:
        """
        Run a complete combat simulation using the provided strategy.
        
        The strategy function receives current state and returns which weapon
        to fire. Simulation runs until all UAVs are neutralized or max steps reached.
        """
        # Reset state
        self.step_count = 0
        self.total_penetration_cost = 0.0
        self.total_miss_penalty_cost = 0.0
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
            
            # ONE-SHOT-PER-DRONE MECHANIC
            if action != WeaponType.SKIP and had_ammunition:
                # Get nearest untargeted UAV
                target_uav = self.swarm.get_nearest_untargeted()
                
                if target_uav is not None:
                    # Mark as targeted
                    self.swarm.targeted_uavs[target_uav] = True
                    
                    if success:
                        # HIT: Destroy the UAV
                        self.swarm.destroy_uav(target_uav)
                    else:
                        # MISS: Apply immediate penalty, mark as passed
                        self.swarm.mark_as_passed(target_uav)
                        miss_penalty = self._sample_penetration_penalty()
                        self.total_miss_penalty_cost += miss_penalty
            
            # Advance the swarm
            penetrated_this_step = self.swarm.advance_swarm(self.distance_per_step)
            
            # Sample penalty for each penetration this step
            if penetrated_this_step > 0:
                step_penetration_cost = sum(
                    self._sample_penetration_penalty() 
                    for _ in range(penetrated_this_step)
                )
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
        """
        return {
            'initial_swarm_size': self.swarm.initial_size,
            'uavs_remaining': self.swarm.get_alive_count(),
            'remaining_uavs': self.swarm.get_alive_count(),
            'remaining_kinetic': self.defense.kinetic_remaining,
            'remaining_de': self.defense.de_remaining,
            'kinetic_remaining': self.defense.kinetic_remaining,
            'de_remaining': self.defense.de_remaining,
            'cumulative_cost': self.defense.total_cost_spent,
            'nearest_distance': self.swarm.get_nearest_threat_distance(),
            'step': self.step_count,
            'current_step': self.step_count,
            'max_steps': self.max_steps,
            'penetrated_count': self.swarm.penetration_count,
            'passed_count': self.swarm.passed_count if hasattr(self.swarm, 'passed_count') else 0
        }
    
    def _compile_results(self) -> Dict[str, Any]:
        """
        Compile comprehensive results from the simulation.
        """
        # Get statistics from components
        defense_stats = self.defense.get_engagement_statistics()
        swarm_status = self.swarm.get_status_summary()
        
        # Calculate key metrics
        attack_cost = swarm_status['total_cost']
        defense_cost = self.defense.total_cost_spent
        total_cost = defense_cost + self.total_penetration_cost + self.total_miss_penalty_cost
        
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
            'miss_penalty_cost': self.total_miss_penalty_cost,
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
    from baseline_strategy import adaptive_10percent_baseline
    sim = CombatSimulation(swarm_size=100, random_seed=42)
    results = sim.run_simulation(adaptive_10percent_baseline, verbose=True)
    print(f"Penetration rate: {results['penetration_rate']*100:.1f}%")
    print(f"Defense cost: ${results['defense_cost']:,.0f}")
