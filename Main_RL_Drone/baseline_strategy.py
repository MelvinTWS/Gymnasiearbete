"""
Baseline Strategy Module

This module implements the baseline nearest-threat-first heuristic strategy
for comparison against the RL agent. This is the traditional rule-based approach.

Strategy Logic:
1. If directed energy available → Use directed energy
2. Elif kinetic available → Use kinetic
3. Else → Skip (conserve resources)

This greedy heuristic prioritizes cheap weapons and always engages the nearest
threat without considering future scenarios or resource optimization.

Author: Master's Thesis Project
Date: January 2026
"""

from typing import Dict, Any
from defense_system import WeaponType


def nearest_threat_first_strategy(state: Dict[str, Any]) -> WeaponType:
    """
    Baseline strategy: Always engage nearest threat with cheapest available weapon.
    
    This is the traditional heuristic approach used as a benchmark for comparing
    the RL agent's performance. The strategy is simple and greedy:
    - Prioritizes cheap directed energy weapons
    - Falls back to expensive kinetic interceptors
    - Never intentionally skips (only when no weapons remain)
    
    This strategy does NOT consider:
    - Future resource needs
    - Cost-benefit analysis
    - Threat distance or urgency
    - Remaining swarm size
    
    Args:
        state: Dictionary containing current simulation state with keys:
            - remaining_uavs (int): Number of alive UAVs
            - remaining_kinetic (int): Kinetic interceptors available
            - remaining_de (int): Directed energy shots available
            - cumulative_cost (float): Total cost spent so far
            - nearest_distance (float or None): Distance to nearest UAV in km
            - step (int): Current simulation step number
            
    Returns:
        WeaponType: Action to take (DIRECTED_ENERGY, KINETIC, or SKIP)
        
    Examples:
        >>> state = {'remaining_de': 50, 'remaining_kinetic': 10, ...}
        >>> action = nearest_threat_first_strategy(state)
        >>> print(action)  # WeaponType.DIRECTED_ENERGY
        
        >>> state = {'remaining_de': 0, 'remaining_kinetic': 10, ...}
        >>> action = nearest_threat_first_strategy(state)
        >>> print(action)  # WeaponType.KINETIC
        
        >>> state = {'remaining_de': 0, 'remaining_kinetic': 0, ...}
        >>> action = nearest_threat_first_strategy(state)
        >>> print(action)  # WeaponType.SKIP
    """
    # Priority 1: Use cheap directed energy if available
    if state['remaining_de'] > 0:
        return WeaponType.DIRECTED_ENERGY
    
    # Priority 2: Use expensive kinetic as fallback
    elif state['remaining_kinetic'] > 0:
        return WeaponType.KINETIC
    
    # Priority 3: No weapons left, must skip
    else:
        return WeaponType.SKIP


def get_baseline_strategy():
    """
    Factory function to get the baseline strategy.
    
    Useful for programmatic access and consistency across codebase.
    
    Returns:
        Callable strategy function
        
    Example:
        >>> strategy = get_baseline_strategy()
        >>> from combat_simulation import CombatSimulation
        >>> sim = CombatSimulation(swarm_size=100)
        >>> results = sim.run_simulation(strategy)
    """
    return nearest_threat_first_strategy


def get_strategy_description() -> str:
    """
    Get a human-readable description of the baseline strategy.
    
    Returns:
        String description of the strategy logic
    """
    return """
    Nearest-Threat-First Baseline Strategy
    ======================================
    
    Logic:
    1. IF directed energy available → FIRE directed energy
    2. ELIF kinetic available → FIRE kinetic interceptor
    3. ELSE → SKIP (no weapons remain)
    
    Characteristics:
    - Greedy: Always fires if weapons available
    - Cost-naive: Doesn't optimize for cost-efficiency
    - Non-adaptive: Same logic regardless of scenario
    - Deterministic: Same inputs always produce same output
    
    Purpose:
    Serves as traditional rule-based baseline for comparing
    against RL agent's learned strategy.
    """


if __name__ == "__main__":
    """
    Test and demonstration code for baseline strategy.
    """
    print("=" * 70)
    print("BASELINE STRATEGY MODULE - DEMONSTRATION")
    print("=" * 70)
    
    print(get_strategy_description())
    
    # Test 1: Strategy behavior with different states
    print("\n[Test 1] Strategy behavior with different ammunition states:")
    
    test_states = [
        {
            'remaining_uavs': 100,
            'remaining_kinetic': 50,
            'remaining_de': 100,
            'cumulative_cost': 0,
            'nearest_distance': 50.0,
            'step': 1
        },
        {
            'remaining_uavs': 100,
            'remaining_kinetic': 50,
            'remaining_de': 0,  # DE depleted
            'cumulative_cost': 1000000,
            'nearest_distance': 30.0,
            'step': 50
        },
        {
            'remaining_uavs': 100,
            'remaining_kinetic': 0,  # Both depleted
            'remaining_de': 0,
            'cumulative_cost': 10000000,
            'nearest_distance': 10.0,
            'step': 100
        },
        {
            'remaining_uavs': 50,
            'remaining_kinetic': 0,
            'remaining_de': 5,  # Low on DE, no kinetic
            'cumulative_cost': 50000000,
            'nearest_distance': 5.0,
            'step': 200
        }
    ]
    
    for i, state in enumerate(test_states, 1):
        action = nearest_threat_first_strategy(state)
        print(f"\n  State {i}:")
        print(f"    Kinetic: {state['remaining_kinetic']}, DE: {state['remaining_de']}")
        print(f"    UAVs: {state['remaining_uavs']}, Distance: {state['nearest_distance']} km")
        print(f"    Action: {action.value}")
    
    # Test 2: Run full simulation with baseline strategy
    print("\n[Test 2] Full simulation with baseline strategy:")
    
    from combat_simulation import CombatSimulation
    
    sim = CombatSimulation(swarm_size=100, random_seed=42)
    results = sim.run_simulation(nearest_threat_first_strategy, verbose=False)
    
    print(f"\n  Simulation Results:")
    print(f"    Swarm size: {results['swarm_size']}")
    print(f"    UAVs destroyed: {results['uavs_destroyed']}")
    print(f"    UAVs penetrated: {results['uavs_penetrated']}")
    print(f"    Penetration rate: {results['penetration_rate']*100:.2f}%")
    print(f"    Defense cost: ${results['defense_cost']:,.0f}")
    print(f"    Penetration cost: ${results['penetration_cost']:,.0f}")
    print(f"    Total cost: ${results['total_cost']:,.0f}")
    print(f"    Cost-exchange ratio: {results['cost_exchange_ratio']:.2f}")
    print(f"    Kinetic fired: {results['kinetic_fired']} (hits: {results['kinetic_hits']})")
    print(f"    DE fired: {results['de_fired']} (hits: {results['de_hits']})")
    print(f"    Steps: {results['steps']}")
    
    # Test 3: Multiple runs with different swarm sizes
    print("\n[Test 3] Baseline performance across different swarm sizes:")
    
    swarm_sizes = [50, 100, 200, 300, 400]
    
    print(f"\n  {'Swarm Size':>12} {'Penetrated':>12} {'Pen Rate':>10} {'Cost-Exch':>12} {'Total Cost':>15}")
    print(f"  {'-'*12} {'-'*12} {'-'*10} {'-'*12} {'-'*15}")
    
    for size in swarm_sizes:
        sim = CombatSimulation(swarm_size=size, random_seed=123)
        results = sim.run_simulation(nearest_threat_first_strategy, verbose=False)
        
        print(f"  {size:>12} {results['uavs_penetrated']:>12} "
              f"{results['penetration_rate']*100:>9.1f}% "
              f"{results['cost_exchange_ratio']:>12.2f} "
              f"${results['total_cost']:>14,.0f}")
    
    # Test 4: Consistency check (deterministic behavior)
    print("\n[Test 4] Consistency check - strategy is deterministic:")
    
    state = {
        'remaining_uavs': 150,
        'remaining_kinetic': 25,
        'remaining_de': 50,
        'cumulative_cost': 5000000,
        'nearest_distance': 75.0,
        'step': 10
    }
    
    actions = []
    for _ in range(10):
        action = nearest_threat_first_strategy(state)
        actions.append(action)
    
    all_same = all(a == actions[0] for a in actions)
    print(f"  Same state tested 10 times: All actions identical? {all_same}")
    print(f"  Action: {actions[0].value}")
    
    # Test 5: Compare with other strategies
    print("\n[Test 5] Baseline vs alternative strategies (100 UAVs):")
    
    def kinetic_first_strategy(state):
        """Alternative: prioritize kinetic over DE."""
        if state['remaining_kinetic'] > 0:
            return WeaponType.KINETIC
        elif state['remaining_de'] > 0:
            return WeaponType.DIRECTED_ENERGY
        else:
            return WeaponType.SKIP
    
    def conservative_strategy(state):
        """Alternative: skip some threats."""
        if state['nearest_distance'] is not None and state['nearest_distance'] > 50:
            return WeaponType.SKIP
        elif state['remaining_de'] > 0:
            return WeaponType.DIRECTED_ENERGY
        elif state['remaining_kinetic'] > 0:
            return WeaponType.KINETIC
        else:
            return WeaponType.SKIP
    
    strategies = {
        'Baseline (DE first)': nearest_threat_first_strategy,
        'Kinetic first': kinetic_first_strategy,
        'Conservative (skip far)': conservative_strategy
    }
    
    print(f"\n  {'Strategy':<25} {'Pen Rate':>10} {'Cost-Exch':>12} {'Total Cost':>15}")
    print(f"  {'-'*25} {'-'*10} {'-'*12} {'-'*15}")
    
    for name, strategy in strategies.items():
        sim = CombatSimulation(swarm_size=100, random_seed=999)
        results = sim.run_simulation(strategy, verbose=False)
        
        print(f"  {name:<25} {results['penetration_rate']*100:>9.1f}% "
              f"{results['cost_exchange_ratio']:>12.2f} "
              f"${results['total_cost']:>14,.0f}")
    
    # Test 6: Edge cases
    print("\n[Test 6] Edge case handling:")
    
    # Empty state (no UAVs, no weapons)
    empty_state = {
        'remaining_uavs': 0,
        'remaining_kinetic': 0,
        'remaining_de': 0,
        'cumulative_cost': 0,
        'nearest_distance': None,
        'step': 0
    }
    action = nearest_threat_first_strategy(empty_state)
    print(f"  Empty state (no UAVs, no weapons): {action.value}")
    
    # Only kinetic available
    kinetic_only_state = {
        'remaining_uavs': 100,
        'remaining_kinetic': 10,
        'remaining_de': 0,
        'cumulative_cost': 0,
        'nearest_distance': 20.0,
        'step': 5
    }
    action = nearest_threat_first_strategy(kinetic_only_state)
    print(f"  Only kinetic available: {action.value}")
    
    # Only DE available
    de_only_state = {
        'remaining_uavs': 100,
        'remaining_kinetic': 0,
        'remaining_de': 10,
        'cumulative_cost': 0,
        'nearest_distance': 20.0,
        'step': 5
    }
    action = nearest_threat_first_strategy(de_only_state)
    print(f"  Only DE available: {action.value}")
    
    # Test 7: Factory function
    print("\n[Test 7] Factory function test:")
    strategy_func = get_baseline_strategy()
    print(f"  Strategy function retrieved: {strategy_func.__name__}")
    
    test_state = {
        'remaining_uavs': 50,
        'remaining_kinetic': 10,
        'remaining_de': 20,
        'cumulative_cost': 1000000,
        'nearest_distance': 40.0,
        'step': 15
    }
    action = strategy_func(test_state)
    print(f"  Action from factory function: {action.value}")
    
    print("\n" + "=" * 70)
    print("ALL TESTS COMPLETED SUCCESSFULLY")
    print("=" * 70)
    print("\nBaseline strategy ready for evaluation against RL agent!")
