"""
Baseline Strategy Module

This module implements the baseline nearest-threat-first heuristic strategy
for comparison against the RL agent. This is the traditional rule-based approach.

Strategy Logic:
1. If kinetic available → Use kinetic
2. Elif directed energy available → Use directed energy
3. Else → Skip (conserve resources)

This greedy heuristic prioritizes reliability first, then cost, and always engages
the nearest threat without considering future scenarios or resource optimization.
"""

from typing import Dict, Any
from defense_system import WeaponType


def adaptive_10percent_baseline(state: Dict[str, Any]) -> WeaponType:
    """
    Adaptive strategy targeting ≤10% penetration while minimizing cost.
    
    Strategy logic:
    - Target: Keep penetration at or below 10% of initial swarm
    - Cost-conscious: Prefer cheap DE when ahead of target
    - Aggressive: Switch to expensive kinetic when falling behind target
    - Adaptive: Adjusts based on current progress and remaining time
    
    This represents a "Ukrainian-style" maximum interception approach
    with cost awareness.
    """
    initial_swarm = state.get('initial_swarm_size', 300)
    remaining_uavs = state.get('uavs_remaining', 0)
    current_step = state.get('current_step', 0)
    max_steps = state.get('max_steps', 20)
    kinetic_ammo = state.get('kinetic_remaining', 0)
    de_ammo = state.get('de_remaining', 0)
    penetrated = state.get('penetrated_count', 0)
    passed = state.get('passed_count', 0)
    
    # Calculate 10% penetration target
    max_acceptable_penetrations = initial_swarm * 0.10
    
    # Total "lost" drones (penetrated + passed/missed)
    total_losses = penetrated + passed
    
    # Penetration budget remaining
    penetration_budget = max(0, max_acceptable_penetrations - total_losses)
    
    # If no UAVs left, skip
    if remaining_uavs == 0:
        return WeaponType.SKIP
    
    # Critical: If we're over budget or at limit, fire everything
    if total_losses >= max_acceptable_penetrations:
        # Already exceeded target, use all firepower
        if kinetic_ammo > 0:
            return WeaponType.KINETIC
        elif de_ammo > 0:
            return WeaponType.DIRECTED_ENERGY
        else:
            return WeaponType.SKIP
    
    # If remaining UAVs exceed our penetration budget, must engage aggressively
    if remaining_uavs > penetration_budget:
        # Need to reduce UAV count, prefer reliable kinetic
        if kinetic_ammo > 0:
            return WeaponType.KINETIC
        elif de_ammo > 0:
            return WeaponType.DIRECTED_ENERGY
        else:
            return WeaponType.SKIP
    
    # We have budget cushion - use cheaper DE when available
    if de_ammo > 0:
        return WeaponType.DIRECTED_ENERGY
    
    # Fallback to kinetic
    if kinetic_ammo > 0:
        return WeaponType.KINETIC
    
    # No ammo left
    return WeaponType.SKIP


def nearest_threat_first_strategy(state: Dict[str, Any]) -> WeaponType:
    """
    Baseline strategy: Always engage nearest threat with cheapest available weapon.
    
    This is the traditional heuristic approach used as a benchmark for comparing
    the RL agent's performance. The strategy is simple and greedy:
    - Prioritizes reliable kinetic interceptors
    - Falls back to cheaper directed energy weapons
    - Never intentionally skips (only when no weapons remain)
    
    This strategy does NOT consider:
    - Future resource needs
    - Cost-benefit analysis
    - Threat distance or urgency
    - Remaining swarm size
    """
    # Priority 1: Use reliable kinetic if available
    if state['remaining_kinetic'] > 0:
        return WeaponType.KINETIC
    
    # Priority 2: Use cheaper directed energy as fallback
    elif state['remaining_de'] > 0:
        return WeaponType.DIRECTED_ENERGY
    
    # Priority 3: No weapons left, must skip
    else:
        return WeaponType.SKIP


def get_baseline_strategy():
    """
    Factory function to get the baseline strategy.
    
    Useful for programmatic access and consistency across codebase.
    """
    return nearest_threat_first_strategy


def get_strategy_description() -> str:
    """
    Get a human-readable description of the baseline strategy.
    """
    return """
    Nearest-Threat-First Baseline Strategy
    ======================================
    
    Logic:
    1. IF kinetic available → FIRE kinetic interceptor
    2. ELIF directed energy available → FIRE directed energy
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
    # Test strategy with a sample state
    state = {
        "initial_swarm_size": 100, "uavs_remaining": 80,
        "current_step": 10, "max_steps": 200,
        "kinetic_remaining": 600, "de_remaining": 100,
        "penetrated_count": 3, "passed_count": 2
    }
    action = adaptive_10percent_baseline(state)
    print(f"Action: {action}")
