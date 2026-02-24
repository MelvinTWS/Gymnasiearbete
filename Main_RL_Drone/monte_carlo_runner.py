"""
Monte Carlo Scenario Runner Module

This module provides batch execution of combat simulations for statistical
evaluation. Runs multiple scenarios with different random seeds to generate
robust performance metrics.
"""

import numpy as np
import pandas as pd
from typing import Callable, Optional, Dict, Any, List
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
import time

from combat_simulation import CombatSimulation
from defense_system import WeaponType


def run_scenarios(
    strategy: Callable[[Dict[str, Any]], WeaponType],
    num_scenarios: int = 1000,
    swarm_size_range: tuple = (100, 500),
    kinetic_quantity: int = 800,
    de_quantity: int = 150,
    random_seed: Optional[int] = None,
    verbose: bool = True,
    strategy_name: str = "Unknown"
) -> pd.DataFrame:
    """
    Run multiple Monte Carlo scenarios with different random configurations.
    
    Each scenario uses a different random seed to generate:
    - Random swarm size (uniform distribution within range)
    - Random weapon success rates (stochastic)
    - Random engagement outcomes
    """
    # Validate inputs
    if num_scenarios <= 0:
        raise ValueError(f"Number of scenarios must be positive, got {num_scenarios}")
    
    if len(swarm_size_range) != 2:
        raise ValueError(f"Swarm size range must be (min, max), got {swarm_size_range}")
    
    min_swarm, max_swarm = swarm_size_range
    if min_swarm <= 0 or max_swarm < min_swarm:
        raise ValueError(f"Invalid swarm size range: {swarm_size_range}")
    
    if kinetic_quantity < 0 or de_quantity < 0:
        raise ValueError(f"Weapon quantities cannot be negative")
    
    # Set base random seed
    if random_seed is None:
        random_seed = int(time.time())
    
    np.random.seed(random_seed)
    
    # Generate unique seeds for each scenario
    scenario_seeds = np.random.randint(0, 2**31 - 1, size=num_scenarios)
    
    # Initialize results storage
    results_list = []
    
    # Progress tracking
    if verbose:
        print(f"\n{'='*70}")
        print(f"MONTE CARLO SIMULATION - {strategy_name}")
        print(f"{'='*70}")
        print(f"Scenarios: {num_scenarios}")
        print(f"Swarm size range: {min_swarm}-{max_swarm} UAVs")
        print(f"Weapons: {kinetic_quantity} kinetic, {de_quantity} DE")
        print(f"Base seed: {random_seed}")
        print(f"{'='*70}\n")
        
        pbar = tqdm(total=num_scenarios, desc="Running scenarios", unit="scenario")
    
    # Run scenarios
    start_time = time.time()
    
    for scenario_id in range(num_scenarios):
        seed = int(scenario_seeds[scenario_id])
        
        # Set seed for this scenario
        np.random.seed(seed)
        
        # Randomly determine swarm size for this scenario
        swarm_size = np.random.randint(min_swarm, max_swarm + 1)
        
        # Create and run simulation
        sim = CombatSimulation(
            swarm_size=swarm_size,
            random_seed=seed,
            distance_per_step=5.0,
            penalty_random_range=(1_000_000.0, 6_000_000.0),
            critical_penalty=10_000_000.0,
            critical_probability=0.2,
            max_steps=200
        )
        
        # Ensure correct weapon quantities
        sim.defense.reset(kinetic_quantity=kinetic_quantity, de_quantity=de_quantity)
        
        # Run the simulation
        try:
            result = sim.run_simulation(strategy, verbose=False)
            
            # Add metadata
            result['scenario_id'] = scenario_id
            result['random_seed'] = seed
            result['strategy_name'] = strategy_name
            result['timestamp'] = datetime.now().isoformat()
            
            results_list.append(result)
            
        except Exception as e:
            if verbose:
                pbar.close()
            raise RuntimeError(
                f"Scenario {scenario_id} (seed={seed}) failed: {e}"
            ) from e
        
        if verbose:
            pbar.update(1)
    
    if verbose:
        pbar.close()
    
    elapsed_time = time.time() - start_time
    
    # Convert to DataFrame
    df = pd.DataFrame(results_list)
    
    # Reorder columns for better readability
    column_order = [
        'scenario_id', 'strategy_name', 'random_seed', 'swarm_size',
        'uavs_destroyed', 'uavs_penetrated', 'penetration_rate',
        'defense_cost', 'attack_cost', 'penetration_cost', 'total_cost',
        'cost_exchange_ratio', 'kinetic_fired', 'de_fired',
        'kinetic_hits', 'de_hits', 'kinetic_accuracy', 'de_accuracy',
        'overall_accuracy', 'steps', 'termination_reason',
        'timestamp', 'simulation_completed', 'total_shots_fired'
    ]
    
    # Only include columns that exist
    column_order = [col for col in column_order if col in df.columns]
    other_columns = [col for col in df.columns if col not in column_order]
    df = df[column_order + other_columns]
    
    # Print summary statistics
    if verbose:
        print(f"\n{'='*70}")
        print(f"SCENARIOS COMPLETED")
        print(f"{'='*70}")
        print(f"Total time: {elapsed_time:.2f} seconds")
        print(f"Average time per scenario: {elapsed_time/num_scenarios:.3f} seconds")
        print(f"\n{_get_summary_statistics(df)}")
        print(f"{'='*70}\n")
    
    return df


def _get_summary_statistics(df: pd.DataFrame) -> str:
    """
    Generate summary statistics from scenario results.
    """
    stats = []
    stats.append("SUMMARY STATISTICS")
    stats.append("-" * 70)
    
    # Swarm metrics
    stats.append(f"\nSwarm Size:")
    stats.append(f"  Mean: {df['swarm_size'].mean():.1f}")
    stats.append(f"  Std: {df['swarm_size'].std():.1f}")
    stats.append(f"  Range: {df['swarm_size'].min()}-{df['swarm_size'].max()}")
    
    # Penetration metrics
    stats.append(f"\nPenetration Rate:")
    stats.append(f"  Mean: {df['penetration_rate'].mean()*100:.2f}%")
    stats.append(f"  Std: {df['penetration_rate'].std()*100:.2f}%")
    stats.append(f"  Median: {df['penetration_rate'].median()*100:.2f}%")
    stats.append(f"  Range: {df['penetration_rate'].min()*100:.2f}%-{df['penetration_rate'].max()*100:.2f}%")
    
    # Cost-exchange ratio
    stats.append(f"\nCost-Exchange Ratio:")
    stats.append(f"  Mean: {df['cost_exchange_ratio'].mean():.2f}")
    stats.append(f"  Std: {df['cost_exchange_ratio'].std():.2f}")
    stats.append(f"  Median: {df['cost_exchange_ratio'].median():.2f}")
    stats.append(f"  Range: {df['cost_exchange_ratio'].min():.2f}-{df['cost_exchange_ratio'].max():.2f}")
    
    # Costs
    stats.append(f"\nTotal Cost:")
    stats.append(f"  Mean: ${df['total_cost'].mean():,.0f}")
    stats.append(f"  Median: ${df['total_cost'].median():,.0f}")
    
    stats.append(f"\nDefense Cost:")
    stats.append(f"  Mean: ${df['defense_cost'].mean():,.0f}")
    stats.append(f"  Median: ${df['defense_cost'].median():,.0f}")
    
    # Weapon usage
    stats.append(f"\nWeapon Usage:")
    stats.append(f"  Kinetic fired: {df['kinetic_fired'].mean():.1f} avg ({df['kinetic_fired'].sum()} total)")
    stats.append(f"  DE fired: {df['de_fired'].mean():.1f} avg ({df['de_fired'].sum()} total)")
    
    # Accuracy (handle None values)
    kinetic_acc = df['kinetic_accuracy'].dropna()
    de_acc = df['de_accuracy'].dropna()
    overall_acc = df['overall_accuracy'].dropna()
    
    if len(kinetic_acc) > 0:
        stats.append(f"\nKinetic Accuracy: {kinetic_acc.mean()*100:.2f}% (n={len(kinetic_acc)})")
    if len(de_acc) > 0:
        stats.append(f"DE Accuracy: {de_acc.mean()*100:.2f}% (n={len(de_acc)})")
    if len(overall_acc) > 0:
        stats.append(f"Overall Accuracy: {overall_acc.mean()*100:.2f}% (n={len(overall_acc)})")
    
    return "\n".join(stats)


def save_results(
    df: pd.DataFrame,
    filepath: str,
    include_timestamp: bool = True
) -> Path:
    """
    Save scenario results to CSV file.
    """
    filepath = Path(filepath)
    
    # Ensure .csv extension
    if filepath.suffix != '.csv':
        filepath = filepath.with_suffix('.csv')
    
    # Add timestamp if requested
    if include_timestamp:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        stem = filepath.stem
        filepath = filepath.with_name(f"{stem}_{timestamp}.csv")
    
    # Create directory if needed
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    # Save to CSV
    df.to_csv(filepath, index=False)
    
    return filepath


def load_results(filepath: str) -> pd.DataFrame:
    """
    Load scenario results from CSV file.
    """
    return pd.read_csv(filepath)


def compare_strategies(
    df1: pd.DataFrame,
    df2: pd.DataFrame,
    name1: str = "Strategy 1",
    name2: str = "Strategy 2"
) -> str:
    """
    Compare results from two strategies.
    """
    comparison = []
    comparison.append(f"\n{'='*70}")
    comparison.append(f"STRATEGY COMPARISON: {name1} vs {name2}")
    comparison.append(f"{'='*70}")
    
    metrics = [
        ('penetration_rate', 'Penetration Rate', '%', 100),
        ('cost_exchange_ratio', 'Cost-Exchange Ratio', '', 1),
        ('total_cost', 'Total Cost', '$', 1),
        ('defense_cost', 'Defense Cost', '$', 1)
    ]
    
    comparison.append(f"\n{'Metric':<25} {name1:>20} {name2:>20}")
    comparison.append(f"{'-'*25} {'-'*20} {'-'*20}")
    
    for metric, label, prefix, multiplier in metrics:
        val1 = df1[metric].mean() * multiplier
        val2 = df2[metric].mean() * multiplier
        
        if prefix == '$':
            comparison.append(f"{label:<25} {prefix}{val1:>19,.0f} {prefix}{val2:>19,.0f}")
        elif prefix == '%':
            comparison.append(f"{label:<25} {val1:>19.2f}{prefix} {val2:>19.2f}{prefix}")
        else:
            comparison.append(f"{label:<25} {val1:>20.2f} {val2:>20.2f}")
    
    comparison.append(f"{'='*70}\n")
    
    return "\n".join(comparison)


if __name__ == "__main__":
    from baseline_strategy import adaptive_10percent_baseline
    results = run_scenarios(
        strategy=adaptive_10percent_baseline,
        num_scenarios=20,
        swarm_size_range=(50, 150),
        random_seed=42,
        verbose=True,
        strategy_name="Baseline"
    )
    print(results[["penetration_rate", "defense_cost", "cost_exchange_ratio"]].describe())
