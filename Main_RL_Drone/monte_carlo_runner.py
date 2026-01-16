"""
Monte Carlo Scenario Runner Module

This module provides batch execution of combat simulations for statistical
evaluation. Runs multiple scenarios with different random seeds to generate
robust performance metrics.

Author: Master's Thesis Project
Date: January 2026
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
    kinetic_quantity: int = 50,
    de_quantity: int = 100,
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
    
    Args:
        strategy: Strategy function that takes state dict and returns WeaponType
        num_scenarios: Number of scenarios to run
        swarm_size_range: Tuple of (min, max) swarm size (inclusive)
        kinetic_quantity: Initial kinetic interceptors per scenario
        de_quantity: Initial directed energy shots per scenario
        random_seed: Base random seed (if None, uses current time)
        verbose: If True, show progress bar and summary
        strategy_name: Name of strategy for reporting
        
    Returns:
        pandas DataFrame with one row per scenario containing all metrics:
            - scenario_id: Scenario number (0 to num_scenarios-1)
            - random_seed: Seed used for this scenario
            - swarm_size: Number of UAVs in swarm
            - uavs_destroyed: UAVs destroyed by weapons
            - uavs_penetrated: UAVs that reached target
            - penetration_rate: Fraction penetrated (0-1)
            - defense_cost: Cost of weapons fired
            - attack_cost: Cost of attacking swarm
            - penetration_cost: Penalty for penetrations
            - total_cost: defense_cost + penetration_cost
            - cost_exchange_ratio: total_cost / attack_cost
            - kinetic_fired: Kinetic interceptors used
            - de_fired: Directed energy shots used
            - kinetic_hits: Successful kinetic intercepts
            - de_hits: Successful DE intercepts
            - kinetic_accuracy: Kinetic hit rate (or None)
            - de_accuracy: DE hit rate (or None)
            - overall_accuracy: Combined hit rate (or None)
            - steps: Simulation steps
            - termination_reason: Why simulation ended
            - strategy_name: Name of strategy used
            - timestamp: When scenario was run
            
    Raises:
        ValueError: If parameters are invalid
        
    Example:
        >>> from baseline_strategy import nearest_threat_first_strategy
        >>> results = run_scenarios(
        ...     strategy=nearest_threat_first_strategy,
        ...     num_scenarios=100,
        ...     strategy_name="Baseline"
        ... )
        >>> print(results['cost_exchange_ratio'].mean())
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
            random_seed=seed
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
    
    Args:
        df: DataFrame of scenario results
        
    Returns:
        Formatted string with summary statistics
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
    
    Args:
        df: DataFrame of results to save
        filepath: Path to save file (can include or exclude .csv extension)
        include_timestamp: If True, append timestamp to filename
        
    Returns:
        Path object of saved file
        
    Example:
        >>> results = run_scenarios(...)
        >>> path = save_results(results, "baseline_results")
        >>> print(f"Saved to {path}")
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
    
    Args:
        filepath: Path to CSV file
        
    Returns:
        DataFrame of results
        
    Example:
        >>> df = load_results("baseline_results.csv")
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
    
    Args:
        df1: Results from first strategy
        df2: Results from second strategy
        name1: Name of first strategy
        name2: Name of second strategy
        
    Returns:
        Formatted comparison string
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
    """
    Test and demonstration code for Monte Carlo runner.
    """
    print("=" * 70)
    print("MONTE CARLO RUNNER MODULE - DEMONSTRATION")
    print("=" * 70)
    
    from baseline_strategy import nearest_threat_first_strategy
    
    # Test 1: Run small batch of scenarios
    print("\n[Test 1] Running 10 scenarios with baseline strategy:")
    
    results = run_scenarios(
        strategy=nearest_threat_first_strategy,
        num_scenarios=10,
        swarm_size_range=(50, 150),
        random_seed=42,
        verbose=True,
        strategy_name="Baseline"
    )
    
    print(f"\nDataFrame shape: {results.shape}")
    print(f"Columns: {list(results.columns[:10])}...")
    
    # Test 2: Examine individual scenarios
    print("\n[Test 2] First 5 scenarios summary:")
    print(results[['scenario_id', 'swarm_size', 'penetration_rate', 
                   'cost_exchange_ratio', 'total_cost']].head())
    
    # Test 3: Save and load results
    print("\n[Test 3] Save and load results:")
    
    save_path = save_results(
        results,
        "test_results",
        include_timestamp=False
    )
    print(f"  Saved to: {save_path}")
    
    loaded_results = load_results(save_path)
    print(f"  Loaded shape: {loaded_results.shape}")
    print(f"  Data integrity check: {results.equals(loaded_results)}")
    
    # Clean up test file
    if save_path.exists():
        save_path.unlink()
        print(f"  Test file deleted")
    
    # Test 4: Larger batch for statistics
    print("\n[Test 4] Running 100 scenarios for robust statistics:")
    
    results_large = run_scenarios(
        strategy=nearest_threat_first_strategy,
        num_scenarios=100,
        swarm_size_range=(100, 500),
        random_seed=123,
        verbose=True,
        strategy_name="Baseline-100"
    )
    
    # Test 5: Compare two different configurations
    print("\n[Test 5] Comparing two weapon configurations:")
    
    # Configuration 1: Standard (50 kinetic, 100 DE)
    results_config1 = run_scenarios(
        strategy=nearest_threat_first_strategy,
        num_scenarios=50,
        swarm_size_range=(100, 300),
        kinetic_quantity=50,
        de_quantity=100,
        random_seed=200,
        verbose=False,
        strategy_name="Standard"
    )
    
    # Configuration 2: More weapons (100 kinetic, 200 DE)
    results_config2 = run_scenarios(
        strategy=nearest_threat_first_strategy,
        num_scenarios=50,
        swarm_size_range=(100, 300),
        kinetic_quantity=100,
        de_quantity=200,
        random_seed=200,  # Same seed for fair comparison
        verbose=False,
        strategy_name="Enhanced"
    )
    
    print(compare_strategies(results_config1, results_config2, "Standard", "Enhanced"))
    
    # Test 6: Distribution analysis
    print("\n[Test 6] Analyzing metric distributions:")
    
    print(f"\nPenetration Rate Distribution:")
    print(results_large['penetration_rate'].describe())
    
    print(f"\nCost-Exchange Ratio Distribution:")
    print(results_large['cost_exchange_ratio'].describe())
    
    # Test 7: Identify best and worst scenarios
    print("\n[Test 7] Best and worst performing scenarios:")
    
    best_scenario = results_large.loc[results_large['cost_exchange_ratio'].idxmin()]
    worst_scenario = results_large.loc[results_large['cost_exchange_ratio'].idxmax()]
    
    print(f"\n  Best (lowest cost-exchange):")
    print(f"    Scenario: {best_scenario['scenario_id']}")
    print(f"    Swarm: {best_scenario['swarm_size']} UAVs")
    print(f"    Penetration: {best_scenario['penetration_rate']*100:.1f}%")
    print(f"    Cost-exchange: {best_scenario['cost_exchange_ratio']:.2f}")
    
    print(f"\n  Worst (highest cost-exchange):")
    print(f"    Scenario: {worst_scenario['scenario_id']}")
    print(f"    Swarm: {worst_scenario['swarm_size']} UAVs")
    print(f"    Penetration: {worst_scenario['penetration_rate']*100:.1f}%")
    print(f"    Cost-exchange: {worst_scenario['cost_exchange_ratio']:.2f}")
    
    # Test 8: Error handling
    print("\n[Test 8] Error handling:")
    
    try:
        run_scenarios(nearest_threat_first_strategy, num_scenarios=-10)
    except ValueError as e:
        print(f"  ✓ Caught expected error: {e}")
    
    try:
        run_scenarios(nearest_threat_first_strategy, swarm_size_range=(500, 100))
    except ValueError as e:
        print(f"  ✓ Caught expected error: {e}")
    
    print("\n" + "=" * 70)
    print("ALL TESTS COMPLETED SUCCESSFULLY")
    print("=" * 70)
    print("\nMonte Carlo runner ready for full evaluation!")
