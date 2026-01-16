"""
Strategy Evaluation Module

This module evaluates and compares the RL agent against the baseline strategy
on standardized test scenarios for the master's thesis.

Author: Master's Thesis Project
Date: January 2026
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, Callable, Optional
from datetime import datetime

from stable_baselines3 import DQN
from monte_carlo_runner import run_scenarios, save_results
from baseline_strategy import nearest_threat_first_strategy
from defense_system import WeaponType
from rl_environment import AirDefenseEnv


class RLStrategyWrapper:
    """
    Wrapper to use a trained RL model as a strategy function.
    
    Converts the Gym observation format to state dict and back,
    allowing the RL model to be used with the combat simulation
    and Monte Carlo runner.
    """
    
    def __init__(
        self,
        model: DQN,
        swarm_size_range: tuple = (100, 500),
        deterministic: bool = False
    ):
        """
        Initialize the RL strategy wrapper.
        
        Args:
            model: Trained DQN model
            swarm_size_range: Range used during training
            deterministic: If True, use deterministic policy (no exploration)
        """
        self.model = model
        self.swarm_size_range = swarm_size_range
        self.deterministic = deterministic
        
        # Create environment for observation conversion
        self.env = AirDefenseEnv(swarm_size_range=swarm_size_range)
        
        # Track current episode state
        self.current_swarm_size = None
    
    def __call__(self, state: Dict[str, Any]) -> WeaponType:
        """
        Strategy function interface.
        
        Args:
            state: State dictionary from combat simulation
            
        Returns:
            WeaponType action to take
        """
        # Convert state dict to observation array
        obs = self._state_to_observation(state)
        
        # Get action from model
        action, _ = self.model.predict(obs, deterministic=self.deterministic)
        
        # Convert action index to WeaponType
        action_map = {
            0: WeaponType.KINETIC,
            1: WeaponType.DIRECTED_ENERGY,
            2: WeaponType.SKIP
        }
        
        return action_map[int(action)]
    
    def _state_to_observation(self, state: Dict[str, Any]) -> np.ndarray:
        """
        Convert state dictionary to normalized observation array.
        
        Args:
            state: State dictionary with keys:
                - remaining_uavs
                - remaining_kinetic
                - remaining_de
                - cumulative_cost
                - nearest_distance
                
        Returns:
            Normalized observation array [5,]
        """
        # Infer swarm size if not set (from first state of episode)
        if self.current_swarm_size is None or state['step'] == 1:
            # Estimate from remaining UAVs (assumes episode start)
            self.current_swarm_size = max(
                state['remaining_uavs'],
                self.swarm_size_range[0]
            )
        
        # Normalize observation components
        remaining_uavs_norm = state['remaining_uavs'] / self.current_swarm_size
        remaining_kinetic_norm = state['remaining_kinetic'] / 50  # Initial kinetic
        remaining_de_norm = state['remaining_de'] / 100  # Initial DE
        
        # Normalize cost (using max expected cost)
        max_cost = 50 * 1_000_000 + 100 * 10_000  # Kinetic + DE
        cumulative_cost_norm = min(state['cumulative_cost'] / max_cost, 1.0)
        
        # Normalize distance
        nearest_distance = state['nearest_distance'] if state['nearest_distance'] is not None else 0.0
        nearest_distance_norm = nearest_distance / 100.0  # Initial distance
        
        obs = np.array([
            remaining_uavs_norm,
            remaining_kinetic_norm,
            remaining_de_norm,
            cumulative_cost_norm,
            nearest_distance_norm
        ], dtype=np.float32)
        
        return obs


def create_rl_strategy(
    model_path: str,
    deterministic: bool = False
) -> Callable[[Dict[str, Any]], WeaponType]:
    """
    Create an RL strategy function from a saved model.
    
    Args:
        model_path: Path to saved DQN model
        deterministic: If True, use deterministic policy
        
    Returns:
        Strategy function compatible with combat simulation
    """
    model = DQN.load(model_path)
    wrapper = RLStrategyWrapper(model, deterministic=deterministic)
    return wrapper


def evaluate_baseline(
    num_scenarios: int = 1000,
    swarm_size_range: tuple = (100, 500),
    random_seed: int = 42,
    save_path: Optional[str] = None,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Evaluate baseline strategy on test scenarios.
    
    Args:
        num_scenarios: Number of test scenarios
        swarm_size_range: Range of swarm sizes
        random_seed: Random seed for reproducibility
        save_path: Path to save results (if provided)
        verbose: Show progress
        
    Returns:
        DataFrame of results
    """
    if verbose:
        print("\n" + "=" * 70)
        print("EVALUATING BASELINE STRATEGY")
        print("=" * 70)
    
    results = run_scenarios(
        strategy=nearest_threat_first_strategy,
        num_scenarios=num_scenarios,
        swarm_size_range=swarm_size_range,
        random_seed=random_seed,
        verbose=verbose,
        strategy_name="Baseline"
    )
    
    if save_path:
        path = save_results(results, save_path, include_timestamp=True)
        if verbose:
            print(f"\nBaseline results saved to: {path}")
    
    return results


def evaluate_rl_agent(
    model_path: str,
    num_scenarios: int = 1000,
    swarm_size_range: tuple = (100, 500),
    random_seed: int = 42,
    deterministic: bool = False,
    save_path: Optional[str] = None,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Evaluate trained RL agent on test scenarios.
    
    Args:
        model_path: Path to trained DQN model
        num_scenarios: Number of test scenarios
        swarm_size_range: Range of swarm sizes
        random_seed: Random seed for reproducibility
        deterministic: Use deterministic policy
        save_path: Path to save results (if provided)
        verbose: Show progress
        
    Returns:
        DataFrame of results
    """
    if verbose:
        print("\n" + "=" * 70)
        print("EVALUATING RL AGENT")
        print("=" * 70)
        print(f"Model: {model_path}")
        print(f"Deterministic: {deterministic}")
    
    # Load model and create strategy
    rl_strategy = create_rl_strategy(model_path, deterministic=deterministic)
    
    results = run_scenarios(
        strategy=rl_strategy,
        num_scenarios=num_scenarios,
        swarm_size_range=swarm_size_range,
        random_seed=random_seed,
        verbose=verbose,
        strategy_name="RL_Agent"
    )
    
    if save_path:
        path = save_results(results, save_path, include_timestamp=True)
        if verbose:
            print(f"\nRL agent results saved to: {path}")
    
    return results


def compare_strategies(
    baseline_results: pd.DataFrame,
    rl_results: pd.DataFrame,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Compare baseline and RL agent results.
    
    Args:
        baseline_results: DataFrame from baseline evaluation
        rl_results: DataFrame from RL evaluation
        verbose: Print comparison summary
        
    Returns:
        Dictionary with comparison statistics
    """
    comparison = {}
    
    # Key metrics to compare
    metrics = [
        ('penetration_rate', 'Penetration Rate', '%', 100, 'lower'),
        ('cost_exchange_ratio', 'Cost-Exchange Ratio', '', 1, 'lower'),
        ('total_cost', 'Total Cost', '$', 1, 'lower'),
        ('defense_cost', 'Defense Cost', '$', 1, 'lower'),
        ('uavs_destroyed', 'UAVs Destroyed', '', 1, 'higher'),
        ('kinetic_fired', 'Kinetic Fired', '', 1, 'lower'),
        ('de_fired', 'DE Fired', '', 1, 'lower')
    ]
    
    for metric, label, unit, multiplier, better in metrics:
        baseline_mean = baseline_results[metric].mean() * multiplier
        baseline_std = baseline_results[metric].std() * multiplier
        
        rl_mean = rl_results[metric].mean() * multiplier
        rl_std = rl_results[metric].std() * multiplier
        
        # Calculate improvement
        if better == 'lower':
            improvement_pct = ((baseline_mean - rl_mean) / baseline_mean) * 100
        else:
            improvement_pct = ((rl_mean - baseline_mean) / baseline_mean) * 100
        
        comparison[metric] = {
            'baseline_mean': baseline_mean,
            'baseline_std': baseline_std,
            'rl_mean': rl_mean,
            'rl_std': rl_std,
            'improvement_pct': improvement_pct,
            'better': better
        }
    
    if verbose:
        print("\n" + "=" * 70)
        print("STRATEGY COMPARISON")
        print("=" * 70)
        print(f"\nNumber of scenarios: {len(baseline_results)}")
        print(f"\n{'Metric':<25} {'Baseline':>20} {'RL Agent':>20} {'Improvement':>15}")
        print("-" * 85)
        
        for metric, label, unit, multiplier, better in metrics:
            comp = comparison[metric]
            baseline_val = comp['baseline_mean']
            rl_val = comp['rl_mean']
            improvement = comp['improvement_pct']
            
            if unit == '$':
                baseline_str = f"${baseline_val:>19,.0f}"
                rl_str = f"${rl_val:>19,.0f}"
            elif unit == '%':
                baseline_str = f"{baseline_val:>19.2f}%"
                rl_str = f"{rl_val:>19.2f}%"
            else:
                baseline_str = f"{baseline_val:>20.2f}"
                rl_str = f"{rl_val:>20.2f}"
            
            improvement_str = f"{improvement:>+14.2f}%"
            
            print(f"{label:<25} {baseline_str} {rl_str} {improvement_str}")
        
        print("=" * 70)
        print("\nKey Findings:")
        print(f"  Penetration rate improvement: {comparison['penetration_rate']['improvement_pct']:+.2f}%")
        print(f"  Cost-exchange improvement: {comparison['cost_exchange_ratio']['improvement_pct']:+.2f}%")
        print(f"  Total cost improvement: {comparison['total_cost']['improvement_pct']:+.2f}%")
        print("=" * 70 + "\n")
    
    return comparison


def evaluate_both_strategies(
    model_path: str,
    num_scenarios: int = 1000,
    swarm_size_range: tuple = (100, 500),
    random_seed: int = 42,
    deterministic: bool = False,
    save_dir: str = "evaluation_results",
    verbose: bool = True
) -> tuple:
    """
    Evaluate both baseline and RL strategies and compare.
    
    Args:
        model_path: Path to trained DQN model
        num_scenarios: Number of test scenarios
        swarm_size_range: Range of swarm sizes
        random_seed: Random seed for reproducibility
        deterministic: Use deterministic RL policy
        save_dir: Directory to save results
        verbose: Show progress and results
        
    Returns:
        Tuple of (baseline_results, rl_results, comparison)
    """
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    
    # Evaluate baseline
    baseline_results = evaluate_baseline(
        num_scenarios=num_scenarios,
        swarm_size_range=swarm_size_range,
        random_seed=random_seed,
        save_path=str(save_path / "baseline_results"),
        verbose=verbose
    )
    
    # Evaluate RL agent
    rl_results = evaluate_rl_agent(
        model_path=model_path,
        num_scenarios=num_scenarios,
        swarm_size_range=swarm_size_range,
        random_seed=random_seed,
        deterministic=deterministic,
        save_path=str(save_path / "rl_agent_results"),
        verbose=verbose
    )
    
    # Compare results
    comparison = compare_strategies(baseline_results, rl_results, verbose=verbose)
    
    # Save comparison
    comparison_path = save_path / f"comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    import json
    with open(comparison_path, 'w') as f:
        # Convert to serializable format
        serializable_comp = {}
        for key, value in comparison.items():
            serializable_comp[key] = {k: float(v) if isinstance(v, (np.floating, np.integer)) else v 
                                     for k, v in value.items()}
        json.dump(serializable_comp, f, indent=2)
    
    if verbose:
        print(f"Comparison saved to: {comparison_path}")
    
    return baseline_results, rl_results, comparison


if __name__ == "__main__":
    """
    Test and demonstration code.
    """
    print("=" * 70)
    print("STRATEGY EVALUATION MODULE - DEMONSTRATION")
    print("=" * 70)
    
    # Test 1: Create a simple test RL model
    print("\n[Test 1] Training a small test RL model:")
    
    from train_agent import train_dqn_agent
    
    model, metrics, model_path = train_dqn_agent(
        total_timesteps=5_000,
        swarm_size_range=(50, 100),
        save_dir="test_eval",
        model_name="test_model",
        checkpoint_freq=10000,
        eval_freq=10000,
        verbose=0
    )
    
    print(f"  Test model saved to: {model_path}")
    
    # Test 2: Create RL strategy wrapper
    print("\n[Test 2] Testing RL strategy wrapper:")
    
    rl_strategy = create_rl_strategy(str(model_path), deterministic=True)
    
    # Test with sample state
    test_state = {
        'remaining_uavs': 50,
        'remaining_kinetic': 25,
        'remaining_de': 50,
        'cumulative_cost': 1000000,
        'nearest_distance': 40.0,
        'step': 5
    }
    
    action = rl_strategy(test_state)
    print(f"  Sample state: {test_state}")
    print(f"  RL action: {action.value}")
    
    # Test 3: Evaluate baseline on small set
    print("\n[Test 3] Evaluating baseline (10 scenarios):")
    
    baseline_results = evaluate_baseline(
        num_scenarios=10,
        swarm_size_range=(50, 100),
        random_seed=42,
        save_path=None,
        verbose=False
    )
    
    print(f"  Scenarios: {len(baseline_results)}")
    print(f"  Mean penetration: {baseline_results['penetration_rate'].mean()*100:.2f}%")
    print(f"  Mean cost-exchange: {baseline_results['cost_exchange_ratio'].mean():.2f}")
    
    # Test 4: Evaluate RL agent on small set
    print("\n[Test 4] Evaluating RL agent (10 scenarios):")
    
    rl_results = evaluate_rl_agent(
        model_path=str(model_path),
        num_scenarios=10,
        swarm_size_range=(50, 100),
        random_seed=42,
        deterministic=True,
        save_path=None,
        verbose=False
    )
    
    print(f"  Scenarios: {len(rl_results)}")
    print(f"  Mean penetration: {rl_results['penetration_rate'].mean()*100:.2f}%")
    print(f"  Mean cost-exchange: {rl_results['cost_exchange_ratio'].mean():.2f}")
    
    # Test 5: Compare strategies
    print("\n[Test 5] Comparing strategies:")
    
    comparison = compare_strategies(baseline_results, rl_results, verbose=True)
    
    # Test 6: Full evaluation pipeline
    print("\n[Test 6] Full evaluation pipeline (20 scenarios):")
    
    baseline_full, rl_full, comparison_full = evaluate_both_strategies(
        model_path=str(model_path),
        num_scenarios=20,
        swarm_size_range=(50, 100),
        random_seed=999,
        deterministic=True,
        save_dir="test_eval_results",
        verbose=True
    )
    
    # Test 7: Verify saved files
    print("\n[Test 7] Verify saved files:")
    
    eval_dir = Path("test_eval_results")
    if eval_dir.exists():
        files = list(eval_dir.glob("*.csv"))
        print(f"  CSV files created: {len(files)}")
        for f in files:
            print(f"    - {f.name}")
    
    # Cleanup
    print("\n[Test 8] Cleanup test files:")
    import shutil
    
    for cleanup_dir in ["test_eval", "test_eval_results"]:
        if Path(cleanup_dir).exists():
            shutil.rmtree(cleanup_dir)
            print(f"  Removed: {cleanup_dir}")
    
    print("\n" + "=" * 70)
    print("ALL TESTS COMPLETED SUCCESSFULLY")
    print("=" * 70)
    print("\nReady for full evaluation with:")
    print("  from evaluate_strategies import evaluate_both_strategies")
    print("  baseline, rl, comp = evaluate_both_strategies(")
    print("      model_path='trained_models/dqn_final.zip',")
    print("      num_scenarios=1000")
    print("  )")
