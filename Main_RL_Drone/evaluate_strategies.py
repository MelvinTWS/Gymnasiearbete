"""
Strategy Evaluation Module

This module evaluates and compares the RL agent against the baseline strategy
on standardized test scenarios for the master's thesis.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, Callable, Optional
from datetime import datetime

from stable_baselines3 import DQN
from monte_carlo_runner import run_scenarios, save_results
from baseline_strategy import adaptive_10percent_baseline
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
        swarm_size_range: tuple = (50, 150),
        deterministic: bool = False
    ):
        """
        Initialize the RL strategy wrapper.
        """
        self.model = model
        self.swarm_size_range = swarm_size_range
        self.deterministic = deterministic

        # Create environment for observation conversion
        self.env = AirDefenseEnv(swarm_size_range=swarm_size_range)

        # Track current episode state
        self.current_swarm_size = None
        self._initial_kinetic = 50   # Updated at step 1 of each episode
        self._initial_de = 100       # Updated at step 1 of each episode
    
    def __call__(self, state: Dict[str, Any]) -> WeaponType:
        """
        Strategy function interface.
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
        """
        # Infer swarm size and initial ammo at the first step of each episode
        if self.current_swarm_size is None or state['step'] == 1:
            self.current_swarm_size = max(
                state['remaining_uavs'],
                self.swarm_size_range[0]
            )
            # Record initial ammo so normalization matches regardless of quantity
            self._initial_kinetic = max(state['remaining_kinetic'], 1)
            self._initial_de = max(state['remaining_de'], 1)

        # Normalize observation components (fraction of remaining vs initial)
        remaining_uavs_norm = state['remaining_uavs'] / self.current_swarm_size
        remaining_kinetic_norm = state['remaining_kinetic'] / self._initial_kinetic
        remaining_de_norm = state['remaining_de'] / self._initial_de

        # Normalize cost (using max expected cost based on initial ammo)
        max_cost = self._initial_kinetic * 1_000_000 + self._initial_de * 10_000
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
    """
    model = DQN.load(model_path)
    wrapper = RLStrategyWrapper(model, deterministic=deterministic)
    return wrapper


def evaluate_baseline(
    num_scenarios: int = 1000,
    swarm_size_range: tuple = (50, 150),
    random_seed: int = 42,
    save_path: Optional[str] = None,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Evaluate baseline strategy on test scenarios.
    """
    if verbose:
        print("\n" + "=" * 70)
        print("EVALUATING BASELINE STRATEGY (Adaptive 10% Target)")
        print("=" * 70)
    
    results = run_scenarios(
        strategy=adaptive_10percent_baseline,
        num_scenarios=num_scenarios,
        swarm_size_range=swarm_size_range,
        kinetic_quantity=800,
        de_quantity=150,
        random_seed=random_seed,
        verbose=verbose,
        strategy_name="Adaptive_10pct_Baseline"
    )
    
    if save_path:
        path = save_results(results, save_path, include_timestamp=True)
        if verbose:
            print(f"\nBaseline results saved to: {path}")
    
    return results


def evaluate_rl_agent(
    model_path: str,
    num_scenarios: int = 1000,
    swarm_size_range: tuple = (50, 150),
    random_seed: int = 42,
    deterministic: bool = False,
    save_path: Optional[str] = None,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Evaluate trained RL agent on test scenarios.
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
        kinetic_quantity=800,
        de_quantity=150,
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
    swarm_size_range: tuple = (50, 150),
    random_seed: int = 42,
    deterministic: bool = False,
    save_dir: str = "evaluation_results",
    verbose: bool = True
) -> tuple:
    """
    Evaluate both baseline and RL strategies and compare.
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
    import os
    model_path = "thesis_results/trained_models/best_model/best_model.zip"
    if os.path.exists(model_path):
        baseline, rl, comp = evaluate_both_strategies(
            model_path=model_path,
            num_scenarios=100,
            random_seed=42,
            verbose=True
        )
    else:
        print("Train an agent first with run_complete_experiment.py")
