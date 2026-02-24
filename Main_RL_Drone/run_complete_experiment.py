"""
Master Experiment Runner

This script executes the complete thesis experimental pipeline:
1. Train RL agent (DQN)
2. Evaluate baseline strategy
3. Evaluate RL strategy
4. Perform statistical analysis
5. Generate all visualizations
6. Create summary report
"""

import time
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any
import pandas as pd

# Import all modules
from train_agent import train_dqn_agent
from evaluate_strategies import evaluate_baseline, evaluate_rl_agent
from statistical_analysis import analyze_and_save, interpret_results
from visualizations import create_all_visualizations


def print_header(text: str, char: str = "="):
    """Print formatted header."""
    width = 70
    print("\n" + char * width)
    print(text.center(width))
    print(char * width + "\n")


def print_section(text: str):
    """Print section header."""
    print(f"\n{'='*70}")
    print(f"  {text}")
    print(f"{'='*70}\n")


def create_experiment_directories(base_dir: str = "thesis_results") -> Dict[str, Path]:
    """Create directory structure for experiment results."""
    base_path = Path(base_dir)
    
    dirs = {
        'base': base_path,
        'models': base_path / 'trained_models',
        'training': base_path / 'training_metrics',
        'evaluation': base_path / 'evaluation_results',
        'analysis': base_path / 'statistical_analysis',
        'figures': base_path / 'visualizations',
        'reports': base_path / 'reports'
    }
    
    for dir_path in dirs.values():
        dir_path.mkdir(parents=True, exist_ok=True)
    
    return dirs


def run_complete_experiment(
    training_timesteps: int = 50000,
    eval_scenarios: int = 1000,
    random_seed: int = 42,
    base_dir: str = "thesis_results"
) -> Dict[str, Any]:
    """
    Run the complete experimental pipeline.
    """
    start_time = time.time()
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    print_header("MASTER'S THESIS EXPERIMENTAL PIPELINE")
    print(f"Timestamp: {timestamp}")
    print(f"Configuration:")
    print(f"  - Training timesteps: {training_timesteps:,}")
    print(f"  - Evaluation scenarios: {eval_scenarios:,}")
    print(f"  - Random seed: {random_seed}")
    print(f"  - Output directory: {base_dir}")
    
    # Create directories
    print_section("PHASE 0: Setup")
    dirs = create_experiment_directories(base_dir)
    print(f"Created directory structure:")
    for name, path in dirs.items():
        print(f"  - {name}: {path}")
    
    results = {
        'timestamp': timestamp,
        'config': {
            'training_timesteps': training_timesteps,
            'eval_scenarios': eval_scenarios,
            'random_seed': random_seed
        },
        'directories': {k: str(v) for k, v in dirs.items()},
        'files': {},
        'metrics': {},
        'timings': {}
    }
    
    # Phase 1: Train RL Agent
    print_section("PHASE 1: Training RL Agent")
    phase_start = time.time()
    
    print(f"Training DQN agent for {training_timesteps:,} timesteps...")
    print(f"Models will be saved to: {dirs['models']}")
    
    model, training_metrics, model_save_path = train_dqn_agent(
        total_timesteps=training_timesteps,
        save_dir=str(dirs['models']),
        model_name="dqn_agent",
        random_seed=random_seed,
        verbose=1
    )
    
    # Convert training metrics dict to DataFrame and save
    metrics_path = str(dirs['training'] / f"training_metrics_{timestamp}.csv")
    metrics_df = pd.DataFrame(training_metrics)
    metrics_df.to_csv(metrics_path, index=False)
    
    phase_time = time.time() - phase_start
    results['timings']['training'] = phase_time
    results['files']['model'] = str(model_save_path)
    results['files']['training_metrics'] = metrics_path
    results['metrics']['training_episodes'] = len(metrics_df)
    
    print(f"\n✓ Training complete in {phase_time:.1f}s")
    print(f"  - Episodes: {len(metrics_df)}")
    if len(metrics_df) > 0 and 'episode_rewards' in training_metrics:
        print(f"  - Final reward: {metrics_df['episode_rewards'].iloc[-10:].mean():.2f}")
    
    # Phase 2: Evaluate Baseline Strategy
    print_section("PHASE 2: Evaluating Baseline Strategy")
    phase_start = time.time()
    
    baseline_results_path = str(dirs['evaluation'] / f"baseline_results_{timestamp}.csv")
    
    print(f"Running {eval_scenarios:,} Monte Carlo scenarios with baseline strategy...")
    
    baseline_results = evaluate_baseline(
        num_scenarios=eval_scenarios,
        save_path=baseline_results_path,
        random_seed=random_seed
    )
    
    phase_time = time.time() - phase_start
    results['timings']['baseline_eval'] = phase_time
    results['files']['baseline_results'] = baseline_results_path
    
    # Calculate baseline metrics
    baseline_metrics = {
        'penetration_rate': float(baseline_results['penetration_rate'].mean()),
        'cost_exchange_ratio': float(baseline_results['cost_exchange_ratio'].mean()),
        'total_cost': float(baseline_results['total_cost'].mean()),
        'uavs_destroyed': float(baseline_results['uavs_destroyed'].mean())
    }
    results['metrics']['baseline'] = baseline_metrics
    
    print(f"\n✓ Baseline evaluation complete in {phase_time:.1f}s")
    print(f"  - Penetration rate: {baseline_metrics['penetration_rate']*100:.2f}%")
    print(f"  - Cost-exchange ratio: {baseline_metrics['cost_exchange_ratio']:.2f}")
    print(f"  - Total cost: ${baseline_metrics['total_cost']/1e6:.2f}M")
    
    # Phase 3: Evaluate RL Strategy
    print_section("PHASE 3: Evaluating RL Agent")
    phase_start = time.time()
    
    rl_results_path = str(dirs['evaluation'] / f"rl_results_{timestamp}.csv")
    
    print(f"Running {eval_scenarios:,} Monte Carlo scenarios with RL agent...")
    
    rl_results = evaluate_rl_agent(
        model_path=results['files']['model'],
        num_scenarios=eval_scenarios,
        save_path=rl_results_path,
        random_seed=random_seed
    )
    
    phase_time = time.time() - phase_start
    results['timings']['rl_eval'] = phase_time
    results['files']['rl_results'] = rl_results_path
    
    # Calculate RL metrics
    rl_metrics = {
        'penetration_rate': float(rl_results['penetration_rate'].mean()),
        'cost_exchange_ratio': float(rl_results['cost_exchange_ratio'].mean()),
        'total_cost': float(rl_results['total_cost'].mean()),
        'uavs_destroyed': float(rl_results['uavs_destroyed'].mean())
    }
    results['metrics']['rl'] = rl_metrics
    
    print(f"\n✓ RL evaluation complete in {phase_time:.1f}s")
    print(f"  - Penetration rate: {rl_metrics['penetration_rate']*100:.2f}%")
    print(f"  - Cost-exchange ratio: {rl_metrics['cost_exchange_ratio']:.2f}")
    print(f"  - Total cost: ${rl_metrics['total_cost']/1e6:.2f}M")
    
    # Phase 4: Statistical Analysis
    print_section("PHASE 4: Statistical Analysis")
    phase_start = time.time()
    
    print(f"Performing t-tests and generating statistical tables...")
    
    analysis_results = analyze_and_save(
        baseline_results,
        rl_results,
        save_dir=str(dirs['analysis']),
        alpha=0.05,
        verbose=False
    )
    
    # Get interpretation
    interpretation = interpret_results(analysis_results)
    
    phase_time = time.time() - phase_start
    results['timings']['statistical_analysis'] = phase_time
    results['metrics']['statistical'] = analysis_results
    
    # Save interpretation
    interp_path = dirs['reports'] / f"interpretation_{timestamp}.txt"
    with open(interp_path, 'w', encoding='utf-8') as f:
        f.write(interpretation)
    results['files']['interpretation'] = str(interp_path)
    
    print(interpretation)
    print(f"\n✓ Statistical analysis complete in {phase_time:.1f}s")
    
    # Phase 5: Generate Visualizations
    print_section("PHASE 5: Generating Visualizations")
    phase_start = time.time()
    
    print(f"Creating publication-quality figures...")
    
    viz_files = create_all_visualizations(
        baseline_results,
        rl_results,
        training_metrics_file=results['files']['training_metrics'],
        save_dir=str(dirs['figures']),
        show=False
    )
    
    phase_time = time.time() - phase_start
    results['timings']['visualization'] = phase_time
    results['files']['visualizations'] = viz_files
    
    print(f"\n✓ Visualization complete in {phase_time:.1f}s")
    
    # Phase 6: Generate Summary Report
    print_section("PHASE 6: Generating Summary Report")
    
    total_time = time.time() - start_time
    results['timings']['total'] = total_time
    
    # Create summary report
    summary_path = dirs['reports'] / f"experiment_summary_{timestamp}.txt"
    
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write("="*70 + "\n")
        f.write("MASTER'S THESIS EXPERIMENT SUMMARY\n")
        f.write("="*70 + "\n\n")
        f.write(f"Timestamp: {timestamp}\n")
        f.write(f"Total execution time: {total_time:.1f}s ({total_time/60:.1f} minutes)\n\n")
        
        f.write("CONFIGURATION\n")
        f.write("-"*70 + "\n")
        f.write(f"  Training timesteps: {training_timesteps:,}\n")
        f.write(f"  Evaluation scenarios: {eval_scenarios:,}\n")
        f.write(f"  Random seed: {random_seed}\n\n")
        
        f.write("PERFORMANCE COMPARISON\n")
        f.write("-"*70 + "\n")
        f.write(f"{'Metric':<30} {'Baseline':>15} {'RL Agent':>15} {'Improvement':>15}\n")
        f.write("-"*70 + "\n")
        
        pen_imp = ((baseline_metrics['penetration_rate'] - rl_metrics['penetration_rate']) / 
                   baseline_metrics['penetration_rate']) * 100
        f.write(f"{'Penetration Rate':<30} {baseline_metrics['penetration_rate']*100:>14.2f}% "
                f"{rl_metrics['penetration_rate']*100:>14.2f}% {pen_imp:>14.2f}%\n")
        
        cost_imp = ((baseline_metrics['cost_exchange_ratio'] - rl_metrics['cost_exchange_ratio']) / 
                    baseline_metrics['cost_exchange_ratio']) * 100
        f.write(f"{'Cost-Exchange Ratio':<30} {baseline_metrics['cost_exchange_ratio']:>15.2f} "
                f"{rl_metrics['cost_exchange_ratio']:>15.2f} {cost_imp:>14.2f}%\n")
        
        total_cost_imp = ((baseline_metrics['total_cost'] - rl_metrics['total_cost']) / 
                          baseline_metrics['total_cost']) * 100
        f.write(f"{'Total Cost (M$)':<30} {baseline_metrics['total_cost']/1e6:>15.2f} "
                f"{rl_metrics['total_cost']/1e6:>15.2f} {total_cost_imp:>14.2f}%\n\n")
        
        f.write("EXECUTION TIMINGS\n")
        f.write("-"*70 + "\n")
        for phase, timing in results['timings'].items():
            pct = (timing / total_time) * 100
            f.write(f"  {phase:<25} {timing:>8.1f}s ({pct:>5.1f}%)\n")
        f.write("\n")
        
        f.write("OUTPUT FILES\n")
        f.write("-"*70 + "\n")
        f.write(f"\nModels:\n")
        f.write(f"  - {results['files']['model']}\n")
        f.write(f"\nTraining Data:\n")
        f.write(f"  - {results['files']['training_metrics']}\n")
        f.write(f"\nEvaluation Results:\n")
        f.write(f"  - {results['files']['baseline_results']}\n")
        f.write(f"  - {results['files']['rl_results']}\n")
        f.write(f"\nStatistical Analysis:\n")
        f.write(f"  - {dirs['analysis']}\n")
        f.write(f"\nVisualizations:\n")
        for name, path in viz_files.items():
            f.write(f"  - {Path(path).name}\n")
        f.write(f"\nReports:\n")
        f.write(f"  - {summary_path.name}\n")
        f.write(f"  - {interp_path.name}\n")
        f.write("\n" + "="*70 + "\n")
    
    results['files']['summary'] = str(summary_path)
    
    # Save complete results as JSON
    json_path = dirs['reports'] / f"experiment_results_{timestamp}.json"
    
    # Convert Path objects to strings for JSON serialization
    json_results = {
        'timestamp': results['timestamp'],
        'config': results['config'],
        'directories': results['directories'],
        'files': results['files'],
        'metrics': {
            'baseline': results['metrics']['baseline'],
            'rl': results['metrics']['rl'],
            'training_episodes': results['metrics']['training_episodes']
        },
        'timings': results['timings']
    }
    
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(json_results, f, indent=2)
    
    results['files']['json_results'] = str(json_path)
    
    # Print final summary
    print_header("EXPERIMENT COMPLETE!")
    print(f"Total time: {total_time:.1f}s ({total_time/60:.1f} minutes)")
    print(f"\nKey Results:")
    print(f"  Penetration improvement: {pen_imp:+.2f}%")
    print(f"  Cost-exchange improvement: {cost_imp:+.2f}%")
    print(f"  Total cost savings: ${(baseline_metrics['total_cost'] - rl_metrics['total_cost'])/1e6:.2f}M ({total_cost_imp:+.2f}%)")
    print(f"\nAll results saved to: {base_dir}")
    print(f"Summary report: {summary_path.name}")
    
    return results


if __name__ == "__main__":
    """
    Run the complete experimental pipeline with default parameters.
    
    Modify these parameters as needed for your thesis:
    - training_timesteps: More training may improve performance
    - eval_scenarios: 1000 is recommended for statistical significance
    - random_seed: Change for different random scenarios
    """
    
    # Configuration
    TRAINING_TIMESTEPS = 50000  # 50K timesteps (~60-75 seconds)
    EVAL_SCENARIOS = 1000       # 1000 scenarios per strategy
    RANDOM_SEED = 42            # For reproducibility
    OUTPUT_DIR = "thesis_results"
    
    print("="*70)
    print("STARTING COMPLETE EXPERIMENTAL PIPELINE")
    print("="*70)
    print("\nThis will:")
    print("  1. Train RL agent (DQN)")
    print("  2. Evaluate baseline strategy")
    print("  3. Evaluate RL strategy")
    print("  4. Perform statistical analysis")
    print("  5. Generate visualizations")
    print("  6. Create summary reports")
    print(f"\nEstimated time: ~70-90 seconds\n")
    
    # Remove input() to allow automated running
    # input("Press Enter to begin...")
    
    # Run experiment
    results = run_complete_experiment(
        training_timesteps=TRAINING_TIMESTEPS,
        eval_scenarios=EVAL_SCENARIOS,
        random_seed=RANDOM_SEED,
        base_dir=OUTPUT_DIR
    )
    
    print("\n" + "="*70)
    print("Experiment complete! Check the results directory for all outputs.")
    print("="*70)
