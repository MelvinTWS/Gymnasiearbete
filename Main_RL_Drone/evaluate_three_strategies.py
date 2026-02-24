import pandas as pd
from pathlib import Path
from typing import Dict, Any
from datetime import datetime
from stable_baselines3 import DQN
from monte_carlo_runner import run_scenarios
from baseline_strategy import kinetic_first_strategy, de_first_strategy
from evaluate_strategies import RLStrategyWrapper


def evaluate_all_strategies(
    model_path: str,
    num_scenarios: int = 100,
    swarm_size_range: tuple = (50, 150),
    kinetic_quantity: int = 800,
    de_quantity: int = 150,
    random_seed: int = 42,
    save_dir: str = "evaluation_results"
) -> Dict[str, Any]:
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    model = DQN.load(model_path)
    rl_strategy = RLStrategyWrapper(model, swarm_size_range=swarm_size_range, deterministic=True)

    kinetic_results = run_scenarios(
        strategy=kinetic_first_strategy,
        num_scenarios=num_scenarios,
        swarm_size_range=swarm_size_range,
        kinetic_quantity=kinetic_quantity,
        de_quantity=de_quantity,
        random_seed=random_seed,
        verbose=False,
        strategy_name="Kinetic_First"
    )

    de_results = run_scenarios(
        strategy=de_first_strategy,
        num_scenarios=num_scenarios,
        swarm_size_range=swarm_size_range,
        kinetic_quantity=kinetic_quantity,
        de_quantity=de_quantity,
        random_seed=random_seed,
        verbose=False,
        strategy_name="DE_First"
    )

    rl_results = run_scenarios(
        strategy=rl_strategy,
        num_scenarios=num_scenarios,
        swarm_size_range=swarm_size_range,
        kinetic_quantity=kinetic_quantity,
        de_quantity=de_quantity,
        random_seed=random_seed,
        verbose=False,
        strategy_name="RL_Agent"
    )

    kinetic_results.to_csv(save_path / f"kinetic_first_{timestamp}.csv", index=False)
    de_results.to_csv(save_path / f"de_first_{timestamp}.csv", index=False)
    rl_results.to_csv(save_path / f"rl_agent_{timestamp}.csv", index=False)

    metrics = ['penetration_rate', 'cost_exchange_ratio', 'defense_cost', 'total_cost', 'uavs_destroyed', 'total_shots_fired']
    results_summary = {'kinetic_first': {}, 'de_first': {}, 'rl_agent': {}}

    for metric in metrics:
        results_summary['kinetic_first'][metric] = {'mean': float(kinetic_results[metric].mean()), 'std': float(kinetic_results[metric].std())}
        results_summary['de_first'][metric] = {'mean': float(de_results[metric].mean()), 'std': float(de_results[metric].std())}
        results_summary['rl_agent'][metric] = {'mean': float(rl_results[metric].mean()), 'std': float(rl_results[metric].std())}

    summary_file = save_path / f"summary_{timestamp}.txt"
    with open(summary_file, 'w') as f:
        f.write(f"Evaluation Summary - {num_scenarios} scenarios\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"{'Metric':<25} {'Strategy':<15} {'Mean':>15} {'Std':>15}\n")
        f.write("-" * 80 + "\n")
        for metric in metrics:
            for strategy_name, display_name in [('kinetic_first', 'Kinetic-First'), ('de_first', 'DE-First'), ('rl_agent', 'RL-Agent')]:
                mean_val = results_summary[strategy_name][metric]['mean']
                std_val = results_summary[strategy_name][metric]['std']
                if metric in ['defense_cost', 'total_cost']:
                    f.write(f"{metric:<25} {display_name:<15} ${mean_val:>14,.2f} ${std_val:>14,.2f}\n")
                elif metric == 'penetration_rate':
                    f.write(f"{metric:<25} {display_name:<15} {mean_val*100:>14.2f}% {std_val*100:>14.2f}%\n")
                else:
                    f.write(f"{metric:<25} {display_name:<15} {mean_val:>15.2f} {std_val:>15.2f}\n")
            f.write("\n")

    return {'kinetic_first_results': kinetic_results, 'de_first_results': de_results, 'rl_results': rl_results, 'summary': results_summary, 'summary_file': str(summary_file)}


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        sys.exit(1)
    model_path = sys.argv[1]
    num_scenarios = int(sys.argv[2]) if len(sys.argv) > 2 else 100
    results = evaluate_all_strategies(model_path=model_path, num_scenarios=num_scenarios)
    with open(results['summary_file'], 'r') as f:
        print(f.read())
