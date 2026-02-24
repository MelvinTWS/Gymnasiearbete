import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def create_comparison_graphs(kinetic_csv, de_csv, rl_csv, save_dir="graphs"):
    Path(save_dir).mkdir(exist_ok=True)

    kinetic_df = pd.read_csv(kinetic_csv)
    de_df = pd.read_csv(de_csv)
    rl_df = pd.read_csv(rl_csv)

    plt.style.use('seaborn-v0_8-darkgrid')
    colors = ['#2E86AB', '#A23B72', '#F18F01']

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))

    strategies = ['Kinetic-First', 'DE-First', 'RL-Agent']
    x_pos = np.arange(len(strategies))

    pen_means = [kinetic_df['penetration_rate'].mean() * 100,
                 de_df['penetration_rate'].mean() * 100,
                 rl_df['penetration_rate'].mean() * 100]
    pen_stds = [kinetic_df['penetration_rate'].std() * 100,
                de_df['penetration_rate'].std() * 100,
                rl_df['penetration_rate'].std() * 100]

    bars1 = ax1.bar(x_pos, pen_means, yerr=pen_stds, capsize=5, color=colors, alpha=0.8, edgecolor='black')
    ax1.set_ylabel('Penetration Rate (%)', fontsize=12, fontweight='bold')
    ax1.set_title('UAV Penetration Rate Comparison', fontsize=14, fontweight='bold')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(strategies, fontsize=11)
    ax1.grid(axis='y', alpha=0.3)
    for i, (mean, std) in enumerate(zip(pen_means, pen_stds)):
        ax1.text(i, mean + std + 0.5, f'{mean:.1f}%\n±{std:.1f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')

    cost_means = [kinetic_df['cost_exchange_ratio'].mean(),
                  de_df['cost_exchange_ratio'].mean(),
                  rl_df['cost_exchange_ratio'].mean()]
    cost_stds = [kinetic_df['cost_exchange_ratio'].std(),
                 de_df['cost_exchange_ratio'].std(),
                 rl_df['cost_exchange_ratio'].std()]

    bars2 = ax2.bar(x_pos, cost_means, yerr=cost_stds, capsize=5, color=colors, alpha=0.8, edgecolor='black')
    ax2.set_ylabel('Cost-Exchange Ratio', fontsize=12, fontweight='bold')
    ax2.set_title('Cost-Exchange Ratio Comparison', fontsize=14, fontweight='bold')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(strategies, fontsize=11)
    ax2.grid(axis='y', alpha=0.3)
    for i, (mean, std) in enumerate(zip(cost_means, cost_stds)):
        ax2.text(i, mean + std + 1, f'1:{mean:.1f}\n±{std:.1f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

    total_cost_means = [kinetic_df['total_cost'].mean() / 1e6,
                        de_df['total_cost'].mean() / 1e6,
                        rl_df['total_cost'].mean() / 1e6]
    total_cost_stds = [kinetic_df['total_cost'].std() / 1e6,
                       de_df['total_cost'].std() / 1e6,
                       rl_df['total_cost'].std() / 1e6]

    bars3 = ax3.bar(x_pos, total_cost_means, yerr=total_cost_stds, capsize=5, color=colors, alpha=0.8, edgecolor='black')
    ax3.set_ylabel('Total Cost (Million $)', fontsize=12, fontweight='bold')
    ax3.set_title('Total Cost Comparison', fontsize=14, fontweight='bold')
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels(strategies, fontsize=11)
    ax3.grid(axis='y', alpha=0.3)
    for i, (mean, std) in enumerate(zip(total_cost_means, total_cost_stds)):
        ax3.text(i, mean + std + 5, f'${mean:.1f}M\n±${std:.1f}M', ha='center', va='bottom', fontsize=9, fontweight='bold')

    uavs_means = [kinetic_df['uavs_destroyed'].mean(),
                  de_df['uavs_destroyed'].mean(),
                  rl_df['uavs_destroyed'].mean()]
    uavs_stds = [kinetic_df['uavs_destroyed'].std(),
                 de_df['uavs_destroyed'].std(),
                 rl_df['uavs_destroyed'].std()]

    bars4 = ax4.bar(x_pos, uavs_means, yerr=uavs_stds, capsize=5, color=colors, alpha=0.8, edgecolor='black')
    ax4.set_ylabel('UAVs Destroyed', fontsize=12, fontweight='bold')
    ax4.set_title('UAVs Destroyed Comparison', fontsize=14, fontweight='bold')
    ax4.set_xticks(x_pos)
    ax4.set_xticklabels(strategies, fontsize=11)
    ax4.grid(axis='y', alpha=0.3)
    for i, (mean, std) in enumerate(zip(uavs_means, uavs_stds)):
        ax4.text(i, mean + std + 1, f'{mean:.1f}\n±{std:.1f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

    plt.tight_layout()
    plt.savefig(f"{save_dir}/strategy_comparison_4panel.png", dpi=300, bbox_inches='tight')
    print(f"Saved: {save_dir}/strategy_comparison_4panel.png")

    fig, ax = plt.subplots(figsize=(12, 8))

    metrics = ['Penetration\nRate (%)', 'Cost-Exchange\nRatio', 'Total Cost\n(M$)', 'UAVs\nDestroyed']
    x = np.arange(len(metrics))
    width = 0.25

    kinetic_vals = [pen_means[0], cost_means[0], total_cost_means[0], uavs_means[0]]
    de_vals = [pen_means[1], cost_means[1], total_cost_means[1], uavs_means[1]]
    rl_vals = [pen_means[2], cost_means[2], total_cost_means[2], uavs_means[2]]

    kinetic_errs = [pen_stds[0], cost_stds[0], total_cost_stds[0], uavs_stds[0]]
    de_errs = [pen_stds[1], cost_stds[1], total_cost_stds[1], uavs_stds[1]]
    rl_errs = [pen_stds[2], cost_stds[2], total_cost_stds[2], uavs_stds[2]]

    for i in range(len(metrics)):
        max_val = max(kinetic_vals[i], de_vals[i], rl_vals[i])
        kinetic_vals[i] = (kinetic_vals[i] / max_val) * 100
        de_vals[i] = (de_vals[i] / max_val) * 100
        rl_vals[i] = (rl_vals[i] / max_val) * 100
        kinetic_errs[i] = (kinetic_errs[i] / max_val) * 100
        de_errs[i] = (de_errs[i] / max_val) * 100
        rl_errs[i] = (rl_errs[i] / max_val) * 100

    ax.bar(x - width, kinetic_vals, width, label='Kinetic-First', color=colors[0], alpha=0.8, edgecolor='black', yerr=kinetic_errs, capsize=4)
    ax.bar(x, de_vals, width, label='DE-First', color=colors[1], alpha=0.8, edgecolor='black', yerr=de_errs, capsize=4)
    ax.bar(x + width, rl_vals, width, label='RL-Agent', color=colors[2], alpha=0.8, edgecolor='black', yerr=rl_errs, capsize=4)

    ax.set_ylabel('Normalized Performance (%)', fontsize=13, fontweight='bold')
    ax.set_title('Comprehensive Strategy Performance Comparison\n(Normalized to Maximum)', fontsize=15, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, fontsize=12)
    ax.legend(fontsize=11, loc='upper right', framealpha=0.9)
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim(0, 120)

    plt.tight_layout()
    plt.savefig(f"{save_dir}/comprehensive_normalized_comparison.png", dpi=300, bbox_inches='tight')
    print(f"Saved: {save_dir}/comprehensive_normalized_comparison.png")
    plt.close('all')

if __name__ == "__main__":
    create_comparison_graphs(
        "evaluation_results/kinetic_first_20260222_132347.csv",
        "evaluation_results/de_first_20260222_132347.csv",
        "evaluation_results/rl_agent_20260222_132347.csv",
        save_dir="graphs"
    )
