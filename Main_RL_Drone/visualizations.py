"""
Visualization Module

This module creates publication-quality visualizations for comparing
baseline and RL strategies for the master's thesis.

Author: Master's Thesis Project
Date: January 2026
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple
from datetime import datetime


# Set publication-quality defaults
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['figure.titlesize'] = 16

# Use seaborn style for better aesthetics
sns.set_style("whitegrid")
sns.set_palette("Set2")


def plot_penetration_comparison(
    baseline_results: pd.DataFrame,
    rl_results: pd.DataFrame,
    save_path: Optional[str] = None,
    show: bool = True
) -> plt.Figure:
    """
    Create box plot comparing penetration rates.
    
    Args:
        baseline_results: Baseline strategy results DataFrame
        rl_results: RL strategy results DataFrame
        save_path: Path to save figure (optional)
        show: Whether to display the plot
        
    Returns:
        Matplotlib figure object
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Prepare data
    data = pd.DataFrame({
        'Penetration Rate (%)': np.concatenate([
            baseline_results['penetration_rate'].values * 100,
            rl_results['penetration_rate'].values * 100
        ]),
        'Strategy': ['Baseline'] * len(baseline_results) + ['RL Agent'] * len(rl_results)
    })
    
    # Create box plot
    sns.boxplot(
        data=data,
        x='Strategy',
        y='Penetration Rate (%)',
        ax=ax,
        palette=['#E74C3C', '#3498DB'],
        width=0.5
    )
    
    # Add individual points with jitter
    sns.stripplot(
        data=data,
        x='Strategy',
        y='Penetration Rate (%)',
        ax=ax,
        color='black',
        alpha=0.3,
        size=3,
        jitter=True
    )
    
    # Calculate and display means
    baseline_mean = baseline_results['penetration_rate'].mean() * 100
    rl_mean = rl_results['penetration_rate'].mean() * 100
    
    ax.axhline(y=baseline_mean, color='#E74C3C', linestyle='--', alpha=0.7, linewidth=1.5, label=f'Baseline Mean: {baseline_mean:.2f}%')
    ax.axhline(y=rl_mean, color='#3498DB', linestyle='--', alpha=0.7, linewidth=1.5, label=f'RL Agent Mean: {rl_mean:.2f}%')
    
    # Formatting
    ax.set_title('UAV Penetration Rate Comparison', fontweight='bold', pad=20)
    ax.set_ylabel('Penetration Rate (%)', fontweight='bold')
    ax.set_xlabel('Strategy', fontweight='bold')
    ax.legend(loc='upper right', framealpha=0.9)
    ax.grid(True, alpha=0.3)
    
    # Add improvement annotation
    improvement = ((baseline_mean - rl_mean) / baseline_mean) * 100
    ax.text(
        0.5, 0.02,
        f'Improvement: {improvement:+.2f}%',
        transform=ax.transAxes,
        ha='center',
        fontsize=11,
        fontweight='bold',
        bbox=dict(boxstyle='round', facecolor='lightgreen' if improvement > 0 else 'lightcoral', alpha=0.7)
    )
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()
    
    return fig


def plot_cost_exchange_comparison(
    baseline_results: pd.DataFrame,
    rl_results: pd.DataFrame,
    save_path: Optional[str] = None,
    show: bool = True
) -> plt.Figure:
    """
    Create box plot comparing cost-exchange ratios.
    
    Args:
        baseline_results: Baseline strategy results DataFrame
        rl_results: RL strategy results DataFrame
        save_path: Path to save figure (optional)
        show: Whether to display the plot
        
    Returns:
        Matplotlib figure object
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Prepare data
    data = pd.DataFrame({
        'Cost-Exchange Ratio': np.concatenate([
            baseline_results['cost_exchange_ratio'].values,
            rl_results['cost_exchange_ratio'].values
        ]),
        'Strategy': ['Baseline'] * len(baseline_results) + ['RL Agent'] * len(rl_results)
    })
    
    # Create box plot
    sns.boxplot(
        data=data,
        x='Strategy',
        y='Cost-Exchange Ratio',
        ax=ax,
        palette=['#E74C3C', '#3498DB'],
        width=0.5
    )
    
    # Add individual points with jitter
    sns.stripplot(
        data=data,
        x='Strategy',
        y='Cost-Exchange Ratio',
        ax=ax,
        color='black',
        alpha=0.3,
        size=3,
        jitter=True
    )
    
    # Calculate and display means
    baseline_mean = baseline_results['cost_exchange_ratio'].mean()
    rl_mean = rl_results['cost_exchange_ratio'].mean()
    
    ax.axhline(y=baseline_mean, color='#E74C3C', linestyle='--', alpha=0.7, linewidth=1.5, label=f'Baseline Mean: {baseline_mean:.2f}')
    ax.axhline(y=rl_mean, color='#3498DB', linestyle='--', alpha=0.7, linewidth=1.5, label=f'RL Agent Mean: {rl_mean:.2f}')
    
    # Formatting
    ax.set_title('Cost-Exchange Ratio Comparison', fontweight='bold', pad=20)
    ax.set_ylabel('Cost-Exchange Ratio (Penetration Cost / Defense Cost)', fontweight='bold')
    ax.set_xlabel('Strategy', fontweight='bold')
    ax.legend(loc='upper right', framealpha=0.9)
    ax.grid(True, alpha=0.3)
    
    # Add improvement annotation
    improvement = ((baseline_mean - rl_mean) / baseline_mean) * 100
    ax.text(
        0.5, 0.02,
        f'Improvement: {improvement:+.2f}%',
        transform=ax.transAxes,
        ha='center',
        fontsize=11,
        fontweight='bold',
        bbox=dict(boxstyle='round', facecolor='lightgreen' if improvement > 0 else 'lightcoral', alpha=0.7)
    )
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()
    
    return fig


def plot_cost_vs_penetration_scatter(
    baseline_results: pd.DataFrame,
    rl_results: pd.DataFrame,
    save_path: Optional[str] = None,
    show: bool = True
) -> plt.Figure:
    """
    Create scatter plot of total cost vs penetration rate.
    
    Args:
        baseline_results: Baseline strategy results DataFrame
        rl_results: RL strategy results DataFrame
        save_path: Path to save figure (optional)
        show: Whether to display the plot
        
    Returns:
        Matplotlib figure object
    """
    fig, ax = plt.subplots(figsize=(10, 7))
    
    # Baseline scatter
    ax.scatter(
        baseline_results['penetration_rate'] * 100,
        baseline_results['total_cost'] / 1e6,
        alpha=0.6,
        s=50,
        color='#E74C3C',
        label='Baseline',
        edgecolors='black',
        linewidth=0.5
    )
    
    # RL scatter
    ax.scatter(
        rl_results['penetration_rate'] * 100,
        rl_results['total_cost'] / 1e6,
        alpha=0.6,
        s=50,
        color='#3498DB',
        label='RL Agent',
        edgecolors='black',
        linewidth=0.5,
        marker='s'
    )
    
    # Add means with error bars
    baseline_pen_mean = baseline_results['penetration_rate'].mean() * 100
    baseline_cost_mean = baseline_results['total_cost'].mean() / 1e6
    baseline_pen_std = baseline_results['penetration_rate'].std() * 100
    baseline_cost_std = baseline_results['total_cost'].std() / 1e6
    
    rl_pen_mean = rl_results['penetration_rate'].mean() * 100
    rl_cost_mean = rl_results['total_cost'].mean() / 1e6
    rl_pen_std = rl_results['penetration_rate'].std() * 100
    rl_cost_std = rl_results['total_cost'].std() / 1e6
    
    ax.errorbar(
        baseline_pen_mean, baseline_cost_mean,
        xerr=baseline_pen_std, yerr=baseline_cost_std,
        fmt='D', color='#C0392B', markersize=10,
        linewidth=2, capsize=5, capthick=2,
        label='Baseline Mean ± SD'
    )
    
    ax.errorbar(
        rl_pen_mean, rl_cost_mean,
        xerr=rl_pen_std, yerr=rl_cost_std,
        fmt='D', color='#2874A6', markersize=10,
        linewidth=2, capsize=5, capthick=2,
        label='RL Agent Mean ± SD'
    )
    
    # Formatting
    ax.set_title('Total Cost vs Penetration Rate Trade-off', fontweight='bold', pad=20)
    ax.set_xlabel('Penetration Rate (%)', fontweight='bold')
    ax.set_ylabel('Total Cost ($ Millions)', fontweight='bold')
    ax.legend(loc='best', framealpha=0.9)
    ax.grid(True, alpha=0.3)
    
    # Add ideal region annotation
    ax.axvline(x=50, color='green', linestyle=':', alpha=0.3, linewidth=2)
    ax.text(
        50, ax.get_ylim()[1] * 0.95,
        'Lower is better',
        rotation=90,
        va='top',
        ha='right',
        color='green',
        fontweight='bold'
    )
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()
    
    return fig


def plot_training_metrics(
    metrics_file: str,
    save_path: Optional[str] = None,
    show: bool = True
) -> plt.Figure:
    """
    Plot training metrics over time.
    
    Args:
        metrics_file: Path to CSV file with training metrics
        save_path: Path to save figure (optional)
        show: Whether to display the plot
        
    Returns:
        Matplotlib figure object
    """
    # Load metrics
    metrics_df = pd.read_csv(metrics_file)
    
    # Create episode index if not present
    if 'timestep' not in metrics_df.columns:
        metrics_df['episode'] = range(len(metrics_df))
        x_col = 'episode'
        x_label = 'Episode'
    else:
        x_col = 'timestep'
        x_label = 'Timestep'
    
    # Map column names
    reward_col = 'episode_reward' if 'episode_reward' in metrics_df.columns else 'episode_rewards'
    length_col = 'episode_length' if 'episode_length' in metrics_df.columns else 'episode_lengths'
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Episode Reward
    ax = axes[0, 0]
    ax.plot(metrics_df[x_col], metrics_df[reward_col], alpha=0.3, color='blue')
    # Rolling average
    window = min(50, len(metrics_df) // 10)
    if window > 1:
        rolling_reward = metrics_df[reward_col].rolling(window=window, center=True).mean()
        ax.plot(metrics_df[x_col], rolling_reward, color='darkblue', linewidth=2, label=f'{window}-episode MA')
    ax.set_xlabel(x_label)
    ax.set_ylabel('Episode Reward')
    ax.set_title('Training Reward Progress', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Episode Length
    ax = axes[0, 1]
    ax.plot(metrics_df[x_col], metrics_df[length_col], alpha=0.3, color='green')
    if window > 1:
        rolling_length = metrics_df[length_col].rolling(window=window, center=True).mean()
        ax.plot(metrics_df[x_col], rolling_length, color='darkgreen', linewidth=2, label=f'{window}-episode MA')
    ax.set_xlabel(x_label)
    ax.set_ylabel('Episode Length (steps)')
    ax.set_title('Episode Length Over Time', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Penetration Rate (if available)
    ax = axes[1, 0]
    if 'penetration_rates' in metrics_df.columns or 'penetration_rate' in metrics_df.columns:
        pen_col = 'penetration_rate' if 'penetration_rate' in metrics_df.columns else 'penetration_rates'
        ax.plot(metrics_df[x_col], metrics_df[pen_col] * 100, alpha=0.3, color='red')
        if window > 1:
            rolling_pen = (metrics_df[pen_col] * 100).rolling(window=window, center=True).mean()
            ax.plot(metrics_df[x_col], rolling_pen, color='darkred', linewidth=2, label=f'{window}-episode MA')
        ax.set_xlabel(x_label)
        ax.set_ylabel('Penetration Rate (%)')
        ax.set_title('Penetration Rate Over Training', fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
    else:
        ax.text(0.5, 0.5, 'Penetration rate data not available', 
                ha='center', va='center', transform=ax.transAxes)
        ax.axis('off')
    
    # Plot 4: Defense Cost (if available)
    ax = axes[1, 1]
    if 'defense_costs' in metrics_df.columns or 'defense_cost' in metrics_df.columns:
        def_col = 'defense_cost' if 'defense_cost' in metrics_df.columns else 'defense_costs'
        ax.plot(metrics_df[x_col], metrics_df[def_col] / 1e6, alpha=0.3, color='orange')
        if window > 1:
            rolling_def = (metrics_df[def_col] / 1e6).rolling(window=window, center=True).mean()
            ax.plot(metrics_df[x_col], rolling_def, color='darkorange', linewidth=2, label=f'{window}-episode MA')
        ax.set_xlabel(x_label)
        ax.set_ylabel('Defense Cost ($ Millions)')
        ax.set_title('Defense Cost Over Training', fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
    else:
        ax.text(0.5, 0.5, 'Defense cost data not available', 
                ha='center', va='center', transform=ax.transAxes)
        ax.axis('off')
    
    plt.suptitle('RL Agent Training Metrics', fontsize=16, fontweight='bold', y=1.00)
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()
    
    return fig


def plot_weapon_usage_comparison(
    baseline_results: pd.DataFrame,
    rl_results: pd.DataFrame,
    save_path: Optional[str] = None,
    show: bool = True
) -> plt.Figure:
    """
    Create comparison of weapon usage patterns.
    
    Args:
        baseline_results: Baseline strategy results DataFrame
        rl_results: RL strategy results DataFrame
        save_path: Path to save figure (optional)
        show: Whether to display the plot
        
    Returns:
        Matplotlib figure object
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Kinetic missiles
    ax = axes[0]
    data_kinetic = pd.DataFrame({
        'Kinetic Missiles Fired': np.concatenate([
            baseline_results['kinetic_fired'].values,
            rl_results['kinetic_fired'].values
        ]),
        'Strategy': ['Baseline'] * len(baseline_results) + ['RL Agent'] * len(rl_results)
    })
    
    sns.boxplot(
        data=data_kinetic,
        x='Strategy',
        y='Kinetic Missiles Fired',
        ax=ax,
        palette=['#E74C3C', '#3498DB'],
        width=0.5
    )
    
    ax.set_title('Kinetic Missile Usage', fontweight='bold', pad=15)
    ax.set_ylabel('Number of Kinetic Missiles', fontweight='bold')
    ax.set_xlabel('Strategy', fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Add mean annotations
    baseline_kinetic_mean = baseline_results['kinetic_fired'].mean()
    rl_kinetic_mean = rl_results['kinetic_fired'].mean()
    ax.text(0, ax.get_ylim()[1] * 0.95, f'μ = {baseline_kinetic_mean:.1f}', 
            ha='center', fontweight='bold', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    ax.text(1, ax.get_ylim()[1] * 0.95, f'μ = {rl_kinetic_mean:.1f}', 
            ha='center', fontweight='bold', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Directed Energy
    ax = axes[1]
    data_de = pd.DataFrame({
        'DE Weapons Fired': np.concatenate([
            baseline_results['de_fired'].values,
            rl_results['de_fired'].values
        ]),
        'Strategy': ['Baseline'] * len(baseline_results) + ['RL Agent'] * len(rl_results)
    })
    
    sns.boxplot(
        data=data_de,
        x='Strategy',
        y='DE Weapons Fired',
        ax=ax,
        palette=['#E74C3C', '#3498DB'],
        width=0.5
    )
    
    ax.set_title('Directed Energy Weapon Usage', fontweight='bold', pad=15)
    ax.set_ylabel('Number of DE Shots', fontweight='bold')
    ax.set_xlabel('Strategy', fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Add mean annotations
    baseline_de_mean = baseline_results['de_fired'].mean()
    rl_de_mean = rl_results['de_fired'].mean()
    ax.text(0, ax.get_ylim()[1] * 0.95, f'μ = {baseline_de_mean:.1f}', 
            ha='center', fontweight='bold', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    ax.text(1, ax.get_ylim()[1] * 0.95, f'μ = {rl_de_mean:.1f}', 
            ha='center', fontweight='bold', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.suptitle('Weapon System Usage Comparison', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()
    
    return fig


def plot_comprehensive_comparison(
    baseline_results: pd.DataFrame,
    rl_results: pd.DataFrame,
    save_path: Optional[str] = None,
    show: bool = True
) -> plt.Figure:
    """
    Create comprehensive multi-panel comparison figure.
    
    Args:
        baseline_results: Baseline strategy results DataFrame
        rl_results: RL strategy results DataFrame
        save_path: Path to save figure (optional)
        show: Whether to display the plot
        
    Returns:
        Matplotlib figure object
    """
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
    
    # 1. Penetration Rate Box Plot
    ax1 = fig.add_subplot(gs[0, 0])
    data_pen = pd.DataFrame({
        'Value': np.concatenate([
            baseline_results['penetration_rate'].values * 100,
            rl_results['penetration_rate'].values * 100
        ]),
        'Strategy': ['Baseline'] * len(baseline_results) + ['RL Agent'] * len(rl_results)
    })
    sns.boxplot(data=data_pen, x='Strategy', y='Value', ax=ax1, palette=['#E74C3C', '#3498DB'])
    ax1.set_ylabel('Penetration Rate (%)', fontweight='bold')
    ax1.set_xlabel('')
    ax1.set_title('(A) Penetration Rate', fontweight='bold', loc='left')
    ax1.grid(True, alpha=0.3)
    
    # 2. Cost-Exchange Ratio Box Plot
    ax2 = fig.add_subplot(gs[0, 1])
    data_cost = pd.DataFrame({
        'Value': np.concatenate([
            baseline_results['cost_exchange_ratio'].values,
            rl_results['cost_exchange_ratio'].values
        ]),
        'Strategy': ['Baseline'] * len(baseline_results) + ['RL Agent'] * len(rl_results)
    })
    sns.boxplot(data=data_cost, x='Strategy', y='Value', ax=ax2, palette=['#E74C3C', '#3498DB'])
    ax2.set_ylabel('Cost-Exchange Ratio', fontweight='bold')
    ax2.set_xlabel('')
    ax2.set_title('(B) Cost-Exchange Ratio', fontweight='bold', loc='left')
    ax2.grid(True, alpha=0.3)
    
    # 3. Total Cost Box Plot
    ax3 = fig.add_subplot(gs[1, 0])
    data_total = pd.DataFrame({
        'Value': np.concatenate([
            baseline_results['total_cost'].values / 1e6,
            rl_results['total_cost'].values / 1e6
        ]),
        'Strategy': ['Baseline'] * len(baseline_results) + ['RL Agent'] * len(rl_results)
    })
    sns.boxplot(data=data_total, x='Strategy', y='Value', ax=ax3, palette=['#E74C3C', '#3498DB'])
    ax3.set_ylabel('Total Cost ($ Millions)', fontweight='bold')
    ax3.set_xlabel('')
    ax3.set_title('(C) Total Cost', fontweight='bold', loc='left')
    ax3.grid(True, alpha=0.3)
    
    # 4. UAVs Destroyed
    ax4 = fig.add_subplot(gs[1, 1])
    data_destroyed = pd.DataFrame({
        'Value': np.concatenate([
            baseline_results['uavs_destroyed'].values,
            rl_results['uavs_destroyed'].values
        ]),
        'Strategy': ['Baseline'] * len(baseline_results) + ['RL Agent'] * len(rl_results)
    })
    sns.boxplot(data=data_destroyed, x='Strategy', y='Value', ax=ax4, palette=['#E74C3C', '#3498DB'])
    ax4.set_ylabel('UAVs Destroyed', fontweight='bold')
    ax4.set_xlabel('')
    ax4.set_title('(D) UAVs Destroyed', fontweight='bold', loc='left')
    ax4.grid(True, alpha=0.3)
    
    # 5. Scatter: Cost vs Penetration
    ax5 = fig.add_subplot(gs[2, :])
    ax5.scatter(
        baseline_results['penetration_rate'] * 100,
        baseline_results['total_cost'] / 1e6,
        alpha=0.5, s=40, color='#E74C3C', label='Baseline'
    )
    ax5.scatter(
        rl_results['penetration_rate'] * 100,
        rl_results['total_cost'] / 1e6,
        alpha=0.5, s=40, color='#3498DB', label='RL Agent', marker='s'
    )
    ax5.set_xlabel('Penetration Rate (%)', fontweight='bold')
    ax5.set_ylabel('Total Cost ($ Millions)', fontweight='bold')
    ax5.set_title('(E) Cost vs Penetration Trade-off', fontweight='bold', loc='left')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    plt.suptitle('Comprehensive Strategy Comparison', fontsize=18, fontweight='bold', y=0.995)
    
    if save_path:
        fig.savefig(save_path, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()
    
    return fig


def create_all_visualizations(
    baseline_results: pd.DataFrame,
    rl_results: pd.DataFrame,
    training_metrics_file: Optional[str] = None,
    save_dir: str = "visualizations",
    show: bool = False
) -> Dict[str, str]:
    """
    Create all visualizations and save to directory.
    
    Args:
        baseline_results: Baseline strategy results DataFrame
        rl_results: RL strategy results DataFrame
        training_metrics_file: Path to training metrics CSV (optional)
        save_dir: Directory to save all figures
        show: Whether to display plots
        
    Returns:
        Dictionary mapping plot names to file paths
    """
    # Create save directory
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    saved_files = {}
    
    print("\n" + "="*70)
    print("GENERATING VISUALIZATIONS")
    print("="*70)
    
    # 1. Penetration comparison
    print("\n[1/6] Creating penetration rate comparison...")
    file_path = save_path / f"penetration_comparison_{timestamp}.png"
    plot_penetration_comparison(baseline_results, rl_results, save_path=str(file_path), show=show)
    saved_files['penetration_comparison'] = str(file_path)
    
    # 2. Cost-exchange comparison
    print("[2/6] Creating cost-exchange ratio comparison...")
    file_path = save_path / f"cost_exchange_comparison_{timestamp}.png"
    plot_cost_exchange_comparison(baseline_results, rl_results, save_path=str(file_path), show=show)
    saved_files['cost_exchange_comparison'] = str(file_path)
    
    # 3. Scatter plot
    print("[3/6] Creating cost vs penetration scatter plot...")
    file_path = save_path / f"cost_vs_penetration_{timestamp}.png"
    plot_cost_vs_penetration_scatter(baseline_results, rl_results, save_path=str(file_path), show=show)
    saved_files['cost_vs_penetration'] = str(file_path)
    
    # 4. Weapon usage
    print("[4/6] Creating weapon usage comparison...")
    file_path = save_path / f"weapon_usage_{timestamp}.png"
    plot_weapon_usage_comparison(baseline_results, rl_results, save_path=str(file_path), show=show)
    saved_files['weapon_usage'] = str(file_path)
    
    # 5. Comprehensive comparison
    print("[5/6] Creating comprehensive comparison figure...")
    file_path = save_path / f"comprehensive_comparison_{timestamp}.png"
    plot_comprehensive_comparison(baseline_results, rl_results, save_path=str(file_path), show=show)
    saved_files['comprehensive_comparison'] = str(file_path)
    
    # 6. Training metrics (if available)
    if training_metrics_file and Path(training_metrics_file).exists():
        print("[6/6] Creating training metrics plots...")
        file_path = save_path / f"training_metrics_{timestamp}.png"
        plot_training_metrics(training_metrics_file, save_path=str(file_path), show=show)
        saved_files['training_metrics'] = str(file_path)
    else:
        print("[6/6] Skipping training metrics (file not provided)")
    
    print("\n" + "="*70)
    print(f"ALL VISUALIZATIONS SAVED TO: {save_dir}")
    print("="*70)
    print("\nGenerated files:")
    for name, path in saved_files.items():
        print(f"  - {name}: {Path(path).name}")
    
    return saved_files


if __name__ == "__main__":
    """
    Test and demonstration code.
    """
    print("="*70)
    print("VISUALIZATION MODULE - DEMONSTRATION")
    print("="*70)
    
    # Test 1: Generate synthetic test data
    print("\n[Test 1] Generate synthetic test data:")
    
    np.random.seed(42)
    
    # Simulate baseline results
    n_scenarios = 100
    baseline_data = {
        'penetration_rate': np.random.normal(0.85, 0.05, n_scenarios),
        'cost_exchange_ratio': np.random.normal(26.0, 2.0, n_scenarios),
        'total_cost': np.random.normal(200e6, 20e6, n_scenarios),
        'defense_cost': np.random.normal(1e6, 200e3, n_scenarios),
        'penetration_cost': np.random.normal(199e6, 20e6, n_scenarios),
        'uavs_destroyed': np.random.normal(50, 10, n_scenarios),
        'uavs_penetrated': np.random.normal(250, 20, n_scenarios),
        'kinetic_fired': np.random.normal(10, 3, n_scenarios),
        'de_fired': np.random.normal(20, 5, n_scenarios)
    }
    
    # Simulate RL results (better performance)
    rl_data = {
        'penetration_rate': np.random.normal(0.75, 0.05, n_scenarios),
        'cost_exchange_ratio': np.random.normal(22.0, 2.0, n_scenarios),
        'total_cost': np.random.normal(175e6, 18e6, n_scenarios),
        'defense_cost': np.random.normal(1.2e6, 250e3, n_scenarios),
        'penetration_cost': np.random.normal(174e6, 18e6, n_scenarios),
        'uavs_destroyed': np.random.normal(75, 12, n_scenarios),
        'uavs_penetrated': np.random.normal(225, 18, n_scenarios),
        'kinetic_fired': np.random.normal(12, 4, n_scenarios),
        'de_fired': np.random.normal(18, 5, n_scenarios)
    }
    
    baseline_df = pd.DataFrame(baseline_data)
    rl_df = pd.DataFrame(rl_data)
    
    print(f"  Baseline scenarios: {len(baseline_df)}")
    print(f"  RL scenarios: {len(rl_df)}")
    print(f"  Baseline penetration: {baseline_df['penetration_rate'].mean()*100:.2f}%")
    print(f"  RL penetration: {rl_df['penetration_rate'].mean()*100:.2f}%")
    
    # Test 2: Create synthetic training metrics
    print("\n[Test 2] Create synthetic training metrics:")
    
    n_episodes = 500
    training_data = {
        'timestep': np.arange(1, n_episodes + 1) * 100,
        'episode_reward': -1e7 + np.cumsum(np.random.randn(n_episodes) * 1e5 + 2e4),
        'episode_length': np.random.randint(80, 120, n_episodes),
        'penetration_cost': 200e6 - np.cumsum(np.random.randn(n_episodes) * 1e5 + 5e4),
        'defense_cost': np.random.normal(1e6, 200e3, n_episodes)
    }
    
    training_df = pd.DataFrame(training_data)
    training_file = "test_training_metrics.csv"
    training_df.to_csv(training_file, index=False)
    print(f"  Created: {training_file}")
    print(f"  Episodes: {len(training_df)}")
    
    # Test 3: Individual plot tests
    print("\n[Test 3] Test individual plot functions:")
    print("  (Plots will not be displayed, only saved)")
    
    test_dir = Path("test_visualizations")
    test_dir.mkdir(exist_ok=True)
    
    print("\n  Testing penetration comparison...")
    plot_penetration_comparison(baseline_df, rl_df, 
                                save_path=str(test_dir / "test_penetration.png"), 
                                show=False)
    
    print("  Testing cost-exchange comparison...")
    plot_cost_exchange_comparison(baseline_df, rl_df,
                                  save_path=str(test_dir / "test_cost_exchange.png"),
                                  show=False)
    
    print("  Testing scatter plot...")
    plot_cost_vs_penetration_scatter(baseline_df, rl_df,
                                     save_path=str(test_dir / "test_scatter.png"),
                                     show=False)
    
    print("  Testing weapon usage...")
    plot_weapon_usage_comparison(baseline_df, rl_df,
                                save_path=str(test_dir / "test_weapons.png"),
                                show=False)
    
    print("  Testing comprehensive comparison...")
    plot_comprehensive_comparison(baseline_df, rl_df,
                                 save_path=str(test_dir / "test_comprehensive.png"),
                                 show=False)
    
    print("  Testing training metrics...")
    plot_training_metrics(training_file,
                         save_path=str(test_dir / "test_training.png"),
                         show=False)
    
    # Test 4: Create all visualizations at once
    print("\n[Test 4] Test create_all_visualizations():")
    
    saved_files = create_all_visualizations(
        baseline_df,
        rl_df,
        training_metrics_file=training_file,
        save_dir=str(test_dir),
        show=False
    )
    
    # Verify all files created
    print("\n[Test 5] Verify all files created:")
    all_files = list(test_dir.glob("*.png"))
    print(f"  Total PNG files: {len(all_files)}")
    
    for f in sorted(all_files):
        size_kb = f.stat().st_size / 1024
        print(f"    - {f.name} ({size_kb:.1f} KB)")
    
    # Cleanup
    print("\n[Test 6] Cleanup test files:")
    import shutil
    
    if test_dir.exists():
        shutil.rmtree(test_dir)
        print(f"  Removed: {test_dir}")
    
    if Path(training_file).exists():
        Path(training_file).unlink()
        print(f"  Removed: {training_file}")
    
    print("\n" + "="*70)
    print("ALL TESTS COMPLETED SUCCESSFULLY")
    print("="*70)
    print("\nReady for thesis visualization with:")
    print("  from visualizations import create_all_visualizations")
    print("  files = create_all_visualizations(baseline_df, rl_df, save_dir='figures')")
