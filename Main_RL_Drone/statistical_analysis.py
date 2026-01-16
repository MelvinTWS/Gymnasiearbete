"""
Statistical Analysis Module

This module performs statistical tests to compare baseline and RL strategies
for the master's thesis. Uses t-tests to determine statistical significance.

Author: Master's Thesis Project
Date: January 2026
"""

import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, Any, Optional, Tuple
from pathlib import Path
from datetime import datetime


def perform_t_test(
    baseline_data: np.ndarray,
    rl_data: np.ndarray,
    alternative: str = 'two-sided',
    alpha: float = 0.05
) -> Dict[str, Any]:
    """
    Perform independent samples t-test.
    
    Args:
        baseline_data: Baseline strategy metric values
        rl_data: RL strategy metric values
        alternative: Type of test ('two-sided', 'less', 'greater')
        alpha: Significance level (default 0.05)
        
    Returns:
        Dictionary containing:
            - t_statistic: T-test statistic
            - p_value: P-value
            - significant: Whether result is statistically significant
            - baseline_mean: Mean of baseline data
            - rl_mean: Mean of RL data
            - baseline_std: Standard deviation of baseline
            - rl_std: Standard deviation of RL
            - effect_size: Cohen's d effect size
    """
    # Perform t-test
    t_stat, p_value = stats.ttest_ind(baseline_data, rl_data, alternative=alternative)
    
    # Calculate descriptive statistics
    baseline_mean = np.mean(baseline_data)
    baseline_std = np.std(baseline_data, ddof=1)
    rl_mean = np.mean(rl_data)
    rl_std = np.std(rl_data, ddof=1)
    
    # Calculate Cohen's d (effect size)
    pooled_std = np.sqrt(((len(baseline_data) - 1) * baseline_std**2 + 
                          (len(rl_data) - 1) * rl_std**2) / 
                         (len(baseline_data) + len(rl_data) - 2))
    cohens_d = (baseline_mean - rl_mean) / pooled_std if pooled_std > 0 else 0
    
    return {
        't_statistic': float(t_stat),
        'p_value': float(p_value),
        'significant': bool(p_value < alpha),
        'baseline_mean': float(baseline_mean),
        'rl_mean': float(rl_mean),
        'baseline_std': float(baseline_std),
        'rl_std': float(rl_std),
        'effect_size': float(cohens_d),
        'alpha': float(alpha),
        'n_baseline': int(len(baseline_data)),
        'n_rl': int(len(rl_data))
    }


def analyze_strategies(
    baseline_results: pd.DataFrame,
    rl_results: pd.DataFrame,
    alpha: float = 0.05
) -> Dict[str, Dict[str, Any]]:
    """
    Perform comprehensive statistical analysis comparing strategies.
    
    Args:
        baseline_results: DataFrame with baseline strategy results
        rl_results: DataFrame with RL strategy results
        alpha: Significance level for hypothesis tests
        
    Returns:
        Dictionary of statistical test results for each metric
    """
    analysis = {}
    
    # Define metrics and their test directions
    # 'less' means we expect RL < baseline (RL is better)
    # 'greater' means we expect RL > baseline (RL is better)
    metrics = [
        ('penetration_rate', 'less', 'Penetration Rate'),
        ('cost_exchange_ratio', 'less', 'Cost-Exchange Ratio'),
        ('total_cost', 'less', 'Total Cost'),
        ('defense_cost', 'less', 'Defense Cost'),
        ('penetration_cost', 'less', 'Penetration Cost'),
        ('uavs_destroyed', 'greater', 'UAVs Destroyed'),
        ('uavs_penetrated', 'less', 'UAVs Penetrated'),
        ('kinetic_fired', 'two-sided', 'Kinetic Missiles Fired'),
        ('de_fired', 'two-sided', 'Directed Energy Fired')
    ]
    
    for metric, alternative, label in metrics:
        if metric in baseline_results.columns and metric in rl_results.columns:
            baseline_data = baseline_results[metric].values
            rl_data = rl_results[metric].values
            
            result = perform_t_test(baseline_data, rl_data, alternative, alpha)
            result['metric_name'] = label
            result['alternative'] = alternative
            
            analysis[metric] = result
    
    return analysis


def generate_summary_table(
    analysis: Dict[str, Dict[str, Any]],
    format: str = 'text'
) -> str:
    """
    Generate formatted summary table of statistical analysis.
    
    Args:
        analysis: Dictionary of analysis results from analyze_strategies
        format: Output format ('text', 'latex', 'markdown')
        
    Returns:
        Formatted table string
    """
    if format == 'text':
        return _generate_text_table(analysis)
    elif format == 'latex':
        return _generate_latex_table(analysis)
    elif format == 'markdown':
        return _generate_markdown_table(analysis)
    else:
        raise ValueError(f"Unknown format: {format}")


def _generate_text_table(analysis: Dict[str, Dict[str, Any]]) -> str:
    """Generate plain text summary table."""
    lines = []
    lines.append("\n" + "=" * 100)
    lines.append("STATISTICAL ANALYSIS SUMMARY")
    lines.append("=" * 100)
    
    lines.append(f"\n{'Metric':<30} {'Baseline Mean':>15} {'RL Mean':>15} {'p-value':>12} {'Sig.':>6} {'Effect Size':>12}")
    lines.append("-" * 100)
    
    for metric, result in analysis.items():
        baseline_mean = result['baseline_mean']
        rl_mean = result['rl_mean']
        p_value = result['p_value']
        significant = "YES" if result['significant'] else "NO"
        effect_size = result['effect_size']
        metric_name = result['metric_name']
        
        # Format values based on metric
        if 'rate' in metric.lower():
            baseline_str = f"{baseline_mean*100:>14.2f}%"
            rl_str = f"{rl_mean*100:>14.2f}%"
        elif 'cost' in metric.lower():
            baseline_str = f"${baseline_mean/1e6:>13.2f}M"
            rl_str = f"${rl_mean/1e6:>13.2f}M"
        else:
            baseline_str = f"{baseline_mean:>15.2f}"
            rl_str = f"{rl_mean:>15.2f}"
        
        p_str = f"{p_value:>12.6f}" if p_value >= 0.001 else f"{'<0.001':>12}"
        effect_str = f"{effect_size:>12.3f}"
        
        lines.append(f"{metric_name:<30} {baseline_str} {rl_str} {p_str} {significant:>6} {effect_str}")
    
    lines.append("=" * 100)
    lines.append("\nNotes:")
    lines.append(f"  - Significance level (alpha): {list(analysis.values())[0]['alpha']}")
    lines.append(f"  - Number of scenarios: {list(analysis.values())[0]['n_baseline']}")
    lines.append("  - Effect size is Cohen's d (small: 0.2, medium: 0.5, large: 0.8)")
    lines.append("  - 'Sig.' indicates statistical significance at specified alpha level")
    lines.append("=" * 100 + "\n")
    
    return "\n".join(lines)


def _generate_latex_table(analysis: Dict[str, Dict[str, Any]]) -> str:
    """Generate LaTeX table for thesis."""
    lines = []
    lines.append("\\begin{table}[htbp]")
    lines.append("\\centering")
    lines.append("\\caption{Statistical Comparison of Baseline and RL Strategies}")
    lines.append("\\label{tab:statistical_comparison}")
    lines.append("\\begin{tabular}{lrrrrr}")
    lines.append("\\toprule")
    lines.append("Metric & Baseline Mean & RL Mean & $p$-value & Significant & Effect Size \\\\")
    lines.append("\\midrule")
    
    for metric, result in analysis.items():
        baseline_mean = result['baseline_mean']
        rl_mean = result['rl_mean']
        p_value = result['p_value']
        significant = "Yes" if result['significant'] else "No"
        effect_size = result['effect_size']
        metric_name = result['metric_name']
        
        # Format values
        if 'rate' in metric.lower():
            baseline_str = f"{baseline_mean*100:.2f}\\%"
            rl_str = f"{rl_mean*100:.2f}\\%"
        elif 'cost' in metric.lower():
            baseline_str = f"\\${baseline_mean/1e6:.2f}M"
            rl_str = f"\\${rl_mean/1e6:.2f}M"
        else:
            baseline_str = f"{baseline_mean:.2f}"
            rl_str = f"{rl_mean:.2f}"
        
        p_str = f"{p_value:.4f}" if p_value >= 0.001 else "$<$0.001"
        effect_str = f"{effect_size:.3f}"
        
        lines.append(f"{metric_name} & {baseline_str} & {rl_str} & {p_str} & {significant} & {effect_str} \\\\")
    
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")
    
    return "\n".join(lines)


def _generate_markdown_table(analysis: Dict[str, Dict[str, Any]]) -> str:
    """Generate Markdown table."""
    lines = []
    lines.append("\n## Statistical Analysis Summary\n")
    lines.append("| Metric | Baseline Mean | RL Mean | p-value | Significant | Effect Size |")
    lines.append("|--------|---------------|---------|---------|-------------|-------------|")
    
    for metric, result in analysis.items():
        baseline_mean = result['baseline_mean']
        rl_mean = result['rl_mean']
        p_value = result['p_value']
        significant = "YES" if result['significant'] else "NO"
        effect_size = result['effect_size']
        metric_name = result['metric_name']
        
        # Format values
        if 'rate' in metric.lower():
            baseline_str = f"{baseline_mean*100:.2f}%"
            rl_str = f"{rl_mean*100:.2f}%"
        elif 'cost' in metric.lower():
            baseline_str = f"${baseline_mean/1e6:.2f}M"
            rl_str = f"${rl_mean/1e6:.2f}M"
        else:
            baseline_str = f"{baseline_mean:.2f}"
            rl_str = f"{rl_mean:.2f}"
        
        p_str = f"{p_value:.6f}" if p_value >= 0.001 else "<0.001"
        effect_str = f"{effect_size:.3f}"
        
        lines.append(f"| {metric_name} | {baseline_str} | {rl_str} | {p_str} | {significant} | {effect_str} |")
    
    lines.append("\n**Notes:**")
    lines.append(f"- Significance level (α): {list(analysis.values())[0]['alpha']}")
    lines.append(f"- Sample size: {list(analysis.values())[0]['n_baseline']} scenarios per strategy")
    lines.append("- Effect size is Cohen's d (small: 0.2, medium: 0.5, large: 0.8)\n")
    
    return "\n".join(lines)


def analyze_and_save(
    baseline_results: pd.DataFrame,
    rl_results: pd.DataFrame,
    save_dir: str = "analysis_results",
    alpha: float = 0.05,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Perform analysis and save results to files.
    
    Args:
        baseline_results: Baseline strategy results
        rl_results: RL strategy results
        save_dir: Directory to save results
        alpha: Significance level
        verbose: Print results
        
    Returns:
        Dictionary of analysis results
    """
    # Create save directory
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    
    # Perform analysis
    analysis = analyze_strategies(baseline_results, rl_results, alpha)
    
    # Generate and save text summary
    text_table = generate_summary_table(analysis, format='text')
    text_path = save_path / f"statistical_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    with open(text_path, 'w', encoding='utf-8') as f:
        f.write(text_table)
    
    # Generate and save LaTeX table
    latex_table = generate_summary_table(analysis, format='latex')
    latex_path = save_path / f"statistical_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.tex"
    with open(latex_path, 'w', encoding='utf-8') as f:
        f.write(latex_table)
    
    # Generate and save Markdown table
    markdown_table = generate_summary_table(analysis, format='markdown')
    markdown_path = save_path / f"statistical_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
    with open(markdown_path, 'w', encoding='utf-8') as f:
        f.write(markdown_table)
    
    # Save detailed JSON results
    import json
    json_path = save_path / f"statistical_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(analysis, f, indent=2)
    
    if verbose:
        print(text_table)
        print(f"\nResults saved to:")
        print(f"  - Text: {text_path}")
        print(f"  - LaTeX: {latex_path}")
        print(f"  - Markdown: {markdown_path}")
        print(f"  - JSON: {json_path}")
    
    return analysis


def interpret_results(analysis: Dict[str, Dict[str, Any]]) -> str:
    """
    Generate interpretation of statistical results.
    
    Args:
        analysis: Dictionary of analysis results
        
    Returns:
        Formatted interpretation string
    """
    lines = []
    lines.append("\n" + "=" * 70)
    lines.append("STATISTICAL INTERPRETATION")
    lines.append("=" * 70)
    
    # Key metrics for thesis
    key_metrics = ['penetration_rate', 'cost_exchange_ratio', 'total_cost']
    
    lines.append("\nKey Findings:\n")
    
    for metric in key_metrics:
        if metric in analysis:
            result = analysis[metric]
            metric_name = result['metric_name']
            
            # Calculate improvement
            baseline_mean = result['baseline_mean']
            rl_mean = result['rl_mean']
            improvement = ((baseline_mean - rl_mean) / baseline_mean) * 100
            
            # Interpret effect size
            effect_size = abs(result['effect_size'])
            if effect_size < 0.2:
                effect_interp = "negligible"
            elif effect_size < 0.5:
                effect_interp = "small"
            elif effect_size < 0.8:
                effect_interp = "medium"
            else:
                effect_interp = "large"
            
            # Generate interpretation
            lines.append(f"{metric_name}:")
            
            if 'rate' in metric.lower():
                lines.append(f"  - Baseline: {baseline_mean*100:.2f}%")
                lines.append(f"  - RL Agent: {rl_mean*100:.2f}%")
                lines.append(f"  - Improvement: {improvement:+.2f}%")
            elif 'cost' in metric.lower():
                lines.append(f"  - Baseline: ${baseline_mean:,.0f}")
                lines.append(f"  - RL Agent: ${rl_mean:,.0f}")
                lines.append(f"  - Savings: ${baseline_mean - rl_mean:,.0f} ({improvement:+.2f}%)")
            else:
                lines.append(f"  - Baseline: {baseline_mean:.2f}")
                lines.append(f"  - RL Agent: {rl_mean:.2f}")
                lines.append(f"  - Difference: {rl_mean - baseline_mean:+.2f} ({improvement:+.2f}%)")
            
            lines.append(f"  - p-value: {result['p_value']:.6f}")
            lines.append(f"  - Statistically significant: {'YES' if result['significant'] else 'NO'}")
            lines.append(f"  - Effect size: {result['effect_size']:.3f} ({effect_interp})")
            lines.append("")
    
    # Overall conclusion
    lines.append("Conclusion:")
    
    # Check if RL is significantly better on key metrics
    pen_better = (analysis.get('penetration_rate', {}).get('significant', False) and 
                  analysis.get('penetration_rate', {}).get('rl_mean', 1) < 
                  analysis.get('penetration_rate', {}).get('baseline_mean', 0))
    
    cost_better = (analysis.get('cost_exchange_ratio', {}).get('significant', False) and 
                   analysis.get('cost_exchange_ratio', {}).get('rl_mean', 1) < 
                   analysis.get('cost_exchange_ratio', {}).get('baseline_mean', 0))
    
    if pen_better and cost_better:
        lines.append("  The RL agent demonstrates STATISTICALLY SIGNIFICANT improvements")
        lines.append("  over the baseline strategy on both penetration rate and cost-exchange")
        lines.append("  ratio. The RL approach is superior for air defense optimization.")
    elif cost_better:
        lines.append("  The RL agent achieves STATISTICALLY SIGNIFICANT improvement in")
        lines.append("  cost-exchange ratio, indicating better resource efficiency than")
        lines.append("  the baseline heuristic.")
    elif pen_better:
        lines.append("  The RL agent achieves STATISTICALLY SIGNIFICANT reduction in")
        lines.append("  penetration rate compared to the baseline strategy.")
    else:
        lines.append("  Statistical significance was not achieved on key metrics.")
        lines.append("  Further training or parameter tuning may be required.")
    
    lines.append("=" * 70 + "\n")
    
    return "\n".join(lines)


if __name__ == "__main__":
    """
    Test and demonstration code.
    """
    print("=" * 70)
    print("STATISTICAL ANALYSIS MODULE - DEMONSTRATION")
    print("=" * 70)
    
    # Test 1: Generate synthetic test data
    print("\n[Test 1] Generate synthetic test data:")
    
    np.random.seed(42)
    
    # Simulate baseline: higher penetration, higher cost
    baseline_data = {
        'penetration_rate': np.random.normal(0.85, 0.05, 100),
        'cost_exchange_ratio': np.random.normal(26.0, 2.0, 100),
        'total_cost': np.random.normal(200e6, 20e6, 100),
        'defense_cost': np.random.normal(1e6, 200e3, 100),
        'penetration_cost': np.random.normal(199e6, 20e6, 100),
        'uavs_destroyed': np.random.normal(50, 10, 100),
        'uavs_penetrated': np.random.normal(250, 20, 100),
        'kinetic_fired': np.random.normal(10, 3, 100),
        'de_fired': np.random.normal(20, 5, 100)
    }
    
    # Simulate RL: lower penetration, lower cost (better)
    rl_data = {
        'penetration_rate': np.random.normal(0.75, 0.05, 100),
        'cost_exchange_ratio': np.random.normal(22.0, 2.0, 100),
        'total_cost': np.random.normal(175e6, 18e6, 100),
        'defense_cost': np.random.normal(1.2e6, 250e3, 100),
        'penetration_cost': np.random.normal(174e6, 18e6, 100),
        'uavs_destroyed': np.random.normal(75, 12, 100),
        'uavs_penetrated': np.random.normal(225, 18, 100),
        'kinetic_fired': np.random.normal(12, 4, 100),
        'de_fired': np.random.normal(18, 5, 100)
    }
    
    baseline_df = pd.DataFrame(baseline_data)
    rl_df = pd.DataFrame(rl_data)
    
    print(f"  Generated {len(baseline_df)} baseline scenarios")
    print(f"  Generated {len(rl_df)} RL scenarios")
    
    # Test 2: Perform single t-test
    print("\n[Test 2] Single t-test on penetration rate:")
    
    result = perform_t_test(
        baseline_df['penetration_rate'].values,
        rl_df['penetration_rate'].values,
        alternative='less'
    )
    
    print(f"  Baseline mean: {result['baseline_mean']*100:.2f}%")
    print(f"  RL mean: {result['rl_mean']*100:.2f}%")
    print(f"  p-value: {result['p_value']:.6f}")
    print(f"  Significant (α=0.05): {result['significant']}")
    print(f"  Effect size (Cohen's d): {result['effect_size']:.3f}")
    
    # Test 3: Comprehensive analysis
    print("\n[Test 3] Comprehensive statistical analysis:")
    
    analysis = analyze_strategies(baseline_df, rl_df, alpha=0.05)
    
    print(f"  Metrics analyzed: {len(analysis)}")
    print(f"  Significant results: {sum(1 for r in analysis.values() if r['significant'])}")
    
    # Test 4: Generate text table
    print("\n[Test 4] Generate formatted tables:")
    
    text_table = generate_summary_table(analysis, format='text')
    print(text_table)
    
    # Test 5: Generate LaTeX table
    print("\n[Test 5] LaTeX table (first 5 lines):")
    latex_table = generate_summary_table(analysis, format='latex')
    for line in latex_table.split('\n')[:5]:
        print(f"  {line}")
    print("  ...")
    
    # Test 6: Generate Markdown table
    print("\n[Test 6] Markdown table (first 5 lines):")
    markdown_table = generate_summary_table(analysis, format='markdown')
    for line in markdown_table.split('\n')[:5]:
        print(f"  {line}")
    print("  ...")
    
    # Test 7: Save all formats
    print("\n[Test 7] Save analysis to files:")
    
    analysis_results = analyze_and_save(
        baseline_df,
        rl_df,
        save_dir="test_analysis",
        alpha=0.05,
        verbose=False
    )
    
    analysis_dir = Path("test_analysis")
    if analysis_dir.exists():
        files = list(analysis_dir.glob("*"))
        print(f"  Files created: {len(files)}")
        for f in files:
            print(f"    - {f.name}")
    
    # Test 8: Interpretation
    print("\n[Test 8] Generate interpretation:")
    
    interpretation = interpret_results(analysis)
    print(interpretation)
    
    # Cleanup
    print("\n[Test 9] Cleanup test files:")
    import shutil
    if analysis_dir.exists():
        shutil.rmtree(analysis_dir)
        print(f"  Removed: test_analysis")
    
    print("\n" + "=" * 70)
    print("ALL TESTS COMPLETED SUCCESSFULLY")
    print("=" * 70)
    print("\nReady for thesis analysis with:")
    print("  from statistical_analysis import analyze_and_save")
    print("  analysis = analyze_and_save(baseline_df, rl_df, save_dir='results')")
