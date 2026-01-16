"""
Quick End-to-End Test
Tests the complete workflow with small parameters
"""

print("="*70)
print("QUICK END-TO-END TEST")
print("="*70)

print("\n[1/4] Training agent (2000 timesteps)...")
from train_agent import train_dqn_agent
model, metrics, model_path = train_dqn_agent(total_timesteps=2000, verbose=0)
print(f"✓ Model trained and saved to: {model_path}")
print(f"  Type: {type(model_path)}")
print(f"  Episodes: {len(metrics['episode_rewards'])}")

print("\n[2/4] Evaluating baseline (50 scenarios)...")
from evaluate_strategies import evaluate_baseline
baseline_df = evaluate_baseline(num_scenarios=50, verbose=False)
print(f"✓ Baseline evaluated: {baseline_df.shape}")
print(f"  Penetration: {baseline_df['penetration_rate'].mean()*100:.2f}%")

print("\n[3/4] Evaluating RL agent (50 scenarios)...")
from evaluate_strategies import evaluate_rl_agent
rl_df = evaluate_rl_agent(model_path, num_scenarios=50, verbose=False)
print(f"✓ RL evaluated: {rl_df.shape}")
print(f"  Penetration: {rl_df['penetration_rate'].mean()*100:.2f}%")

print("\n[4/4] Statistical analysis...")
from statistical_analysis import analyze_strategies
analysis = analyze_strategies(baseline_df, rl_df, alpha=0.05)
print(f"✓ Analysis complete: {len(analysis)} metrics")
pen_result = analysis['penetration_rate']
print(f"  Baseline: {pen_result['baseline_mean']*100:.2f}%")
print(f"  RL: {pen_result['rl_mean']*100:.2f}%")
print(f"  p-value: {pen_result['p_value']:.6f}")

print("\n" + "="*70)
print("ALL TESTS PASSED!")
print("="*70)

# Cleanup
import shutil
from pathlib import Path
shutil.rmtree("trained_models", ignore_errors=True)
print("\nCleaned up test files.")
