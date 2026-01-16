# EXPERIMENTAL RESULTS: RL with Reward Shaping Fixes (500K Training)

**Date:** January 12-13, 2026  
**Training Duration:** 500,000 timesteps (~12 minutes)  
**Evaluation:** 1,000 Monte Carlo scenarios per strategy  
**Status:** ⚠️ **DISAPPOINTING - RL Still Not Beating Baseline**

---

## TL;DR - What Happened

**Expected:** RL agent would beat baseline after reward shaping fixes  
**Actual:** RL performed essentially IDENTICAL to baseline (no improvement)  
**Conclusion:** Reward shaping fixes didn't work as expected. More investigation needed.

---

## Statistical Results Summary

### Performance Comparison (1000 scenarios each)

| Metric | Baseline | RL Agent (Fixed) | Difference | p-value | Significant? |
|--------|----------|------------------|------------|---------|--------------|
| **Penetration Rate** | 94.39% | 94.44% | **-0.05%** (worse) | 0.345 | NO |
| **Cost-Exchange Ratio** | $26.99 | $27.01 | **-0.08%** (worse) | 0.276 | NO |
| **Total Cost** | $287.08M | $287.30M | **-$213K** (worse) | 0.483 | NO |
| **Defense Cost** | $200K | $250K | **-$50K** (worse) | <0.001 | YES* |
| **UAVs Destroyed** | 14.03 | 13.87 | **-0.16** (worse) | 0.044 | YES* |
| **Kinetic Fired** | 0.00 | 0.06 | +0.06 | <0.001 | YES* |
| **DE Fired** | 20.00 | 19.88 | -0.12 | <0.001 | YES* |

**\*Note:** Statistically significant differences are TINY and practically meaningless.

### Interpretation

**Bottom Line:** RL agent learned to do **exactly the same thing as baseline** - fire all 20 DE shots, achieve ~94% penetration.

**Effect Sizes:** All negligible (Cohen's d < 0.1), meaning differences are not practically meaningful.

---

## Detailed Breakdown

### What the RL Agent Actually Learned

From evaluation CSV analysis:

```
Average RL Agent Behavior (1000 scenarios):
- Fires: ~20 shots per scenario (19.88 DE + 0.06 kinetic)
- Hits: ~14 UAVs destroyed per scenario
- Penetration: 94.44% of UAVs get through
- Strategy: Fire all DE ammo, almost never use kinetic
```

**This is essentially the SAME as baseline!** ✅ No ammo hoarding ❌ No improvement

### Comparison to Previous (Broken) Version

| Version | Penetration | Shots Fired | Behavior |
|---------|-------------|-------------|----------|
| **Original (50K, $1M penalty)** | 99.78% ❌ | 0.3 shots ❌ | Hoarded ammo |
| **Fixed (500K, $10M penalty)** | 94.44% ⚠️ | 19.94 shots ✓ | Same as baseline |
| **Baseline** | 94.39% | 20.00 shots | Greedy DE-first |

**Progress:** RL no longer hoards ammo ✓  
**Problem:** RL just converged to the baseline strategy (no innovation) ❌

---

## Training Analysis

### Training Timings

- **Total Execution:** 750.6 seconds (12.5 minutes)
  - Training: 733.6s (97.7%)
  - Baseline eval: 0.9s (0.1%)
  - RL eval: 7.9s (1.1%)
  - Analysis + viz: 8.2s (1.1%)

### Training Configuration Used

```python
TRAINING_TIMESTEPS = 500_000        # 10x increase from original
PENETRATION_PENALTY = $10_000_000   # 10x increase from original
HIT_REWARD = +$50_000               # NEW: Immediate positive feedback
EXPLORATION_FRACTION = 0.9          # 90% of training
EXPLORATION_FINAL = 0.01            # Final epsilon
```

---

## Why the Fixes Didn't Improve Performance

### Theory vs. Reality

**Expected Outcome:**
- RL should learn to prioritize kinetic (90% success, $1M) over DE (70% success, $10K)
- Higher accuracy should reduce penetrations below baseline's 94%
- Smart ammo management should beat greedy "fire all DE" strategy

**Actual Outcome:**
- RL learned: "Just fire all my DE shots like baseline does"
- No strategic innovation
- Converged to local optimum (baseline strategy)

### Likely Reasons for Failure

1. **State Space Too Simple**
   - 5D state [UAVs, kinetic, DE, cost, distance] may not provide enough info
   - Agent can't distinguish "save kinetic for large swarms" from "fire DE always"

2. **Reward Signal Still Not Strong Enough**
   - Even $10M penalty may not be enough to incentivize kinetic use
   - Agent learned: "DE is cheap, fire all of it" (correct for small swarms)
   - Never learned: "Save kinetic for when it matters" (large swarms)

3. **Exploration Insufficient**
   - Agent may have found baseline strategy early and stuck with it
   - Need curriculum learning or shaped exploration

4. **Action Space Too Coarse**
   - Discrete actions [KINETIC, DE, SKIP] don't allow nuanced strategies
   - Can't express "use kinetic only for swarms >300 UAVs"

5. **Training Dynamics**
   - Most scenarios have 100-500 UAVs randomly
   - Average swarm ~300 UAVs
   - Baseline strategy (fire all DE) is actually OPTIMAL for small swarms!
   - RL correctly learned this but never discovered kinetic value for large swarms

---

## What This Means for Your Thesis

### Positive Framing

**Research Finding:** "Reward shaping successfully corrected pathological behavior (ammo hoarding) but revealed fundamental limitations of DQN for complex air defense optimization."

**Contributions:**
1. ✅ Identified and fixed reward shaping failure mode
2. ✅ Demonstrated RL can match baseline performance
3. ✅ Showed that simple RL may not beat well-designed heuristics
4. ⚠️ Suggests more sophisticated RL (PPO, multi-objective) may be needed

### Honest Assessment

**What worked:**
- System implementation (10 modules, all functional)
- Experimental framework (reproducible, well-documented)
- Debugging process (identified root causes)

**What didn't work:**
- RL didn't beat baseline (converged to same strategy)
- 500K timesteps not enough for strategic innovation
- Reward shaping alone insufficient

**Why this is still valuable:**
- **Negative results are still results!**
- Shows baseline is actually quite good (hard to beat)
- Demonstrates RL limitations in adversarial domains
- Provides foundation for future work (PPO, curriculum learning, etc.)

---

## Recommendations for Next Steps

### Option 1: Accept Results (Recommended for Thesis Deadline)

**Thesis Angle:** "Comparative Study of RL vs. Heuristic Strategies"

**Narrative:**
> "This thesis investigates whether deep reinforcement learning can outperform hand-crafted heuristics in UAV swarm defense. Initial experiments revealed reward shaping challenges that caused the RL agent to develop pathological behaviors. After implementing corrective measures (increased penetration penalties and immediate positive rewards), the agent successfully learned viable defensive strategies but **converged to the same greedy approach as the baseline**, achieving statistically identical performance (94.4% vs 94.4% penetration). These findings suggest that **well-designed heuristics can be surprisingly competitive** with RL in structured tactical scenarios, and that achieving superhuman performance may require more sophisticated RL algorithms or hybrid approaches."

**Strengths:**
- Honest about results
- Shows rigorous experimental method
- Demonstrates debugging skills
- Identifies future research directions

---

### Option 2: Try More Aggressive Fixes (High Risk, Time-Consuming)

**If you have 1-2 more days, try these escalations:**

#### Fix A: Much Stronger Penalties

```python
# In rl_environment.py
penetration_penalty: float = 50_000_000  # Increase to $50M (5x current)
reward += 200_000  # Increase hit reward to $200K (4x current)

# In run_complete_experiment.py
TRAINING_TIMESTEPS = 2_000_000  # 4x current (will take ~1 hour)
```

#### Fix B: Curriculum Learning

Train on progressively harder scenarios:
1. Train on small swarms (100-200 UAVs) - 100K steps
2. Train on medium swarms (200-350 UAVs) - 200K steps  
3. Train on large swarms (350-500 UAVs) - 200K steps
4. Train on mixed swarms (100-500 UAVs) - 200K steps

#### Fix C: Try Different Algorithm (PPO)

```python
from stable_baselines3 import PPO

model = PPO(
    policy="MlpPolicy",
    env=env,
    learning_rate=0.0003,
    n_steps=2048,
    batch_size=64,
    n_epochs=10,
    gamma=0.99,
    # ... other params
)
```

PPO often works better than DQN for continuous control problems.

---

### Option 3: Hybrid Approach (Safest for Thesis)

**Combine RL + Baseline:**

```python
def hybrid_strategy(state):
    # Use baseline for small swarms
    if state['uavs_remaining'] < 250:
        return baseline_strategy(state)
    # Use RL for large swarms
    else:
        return rl_strategy(state)
```

This might actually beat both individual strategies!

---

## File Structure After Latest Run

```
thesis_results/
├── trained_models/
│   └── dqn_agent_20260112_205352_final.zip (1.1 MB)
├── training_metrics/
│   └── training_metrics_20260112_205352.csv (2.2 MB, 25000 episodes)
├── evaluation_results/
│   ├── baseline_results_20260112_205352.csv
│   └── rl_results_20260112_205352.csv
├── statistical_analysis/
│   └── statistical_analysis_20260112_210615.txt
├── visualizations/
│   ├── penetration_comparison_20260112_210615.png
│   ├── cost_exchange_comparison_20260112_210615.png
│   ├── cost_vs_penetration_20260112_210615.png
│   ├── weapon_usage_20260112_210615.png
│   ├── comprehensive_comparison_20260112_210615.png
│   └── training_metrics_20260112_210615.png
└── reports/
    ├── experiment_summary_20260112_205352.txt
    └── interpretation_20260112_205352.txt
```

---

## Quick Data Summary for Your AI

### Agent Behavior Comparison

**Baseline Strategy (Greedy):**
```
For each timestep:
  1. Fire DE at nearest UAV
  2. Repeat until ammo depleted (20 shots)
  3. All remaining UAVs penetrate
Result: 94.39% penetration, $287M total cost
```

**RL Agent Strategy (After 500K Training):**
```
For each timestep:
  1. Fire DE at nearest UAV (19.88 avg shots)
  2. Very rarely fire kinetic (0.06 avg shots)
  3. All remaining UAVs penetrate
Result: 94.44% penetration, $287M total cost
```

**Conclusion:** RL learned the baseline strategy (not better, not worse - same).

---

## Statistical Significance Interpretation

From the statistical analysis file:

```
Penetration Rate:
  - p-value: 0.345 (NOT significant, need p < 0.05)
  - Effect size: -0.018 (negligible, need d > 0.5 for medium effect)
  - Conclusion: No meaningful difference

Cost-Exchange Ratio:
  - p-value: 0.276 (NOT significant)
  - Effect size: -0.027 (negligible)
  - Conclusion: No meaningful difference

Total Cost:
  - p-value: 0.483 (NOT significant)
  - Effect size: -0.002 (negligible)
  - Conclusion: No meaningful difference
```

**Translation:** RL and baseline are statistically indistinguishable on all key metrics.

---

## Figures Generated

All 6 visualizations show **RL and Baseline boxplots overlapping almost perfectly**:

1. **Penetration Comparison:** Both centered at ~94%
2. **Cost-Exchange Comparison:** Both centered at ~$27
3. **Cost vs Penetration Scatter:** RL and baseline points completely intermixed
4. **Weapon Usage:** Both fire ~20 DE shots, minimal kinetic
5. **Comprehensive 4-Panel:** All panels show identical distributions
6. **Training Metrics:** Shows convergence but to baseline-level performance

---

## What to Tell Your Other AI

### Context Summary

"I applied reward shaping fixes to my RL agent (increased penetration penalty from $1M to $10M, added +$50K immediate rewards for hits, trained for 500K timesteps instead of 50K). The good news: agent stopped hoarding ammo. The bad news: agent converged to the exact same strategy as the baseline - fire all 20 DE shots every scenario, achieve 94.4% penetration. Statistical tests show NO significant difference between RL and baseline on any metric (all p-values > 0.05, effect sizes < 0.1). 

The RL agent essentially re-discovered the greedy baseline strategy through learning but didn't innovate beyond it. This suggests either:
1. Baseline strategy is actually optimal for this problem
2. My state space is too simple for RL to learn better strategies
3. Need different RL algorithm (PPO instead of DQN)
4. Need curriculum learning or more training (2M+ timesteps)

For thesis purposes, I can frame this as: 'RL matched baseline performance, demonstrating that well-designed heuristics can be competitive with learned policies, but did not achieve superior performance.'"

### Key Numbers

- **Training time:** 12.5 minutes (500K timesteps)
- **Performance:** RL = 94.44% penetration, Baseline = 94.39% penetration (difference: 0.05%, p=0.345)
- **Behavior:** RL fires 19.88 DE + 0.06 kinetic, Baseline fires 20 DE + 0 kinetic
- **Conclusion:** Statistically identical, practically equivalent

### Status

✅ System works perfectly (all 10 modules, master script, analysis, visualization)  
✅ Reward shaping fixed ammo hoarding behavior  
❌ RL didn't beat baseline (matched it instead)  
⚠️ Need to decide: accept results or try more aggressive approaches

---

## Bottom Line

**Question:** Did the reward shaping fixes work?  
**Answer:** Yes and no.
- ✅ Fixed the ammo hoarding problem
- ✅ Agent learned to engage threats
- ❌ Didn't achieve better-than-baseline performance
- ⚠️ Converged to local optimum (baseline strategy)

**For thesis:** This is still publishable - negative results showing RL limitations are valuable scientific contributions. The experimental framework is solid, the analysis is rigorous, and the findings are clear.

---

**Last Updated:** January 13, 2026  
**Experiment ID:** 20260112_205352  
**Training Duration:** 500,000 timesteps  
**Evaluation Scenarios:** 2,000 (1000 baseline + 1000 RL)  
**Statistical Significance:** None achieved on key metrics  
**Recommendation:** Accept results or try Option 2 escalations if time permits
