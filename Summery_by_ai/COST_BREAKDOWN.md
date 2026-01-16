# Cost Breakdown - Air Defense Simulation

## UAV (Attacking Drones)

**Shahed-136 Style UAV:**
- **Cost per UAV:** $35,000
- **Typical swarm size:** 100-500 UAVs randomly
- **Example:** 300 UAV swarm = $10,500,000 attack cost

---

## Defense Systems (Weapons to Shoot Down UAVs)

### Option 1: Kinetic Interceptor (Missile)
- **Cost per shot:** $1,000,000 (very expensive!)
- **Success rate:** 90% (very reliable)
- **Available ammunition:** 50 missiles per scenario
- **Total investment:** $50,000,000 worth of missiles available

**Economics:**
- Cost to destroy 1 UAV: $1,000,000 / 0.90 = **$1,111,111 per kill**
- Much more expensive than the $35K UAV it destroys
- Cost-exchange ratio: Poor (spending $1M to stop $35K threat)

---

### Option 2: Directed Energy (DE) - Laser System
- **Cost per shot:** $10,000 (cheap!)
- **Success rate:** 70% (less reliable than kinetic)
- **Available ammunition:** 100 shots per scenario
- **Total investment:** $1,000,000 worth of DE shots available

**Economics:**
- Cost to destroy 1 UAV: $10,000 / 0.70 = **$14,286 per kill**
- Cheaper than the $35K UAV it destroys ✓
- Cost-exchange ratio: Good (spending $14K to stop $35K threat)

---

### Option 3: Skip / Do Nothing
- **Cost per action:** $0
- **Success rate:** 0% (UAV penetrates defenses)
- **Result:** UAV hits target

**Economics:**
- Direct cost: $0
- **Penetration cost:** Varies by scenario (see below)

---

## Penetration Costs (What Happens if UAV Gets Through)

### In the Simulation:

**Original Version (Broken):**
- **Penetration penalty:** $1,000,000 per UAV that gets through
- **Problem:** Same cost as kinetic missile, so agent thought "why bother firing?"

**Fixed Version (Current):**
- **Penetration penalty:** $10,000,000 per UAV that gets through
- **Rationale:** Makes stopping UAVs much more important than saving ammo

**Real-World Interpretation:**
- Each penetrating UAV could hit critical infrastructure
- Damage could be: power plant, fuel depot, ammunition storage, command center
- $10M represents expected damage from successful strike

---

## Cost Comparison Summary

| Action | Direct Cost | Expected Outcome | Total Expected Cost |
|--------|-------------|------------------|---------------------|
| **Fire Kinetic** | $1,000,000 | 90% kill, 10% miss | $1M + ($10M × 10%) = **$2M** |
| **Fire DE** | $10,000 | 70% kill, 30% miss | $10K + ($10M × 30%) = **$3.01M** |
| **Skip** | $0 | 100% penetration | $0 + ($10M × 100%) = **$10M** |

**Paradox:** Kinetic is most expensive per shot ($1M) but cheapest total expected cost ($2M) due to high success rate!

---

## Baseline Strategy Cost Analysis

**Greedy Baseline (fires all DE, no kinetic):**

Typical scenario with 300 UAV swarm:
1. Fire all 100 DE shots = $1,000,000 spent
2. Hit ~70 UAVs (70% success rate) = 70 destroyed
3. Remaining 230 UAVs penetrate = $2,300,000,000 damage
4. **Total cost:** $1M + $2.3B = **$2,301,000,000**

**Cost-exchange ratio:** $2.3B / $10.5M (attack cost) = **~220**
- Defender spends 220x more than attacker
- This is why UAV swarms are so dangerous!

---

## Why the Costs Matter for RL

### Old Reward Function (Failed):
```
Reward = -cost - ($1M × penetrations)
```

**Agent's thinking:**
- "Fire kinetic costs $1M"
- "Letting UAV through costs $1M" 
- "Same price! Why bother firing?"
- **Result:** Agent hoards ammo ❌

### New Reward Function (Fixed):
```
Reward = +$50K (if hit) - cost - ($10M × penetrations)
```

**Agent's thinking:**
- "Fire kinetic: +$50K (hit) - $1M (cost) + $9M saved (prevented penetration) = **+$8M net**"
- "Fire DE: +$50K (hit) - $10K (cost) + $7M saved (30% still penetrate) = **+$7M net**"
- "Skip: lose $10M"
- **Result:** Agent should fire weapons! ✓

(But in practice, agent still just copies baseline and fires all DE)

---

## Actual Simulation Economics

### Baseline Performance (1000 scenarios average):

- **Defense spending:** $200,000 (20 DE shots × $10K)
- **Penetration cost:** $286,880,000 (~28,688 UAVs penetrate)
- **Total cost:** $287,080,000
- **Cost-exchange ratio:** $287M / (attack value) ≈ **$27 per $1 of attack**

### RL Agent Performance (500K training):

- **Defense spending:** $250,000 (19.88 DE + 0.06 kinetic)
- **Penetration cost:** $287,040,000 (~28,704 UAVs penetrate)
- **Total cost:** $287,295,760
- **Cost-exchange ratio:** $287M / (attack value) ≈ **$27 per $1 of attack**

**Conclusion:** Both strategies spend about the same, achieve same results.

---

## Real-World Cost Context

**For Reference (Approximate Real Costs):**

| System | Real Cost per Shot | Simulation Cost |
|--------|-------------------|-----------------|
| **Patriot Missile** | $4,000,000 | $1,000,000 (kinetic) |
| **Iron Dome** | $50,000-$100,000 | $1,000,000 (kinetic) |
| **Directed Energy (Laser)** | $1,000-$10,000 | $10,000 (DE) |
| **Shahed-136 Drone** | $20,000-$50,000 | $35,000 (UAV) |

**The Problem:** 
- Shooting down a $35K drone with a $4M missile is economically devastating
- Cost-exchange ratio: 114:1 in attacker's favor
- This is why laser systems (DE) are being developed - better economics

---

## Bottom Line: The Economics Problem

**Why UAV swarms are scary:**
- Cheap to produce ($35K each)
- Expensive to defend against ($1M kinetic, even $10K DE adds up)
- Swarms of 300+ UAVs overwhelm defenses
- Defender must spend 10-100x more than attacker

**What the simulation shows:**
- DE-only strategy (baseline): ~94% penetration, $287M cost
- RL strategy: ~94% penetration, $287M cost (same!)
- **Neither strategy solves the fundamental economics problem**

**What would help:**
- More ammunition (>100 DE shots)
- Better accuracy (>90% for kinetic, >70% for DE)
- Cheaper kinetic ($500K instead of $1M)
- Early warning (more than 20 timesteps to engage)
- Mix of systems optimized by swarm size

---

## Cost Summary Table

| Item | Cost | Notes |
|------|------|-------|
| **UAV (drone)** | $35,000 | Attacker's cost |
| **Kinetic missile** | $1,000,000 | 90% success rate |
| **DE laser shot** | $10,000 | 70% success rate |
| **Penetration (damage)** | $10,000,000 | Per UAV that gets through |
| **Successful hit reward (RL)** | +$50,000 | Bonus for destroying UAV |
| **Available kinetic ammo** | 50 shots | $50M total value |
| **Available DE ammo** | 100 shots | $1M total value |

---

**Key Insight:** The simulation is modeling an **asymmetric warfare** scenario where the defender is at a massive economic disadvantage. Both baseline and RL strategies struggle because the problem is fundamentally hard - you can't defend 300 UAVs with only 100 DE shots when each shot only works 70% of the time.
