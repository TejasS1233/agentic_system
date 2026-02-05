"""
SIMULATION: LINEAR SCALING (×0.01) with TSU

Formula: half_life = 9400 × (1 + min(freq × 0.01, 2.0))
         score = exp(-TSU / half_life)

Eviction Threshold: 0.01 (tools below this score are evicted)
Survival = -H × ln(0.01) = H × 4.605
"""
import math

def linear_score(tsu_hours, freq):
    """
    half_life = 9400 × (1 + min(freq * 0.01, 2))
    score = exp(-TSU / half_life)
    """
    BASE_HALF_LIFE = 9400.0
    
    if freq == 0:
        return 0.5 if tsu_hours < 6 else 0.05
    
    multiplier = 1.0 + min(freq * 0.01, 2.0)
    effective_half_life = BASE_HALF_LIFE * multiplier
    
    tsu_seconds = tsu_hours * 3600
    return math.exp(-tsu_seconds / effective_half_life)

print("=" * 75)
print("DECAY SIMULATION: Linear Formula (freq × 0.01)")
print("=" * 75)
print()
print("Formula: half_life = 9400 × (1 + min(freq × 0.01, 2.0))")
print("         score = exp(-TSU / half_life)")
print("Eviction Threshold: 0.01")
print()
print(f"{'Freq':<6} {'Multiplier':<12} {'Half-Life':<12} {'12h Score':<12} {'Survives (0.01)'}")
print("-" * 75)

for freq in [0, 1, 5, 10, 50, 100, 200]:
    if freq == 0:
        mult = "N/A"
        hl = "N/A"
        s12 = "0.05" 
        surv = "~6h (grace)"
    else:
        m = 1.0 + min(freq * 0.01, 2.0)
        mult = f"{m:.2f}x"
        
        hl_sec = 9400 * m
        hl = f"{hl_sec/3600:.1f} hours"
        
        # Score after 12 hours inactive
        s = linear_score(12, freq)
        status = "✅" if s >= 0.01 else "❌"
        s12 = f"{s:.4f} {status}"
        
        # Survival time to reach score=0.01:  TSU = -H × ln(0.01) = H × 4.605
        surv_sec = hl_sec * 4.605
        surv = f"~{surv_sec/3600:.1f} hours"

    print(f"{freq:<6} {mult:<12} {hl:<12} {s12:<12} {surv}")

print()
print("=" * 75)
print("KEY INSIGHTS:")
print("  • Base half-life: 9400 seconds (~2.6 hours)")
print("  • At 100 calls: multiplier = 2.0x → half-life = 5.2 hours")
print("  • At 200+ calls: multiplier = 3.0x → half-life = 7.8 hours (max)")
print("  • Survival = H × 4.605 (time to reach score = 0.01)")
print("=" * 75)

