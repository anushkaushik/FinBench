"""
Task 1 (Easy): Portfolio Allocation
====================================
Given a client profile, allocate the portfolio correctly across asset classes.
Grader checks suitability based on risk tolerance, age, investment horizon.
"""
 
from __future__ import annotations
 
 
 
from typing import Dict, Tuple
from ..models import ClientProfile, Reward, _SCORE_EPSILON as STRICT_SCORE_EPSILON
 
 
# ── Ideal allocation ranges per risk profile ─────────────────────────────────
# Each: {asset_class: (min%, max%, ideal%)}
ALLOCATION_PROFILES: Dict[str, Dict[str, Tuple[float, float, float]]] = {
    "conservative": {
        "bonds":            (40, 70, 55),
        "equities":         (10, 30, 20),
        "cash":             (10, 30, 15),
        "real_estate":      (0,  15, 5),
        "commodities":      (0,  10, 5),
    },
    "moderate": {
        "bonds":            (20, 45, 35),
        "equities":         (35, 60, 50),
        "cash":             (5,  15, 8),
        "real_estate":      (0,  15, 5),
        "commodities":      (0,  10, 2),
    },
    "aggressive": {
        "bonds":            (0,  20, 10),
        "equities":         (55, 85, 70),
        "cash":             (0,  10, 5),
        "real_estate":      (0,  20, 10),
        "commodities":      (0,  15, 5),
    },
}
 
AGE_BOND_RULE_BONUS = 5.0   # extra bond % for every 10 years over 50
 
 
# FIX #7 (task grader signature): graders previously accepted task-specific action
# sub-classes (AllocationAction etc.) that don't exist. Updated to accept FinbenchAction
# directly — only the relevant fields are read, so this is fully backward compatible.
def grade(action, client: ClientProfile) -> Reward:
    """
    Score a portfolio allocation 0.0–1.0.
    Partial credit for:
      - Allocations within acceptable ranges      (0.50)
      - Sum-to-100 correctness                    (0.15)
      - Age-appropriate bond weighting            (0.15)
      - Emergency fund prerequisite awareness     (0.10)
      - Reasoning quality (non-empty)             (0.10)
    """
    components: Dict[str, float] = {}
    penalties: Dict[str, float] = {}
 
    allocs = action.allocations
    profile = ALLOCATION_PROFILES[client.risk_tolerance]
 
    # 1. Allocations within acceptable ranges (0.50)
    known_assets = set(profile.keys())
    in_range_count = 0
    total_checked = len(known_assets)
 
    for asset, (lo, hi, _ideal) in profile.items():
        pct = allocs.get(asset, 0.0)
        if lo <= pct <= hi:
            in_range_count += 1
 
    range_score = (in_range_count / total_checked) * 0.50
    components["in_range"] = round(range_score, 4)
 
    # 2. Allocations sum to 100 (±2 tolerance) (0.15)
    total_alloc = sum(allocs.values())
    if abs(total_alloc - 100.0) <= 2.0:
        components["sum_to_100"] = 0.15
    elif abs(total_alloc - 100.0) <= 10.0:
        components["sum_to_100"] = 0.07
    else:
        components["sum_to_100"] = 0.0
        penalties["allocation_sum_error"] = abs(total_alloc - 100.0)
 
    # 3. Age-appropriate bond allocation (0.15)
    bond_alloc = allocs.get("bonds", 0.0)
    # FIX #8: removed dead variable `_ideal_profile_bond` (was computed but never used).
    #         Read the ideal directly from the profile tuple.
    age_adjusted_ideal = profile["bonds"][2]  # ideal% is index 2
    if client.age > 50:
        age_adjusted_ideal += ((client.age - 50) / 10) * AGE_BOND_RULE_BONUS
 
    age_adjusted_ideal = min(age_adjusted_ideal, 80)  # cap
    bond_error = abs(bond_alloc - age_adjusted_ideal)
    if bond_error <= 5:
        components["age_appropriate_bonds"] = 0.15
    elif bond_error <= 15:
        components["age_appropriate_bonds"] = 0.08
    else:
        components["age_appropriate_bonds"] = 0.0
 
    # 4. Emergency fund prerequisite awareness (0.10)
    cash_alloc = allocs.get("cash", 0.0)
    if not client.has_emergency_fund:
        if cash_alloc >= 15:
            components["emergency_fund_awareness"] = 0.10
        elif cash_alloc >= 8:
            components["emergency_fund_awareness"] = 0.05
        else:
            components["emergency_fund_awareness"] = 0.0
    else:
        components["emergency_fund_awareness"] = 0.10  # full marks if fund exists
 
    # 5. Reasoning quality (0.10)
    components["reasoning"] = 0.10 if len(action.reasoning.strip()) >= 30 else 0.04
 
    total = sum(components.values()) - sum(penalties.values())
    total = max(STRICT_SCORE_EPSILON, min(1.0 - STRICT_SCORE_EPSILON, total))
 
    return Reward(
        total=round(total, 4),
        components=components,
        penalties=penalties,
        explanation=(
            f"Risk profile: {client.risk_tolerance} | Age: {client.age} | "
            f"Bond ideal: {age_adjusted_ideal:.1f}% | Agent bond: {bond_alloc}% | "
            f"Alloc sum: {total_alloc:.1f}%"
        ),
    )
