"""
Task 2 (Medium): Risk Assessment & Recommendations
====================================================
Agent analyzes a client's financial situation, identifies risks,
assigns a risk score, and provides actionable recommendations.
"""
 
from __future__ import annotations
 
 
 
from typing import Dict, List, Set
from ..models import ClientProfile, Reward, _SCORE_EPSILON as STRICT_SCORE_EPSILON
 
 
 
# ── Risk factors that should be identified based on client profile ────────────
def expected_risk_flags(client: ClientProfile) -> Dict[str, str]:
    """Return a dict of {flag_key: description} that SHOULD be identified."""
    flags: Dict[str, str] = {}
 
    # Debt risk
    if client.debt_to_income_ratio > 0.36:
        flags["high_debt"] = "debt-to-income ratio exceeds 36%"
 
    # Emergency fund
    if not client.has_emergency_fund:
        flags["no_emergency_fund"] = "no emergency fund"
 
    # Insurance gap
    if not client.has_insurance and client.dependents > 0:
        flags["insurance_gap"] = "has dependents but no insurance"
 
    # Under-saving (simplified: saves < 15% of income)
    monthly_income = client.annual_income / 12
    savings = monthly_income - client.monthly_expenses
    savings_rate = savings / monthly_income if monthly_income > 0 else 0
    if savings_rate < 0.15:
        flags["low_savings_rate"] = f"savings rate ~{savings_rate*100:.0f}% below 15% target"
 
    # Retirement horizon risk
    years_to_65 = max(0, 65 - client.age)
    if years_to_65 < 10 and client.risk_tolerance == "aggressive":
        flags["near_retirement_aggressive"] = "near retirement with aggressive allocation"
 
    # Concentration risk
    if client.existing_portfolio:
        for asset, pct in client.existing_portfolio.items():
            if pct > 60:
                flags[f"concentration_risk_{asset}"] = f"over-concentrated in {asset} ({pct}%)"
 
    # High income + no tax strategy
    if client.annual_income > 150_000 and client.tax_bracket >= 0.32:
        flags["tax_optimization_needed"] = "high income with no apparent tax strategy"
 
    return flags
 
 
def expected_risk_score_range(client: ClientProfile) -> tuple[float, float]:
    """Return acceptable normalized (min, max) risk score based on client profile."""
    flags = expected_risk_flags(client)
    num_flags = len(flags)
 
    # Base from risk tolerance
    base = {"conservative": 0.3, "moderate": 0.5, "aggressive": 0.6}[client.risk_tolerance]
    adjustment = num_flags * 0.08
 
    ideal = min(1.0 - STRICT_SCORE_EPSILON, base + adjustment)
    return max(STRICT_SCORE_EPSILON, ideal - 0.2), min(1.0 - STRICT_SCORE_EPSILON, ideal + 0.2)
 
 
# FIX #7: updated signature to accept FinbenchAction directly instead of
#         RiskAssessmentAction (which doesn't exist). Reads the same fields.
def grade(action, client: ClientProfile) -> Reward:
    """
    Score a risk assessment 0.0–1.0.
    Partial credit for:
      - Identifying expected risk flags        (0.40)
      - Accurate risk score                    (0.20)
      - Quality & specificity of recommendations (0.25)
      - Priority recommendation clarity        (0.15)
    """
    components: Dict[str, float] = {}
    penalties: Dict[str, float] = {}
 
    expected_flags = expected_risk_flags(client)
    expected_keywords = {
        "high_debt": ["debt", "dti", "debt-to-income"],
        "no_emergency_fund": ["emergency", "emergency fund", "liquid"],
        "insurance_gap": ["insurance", "life insurance", "coverage"],
        "low_savings_rate": ["saving", "savings rate", "save more"],
        "near_retirement_aggressive": ["retirement", "conservative", "near retirement"],
        "concentration_risk": ["concentration", "concentrated", "diversif"],
        "tax_optimization_needed": ["tax", "tax-advantaged", "401k", "ira"],
    }
 
    # 1. Risk flag identification (0.40)
    identified_text = " ".join(action.identified_risks).lower()
    flags_caught = 0
    for flag_key in expected_flags:
        keywords = (
            expected_keywords["concentration_risk"]
            if flag_key.startswith("concentration_risk")
            else expected_keywords[flag_key]
        )
        if any(kw in identified_text for kw in keywords):
            flags_caught += 1
 
    if expected_flags:
        flag_score = (flags_caught / len(expected_flags)) * 0.40
    else:
        flag_score = 0.40  # no flags = low risk = full marks if agent finds nothing major
 
    components["risk_flags_identified"] = round(flag_score, 4)
 
    # Penalize false positives (more than 3 extra risks not expected)
    if len(action.identified_risks) > len(expected_flags) + 3:
        penalties["false_positive_risks"] = 0.05
 
    # 2. Risk score accuracy (0.20)
    score_min, score_max = expected_risk_score_range(client)
    if score_min <= action.risk_score <= score_max:
        components["risk_score_accuracy"] = 0.20
    else:
        error = min(
            abs(action.risk_score - score_min),
            abs(action.risk_score - score_max)
        )
        components["risk_score_accuracy"] = max(0, 0.20 - error * 0.03)
 
    # 3. Recommendation quality (0.25)
    rec_text = " ".join(action.recommendations).lower()
    useful_terms = [
        "emergency fund", "insurance", "diversif", "rebalance", "tax",
        "401k", "ira", "debt", "save", "budget", "inflation", "income"
    ]
    hits = sum(1 for term in useful_terms if term in rec_text)
    rec_score = min(0.25, hits * 0.04)
    # Bonus for having 3+ specific recommendations
    if len(action.recommendations) >= 3:
        rec_score = min(0.25, rec_score + 0.05)
    components["recommendation_quality"] = round(rec_score, 4)
 
    # 4. Priority recommendation clarity (0.15)
    priority = action.priority_recommendation.lower()
    clarity_score = 0.0
    if len(priority) >= 20:
        clarity_score += 0.08
    # Should match the most critical flag
    critical_flags_order = [
        "insurance_gap", "no_emergency_fund", "high_debt",
        "concentration_risk", "low_savings_rate"
    ]
    for flag in critical_flags_order:
        if flag in expected_flags:
            keywords = expected_keywords.get(flag, [])
            if any(kw in priority for kw in keywords):
                clarity_score += 0.07
            break
    components["priority_clarity"] = round(min(0.15, clarity_score), 4)
 
    total = sum(components.values()) - sum(penalties.values())
    total = max(STRICT_SCORE_EPSILON, min(1.0 - STRICT_SCORE_EPSILON, total))
 
    return Reward(
        total=round(total, 3),
        components=components,
        penalties=penalties,
        explanation=(
            f"Expected flags: {list(expected_flags.keys())} | "
            f"Caught: {flags_caught}/{len(expected_flags)} | "
            f"Risk score range: {score_min:.2f}–{score_max:.2f} | "
            f"Agent score: {action.risk_score}"
        ),
    )
 
