"""
Task 3 (Hard): Comprehensive Financial Plan
=============================================
Agent must generate a complete, coherent financial plan covering:
- Emergency fund, insurance, debt, investments, retirement,
  tax optimization, goal timelines, rebalancing strategy.
 
This is hard because all components must be internally consistent
and appropriate for the specific client.
"""
 
from __future__ import annotations
 
 
 
from typing import Dict
from ..models import ClientProfile, MarketConditions, Reward, _SCORE_EPSILON as STRICT_SCORE_EPSILON
 
 
 
 
def _retirement_savings_needed(client: ClientProfile, market: MarketConditions) -> float:
    """Estimate minimum monthly retirement savings using simplified FV formula."""
    years = max(1, 65 - client.age)
    monthly_rate = (market.equity_expected_return * 0.6 + market.bond_expected_return * 0.4) / 12
    # Target: 25x annual expenses at retirement (4% rule)
    target = client.monthly_expenses * 12 * 25
    existing_nw = max(0, client.net_worth)
    gap = max(0, target - existing_nw)
 
    if monthly_rate == 0 or years == 0:
        return gap / (years * 12) if years > 0 else 0
 
    months = years * 12
    # PMT formula: gap = PMT * [(1+r)^n - 1] / r
    fv_factor = ((1 + monthly_rate) ** months - 1) / monthly_rate
    return gap / fv_factor if fv_factor > 0 else 0
 
 
# FIX #7: updated signature to accept FinbenchAction directly instead of
#         FinancialPlanAction (which doesn't exist). Reads the same fields.
# FIX (market arg): FinbenchEnvironment passes self._market which is a MarketConditions
#         object — the grader correctly receives it as such, no change needed here.
def grade(
    action,
    client: ClientProfile,
    market: MarketConditions,
) -> Reward:
    """
    Score a comprehensive financial plan 0.0–1.0.
    Components:
      - Emergency fund adequacy           (0.12)
      - Insurance recommendations         (0.08)
      - Debt payoff strategy              (0.10)
      - Investment allocation suitability (0.20)
      - Retirement savings adequacy       (0.15)
      - Tax optimization                  (0.15)
      - Goal timeline realism             (0.10)
      - Internal plan consistency         (0.10)
    """
    components: Dict[str, float] = {}
    penalties: Dict[str, float] = {}
 
    monthly_income = client.annual_income / 12
 
    # ── 1. Emergency Fund (0.12) ──────────────────────────────────────────────
    if not client.has_emergency_fund:
        ideal_months = 6 if client.dependents > 0 else 3
        if action.emergency_fund_months >= ideal_months:
            components["emergency_fund"] = 0.12
        elif action.emergency_fund_months >= ideal_months * 0.5:
            components["emergency_fund"] = 0.06
        else:
            components["emergency_fund"] = 0.0
    else:
        components["emergency_fund"] = 0.12  # already has one
 
    # ── 2. Insurance (0.08) ──────────────────────────────────────────────────
    if not client.has_insurance:
        ins_text = " ".join(action.insurance_recommendations).lower()
        if client.dependents > 0 and "life" in ins_text:
            components["insurance"] = 0.08
        elif any(t in ins_text for t in ["term", "disability", "health", "life"]):
            components["insurance"] = 0.05
        else:
            components["insurance"] = 0.0
    else:
        components["insurance"] = 0.08
 
    # ── 3. Debt Payoff Strategy (0.10) ────────────────────────────────────────
    if client.debt_to_income_ratio > 0.36:
        debt_text = action.debt_payoff_strategy.lower()
        has_strategy = any(
            t in debt_text
            for t in ["avalanche", "snowball", "high interest", "pay off", "consolidat"]
        )
        components["debt_strategy"] = 0.10 if has_strategy else 0.02
    else:
        components["debt_strategy"] = 0.10  # no debt problem = no penalty
 
    # ── 4. Investment Allocation Suitability (0.20) ───────────────────────────
    allocs = action.investment_allocations
    total_alloc = sum(allocs.values())
 
    # Sum check
    if abs(total_alloc - 100.0) > 5:
        penalties["alloc_sum_error"] = 0.10
        components["investment_allocation"] = 0.0
    else:
        equity = allocs.get("equities", allocs.get("stocks", 0))
        bonds = allocs.get("bonds", 0)
        cash = allocs.get("cash", 0)
 
        # Expected equity range by risk
        eq_ranges = {
            "conservative": (10, 30),
            "moderate": (35, 60),
            "aggressive": (55, 85),
        }
        lo, hi = eq_ranges[client.risk_tolerance]
 
        alloc_score = 0.0
        if lo <= equity <= hi:
            alloc_score += 0.10
 
        # Age adjustment: high equity bad near retirement
        if client.age >= 60 and equity > 50:
            penalties["too_aggressive_near_retirement"] = 0.05
        else:
            alloc_score += 0.05
 
        # Cash not excessively high
        if cash <= 20:
            alloc_score += 0.05
 
        components["investment_allocation"] = min(0.20, alloc_score)
 
    # ── 5. Retirement Savings (0.15) ──────────────────────────────────────────
    needed = _retirement_savings_needed(client, market)
    agent_savings = action.retirement_monthly_savings
    income_pct = agent_savings / monthly_income if monthly_income > 0 else 0
 
    # Must save at least 10% of income for partial credit
    if income_pct >= 0.15 and agent_savings >= needed * 0.80:
        components["retirement_savings"] = 0.15
    elif income_pct >= 0.10:
        components["retirement_savings"] = 0.08
    elif income_pct >= 0.05:
        components["retirement_savings"] = 0.04
    else:
        components["retirement_savings"] = 0.0
 
    # ── 6. Tax Optimization (0.15) ────────────────────────────────────────────
    tax_text = " ".join(action.tax_optimization_strategies).lower()
    tax_score = 0.0
    tax_keywords = {
        "401k": 0.04, "ira": 0.04, "roth": 0.03,
        "tax-loss": 0.02, "hsa": 0.02,
        "municipal": 0.02, "capital gains": 0.02,
    }
    for kw, pts in tax_keywords.items():
        if kw in tax_text:
            tax_score += pts
 
    # High earners must mention tax-advantaged accounts
    if client.tax_bracket >= 0.32 and not any(
        t in tax_text for t in ["401k", "ira", "roth", "hsa"]
    ):
        penalties["missed_tax_advantaged"] = 0.05
 
    components["tax_optimization"] = min(0.15, tax_score)
 
    # ── 7. Goal Timeline Realism (0.10) ──────────────────────────────────────
    if client.goals and action.goal_timelines:
        # Check timelines are within investment horizon
        max_timeline = client.investment_horizon_years
        realistic = sum(
            1 for yrs in action.goal_timelines.values()
            if 1 <= yrs <= max_timeline + 5
        )
        timeline_score = (realistic / len(action.goal_timelines)) * 0.10
        components["goal_timelines"] = round(timeline_score, 4)
    else:
        components["goal_timelines"] = 0.05  # partial for missing
 
    # ── 8. Internal Consistency (0.10) ────────────────────────────────────────
    consistency_score = 0.10
    # Inconsistency: saving a lot but no debt payoff strategy with high debt
    if client.debt_to_income_ratio > 0.5 and agent_savings > monthly_income * 0.3:
        penalties["inconsistent_savings_vs_debt"] = 0.03
        consistency_score -= 0.03
    # Inconsistency: aggressive allocation but conservative risk tolerance
    if client.risk_tolerance == "conservative":
        equity = allocs.get("equities", allocs.get("stocks", 0))
        if equity > 40:
            penalties["allocation_risk_mismatch"] = 0.05
            consistency_score -= 0.05
    # Bonus: reasoning present
    if len(action.reasoning.strip()) >= 50:
        consistency_score = min(0.10, consistency_score + 0.02)
 
    components["consistency"] = max(0.0, round(consistency_score, 4))
 
    # ── Final Score ───────────────────────────────────────────────────────────
    total = sum(components.values()) - sum(penalties.values())
    total = max(STRICT_SCORE_EPSILON, min(1.0 - STRICT_SCORE_EPSILON, total))
 
    return Reward(
        total=round(total, 4),
        components=components,
        penalties=penalties,
        explanation=(
            f"Needed monthly retirement savings: ${needed:,.0f} | "
            f"Agent savings: ${agent_savings:,.0f} | "
            f"Tax bracket: {client.tax_bracket*100:.0f}% | "
            f"Goals: {client.goals}"
        ),
    )
