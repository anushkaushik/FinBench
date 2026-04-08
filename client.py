# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Finbench Environment Client."""

from typing import Dict

from openenv.core import EnvClient
from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State

from .models import FinbenchAction, FinbenchObservation


class FinbenchEnv(
    EnvClient[FinbenchAction, FinbenchObservation, State]
):
    """
    Client for the FinBench Financial Advisor Environment.

    This client maintains a persistent WebSocket connection to the environment server,
    enabling efficient multi-step interactions with lower latency.
    Each client instance has its own dedicated environment session on the server.

    Example:
        >>> with FinbenchEnv(base_url="http://localhost:8000") as client:
        ...     result = client.reset()
        ...     result = client.step(FinbenchAction(
        ...         action_type="allocate",
        ...         allocations={"equities": 20, "bonds": 55, "cash": 15, "real_estate": 5, "commodities": 5},
        ...         reasoning="Conservative client near retirement."
        ...     ))
        ...     print(result.observation.feedback)
    """

    def _step_payload(self, action: FinbenchAction) -> Dict:
        """Convert FinbenchAction to JSON payload for step message."""
        return action.model_dump()

    def _parse_result(self, payload: Dict) -> StepResult[FinbenchObservation]:
        """Parse server response into StepResult[FinbenchObservation]."""
        obs_data = payload.get("observation", {})
        observation = FinbenchObservation(
            client=obs_data.get("client", {}),
            market_conditions=obs_data.get("market_conditions", {}),
            task_id=obs_data.get("task_id", ""),
            task_description=obs_data.get("task_description", ""),
            step_number=obs_data.get("step_number", 0),
            previous_actions=obs_data.get("previous_actions", []),
            feedback=obs_data.get("feedback", ""),
            done=payload.get("done", False),
            reward=payload.get("reward"),
            metadata=obs_data.get("metadata", {}),
        )
        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict) -> State:
        """Parse server response into State object."""
        return State(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
        )


# ── Rule-based demo agents ────────────────────────────────────────────────────

def rule_based_agent_task1(obs: FinbenchObservation) -> FinbenchAction:
    """Simple rule-based agent for portfolio allocation."""
    client = obs.client
    rt = client["risk_tolerance"]
    age = client["age"]
    has_emergency_fund = client["has_emergency_fund"]

    profiles = {
        "conservative": {"equities": 18, "bonds": 57, "cash": 15, "real_estate": 5, "commodities": 5},
        "moderate": {"equities": 50, "bonds": 35, "cash": 8, "real_estate": 4, "commodities": 3},
        "aggressive": {"equities": 70, "bonds": 12, "cash": 5, "real_estate": 8, "commodities": 5},
    }
    allocs = profiles[rt].copy()

    if age >= 60 and rt != "conservative":
        allocs["bonds"] += 10
        allocs["equities"] -= 10

    if not has_emergency_fund:
        allocs["cash"] = max(allocs["cash"], 15)
        diff = sum(allocs.values()) - 100
        allocs["equities"] -= diff

    return FinbenchAction(
        action_type="allocate",
        allocations=allocs,
        reasoning=(
            f"Client is {age} years old with {rt} risk tolerance. "
            f"Adjusted allocation for age and emergency fund status."
        ),
    )


def rule_based_agent_task2(obs: FinbenchObservation) -> FinbenchAction:
    """Simple rule-based agent for risk assessment."""
    client = obs.client
    risks = []
    recs = []
    score = {
        "conservative": 3.0,
        "moderate": 5.0,
        "aggressive": 6.0,
    }[client["risk_tolerance"]]

    if not client["has_emergency_fund"]:
        risks.append("No emergency fund - financially vulnerable to unexpected expenses")
        recs.append("Build a 3-6 month emergency fund in a high-yield savings account")
        score += 1.5

    if not client["has_insurance"] and client["dependents"] > 0:
        risks.append(f"No insurance with {client['dependents']} dependent(s) - major risk")
        recs.append("Obtain term life insurance immediately to protect dependents")
        score += 2.0

    if client["debt_to_income_ratio"] > 0.36:
        risks.append(f"High debt-to-income ratio ({client['debt_to_income_ratio']:.0%}) above 36% threshold")
        recs.append("Implement debt avalanche strategy - pay highest-interest debt first")
        score += 1.5

    monthly = client["annual_income"] / 12
    savings_rate = (monthly - client["monthly_expenses"]) / monthly if monthly > 0 else 0
    if savings_rate < 0.15:
        risks.append(f"Low savings rate (~{savings_rate*100:.0f}%) - below recommended 15%")
        recs.append("Reduce discretionary spending to achieve at least 15% savings rate")
        score += 1.0

    for asset, pct in client.get("existing_portfolio", {}).items():
        if pct > 60:
            risks.append(f"Over-concentration in {asset} ({pct}%) - lacks diversification")
            recs.append(f"Rebalance away from {asset} to diversify risk")
            score += 1.0

    if client["tax_bracket"] >= 0.32:
        risks.append("High income with potential tax inefficiency")
        recs.append("Maximize tax-advantaged accounts: 401k, IRA, HSA")
        score += 0.5

    if not risks:
        risks.append("Client profile appears generally sound with few major risk flags")
        recs.append("Continue current strategy with annual review")

    return FinbenchAction(
        action_type="assess_risk",
        identified_risks=risks,
        risk_score=round(min(10.0, score), 1),
        recommendations=recs,
        priority_recommendation=recs[0] if recs else "Review financial plan annually.",
    )


def rule_based_agent_task3(obs: FinbenchObservation) -> FinbenchAction:
    """Simple rule-based agent for comprehensive financial planning."""
    client = obs.client
    monthly = client["annual_income"] / 12
    rt = client["risk_tolerance"]

    alloc_map = {
        "conservative": {"equities": 20, "bonds": 55, "cash": 15, "real_estate": 5, "commodities": 5},
        "moderate": {"equities": 50, "bonds": 32, "cash": 8, "real_estate": 7, "commodities": 3},
        "aggressive": {"equities": 68, "bonds": 15, "cash": 7, "real_estate": 7, "commodities": 3},
    }

    insurance = []
    if not client["has_insurance"]:
        insurance.append("Term life insurance - $500,000 coverage")
        if client["dependents"] > 0:
            insurance.append("Disability insurance to protect income")

    debt_strategy = ""
    if client["debt_to_income_ratio"] > 0.36:
        debt_strategy = (
            "Avalanche method: list all debts by interest rate, "
            "pay minimums on all but highest-rate debt, apply all extra cash there. "
            "Target DTI below 0.36 within 24-36 months."
        )

    tax_strategies = ["Maximize 401k contribution ($23,000/year)"]
    if client["tax_bracket"] >= 0.24:
        tax_strategies.append("Open and fund Roth IRA ($7,000/year)")
    if client["tax_bracket"] >= 0.32:
        tax_strategies.append("HSA contributions for triple tax advantage")
        tax_strategies.append("Tax-loss harvesting in taxable accounts")
        tax_strategies.append("Consider municipal bonds for tax-efficient fixed income")

    goal_timelines = {}
    for goal in client.get("goals", []):
        if "retire" in goal.lower():
            goal_timelines[goal] = max(65 - client["age"], 1)
        elif "home" in goal.lower() or "house" in goal.lower():
            goal_timelines[goal] = 5
        elif "education" in goal.lower() or "college" in goal.lower():
            goal_timelines[goal] = 18
        else:
            goal_timelines[goal] = 10

    return FinbenchAction(
        action_type="financial_plan",
        emergency_fund_months=6.0 if client["dependents"] > 0 else 3.0,
        insurance_recommendations=insurance,
        debt_payoff_strategy=debt_strategy,
        investment_allocations=alloc_map[rt],
        retirement_monthly_savings=round(max(monthly * 0.15, 500), 2),
        tax_optimization_strategies=tax_strategies,
        goal_timelines=goal_timelines,
        rebalancing_frequency="quarterly",
        reasoning=(
            f"Comprehensive plan for {client['age']}-year-old {rt} investor. "
            f"Annual income ${client['annual_income']:,.0f}, net worth ${client['net_worth']:,.0f}. "
            f"Priority order: emergency fund -> insurance -> debt -> invest -> tax optimize."
        ),
    )


# ── Demo runner ───────────────────────────────────────────────────────────────

AGENTS = {
    "task1_allocation": rule_based_agent_task1,
    "task2_risk": rule_based_agent_task2,
    "task3_plan": rule_based_agent_task3,
}

TASK_LABELS = {
    "task1_allocation": "Task 1 - Portfolio Allocation    [EASY]",
    "task2_risk": "Task 2 - Risk Assessment         [MEDIUM]",
    "task3_plan": "Task 3 - Comprehensive Plan      [HARD]",
}


def run_demo(base_url: str = "http://localhost:8000"):
    from .server.FinBench_environment import FinbenchEnvironment

    print("\n" + "=" * 60)
    print("  FinBench - Rule-Based Agent Demo")
    print("=" * 60)

    all_scores = []

    for task_id, agent_fn in AGENTS.items():
        print(f"\n{'-' * 60}")
        print(f"  {TASK_LABELS[task_id]}")
        print(f"{'-' * 60}")

        task_scores = []
        for scenario_idx in range(3):
            env = FinbenchEnvironment(task_id=task_id, scenario_index=scenario_idx)
            obs = env.reset()
            client = obs.client

            print(f"\n  Scenario {scenario_idx + 1}: {client.get('client_id')}")
            print(
                f"  Age={client.get('age')} | Risk={client.get('risk_tolerance')} | "
                f"Income=${client.get('annual_income'):,.0f} | DTI={client.get('debt_to_income_ratio'):.2f}"
            )

            best_reward = 0.0
            done = False
            step = 0

            while not done:
                action = agent_fn(obs)
                obs = env.step(action)
                done = obs.done
                reward = obs.reward or 0.0
                best_reward = max(best_reward, reward)
                step += 1
                print(f"  Step {step}: reward={reward:.4f}")

            task_scores.append(best_reward)
            print(f"  -> Best: {best_reward:.4f}")

        avg = sum(task_scores) / len(task_scores)
        all_scores.append(avg)
        print(f"\n  Task Average: {avg:.4f}")

    print(f"\n{'=' * 60}")
    print("  SUMMARY")
    print(f"{'=' * 60}")
    for task_id, score in zip(AGENTS.keys(), all_scores):
        bar = "#" * int(score * 20) + "." * (20 - int(score * 20))
        print(f"  {TASK_LABELS[task_id]}: {bar} {score:.4f}")

    overall = sum(all_scores) / len(all_scores)
    print(f"\n  Overall Average: {overall:.4f}")
    print("\n  Environment working correctly!\n")


if __name__ == "__main__":
    run_demo()
