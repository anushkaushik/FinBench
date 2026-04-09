"""
inference.py — FinBench Financial Advisor OpenEnv
===================================================
Runs LLM inference against all 3 tasks across all scenarios.

MANDATORY environment variables:
    API_BASE_URL        The API endpoint for the LLM.
    MODEL_NAME          The model identifier to use for inference.
    HF_TOKEN            Your Hugging Face / API key.

Defaults:
    API_BASE_URL = "https://router.huggingface.co/v1"
    MODEL_NAME   = "Qwen/Qwen2.5-72B-Instruct"

Usage:
    export API_BASE_URL=https://api.openai.com/v1
    export MODEL_NAME=gpt-4o
    export HF_TOKEN=sk-...
    python inference.py
    python inference.py --task task1_allocation
"""

from __future__ import annotations

import argparse
import json
import time
from typing import List, Optional
import os
import sys
from openai import OpenAI



from FinBench.server.FinBench_environment import FinbenchEnvironment, TaskId
from FinBench.models import FinbenchAction, FinbenchObservation

# FIX #3: FinbenchObservation imported here and used as the correct type hint below
from FinBench.models import FinbenchAction, FinbenchObservation


# ── Environment variables ──────────────────────────────────────────────────────
API_BASE_URL = os.getenv("API_BASE_URL") or "https://router.huggingface.co/v1"
MODEL_NAME   = os.getenv("MODEL_NAME")   or "Qwen/Qwen2.5-72B-Instruct"
HF_TOKEN     = os.getenv("HF_TOKEN")     or os.getenv("OPENAI_API_KEY", "")

BENCHMARK    = "finbench"
NUM_SCENARIOS = 6
MAX_STEPS     = 3  # matches openenv.yaml max_steps
STRICT_SCORE_EPSILON = 0.0001

# ── Logging helpers (mandatory stdout format) ─────────────────────────────────

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val  = str(done).lower()
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}",
        flush=True,
    )


# ── System prompt ──────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are an expert financial advisor AI. You will be given a client profile
and a specific task. You must respond with a valid JSON object matching the required schema.

IMPORTANT:
- All portfolio allocations must sum to exactly 100.
- Be specific and detailed in all reasoning fields.
- Consider the client's age, risk tolerance, income, dependents, and goals holistically.
- Respond ONLY with the JSON object — no markdown, no explanation outside JSON.
"""

SCHEMA_PROMPTS = {
    "task1_allocation": """
Respond with JSON matching this schema exactly:
{
  "action_type": "allocate",
  "allocations": {
    "equities": <float 0-100>,
    "bonds": <float 0-100>,
    "cash": <float 0-100>,
    "real_estate": <float 0-100>,
    "commodities": <float 0-100>
  },
  "reasoning": "<detailed explanation>"
}
Allocations must sum to exactly 100.
""",
    "task2_risk": """
Respond with JSON matching this schema exactly:
{
  "action_type": "assess_risk",
  "identified_risks": ["<risk 1>", "<risk 2>", ...],
  "risk_score": <float 0.0-1.0>,
  "recommendations": ["<rec 1>", "<rec 2>", ...],
  "priority_recommendation": "<single most important action>"
}
""",
    "task3_plan": """
Respond with JSON matching this schema exactly:
{
  "action_type": "financial_plan",
  "emergency_fund_months": <float>,
  "insurance_recommendations": ["<e.g. term life insurance>", ...],
  "debt_payoff_strategy": "<e.g. avalanche method>",
  "investment_allocations": {
    "equities": <float 0-100>,
    "bonds": <float 0-100>,
    "cash": <float 0-100>,
    "real_estate": <float 0-100>,
    "commodities": <float 0-100>
  },
  "retirement_monthly_savings": <float>,
  "tax_optimization_strategies": ["<e.g. maximize 401k>", ...],
  "goal_timelines": {"<goal name>": <years as int>, ...},
  "rebalancing_frequency": "quarterly",
  "reasoning": "<comprehensive explanation>"
}
investment_allocations must sum to exactly 100.
""",
}


# ── Prompt builder ─────────────────────────────────────────────────────────────
# FIX #3: was `obs: Observation` — Observation was never imported anywhere.
#         Corrected to FinbenchObservation (already imported above).
#         obs.client and obs.market_conditions are dicts (from model_dump()),
#         so access fields with dict syntax, not attribute syntax.
def build_prompt(obs: FinbenchObservation, task_id: TaskId) -> str:
    c = obs.client
    m = obs.market_conditions
    prompt = f"""
CLIENT PROFILE:
- Age: {c['age']}
- Annual Income: ${c['annual_income']:,.0f}
- Net Worth: ${c['net_worth']:,.0f}
- Monthly Expenses: ${c['monthly_expenses']:,.0f}
- Dependents: {c['dependents']}
- Risk Tolerance: {c['risk_tolerance']}
- Investment Horizon: {c['investment_horizon_years']} years
- Tax Bracket: {c['tax_bracket']*100:.0f}%
- Debt-to-Income Ratio: {c['debt_to_income_ratio']:.2f}
- Has Emergency Fund: {c['has_emergency_fund']}
- Has Insurance: {c['has_insurance']}
- Existing Portfolio: {c['existing_portfolio']}
- Goals: {c['goals']}

MARKET CONDITIONS:
- Equity Expected Return: {m['equity_expected_return']*100:.0f}%
- Bond Expected Return: {m['bond_expected_return']*100:.0f}%
- Inflation Rate: {m['inflation_rate']*100:.0f}%
- Market Volatility: {m['market_volatility']}

TASK: {obs.task_description}
"""
    if obs.feedback:
        prompt += f"\nPREVIOUS FEEDBACK:\n{obs.feedback}\n"
    return prompt + "\n" + SCHEMA_PROMPTS[task_id]


# ── Action parser ──────────────────────────────────────────────────────────────
# FIX #2: was instantiating AllocationAction / RiskAssessmentAction / FinancialPlanAction
#         which are never defined or imported in this file.
#         FinbenchEnvironment.step() accepts a FinbenchAction directly — use that.
def parse_action(task_id: TaskId, response_text: str) -> FinbenchAction:
    text = response_text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        text = "\n".join(lines[1:-1]) if len(lines) > 2 else text
    data = json.loads(text)
    return FinbenchAction(**data)


# ── Single episode runner ──────────────────────────────────────────────────────
def run_episode(
    client_llm: OpenAI,
    task_id: TaskId,
    scenario_idx: int,
) -> float:
    """Run one episode. Returns best reward (score) for this scenario."""
    # FIX #1: was FinancialAdvisorEnv(...) — class never imported or defined anywhere.
    #         Correct class is FinbenchEnvironment (imported at top of file).
    env = FinbenchEnvironment(task_id=task_id, scenario_index=scenario_idx)
    obs = env.reset()

    rewards: List[float] = []
    steps_taken = 0
    score = STRICT_SCORE_EPSILON
    success = False

    log_start(task=task_id, env=BENCHMARK, model=MODEL_NAME)

    try:
        done = False
        for step in range(1, MAX_STEPS + 1):
            if done:
                break

            user_prompt = build_prompt(obs, task_id)
            error_msg = None
            reward = STRICT_SCORE_EPSILON

            try:
                response = client_llm.chat.completions.create(
                    model=MODEL_NAME,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user",   "content": user_prompt},
                    ],
                    temperature=0.1,
                    max_tokens=1200,
                )
                response_text = response.choices[0].message.content
                action = parse_action(task_id, response_text)
                action_str = action.action_type

            except Exception as e:
                error_msg = str(e)
                action_str = "parse_error"
                log_step(step=step, action=action_str, reward=0.00, done=True, error=error_msg)
                rewards.append(STRICT_SCORE_EPSILON)
                steps_taken = step
                break

            # FIX #4: was `obs, reward_obj, done, info = env.step(action)`
            #         FinbenchEnvironment.step() returns a single FinbenchObservation,
            #         not a 4-tuple. Reward and done are fields on the observation.
            obs = env.step(action)
            reward = obs.reward
            done = obs.done
            rewards.append(reward)
            steps_taken = step

            log_step(step=step, action=action_str, reward=reward, done=done, error=error_msg)

            if not done:
                time.sleep(0.3)

        score = max(rewards) if rewards else STRICT_SCORE_EPSILON
        score = min(max(score, STRICT_SCORE_EPSILON), 1.0 - STRICT_SCORE_EPSILON)
        success = score >= 0.5

    finally:
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

    return score


# ── Task runner ────────────────────────────────────────────────────────────────
def run_task(client_llm: OpenAI, task_id: TaskId) -> dict:
    print(f"\n{'='*60}", flush=True)
    print(f"TASK: {task_id.upper()}", flush=True)
    print(f"{'='*60}", flush=True)

    scenario_scores = []
    for i in range(NUM_SCENARIOS):
        score = run_episode(client_llm, task_id, i)
        # Ensure stored scores are strictly within (0, 1)
        score = min(max(score, STRICT_SCORE_EPSILON), 1.0 - STRICT_SCORE_EPSILON)
        scenario_scores.append(score)

    avg = sum(scenario_scores) / len(scenario_scores)
    print(f"\n  Task Average: {avg:.4f}", flush=True)

    return {
        "task_id": task_id,
        "scenario_scores": scenario_scores,
        "average_score": round(avg, 4),
    }


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="FinBench Inference")
    parser.add_argument(
        "--task",
        choices=["task1_allocation", "task2_risk", "task3_plan", "all"],
        default="all",
    )
    parser.add_argument("--output", default="inference_results.json")
    args = parser.parse_args()

    if not HF_TOKEN:
        print("ERROR: HF_TOKEN (or OPENAI_API_KEY) environment variable not set.")
        sys.exit(1)

    print("\n🏦 FinBench — Financial Advisor OpenEnv Inference")
    print(f"   API Base : {API_BASE_URL}")
    print(f"   Model    : {MODEL_NAME}")
    print(f"   Task     : {args.task}")

    client_llm = OpenAI(api_key=HF_TOKEN, base_url=API_BASE_URL)

    tasks_to_run = (
        ["task1_allocation", "task2_risk", "task3_plan"]
        if args.task == "all"
        else [args.task]
    )

    all_results = []
    for task_id in tasks_to_run:
        result = run_task(client_llm, task_id)
        all_results.append(result)

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("RESULTS SUMMARY")
    print(f"{'='*60}")
    scores = []
    for r in all_results:
        scores.append(r["average_score"])
        print(f"  {r['task_id']:30s} → {r['average_score']:.4f}")

    overall = sum(scores) / len(scores) if scores else STRICT_SCORE_EPSILON
    print(f"\n  Overall Average : {overall:.4f}")

    output = {
        "api_base_url": API_BASE_URL,
        "model": MODEL_NAME,
        "tasks": all_results,
        "overall_average": round(overall, 4),
    }
    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)
    print(f"  Results saved  : {args.output}\n")


if __name__ == "__main__":
    main()
