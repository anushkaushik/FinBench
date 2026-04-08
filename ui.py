import json
import gradio as gr

from FinBench.server.FinBench_environment import FinbenchEnvironment
from FinBench.models import FinbenchAction

env = None
obs = None


def do_reset(task_id, scenario_index):
    global env, obs
    env = FinbenchEnvironment(task_id=task_id, scenario_index=int(scenario_index))
    obs = env.reset()
    return json.dumps(obs.model_dump(), indent=2), "Environment reset"


def do_step(
    action_type,
    allocations,
    reasoning,
    identified_risks,
    risk_score,
    recommendations,
    priority_recommendation,
    emergency_fund_months,
    insurance_recommendations,
    debt_payoff_strategy,
    investment_allocations,
    retirement_monthly_savings,
    tax_optimization_strategies,
    goal_timelines,
    rebalancing_frequency,
):
    global env, obs
    if env is None:
        return "", "Click Reset first"

    action = FinbenchAction(
        action_type=action_type,
        allocations=json.loads(allocations) if allocations.strip() else {},
        reasoning=reasoning,
        identified_risks=json.loads(identified_risks) if identified_risks.strip() else [],
        risk_score=float(risk_score or 0),
        recommendations=json.loads(recommendations) if recommendations.strip() else [],
        priority_recommendation=priority_recommendation,
        emergency_fund_months=float(emergency_fund_months or 0),
        insurance_recommendations=json.loads(insurance_recommendations) if insurance_recommendations.strip() else [],
        debt_payoff_strategy=debt_payoff_strategy,
        investment_allocations=json.loads(investment_allocations) if investment_allocations.strip() else {},
        retirement_monthly_savings=float(retirement_monthly_savings or 0),
        tax_optimization_strategies=json.loads(tax_optimization_strategies) if tax_optimization_strategies.strip() else [],
        goal_timelines=json.loads(goal_timelines) if goal_timelines.strip() else {},
        rebalancing_frequency=rebalancing_frequency,
    )

    obs = env.step(action)
    status = f"Reward: {obs.reward} | Done: {obs.done}"
    return json.dumps(obs.model_dump(), indent=2), status


with gr.Blocks() as demo:
    gr.Markdown("# FinBench Playground")

    with gr.Row():
        task_id = gr.Dropdown(
            choices=["task1_allocation", "task2_risk", "task3_plan"],
            value="task1_allocation",
            label="Task",
        )
        scenario_index = gr.Number(value=0, precision=0, label="Scenario Index")
        reset_btn = gr.Button("Reset")

    action_type = gr.Dropdown(
        choices=["allocate", "assess_risk", "financial_plan"],
        value="allocate",
        label="Action Type",
    )

    allocations = gr.Textbox(label="Allocations JSON")
    reasoning = gr.Textbox(label="Reasoning")
    identified_risks = gr.Textbox(label="Identified Risks JSON")
    risk_score = gr.Number(value=0, label="Risk Score")
    recommendations = gr.Textbox(label="Recommendations JSON")
    priority_recommendation = gr.Textbox(label="Priority Recommendation")
    emergency_fund_months = gr.Number(value=0, label="Emergency Fund Months")
    insurance_recommendations = gr.Textbox(label="Insurance Recommendations JSON")
    debt_payoff_strategy = gr.Textbox(label="Debt Payoff Strategy")
    investment_allocations = gr.Textbox(label="Investment Allocations JSON")
    retirement_monthly_savings = gr.Number(value=0, label="Retirement Monthly Savings")
    tax_optimization_strategies = gr.Textbox(label="Tax Optimization Strategies JSON")
    goal_timelines = gr.Textbox(label="Goal Timelines JSON")
    rebalancing_frequency = gr.Dropdown(
        choices=["monthly", "quarterly", "annually"],
        value="quarterly",
        label="Rebalancing Frequency",
    )

    step_btn = gr.Button("Step")
    status = gr.Textbox(label="Status")
    output = gr.Code(label="Raw JSON", language="json")

    reset_btn.click(do_reset, [task_id, scenario_index], [output, status])
    step_btn.click(
        do_step,
        [
            action_type,
            allocations,
            reasoning,
            identified_risks,
            risk_score,
            recommendations,
            priority_recommendation,
            emergency_fund_months,
            insurance_recommendations,
            debt_payoff_strategy,
            investment_allocations,
            retirement_monthly_savings,
            tax_optimization_strategies,
            goal_timelines,
            rebalancing_frequency,
        ],
        [output, status],
    )

demo.launch(server_name="0.0.0.0", server_port=7860)
