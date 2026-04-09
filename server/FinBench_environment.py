# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""FinBench Financial Advisor Environment Implementation."""

from uuid import uuid4
from typing import Any, Dict, List, Literal, Optional, Tuple

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

try:
    from ..models import FinbenchAction, FinbenchObservation
    from ..data.scenarios import BASELINE_MARKET, TASK1_SCENARIOS, TASK2_SCENARIOS, TASK3_SCENARIOS
    from ..tasks import task1_allocation, task2_risk, task3_plan
except ImportError:
    from FinBench.models import FinbenchAction, FinbenchObservation
    from FinBench.data.scenarios import BASELINE_MARKET, TASK1_SCENARIOS, TASK2_SCENARIOS, TASK3_SCENARIOS
    from FinBench.tasks import task1_allocation, task2_risk, task3_plan


TaskId = Literal["task1_allocation", "task2_risk", "task3_plan"]

TASK_DESCRIPTIONS = {
    "task1_allocation": (
        "TASK 1 — Portfolio Allocation (Easy)\n"
        "Allocate the client's portfolio across: equities, bonds, cash, real_estate, commodities. "
        "Allocations must sum to 100%. Consider risk tolerance, age, and emergency fund status."
    ),
    "task2_risk": (
        "TASK 2 — Risk Assessment (Medium)\n"
        "Identify all risk factors, assign a risk score (0–10), and provide prioritized recommendations."
    ),
    "task3_plan": (
        "TASK 3 — Comprehensive Financial Plan (Hard)\n"
        "Generate a complete plan covering: emergency fund, insurance, debt payoff, investments, "
        "retirement savings, tax optimization, goal timelines, and rebalancing frequency."
    ),
}

MAX_STEPS = 3


class FinbenchEnvironment(Environment):
    """Financial Advisor evaluation environment."""

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self, task_id: TaskId = "task1_allocation", scenario_index: int = 0):
        self._task_id = task_id
        self._scenario_index = scenario_index
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._client = None
        self._market = BASELINE_MARKET
        self._actions_taken: List[Dict[str, Any]] = []
        self._done = False

    def _get_client(self):
        scenarios = {
            "task1_allocation": TASK1_SCENARIOS,
            "task2_risk": TASK2_SCENARIOS,
            "task3_plan": TASK3_SCENARIOS,
        }[self._task_id]
        return scenarios[self._scenario_index % len(scenarios)]

    def reset(self) -> FinbenchObservation:
        self._client = self._get_client()
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._actions_taken = []
        self._done = False

        return FinbenchObservation(
            client=self._client.model_dump(),
            market_conditions=self._market.model_dump(),
            task_id=self._task_id,
            task_description=TASK_DESCRIPTIONS[self._task_id],
            step_number=0,
            previous_actions=[],
            feedback="",
            done=False,
            reward=0.0001,
        )

    def step(self, action: FinbenchAction) -> FinbenchObservation:
        self._state.step_count += 1

        # FIX #5: removed broken local imports of AllocationAction / RiskAssessmentAction /
        #         FinancialPlanAction from models — those classes do not exist in models.py.
        #         The task graders expect ClientProfile objects, not sub-action wrappers.
        #         Pass FinbenchAction directly to each grader; the graders already accept it
        #         because they only read the relevant fields (allocations, identified_risks, etc.).
        if action.action_type == "allocate":
            reward_obj = task1_allocation.grade(action, self._client)
        elif action.action_type == "assess_risk":
            reward_obj = task2_risk.grade(action, self._client)
        else:
            reward_obj = task3_plan.grade(action, self._client, self._market)

        self._actions_taken.append({
            "step": self._state.step_count,
            "action_type": action.action_type,
            "reward": reward_obj.total,
            "components": reward_obj.components,
        })

        done = reward_obj.total >= 0.90 or self._state.step_count >= MAX_STEPS
        self._done = done

        feedback = (
            f"Step {self._state.step_count} reward: {reward_obj.total:.4f}\n"
            f"Components: {reward_obj.components}\n"
            f"{reward_obj.explanation}"
        )

        return FinbenchObservation(
            client=self._client.model_dump(),
            market_conditions=self._market.model_dump(),
            task_id=self._task_id,
            task_description=TASK_DESCRIPTIONS[self._task_id],
            step_number=self._state.step_count,
            previous_actions=self._actions_taken,
            feedback=feedback,
            done=done,
            reward=reward_obj.total,
        )

    @property
    def state(self) -> State:
        return self._state
