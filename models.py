# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Data models for the FinBench Financial Advisor Environment."""

from typing import Dict, List, Literal, Any
from openenv.core.env_server.types import Action, Observation
from pydantic import BaseModel, Field


class ClientProfile(BaseModel):
    client_id: str
    age: int = Field(ge=18, le=100)
    annual_income: float = Field(ge=0)
    net_worth: float
    monthly_expenses: float = Field(ge=0)
    dependents: int = Field(ge=0)
    risk_tolerance: Literal["conservative", "moderate", "aggressive"]
    investment_horizon_years: int = Field(ge=1, le=50)
    existing_portfolio: Dict[str, float] = Field(default_factory=dict)
    goals: List[str] = Field(default_factory=list)
    tax_bracket: float = Field(ge=0.0, le=0.5)
    has_emergency_fund: bool = False
    has_insurance: bool = False
    debt_to_income_ratio: float = Field(ge=0.0)


class FinbenchAction(Action):
    """Action for the FinBench environment."""
    action_type: Literal["allocate", "assess_risk", "financial_plan"]
    # Task 1
    allocations: Dict[str, float] = Field(default_factory=dict)
    reasoning: str = Field(default="")
    # Task 2
    identified_risks: List[str] = Field(default_factory=list)
    risk_score: float = Field(default=0.01, gt=0.0, lt=1.0)
    recommendations: List[str] = Field(default_factory=list)
    priority_recommendation: str = Field(default="")
    # Task 3
    emergency_fund_months: float = Field(default=0.0, ge=0)
    insurance_recommendations: List[str] = Field(default_factory=list)
    debt_payoff_strategy: str = Field(default="")
    investment_allocations: Dict[str, float] = Field(default_factory=dict)
    retirement_monthly_savings: float = Field(default=0.0, ge=0)
    tax_optimization_strategies: List[str] = Field(default_factory=list)
    goal_timelines: Dict[str, int] = Field(default_factory=dict)
    rebalancing_frequency: Literal["monthly", "quarterly", "annually"] = "annually"


class FinbenchObservation(Observation):
    """Observation from the FinBench environment."""
    client: Dict[str, Any] = Field(default_factory=dict)
    market_conditions: Dict[str, Any] = Field(default_factory=dict)
    task_id: str = Field(default="")
    task_description: str = Field(default="")
    step_number: int = Field(default=0)
    previous_actions: List[Dict[str, Any]] = Field(default_factory=list)
    feedback: str = Field(default="")

class MarketConditions(BaseModel):
    equity_expected_return: float
    bond_expected_return: float
    inflation_rate: float
    interest_rate: float
    market_volatility: str


class Reward(BaseModel):
    total: float
    components: Dict[str, float] = Field(default_factory=dict)
    penalties: Dict[str, float] = Field(default_factory=dict)
    explanation: str = ""

