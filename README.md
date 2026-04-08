---
title: FinBench Environment Server
emoji: 🏦
colorFrom: blue
colorTo: indigo
sdk: docker
pinned: false
app_port: 8000
base_path: /web
tags:
  - openenv
  - finance
  - evaluation
---

# FinBench Environment

FinBench is a financial-advisor evaluation environment built with OpenEnv. It tests an agent’s ability to solve three portfolio-planning tasks:

- Portfolio allocation
- Risk assessment
- Comprehensive financial planning

The environment provides structured client profiles and market conditions, then scores the agent’s response with task-specific graders.

Run the server with package-qualified imports:

```bash
python -m FinBench.server.app
```

or

```bash
uvicorn FinBench.server.app:app --host 0.0.0.0 --port 8000
```

## Quick Start

You can use the environment through the `FinbenchEnv` client.

```python
from FinBench import FinbenchAction, FinbenchEnv

with FinbenchEnv(base_url="http://localhost:8000") as env:
    result = env.reset()

    action = FinbenchAction(
        action_type="allocate",
        allocations={
            "equities": 50,
            "bonds": 30,
            "cash": 10,
            "real_estate": 5,
            "commodities": 5,
        },
        reasoning="Balanced allocation based on the client's age, risk tolerance, and liquidity needs."
    )

    result = env.step(action)
    print(result.observation.feedback)
    print(result.reward)
