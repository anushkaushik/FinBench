# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""FinBench — Financial Advisor OpenEnv Environment."""

# FIX #6: removed `from .client import FinbenchEnv` — client.py lives in the project
#         root, not inside the server/ package. Importing it here caused a ModuleNotFoundError.
#         FinbenchEnv is correctly exported from the root __init__.py.
from .FinBench_environment import FinbenchEnvironment, TaskId

__all__ = [
    "FinbenchEnvironment",
    "TaskId",
]

