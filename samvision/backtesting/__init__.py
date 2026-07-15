"""Read-only leave-one-out backtesting of the deterministic comparable engine.

No database writes, no model access, no scoring-weight changes. See
``docs/comparable_sales_backtest_design.md``.
"""
from __future__ import annotations

__all__ = ["models", "metrics", "runner", "reporting"]
