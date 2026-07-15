"""Transparent, deterministic comparable-sales engine.

Built only on approved ``canonical_sales`` (see ``samvision.storage``). It is
decision support, NOT an appraisal, and NOT a trained ML model. No trained model
is used for scoring or valuation; nothing here writes to a database.

See ``docs/comparable_sales_engine_design.md``.
"""
