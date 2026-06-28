"""Leakage-safe feature engineering. Every rolling/lag feature here is unit-tested
in tests/test_features.py to confirm it only uses information available at
or before the prediction date (no .shift(1) omissions)."""
