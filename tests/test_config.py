"""
Unit and Property Tests for Configuration Loading and Validation.
"""

import pytest
from hypothesis import given
import hypothesis.strategies as st
from pydantic import ValidationError

from src.config import UnitPenaltyConfig, load_config


def test_load_config_defaults():
    """Verify loading default configs succeeds and validates all schemas."""
    config = load_config("configs")
    assert config.base.project_name == "credex"
    assert config.base.random_seed == 42
    assert config.base.data.sample_size == 200000
    assert "benchmark_5_to_1" in config.costs.unit_penalties
    assert config.costs.unit_penalties["benchmark_5_to_1"].ratio == 5.0
    assert "logistic_regression" in config.models["baselines"]
    assert "lightgbm" in config.models["baselines"]


def test_unit_penalty_invalid():
    """Verify non-positive ratios raise Pydantic validation error."""
    with pytest.raises(ValidationError):
        UnitPenaltyConfig(cost_fp=1.0, cost_fn=5.0, ratio=-1.0)

    with pytest.raises(ValidationError):
        UnitPenaltyConfig(cost_fp=0.0, cost_fn=5.0, ratio=5.0)


@given(
    cost_fp=st.floats(min_value=0.1, max_value=100.0),
    cost_fn=st.floats(min_value=0.1, max_value=100.0),
)
def test_unit_penalty_property(cost_fp: float, cost_fn: float):
    """Property test: Any positive cost values construct a valid UnitPenaltyConfig."""
    cfg = UnitPenaltyConfig(cost_fp=cost_fp, cost_fn=cost_fn, ratio=cost_fn / cost_fp)
    assert cfg.cost_fp > 0
    assert cfg.cost_fn > 0
    assert cfg.ratio > 0
