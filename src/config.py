"""
Centralized Configuration Loader with Pydantic Validation.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional
import yaml
from pydantic import BaseModel, Field, field_validator


class DataConfig(BaseModel):
    raw_dir: str = "data/raw"
    processed_dir: str = "data/processed"
    lending_club_path: str = "data/raw/lending_club_loans.csv"
    german_credit_path: str = "data/raw/south_german_credit.csv"
    sample_size: int = Field(default=200000, gt=0)
    use_full_dataset: bool = False
    target_column: str = "loan_status"


class ValidationConfig(BaseModel):
    n_splits: int = Field(default=5, ge=2)
    shuffle: bool = True
    use_oot: bool = True
    train_vintage_start: str = "2012-01-01"
    train_vintage_end: str = "2016-12-31"
    test_vintage_start: str = "2017-01-01"
    test_vintage_end: str = "2018-12-31"


class BaseProjectConfig(BaseModel):
    project_name: str = "credex"
    random_seed: int = 42
    data: DataConfig = Field(default_factory=DataConfig)
    validation: ValidationConfig = Field(default_factory=ValidationConfig)


class FeatureSetBaselineConfig(BaseModel):
    numerical: List[str]
    categorical: List[str]


class FeatureSetEnhancedConfig(BaseModel):
    engineered_ratios: List[str]
    include_pricing_features: bool = False
    pricing_features: List[str]


class MutabilityConfig(BaseModel):
    mutable: List[str]
    immutable: List[str]


class FeatureConfig(BaseModel):
    feature_set_baseline: FeatureSetBaselineConfig
    feature_set_enhanced: FeatureSetEnhancedConfig
    mutability: MutabilityConfig


class UnitPenaltyConfig(BaseModel):
    cost_fp: float = Field(default=1.0, gt=0)
    cost_fn: float = Field(default=5.0, gt=0)
    ratio: float = Field(default=5.0, gt=0)

    @field_validator("ratio")
    @classmethod
    def validate_ratio(cls, v: float, values: Any) -> float:
        if v <= 0:
            raise ValueError("Cost penalty ratio must be strictly positive")
        return v


class PortfolioLossConfig(BaseModel):
    lgd_baseline: float = Field(default=0.65, ge=0.0, le=1.0)
    lgd_scenarios: List[float] = Field(default=[0.60, 0.70, 0.75])
    net_interest_margin: float = Field(default=0.035, gt=0.0)
    recovery_rate: float = Field(default=0.35, ge=0.0, le=1.0)


class ThresholdSweepConfig(BaseModel):
    min_tau: float = Field(default=0.01, gt=0.0)
    max_tau: float = Field(default=0.99, lt=1.0)
    step_size: float = Field(default=0.005, gt=0.0)


class CostMatrixConfig(BaseModel):
    unit_penalties: Dict[str, UnitPenaltyConfig]
    portfolio_loss: PortfolioLossConfig
    threshold_sweep: ThresholdSweepConfig


class AppConfig(BaseModel):
    base: BaseProjectConfig
    features: FeatureConfig
    models: Dict[str, Any]
    costs: CostMatrixConfig


def load_config(config_dir: str = "configs") -> AppConfig:
    """Load and validate all configuration files from the config directory."""
    dir_path = Path(config_dir)

    with open(dir_path / "base_config.yaml", "r") as f:
        base_data = yaml.safe_load(f)

    with open(dir_path / "features_config.yaml", "r") as f:
        features_data = yaml.safe_load(f)

    with open(dir_path / "models_config.yaml", "r") as f:
        models_data = yaml.safe_load(f)

    with open(dir_path / "cost_matrix.yaml", "r") as f:
        costs_data = yaml.safe_load(f)

    return AppConfig(
        base=BaseProjectConfig(**base_data),
        features=FeatureConfig(**features_data),
        models=models_data,
        costs=CostMatrixConfig(**costs_data),
    )
