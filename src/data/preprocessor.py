"""
Out-of-Fold Preprocessing & Financial Winsorization Pipeline.
Performs leakage-free median/mode imputation, 99.5th percentile winsorization,
standard scaling, and categorical encoding.
"""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder, StandardScaler


DEFAULT_WINSORIZE_FEATURES = [
    "annual_income",
    "revol_bal",
    "debt_to_income",
    "payment_to_income",
    "loan_to_income",
    "liquidity_buffer",
    "loan_amount",
    "installment",
]


class CreditDataPreprocessor(BaseEstimator, TransformerMixin):
    """
    Leakage-free credit data preprocessor.
    
    Fitted strictly on training folds and transformed onto test folds:
      1. Imputation: Numerical features -> median; Categorical features -> mode.
      2. Winsorization: Financial features capped at the training 99.5th percentile.
      3. Encoding: One-Hot Encoding for categorical features with handle_unknown='ignore'.
      4. Scaling: Standard scaling (mean=0, std=1) on numerical features.
    """

    def __init__(
        self,
        numerical_features: Optional[List[str]] = None,
        categorical_features: Optional[List[str]] = None,
        winsorize_features: Optional[List[str]] = None,
        winsorize_quantile: float = 0.995,
        scale_numerical: bool = True,
        return_df: bool = True,
    ):
        self.numerical_features = numerical_features
        self.categorical_features = categorical_features
        self.winsorize_features = winsorize_features
        self.winsorize_quantile = winsorize_quantile
        self.scale_numerical = scale_numerical
        self.return_df = return_df

        # Fitted state
        self.numerical_cols_: List[str] = []
        self.categorical_cols_: List[str] = []
        self.active_winsorize_cols_: List[str] = []
        self.numerical_medians_: Dict[str, float] = {}
        self.categorical_modes_: Dict[str, str] = {}
        self.winsorize_caps_: Dict[str, float] = {}
        self.scaler_: Optional[StandardScaler] = None
        self.encoder_: Optional[OneHotEncoder] = None
        self.feature_names_out_: List[str] = []

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> "CreditDataPreprocessor":
        """
        Fit preprocessing statistics strictly on training data X.
        
        Parameters:
        -----------
        X : pd.DataFrame
            Training feature matrix.
        y : Optional[pd.Series]
            Target vector (unused, maintained for scikit-learn API compatibility).
        """
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        # Infer numerical and categorical columns if not explicitly supplied
        if self.numerical_features is not None:
            self.numerical_cols_ = [c for c in self.numerical_features if c in X.columns]
        else:
            self.numerical_cols_ = X.select_dtypes(include=[np.number]).columns.tolist()

        if self.categorical_features is not None:
            self.categorical_cols_ = [c for c in self.categorical_features if c in X.columns]
        else:
            self.categorical_cols_ = X.select_dtypes(exclude=[np.number]).columns.tolist()

        # Winsorization target columns
        if self.winsorize_features is not None:
            self.active_winsorize_cols_ = [
                c for c in self.winsorize_features if c in self.numerical_cols_
            ]
        else:
            self.active_winsorize_cols_ = [
                c for c in DEFAULT_WINSORIZE_FEATURES if c in self.numerical_cols_
            ]

        # 1. Compute training median for numerical features
        self.numerical_medians_ = {}
        for col in self.numerical_cols_:
            median_val = X[col].median()
            self.numerical_medians_[col] = 0.0 if pd.isna(median_val) else float(median_val)

        # 2. Compute training mode for categorical features
        self.categorical_modes_ = {}
        for col in self.categorical_cols_:
            modes = X[col].dropna().mode()
            self.categorical_modes_[col] = str(modes.iloc[0]) if not modes.empty else "missing"

        # 3. Compute exact 99.5th percentile caps on imputed training data
        self.winsorize_caps_ = {}
        for col in self.active_winsorize_cols_:
            col_imputed = X[col].fillna(self.numerical_medians_[col])
            cap_val = col_imputed.quantile(self.winsorize_quantile)
            self.winsorize_caps_[col] = float(cap_val)

        # 4. Prepare training numerical matrix for scaler
        X_num = X[self.numerical_cols_].copy()
        for col in self.numerical_cols_:
            X_num[col] = X_num[col].fillna(self.numerical_medians_[col])
            if col in self.winsorize_caps_:
                X_num[col] = X_num[col].clip(upper=self.winsorize_caps_[col])

        if self.scale_numerical and self.numerical_cols_:
            self.scaler_ = StandardScaler()
            self.scaler_.fit(X_num.values)
        else:
            self.scaler_ = None

        # 5. Prepare training categorical matrix for encoder
        if self.categorical_cols_:
            X_cat = X[self.categorical_cols_].copy()
            for col in self.categorical_cols_:
                X_cat[col] = X_cat[col].fillna(self.categorical_modes_[col]).astype(str)
            self.encoder_ = OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False,
            )
            self.encoder_.fit(X_cat.values)
        else:
            self.encoder_ = None

        # Build feature names out
        feature_names = list(self.numerical_cols_)
        if self.encoder_ is not None:
            encoded_names = self.encoder_.get_feature_names_out(self.categorical_cols_)
            feature_names.extend(encoded_names.tolist())
        self.feature_names_out_ = feature_names

        return self

    def transform(self, X: pd.DataFrame) -> Union[pd.DataFrame, np.ndarray]:
        """
        Transform data X using strictly parameters fitted on training data.
        
        Parameters:
        -----------
        X : pd.DataFrame
            Input feature matrix to transform.
            
        Returns:
        --------
        Union[pd.DataFrame, np.ndarray]
            Processed predictor matrix with zero missing values.
        """
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        # 1. Numerical processing: Impute -> Winsorize -> Scale
        if self.numerical_cols_:
            X_num = X[self.numerical_cols_].copy()
            for col in self.numerical_cols_:
                # Median imputation from training
                med = self.numerical_medians_.get(col, 0.0)
                X_num[col] = X_num[col].fillna(med)

                # Winsorization using training 99.5th percentile cap
                if col in self.winsorize_caps_:
                    cap = self.winsorize_caps_[col]
                    X_num[col] = X_num[col].clip(upper=cap)

            if self.scaler_ is not None:
                scaled_num_arr = self.scaler_.transform(X_num.values)
            else:
                scaled_num_arr = X_num.values
        else:
            scaled_num_arr = np.empty((len(X), 0))

        # 2. Categorical processing: Impute -> One-Hot Encode
        if self.categorical_cols_ and self.encoder_ is not None:
            X_cat = X[self.categorical_cols_].copy()
            for col in self.categorical_cols_:
                mode_val = self.categorical_modes_.get(col, "missing")
                X_cat[col] = X_cat[col].fillna(mode_val).astype(str)
            encoded_cat_arr = self.encoder_.transform(X_cat.values)
        else:
            encoded_cat_arr = np.empty((len(X), 0))

        # Concatenate numerical and categorical arrays
        processed_arr = np.hstack([scaled_num_arr, encoded_cat_arr])

        # Verify zero NaNs remaining
        assert not np.isnan(processed_arr).any(), "Preprocessor output contains NaN values!"

        if self.return_df:
            return pd.DataFrame(
                processed_arr,
                columns=self.feature_names_out_,
                index=X.index,
            )
        return processed_arr

    def get_feature_names_out(self, input_features: Optional[List[str]] = None) -> List[str]:
        """Return output feature names."""
        return list(self.feature_names_out_)


def fit_transform_out_of_fold(
    splits: List[Tuple[np.ndarray, np.ndarray]],
    X: pd.DataFrame,
    y: Optional[pd.Series] = None,
    preprocessor_kwargs: Optional[Dict] = None,
) -> Tuple[List[Union[pd.DataFrame, np.ndarray]], List[Union[pd.DataFrame, np.ndarray]]]:
    """
    Perform out-of-fold preprocessing strictly fitting preprocessor on train folds.
    
    Parameters:
    -----------
    splits : List[Tuple[np.ndarray, np.ndarray]]
        List of (train_idx, val_idx) tuples.
    X : pd.DataFrame
        Full dataset feature matrix.
    y : Optional[pd.Series]
        Target labels.
    preprocessor_kwargs : Optional[Dict]
        Arguments to pass to CreditDataPreprocessor.
        
    Returns:
    --------
    Tuple[List[X_train_transformed], List[X_val_transformed]]
        Out-of-fold transformed splits.
    """
    kwargs = preprocessor_kwargs or {}
    train_transformed_list = []
    val_transformed_list = []

    for train_idx, val_idx in splits:
        preprocessor = CreditDataPreprocessor(**kwargs)
        X_train_fold = X.iloc[train_idx]
        X_val_fold = X.iloc[val_idx]

        # Fit strictly on train fold
        preprocessor.fit(X_train_fold)

        # Transform both
        X_train_trans = preprocessor.transform(X_train_fold)
        X_val_trans = preprocessor.transform(X_val_fold)

        train_transformed_list.append(X_train_trans)
        val_transformed_list.append(X_val_trans)

    return train_transformed_list, val_transformed_list
