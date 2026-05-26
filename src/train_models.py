from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline

try:
    from xgboost import XGBRegressor
except ImportError:  # pragma: no cover
    XGBRegressor = None

try:
    from modeling_utils import (
        DEPLOYED_FEATURE_COLUMNS,
        MODEL_OUTPUT_DIR,
        OUTPUT_DIR,
        PROJECT_ROOT,
        build_long_modeling_table,
        build_modeling_table,
    )
except ModuleNotFoundError:
    from src.modeling_utils import (
        DEPLOYED_FEATURE_COLUMNS,
        MODEL_OUTPUT_DIR,
        OUTPUT_DIR,
        PROJECT_ROOT,
        build_long_modeling_table,
        build_modeling_table,
    )


ASSIGNMENT_DIR = PROJECT_ROOT.parent / "作業"
DEPLOYED_MODEL_NAME = "XGBoost + PREDICTOR_SPATIAL_LAG"
TRAINING_TARGET = "Rx1day change = future Rx1day - OBS_1995-2014"
CV_STRATEGY = "4-fold spatial block CV by longitude quartile"


def rmse_score(y_true, y_pred) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def make_model(algorithm: str):
    if algorithm == "OLS":
        return Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("model", LinearRegression()),
            ]
        )
    if algorithm == "Random Forest":
        return Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    RandomForestRegressor(
                        n_estimators=260,
                        min_samples_leaf=3,
                        random_state=42,
                        n_jobs=-1,
                    ),
                ),
            ]
        )
    if algorithm == "XGBoost" and XGBRegressor is not None:
        return Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    XGBRegressor(
                        n_estimators=360,
                        max_depth=4,
                        learning_rate=0.045,
                        subsample=0.88,
                        colsample_bytree=0.88,
                        objective="reg:squarederror",
                        random_state=42,
                        n_jobs=-1,
                    ),
                ),
            ]
        )
    return make_model("Random Forest")


def spatial_groups(long_table: pd.DataFrame, n_groups: int = 4) -> pd.Series:
    lon_rank = long_table["LON"].rank(method="first")
    return pd.qcut(lon_rank, q=n_groups, labels=False, duplicates="drop").astype(int)


def cross_validate_models(long_table: pd.DataFrame, feature_columns: list[str]) -> pd.DataFrame:
    x = long_table[feature_columns]
    y = long_table["rx1day_change"]
    groups = spatial_groups(long_table)
    group_kfold = GroupKFold(n_splits=min(4, groups.nunique()))

    rows = []
    for algorithm in ["OLS", "Random Forest", "XGBoost"]:
        fold_metrics = []
        for fold_id, (train_idx, test_idx) in enumerate(group_kfold.split(x, y, groups), start=1):
            model = make_model(algorithm)
            model.fit(x.iloc[train_idx], y.iloc[train_idx])
            preds = model.predict(x.iloc[test_idx])
            fold_metrics.append(
                {
                    "fold": fold_id,
                    "r2": r2_score(y.iloc[test_idx], preds),
                    "rmse": rmse_score(y.iloc[test_idx], preds),
                    "mae": mean_absolute_error(y.iloc[test_idx], preds),
                }
            )

        fold_df = pd.DataFrame(fold_metrics)
        rows.append(
            {
                "Model": algorithm,
                "FeatureSet": "PREDICTOR_SPATIAL_LAG" if "tx90p_change_lag" in feature_columns else "NO_SPATIAL",
                "Train_R2": np.nan,
                "Test_R2": fold_df["r2"].mean(),
                "RMSE": fold_df["rmse"].mean(),
                "MAE": fold_df["mae"].mean(),
                "Uses_spatial_lag": "tx90p_change_lag" in feature_columns,
                "Possible_leakage": False,
                "Note": CV_STRATEGY,
            }
        )
    return pd.DataFrame(rows)


def export_feature_importance(model, feature_columns: list[str]) -> pd.DataFrame:
    fitted_model = model.named_steps["model"]
    if hasattr(fitted_model, "feature_importances_"):
        importance = fitted_model.feature_importances_
    elif hasattr(fitted_model, "coef_"):
        importance = np.abs(fitted_model.coef_)
    else:
        importance = np.zeros(len(feature_columns))

    importance_df = pd.DataFrame(
        {"Feature": feature_columns, "Importance": importance.astype(float)}
    ).sort_values("Importance", ascending=False)
    total = importance_df["Importance"].sum()
    if total > 0:
        importance_df["Importance"] = importance_df["Importance"] / total
    return importance_df


def copy_existing_model_reports() -> None:
    report_map = {
        ASSIGNMENT_DIR / "AR6_spatial_block_cv_results.csv": MODEL_OUTPUT_DIR / "spatial_block_cv_results.csv",
        ASSIGNMENT_DIR / "AR6_clean_ML_feature_importance.csv": MODEL_OUTPUT_DIR / "assignment_feature_importance.csv",
        ASSIGNMENT_DIR / "AR6_GWR_MGWR_performance.csv": MODEL_OUTPUT_DIR / "gwr_mgwr_performance.csv",
    }
    for source, target in report_map.items():
        if source.exists():
            shutil.copyfile(source, target)


def write_metadata(feature_columns: list[str], metrics: dict[str, float]) -> None:
    metadata = {
        "deployed_model_name": DEPLOYED_MODEL_NAME,
        "feature_columns": feature_columns,
        "training_target": TRAINING_TARGET,
        "cv_strategy": CV_STRATEGY,
        "evaluation_metrics": metrics,
        "leakage_policy": "TARGET_SPATIAL_LAG_RISKY is display-only and is not used for deployed predictions.",
    }
    (MODEL_OUTPUT_DIR / "model_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def main() -> None:
    MODEL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    wide_table = build_modeling_table()
    long_table = build_long_modeling_table(wide_table)
    feature_columns = [column for column in DEPLOYED_FEATURE_COLUMNS if column in long_table.columns]

    wide_table.to_csv(OUTPUT_DIR / "modeling_table.csv", index=False, encoding="utf-8")
    long_table.to_csv(OUTPUT_DIR / "modeling_table_long.csv", index=False, encoding="utf-8")

    comparison_df = cross_validate_models(long_table, feature_columns)
    assignment_cv = ASSIGNMENT_DIR / "AR6_spatial_block_cv_results.csv"
    if assignment_cv.exists():
        comparison_df = pd.read_csv(assignment_cv)
    comparison_df.to_csv(MODEL_OUTPUT_DIR / "model_comparison.csv", index=False, encoding="utf-8")

    deployed_model = make_model("XGBoost")
    deployed_model.fit(long_table[feature_columns], long_table["rx1day_change"])
    in_sample_pred = deployed_model.predict(long_table[feature_columns])
    deployed_metrics = {
        "in_sample_r2": float(r2_score(long_table["rx1day_change"], in_sample_pred)),
        "in_sample_rmse": rmse_score(long_table["rx1day_change"], in_sample_pred),
        "in_sample_mae": float(mean_absolute_error(long_table["rx1day_change"], in_sample_pred)),
    }

    export_feature_importance(deployed_model, feature_columns).to_csv(
        MODEL_OUTPUT_DIR / "feature_importance.csv",
        index=False,
        encoding="utf-8",
    )
    copy_existing_model_reports()
    write_metadata(feature_columns, deployed_metrics)

    print(f"Created modeling outputs in {OUTPUT_DIR}")
    print(f"Rows: wide={len(wide_table):,}, long={len(long_table):,}")
    print(f"Deployed model: {DEPLOYED_MODEL_NAME}")


if __name__ == "__main__":
    main()
