from __future__ import annotations

import json

import numpy as np
import pandas as pd

try:
    from modeling_utils import (
    DEPLOYED_FEATURE_COLUMNS,
    PREDICTION_OUTPUT_DIR,
    assign_county_regions,
    build_long_modeling_table,
    risk_level_from_percentile,
)
    from train_models import make_model
except ModuleNotFoundError:
    from src.modeling_utils import (
        DEPLOYED_FEATURE_COLUMNS,
        PREDICTION_OUTPUT_DIR,
        assign_county_regions,
        build_long_modeling_table,
        risk_level_from_percentile,
    )
    from src.train_models import make_model


def prepare_predictions() -> pd.DataFrame:
    long_table = build_long_modeling_table()
    feature_columns = [column for column in DEPLOYED_FEATURE_COLUMNS if column in long_table.columns]

    model = make_model("XGBoost")
    model.fit(long_table[feature_columns], long_table["rx1day_change"])

    predictions = long_table.copy()
    predictions["rx1day_change_pred"] = model.predict(long_table[feature_columns])
    predictions["rx1day_future_pred"] = predictions["rx1day_base"] + predictions["rx1day_change_pred"]

    percentile_frames = []
    for (_, _), group in predictions.groupby(["scenario", "period"], sort=False):
        group = group.copy()
        group["risk_percentile"] = group["rx1day_change_pred"].rank(pct=True) * 100
        group["risk_level"] = group["risk_percentile"].apply(risk_level_from_percentile)
        percentile_frames.append(group)

    predictions = pd.concat(percentile_frames, ignore_index=True)
    region_lookup = assign_county_regions(predictions[["LON", "LAT"]])
    predictions = predictions.merge(region_lookup, on=["LON", "LAT"], how="left")
    return predictions


def export_grid_predictions(predictions: pd.DataFrame) -> None:
    grid_df = predictions.rename(
        columns={
            "LON": "lon",
            "LAT": "lat",
        }
    )
    columns = [
        "lon",
        "lat",
        "scenario",
        "period",
        "region_id",
        "region_name",
        "rx1day_base",
        "tx90p_change",
        "prcptot_change",
        "sdii_change",
        "cdd_change",
        "cwd_change",
        "hwdi_change",
        "rx1day_change_pred",
        "rx1day_future_pred",
        "risk_level",
        "risk_percentile",
    ]
    records = grid_df[columns].round(4).to_dict(orient="records")
    (PREDICTION_OUTPUT_DIR / "grid_predictions.json").write_text(
        json.dumps(records, ensure_ascii=False, separators=(",", ":")),
        encoding="utf-8",
    )


def export_region_predictions(predictions: pd.DataFrame) -> None:
    region_df = (
        predictions.groupby(["region_id", "region_name", "scenario", "period"], as_index=False)
        .agg(
            mean_change_pred=("rx1day_change_pred", "mean"),
            mean_future_pred=("rx1day_future_pred", "mean"),
            max_change_pred=("rx1day_change_pred", "max"),
            high_risk_share=("risk_level", lambda values: float(np.mean(values == "高"))),
            n_grid_cells=("rx1day_change_pred", "size"),
        )
        .round(4)
    )
    records = region_df.to_dict(orient="records")
    (PREDICTION_OUTPUT_DIR / "region_predictions.json").write_text(
        json.dumps(records, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def main() -> None:
    PREDICTION_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    predictions = prepare_predictions()
    export_grid_predictions(predictions)
    export_region_predictions(predictions)
    print(f"Exported {len(predictions):,} grid prediction rows.")
    print("Created output/predictions/grid_predictions.json and region_predictions.json")


if __name__ == "__main__":
    main()
