from __future__ import annotations

import json

import numpy as np
import pandas as pd

try:
    from modeling_utils import (
        DEPLOYED_FEATURE_COLUMNS,
        PERIODS,
        PREDICTION_OUTPUT_DIR,
        PREDICTOR_METRICS,
        SCENARIOS,
        assign_county_regions,
        build_long_modeling_table,
        load_county_boundaries,
        risk_level_from_percentile,
    )
    from train_models import export_feature_importance, make_model
except ModuleNotFoundError:
    from src.modeling_utils import (
        DEPLOYED_FEATURE_COLUMNS,
        PERIODS,
        PREDICTION_OUTPUT_DIR,
        PREDICTOR_METRICS,
        SCENARIOS,
        assign_county_regions,
        build_long_modeling_table,
        load_county_boundaries,
        risk_level_from_percentile,
    )
    from src.train_models import export_feature_importance, make_model


METRIC_LABELS = {
    "tx90p": "暖晝天數",
    "prcptot": "雨日總降雨量",
    "sdii": "雨日降雨強度",
    "cdd": "連續不降雨日",
    "cwd": "連續降雨日",
    "hwdi": "高溫持續指數",
}


def prepare_predictions() -> tuple[pd.DataFrame, pd.DataFrame]:
    long_table = build_long_modeling_table()
    feature_columns = [column for column in DEPLOYED_FEATURE_COLUMNS if column in long_table.columns]

    model = make_model("XGBoost")
    model.fit(long_table[feature_columns], long_table["rx1day_change"])
    importance_df = export_feature_importance(model, feature_columns)

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
    return predictions, importance_df


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


def round_coordinates(value, digits: int = 5):
    if isinstance(value, (float, int)):
        return round(float(value), digits)
    return [round_coordinates(item, digits) for item in value]


def export_county_boundaries(predictions: pd.DataFrame) -> None:
    counties = load_county_boundaries()
    if counties is None or counties.empty:
        return

    try:
        counties = counties.to_crs("EPSG:4326")
    except Exception:
        pass

    counties = counties.sort_values("region_name").copy()
    try:
        from shapely.geometry import box

        margin = 0.25
        clip_box = box(
            float(predictions["LON"].min()) - margin,
            float(predictions["LAT"].min()) - margin,
            float(predictions["LON"].max()) + margin,
            float(predictions["LAT"].max()) + margin,
        )
        counties = counties[counties.intersects(clip_box)].copy()
        counties["geometry"] = counties.geometry.intersection(clip_box)
        counties = counties[~counties.geometry.is_empty].copy()
    except Exception:
        pass

    counties["geometry"] = counties.geometry.simplify(0.006, preserve_topology=True)

    features = []
    for row in counties.itertuples(index=False):
        geometry = row.geometry.__geo_interface__
        features.append(
            {
                "type": "Feature",
                "properties": {
                    "region_id": row.region_id,
                    "region_name": row.region_name,
                },
                "geometry": {
                    "type": geometry["type"],
                    "coordinates": round_coordinates(geometry["coordinates"]),
                },
            }
        )

    payload = {"type": "FeatureCollection", "features": features}
    (PREDICTION_OUTPUT_DIR / "county_boundaries.json").write_text(
        json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
        encoding="utf-8",
    )


def metric_from_feature(feature: str) -> str | None:
    for metric in PREDICTOR_METRICS:
        if feature == f"{metric}_change" or feature == f"{metric}_change_lag":
            return metric
    return None


def build_feature_importance_insight(feature_importance: pd.DataFrame) -> list[dict]:
    insight_df = feature_importance.copy()
    insight_df["metric"] = insight_df["Feature"].map(metric_from_feature)
    insight_df = insight_df.dropna(subset=["metric"])
    insight_df = (
        insight_df.groupby("metric", as_index=False)
        .agg(importance=("Importance", "sum"))
        .sort_values("importance", ascending=False)
    )
    total = insight_df["importance"].sum()
    if total > 0:
        insight_df["importance"] = insight_df["importance"] / total
    insight_df["rank"] = range(1, len(insight_df) + 1)
    insight_df["label"] = insight_df["metric"].map(METRIC_LABELS)
    return insight_df[["rank", "metric", "label", "importance"]].round(4).to_dict(orient="records")


def build_region_metric_summary(predictions: pd.DataFrame) -> pd.DataFrame:
    return (
        predictions.groupby(["region_id", "region_name", "scenario", "period"], as_index=False)
        .agg(
            mean_change_pred=("rx1day_change_pred", "mean"),
            mean_future_pred=("rx1day_future_pred", "mean"),
            max_change_pred=("rx1day_change_pred", "max"),
            high_risk_share=("risk_level", lambda values: float(np.mean(values == "高"))),
            mean_tx90p_change=("tx90p_change", "mean"),
            mean_hwdi_change=("hwdi_change", "mean"),
            mean_prcptot_change=("prcptot_change", "mean"),
            mean_sdii_change=("sdii_change", "mean"),
            n_grid_cells=("rx1day_change_pred", "size"),
        )
        .round(6)
    )


def add_percentile_rank(group: pd.DataFrame, source: str, target: str) -> pd.DataFrame:
    group[target] = group[source].rank(pct=True) * 100
    return group


def build_compound_hotspots(region_metrics: pd.DataFrame) -> list[dict]:
    frames = []
    for (_, _), group in region_metrics.groupby(["scenario", "period"], sort=False):
        group = group.copy()
        group = add_percentile_rank(group, "mean_change_pred", "rain_percentile")
        group = add_percentile_rank(group, "mean_tx90p_change", "tx90p_percentile")
        group = add_percentile_rank(group, "mean_hwdi_change", "hwdi_percentile")
        group["temp_percentile"] = (group["tx90p_percentile"] + group["hwdi_percentile"]) / 2
        group["compound_score"] = (group["rain_percentile"] + group["temp_percentile"]) / 2
        group["compound_rank"] = group["compound_score"].rank(method="first", ascending=False).astype(int)
        frames.append(group)

    hotspot_df = pd.concat(frames, ignore_index=True)
    columns = [
        "scenario",
        "period",
        "region_id",
        "region_name",
        "compound_rank",
        "compound_score",
        "rain_percentile",
        "temp_percentile",
        "mean_change_pred",
        "mean_tx90p_change",
        "mean_hwdi_change",
        "high_risk_share",
    ]
    return hotspot_df[columns].round(4).to_dict(orient="records")


def build_scenario_stability(region_metrics: pd.DataFrame) -> dict:
    rows = []
    summaries = []
    for period in PERIODS:
        period_df = region_metrics[region_metrics["period"] == period].copy()
        ranked_frames = []
        for scenario in SCENARIOS:
            scenario_df = period_df[period_df["scenario"] == scenario].copy()
            scenario_df["risk_rank"] = (
                scenario_df["high_risk_share"]
                .rank(method="first", ascending=False)
                .astype(int)
            )
            ranked_frames.append(scenario_df[["region_id", "region_name", "scenario", "risk_rank", "high_risk_share"]])

        ranked = pd.concat(ranked_frames, ignore_index=True)
        rank_pivot = ranked.pivot_table(index=["region_id", "region_name"], columns="scenario", values="risk_rank")
        share_pivot = ranked.pivot_table(index=["region_id", "region_name"], columns="scenario", values="high_risk_share")
        n_regions = max(len(rank_pivot), 1)
        corr_matrix = rank_pivot.corr(method="spearman")
        pairwise = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)).stack()
        mean_corr = float(pairwise.mean()) if not pairwise.empty else np.nan

        for index, rank_values in rank_pivot.iterrows():
            region_id, region_name = index
            valid_ranks = rank_values.dropna()
            rank_range = float(valid_ranks.max() - valid_ranks.min()) if not valid_ranks.empty else np.nan
            stability_score = 1 - (rank_range / max(n_regions - 1, 1)) if np.isfinite(rank_range) else np.nan
            share_values = share_pivot.loc[index].dropna()
            rows.append(
                {
                    "period": period,
                    "region_id": region_id,
                    "region_name": region_name,
                    "mean_rank": float(valid_ranks.mean()) if not valid_ranks.empty else np.nan,
                    "best_rank": float(valid_ranks.min()) if not valid_ranks.empty else np.nan,
                    "worst_rank": float(valid_ranks.max()) if not valid_ranks.empty else np.nan,
                    "rank_range": rank_range,
                    "stability_score": stability_score,
                    "mean_high_risk_share": float(share_values.mean()) if not share_values.empty else np.nan,
                    "scenario_ranks": {
                        scenario: int(rank_values[scenario])
                        for scenario in SCENARIOS
                        if scenario in rank_values and pd.notna(rank_values[scenario])
                    },
                }
            )

        summaries.append(
            {
                "period": period,
                "mean_pairwise_spearman": mean_corr,
                "interpretation": (
                    "排序相當穩定"
                    if np.isfinite(mean_corr) and mean_corr >= 0.7
                    else "排序中度穩定"
                    if np.isfinite(mean_corr) and mean_corr >= 0.45
                    else "排序差異較大"
                ),
            }
        )

    rows = sorted(
        rows,
        key=lambda row: (
            row["period"],
            -(row["mean_high_risk_share"] if np.isfinite(row["mean_high_risk_share"]) else -1),
            row["mean_rank"] if np.isfinite(row["mean_rank"]) else 999,
        ),
    )
    return {
        "summary": pd.DataFrame(summaries).round(4).to_dict(orient="records"),
        "regions": pd.DataFrame(rows).round(4).to_dict(orient="records"),
    }


def build_multi_scenario_high_risk(region_metrics: pd.DataFrame) -> list[dict]:
    rows = []
    for period in PERIODS:
        period_df = region_metrics[region_metrics["period"] == period].copy()
        n_regions = int(period_df["region_id"].nunique())
        top_threshold = max(1, int(np.ceil(n_regions * 0.25)))
        ranked_frames = []
        for scenario in SCENARIOS:
            scenario_df = period_df[period_df["scenario"] == scenario].copy()
            scenario_df["risk_rank"] = (
                scenario_df["high_risk_share"]
                .rank(method="first", ascending=False)
                .astype(int)
            )
            ranked_frames.append(scenario_df)

        ranked = pd.concat(ranked_frames, ignore_index=True)
        ranked["is_top_risk"] = ranked["risk_rank"] <= top_threshold
        for (region_id, region_name), group in ranked.groupby(["region_id", "region_name"], sort=False):
            top_group = group[group["is_top_risk"]]
            rows.append(
                {
                    "period": period,
                    "region_id": region_id,
                    "region_name": region_name,
                    "high_scenario_count": int(top_group["scenario"].nunique()),
                    "high_scenarios": sorted(top_group["scenario"].unique().tolist(), key=SCENARIOS.index),
                    "mean_rank": float(group["risk_rank"].mean()),
                    "mean_change_pred": float(group["mean_change_pred"].mean()),
                    "mean_high_risk_share": float(group["high_risk_share"].mean()),
                    "top_threshold": top_threshold,
                }
            )

    rows = sorted(
        rows,
        key=lambda row: (
            row["period"],
            -row["high_scenario_count"],
            row["mean_rank"],
        ),
    )
    return pd.DataFrame(rows).round(4).to_dict(orient="records")


def export_risk_insights(predictions: pd.DataFrame, feature_importance: pd.DataFrame) -> None:
    region_metrics = build_region_metric_summary(predictions)
    payload = {
        "notes": {
            "target": "Rx1day change: future Rx1day minus OBS_1995-2014 baseline",
            "feature_importance_scope": "XGBoost feature importance aggregated from each climate indicator and its neighboring-grid lag feature.",
            "compound_hotspot_score": "Average of county rain-increase percentile and temperature-change percentile.",
            "multi_scenario_rule": "County is counted as high risk in a scenario when it ranks in the top 25% by high-risk grid-cell share.",
        },
        "feature_importance": build_feature_importance_insight(feature_importance),
        "compound_hotspots": build_compound_hotspots(region_metrics),
        "scenario_stability": build_scenario_stability(region_metrics),
        "multi_scenario_high_risk": build_multi_scenario_high_risk(region_metrics),
    }
    (PREDICTION_OUTPUT_DIR / "risk_insights.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def main() -> None:
    PREDICTION_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    predictions, feature_importance = prepare_predictions()
    export_grid_predictions(predictions)
    export_region_predictions(predictions)
    export_county_boundaries(predictions)
    export_risk_insights(predictions, feature_importance)
    print(f"Exported {len(predictions):,} grid prediction rows.")
    print(
        "Created output/predictions/grid_predictions.json, "
        "region_predictions.json, county_boundaries.json, and risk_insights.json"
    )


if __name__ == "__main__":
    main()
