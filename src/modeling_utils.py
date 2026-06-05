from __future__ import annotations

from io import StringIO
import os
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "output"
MODEL_OUTPUT_DIR = OUTPUT_DIR / "models"
PREDICTION_OUTPUT_DIR = OUTPUT_DIR / "predictions"
COUNTY_BOUNDARY_PATH = PROJECT_ROOT / "COUNTY_MOI_1140318.shp"
TOWN_BOUNDARY_PATH = (
    PROJECT_ROOT
    / "鄉鎮市區界線(TWD97經緯度)"
    / "TOWN_MOI_1120317.shp"
)

BASELINE_COL = "OBS_1995-2014"

SCENARIOS = ["SSP1-2.6", "SSP2-4.5", "SSP3-7.0", "SSP5-8.5"]
PERIODS = ["2021-2040", "2041-2060", "2081-2100"]

METRIC_FILES = {
    "rx1day": DATA_DIR / "rx1day.csv",
    "tx90p": DATA_DIR / "tx90p.csv",
    "prcptot": DATA_DIR / "prcptot.csv",
    "sdii": DATA_DIR / "sdii.csv",
    "cdd": DATA_DIR / "cdd.csv",
    "cwd": DATA_DIR / "cwd.csv",
    "hwdi": DATA_DIR / "hwdi.csv",
}

PREDICTOR_METRICS = ["tx90p", "prcptot", "sdii", "cdd", "cwd", "hwdi"]
DEPLOYED_FEATURE_COLUMNS = [
    "rx1day_base",
    "tx90p_change",
    "prcptot_change",
    "sdii_change",
    "cdd_change",
    "cwd_change",
    "hwdi_change",
    "lon_scaled",
    "lat_scaled",
    "tx90p_change_lag",
    "prcptot_change_lag",
    "sdii_change_lag",
    "cdd_change_lag",
    "cwd_change_lag",
    "hwdi_change_lag",
]

COUNTY_NAME_BY_GEOMETRY_INDEX = {
    0: "連江縣",
    1: "宜蘭縣",
    2: "彰化縣",
    3: "南投縣",
    4: "雲林縣",
    5: "屏東縣",
    6: "基隆市",
    7: "臺北市",
    8: "新北市",
    9: "臺南市",
    10: "桃園市",
    11: "嘉義市",
    12: "嘉義縣",
    13: "金門縣",
    14: "高雄市",
    15: "臺東縣",
    16: "花蓮縣",
    17: "澎湖縣",
    18: "新竹市",
    19: "臺中市",
    20: "苗栗縣",
    21: "新竹縣",
}

COUNTY_NAME_NORMALIZATION = {
    "台北市": "臺北市",
    "台中市": "臺中市",
    "台南市": "臺南市",
    "台東縣": "臺東縣",
}


def normalize_county_name(name: object) -> str:
    normalized = str(name).strip()
    return COUNTY_NAME_NORMALIZATION.get(normalized, normalized)


def scenario_period_key(scenario: str, period: str) -> str:
    return f"{scenario.replace('-', '').replace('.', '')}_{period.replace('-', '')}"


def read_tccip_csv(csv_path: Path) -> pd.DataFrame:
    normalized_lines = [line.rstrip().rstrip(",") for line in csv_path.read_text(encoding="utf-8").splitlines()]
    df = pd.read_csv(StringIO("\n".join(normalized_lines)), skipinitialspace=True)
    df.columns = [column.strip() for column in df.columns]
    return df


def clean_numeric(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    return numeric.mask(np.isclose(numeric, -99.9))


def load_metric(metric: str) -> pd.DataFrame:
    csv_path = METRIC_FILES[metric]
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing metric CSV: {csv_path}")

    df = read_tccip_csv(csv_path)
    required_columns = ["LON", "LAT", BASELINE_COL]
    required_columns.extend(f"{scenario}_{period}" for scenario in SCENARIOS for period in PERIODS)
    missing = [column for column in required_columns if column not in df.columns]
    if missing:
        raise ValueError(f"{csv_path.name} is missing columns: {', '.join(missing)}")

    for column in required_columns:
        df[column] = clean_numeric(df[column])

    keep_columns = ["LON", "LAT", BASELINE_COL, *[f"{scenario}_{period}" for scenario in SCENARIOS for period in PERIODS]]
    return df[keep_columns]


def build_modeling_table() -> pd.DataFrame:
    rx1day = load_metric("rx1day").rename(columns={BASELINE_COL: "rx1day_base"})
    table = rx1day[["LON", "LAT", "rx1day_base"]].copy()

    for scenario in SCENARIOS:
        for period in PERIODS:
            source_col = f"{scenario}_{period}"
            key = scenario_period_key(scenario, period)
            table[f"rx1day_change_{key}"] = rx1day[source_col] - rx1day["rx1day_base"]

    for metric in PREDICTOR_METRICS:
        frame = load_metric(metric)
        metric_table = frame[["LON", "LAT"]].copy()
        for scenario in SCENARIOS:
            for period in PERIODS:
                source_col = f"{scenario}_{period}"
                key = scenario_period_key(scenario, period)
                metric_table[f"{metric}_change_{key}"] = frame[source_col] - frame[BASELINE_COL]
        table = table.merge(metric_table, on=["LON", "LAT"], how="inner")

    value_columns = [column for column in table.columns if column not in {"LON", "LAT"}]
    return table.dropna(subset=["LON", "LAT", *value_columns]).reset_index(drop=True)


def build_long_modeling_table(wide_table: pd.DataFrame | None = None) -> pd.DataFrame:
    if wide_table is None:
        wide_table = build_modeling_table()

    records = []
    for scenario in SCENARIOS:
        for period in PERIODS:
            key = scenario_period_key(scenario, period)
            subset = wide_table[["LON", "LAT", "rx1day_base"]].copy()
            subset["scenario"] = scenario
            subset["period"] = period
            subset["rx1day_change"] = wide_table[f"rx1day_change_{key}"]
            for metric in PREDICTOR_METRICS:
                subset[f"{metric}_change"] = wide_table[f"{metric}_change_{key}"]
            records.append(subset)

    long_table = pd.concat(records, ignore_index=True)
    long_table = add_spatial_predictors(long_table)
    feature_columns = [column for column in DEPLOYED_FEATURE_COLUMNS if column in long_table.columns]
    return long_table.dropna(subset=["rx1day_change", *feature_columns]).reset_index(drop=True)


def add_spatial_predictors(df: pd.DataFrame, neighbor_count: int = 8) -> pd.DataFrame:
    result_frames = []
    for (_, _), group in df.groupby(["scenario", "period"], sort=False):
        group = group.copy().reset_index(drop=True)
        coords = group[["LON", "LAT"]].to_numpy(dtype=float)

        for metric in PREDICTOR_METRICS:
            values = group[f"{metric}_change"].to_numpy(dtype=float)
            lags = np.full(len(group), np.nan)
            for idx, point in enumerate(coords):
                distances = np.sum((coords - point) ** 2, axis=1)
                distances[idx] = np.inf
                neighbor_idx = np.argsort(distances)[:neighbor_count]
                valid_values = values[neighbor_idx][~np.isnan(values[neighbor_idx])]
                if valid_values.size:
                    lags[idx] = float(valid_values.mean())
            group[f"{metric}_change_lag"] = lags

        group["lon_scaled"] = (group["LON"] - group["LON"].mean()) / group["LON"].std(ddof=0)
        group["lat_scaled"] = (group["LAT"] - group["LAT"].mean()) / group["LAT"].std(ddof=0)
        result_frames.append(group)

    return pd.concat(result_frames, ignore_index=True)


def assign_region(lon: float, lat: float) -> tuple[str, str]:
    if lon < 120.0:
        return "islands", "離島"
    if lon >= 121.0 and lat < 24.2:
        return "east", "東部"
    if lat >= 24.2:
        return "north", "北部"
    if lat >= 23.45:
        return "central", "中部"
    return "south", "南部"


def load_county_boundaries():
    os.environ.setdefault("SHAPE_RESTORE_SHX", "YES")
    try:
        import geopandas as gpd
    except ImportError:
        return None

    if TOWN_BOUNDARY_PATH.exists():
        towns = gpd.read_file(TOWN_BOUNDARY_PATH, encoding="utf-8")
        if not towns.empty and "COUNTYNAME" in towns.columns:
            towns["region_name"] = towns["COUNTYNAME"].map(normalize_county_name)
            counties = towns.dissolve(by="region_name", as_index=False)
            counties["region_id"] = counties["region_name"]
            return counties[["region_id", "region_name", "geometry"]]

    if not COUNTY_BOUNDARY_PATH.exists():
        return None

    counties = gpd.read_file(COUNTY_BOUNDARY_PATH)
    if counties.empty:
        return None

    name_column = next(
        (
            column
            for column in counties.columns
            if column.upper() in {"COUNTYNAME", "COUNTY_NAME", "COUNTY", "NAME", "C_NAME"}
        ),
        None,
    )
    if name_column:
        counties["region_name"] = counties[name_column].map(normalize_county_name)
    else:
        counties["region_name"] = [
            COUNTY_NAME_BY_GEOMETRY_INDEX.get(index, f"縣市 {index + 1}")
            for index in range(len(counties))
        ]

    counties["region_id"] = counties["region_name"]
    return counties[["region_id", "region_name", "geometry"]]


def assign_county_regions(points_df: pd.DataFrame) -> pd.DataFrame:
    counties = load_county_boundaries()
    points = points_df[["LON", "LAT"]].drop_duplicates().copy()

    if counties is None:
        fallback = points.apply(lambda row: assign_region(row["LON"], row["LAT"]), axis=1)
        points["region_id"] = [value[0] for value in fallback]
        points["region_name"] = [value[1] for value in fallback]
        return points

    from shapely.geometry import Point

    county_records = counties.to_dict("records")
    assigned = []
    for row in points.itertuples(index=False):
        point = Point(float(row.LON), float(row.LAT))
        match = None
        for county in county_records:
            if county["geometry"].covers(point):
                match = county
                break
        if match is None:
            match = min(
                county_records,
                key=lambda county: county["geometry"].distance(point),
            )
        assigned.append((match["region_id"], match["region_name"]))

    points["region_id"] = [value[0] for value in assigned]
    points["region_name"] = [value[1] for value in assigned]
    return points


def risk_level_from_percentile(percentile: float) -> str:
    if percentile < 33.3333:
        return "低"
    if percentile < 66.6667:
        return "中"
    return "高"
