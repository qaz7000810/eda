from __future__ import annotations

from io import StringIO
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import pydeck as pdk


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
MAP_OUTPUT_DIR = PROJECT_ROOT / "output" / "maps"

RX1DAY_CSV = DATA_DIR / "rx1day.csv"

BASELINE_COL = "OBS_1995-2014"
FUTURE_COL = "SSP5-8.5_2081-2100"
CHANGE_COL = "CHANGE_SSP585_MINUS_OBS"

SEQUENTIAL_COLORS = [
    [24, 54, 110],
    [34, 94, 168],
    [47, 136, 189],
    [74, 185, 176],
    [159, 214, 127],
    [255, 196, 79],
    [241, 108, 32],
]

DIVERGING_COLORS = [
    [49, 54, 149],
    [69, 117, 180],
    [116, 173, 209],
    [224, 224, 224],
    [244, 109, 67],
    [215, 48, 39],
    [165, 0, 38],
]


def read_tccip_csv(csv_path: Path) -> pd.DataFrame:
    normalized_lines = []
    for raw_line in csv_path.read_text(encoding="utf-8").splitlines():
        normalized_lines.append(raw_line.rstrip().rstrip(","))

    df = pd.read_csv(StringIO("\n".join(normalized_lines)), skipinitialspace=True)
    df.columns = [column.strip() for column in df.columns]
    return df


def clean_numeric(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    return numeric.mask(np.isclose(numeric, -99.9))


def load_rx1day_data(csv_path: Path) -> gpd.GeoDataFrame:
    df = read_tccip_csv(csv_path)
    required_columns = ["LON", "LAT", BASELINE_COL, FUTURE_COL]

    for column in required_columns:
        df[column] = clean_numeric(df[column])

    df[CHANGE_COL] = df[FUTURE_COL] - df[BASELINE_COL]

    return gpd.GeoDataFrame(
        df,
        geometry=gpd.points_from_xy(df["LON"], df["LAT"]),
        crs="EPSG:4326",
    )


def compute_breaks(series: pd.Series, diverging: bool) -> np.ndarray:
    valid = series.dropna().astype(float)
    if valid.empty:
        raise ValueError("No valid values were found for the requested map.")

    if diverging:
        max_abs = float(np.nanmax(np.abs(valid)))
        if np.isclose(max_abs, 0):
            max_abs = 1.0
        return np.linspace(-max_abs, max_abs, len(DIVERGING_COLORS) + 1)

    quantiles = np.quantile(valid, np.linspace(0, 1, len(SEQUENTIAL_COLORS) + 1))
    if np.unique(np.round(quantiles, 8)).size < 2:
        min_value = float(valid.min())
        max_value = float(valid.max())
        if np.isclose(min_value, max_value):
            max_value = min_value + 1.0
        return np.linspace(min_value, max_value, len(SEQUENTIAL_COLORS) + 1)
    return quantiles


def assign_color(value: float, breaks: np.ndarray, colors: list[list[int]]) -> list[int]:
    if pd.isna(value):
        return [0, 0, 0, 0]

    for idx, color in enumerate(colors):
        lower = breaks[idx]
        upper = breaks[idx + 1]
        is_last_bin = idx == len(colors) - 1
        if lower <= value < upper or (is_last_bin and lower <= value <= upper):
            return color
    return colors[-1]


def compute_shared_elevation_reference(
    gdf: gpd.GeoDataFrame, value_columns: list[str]
) -> tuple[float, float]:
    combined = pd.concat([gdf[column] for column in value_columns], ignore_index=True).dropna()
    if combined.empty:
        raise ValueError("No valid values were found for 3D elevation scaling.")

    min_value = float(combined.min())
    max_value = float(combined.max())
    if np.isclose(min_value, max_value):
        max_value = min_value + 1.0
    return min_value, max_value


def compute_emphasized_elevation(
    series: pd.Series,
    elevation_reference: tuple[float, float],
    exponent: float = 2.6,
    max_height: float = 7000.0,
) -> pd.Series:
    min_value, max_value = elevation_reference
    normalized = ((series - min_value) / (max_value - min_value)).clip(lower=0, upper=1)
    return np.power(normalized, exponent) * max_height


def prepare_map_frame(
    gdf: gpd.GeoDataFrame,
    value_column: str,
    title: str,
    subtitle: str,
    diverging: bool = False,
    elevation_reference: tuple[float, float] | None = None,
) -> pd.DataFrame:
    map_df = gdf.dropna(subset=["LON", "LAT", value_column]).copy()
    colors = DIVERGING_COLORS if diverging else SEQUENTIAL_COLORS
    breaks = compute_breaks(map_df[value_column], diverging=diverging)

    map_df["fill_color"] = map_df[value_column].apply(
        lambda value: assign_color(value, breaks, colors)
    )
    if elevation_reference is None or diverging:
        map_df["elevation"] = map_df[value_column]
    else:
        map_df["elevation"] = compute_emphasized_elevation(
            map_df[value_column], elevation_reference
        )
    map_df["title"] = title
    map_df["subtitle"] = subtitle
    map_df["value_text"] = map_df[value_column].map(lambda value: f"{value:,.2f}")
    return map_df


def build_view_state(map_df: pd.DataFrame, pitch: int) -> pdk.ViewState:
    return pdk.ViewState(
        longitude=float(map_df["LON"].mean()),
        latitude=float(map_df["LAT"].mean()),
        zoom=6.7,
        min_zoom=5.6,
        max_zoom=12,
        pitch=pitch,
        bearing=0,
    )


def build_2d_map(map_df: pd.DataFrame, radius: int = 3200) -> pdk.Deck:
    layer = pdk.Layer(
        "ScatterplotLayer",
        data=map_df,
        get_position=["LON", "LAT"],
        get_fill_color="fill_color",
        get_radius=radius,
        opacity=0.82,
        stroked=False,
        pickable=True,
    )

    tooltip = {
        "html": "<b>{title}</b><br/>{subtitle}<br/>數值: {value_text}",
        "style": {
            "backgroundColor": "rgba(9, 16, 28, 0.96)",
            "color": "white",
            "fontSize": "13px",
        },
    }

    return pdk.Deck(
        layers=[layer],
        initial_view_state=build_view_state(map_df, pitch=0),
        map_provider="carto",
        map_style=pdk.map_styles.CARTO_DARK,
        tooltip=tooltip,
    )


def build_3d_map(
    map_df: pd.DataFrame,
    elevation_scale: float = 1.0,
    radius: int = 3400,
) -> pdk.Deck:
    layer = pdk.Layer(
        "ColumnLayer",
        data=map_df,
        get_position=["LON", "LAT"],
        get_fill_color="fill_color",
        get_elevation="elevation",
        elevation_scale=elevation_scale,
        radius=radius,
        extruded=True,
        pickable=True,
        auto_highlight=True,
        coverage=0.95,
    )

    tooltip = {
        "html": "<b>{title}</b><br/>{subtitle}<br/>數值: {value_text}",
        "style": {
            "backgroundColor": "rgba(9, 16, 28, 0.96)",
            "color": "white",
            "fontSize": "13px",
        },
    }

    return pdk.Deck(
        layers=[layer],
        initial_view_state=build_view_state(map_df, pitch=45),
        map_provider="carto",
        map_style=pdk.map_styles.CARTO_DARK,
        tooltip=tooltip,
    )


def export_map(deck: pdk.Deck, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    deck.to_html(
        filename=str(output_path),
        open_browser=False,
        notebook_display=False,
        iframe_width="100%",
        iframe_height=720,
    )


def main() -> None:
    gdf = load_rx1day_data(RX1DAY_CSV)
    shared_elevation_reference = compute_shared_elevation_reference(
        gdf, [BASELINE_COL, FUTURE_COL]
    )

    baseline_2d = prepare_map_frame(
        gdf,
        BASELINE_COL,
        title="Rx1day Baseline 2D",
        subtitle="OBS 1995-2014",
    )
    baseline_3d = prepare_map_frame(
        gdf,
        BASELINE_COL,
        title="Rx1day Baseline 3D",
        subtitle="OBS 1995-2014",
        elevation_reference=shared_elevation_reference,
    )
    future_2d = prepare_map_frame(
        gdf,
        FUTURE_COL,
        title="Rx1day Future 2D",
        subtitle="SSP5-8.5 2081-2100",
    )
    future_3d = prepare_map_frame(
        gdf,
        FUTURE_COL,
        title="Rx1day Future 3D",
        subtitle="SSP5-8.5 2081-2100",
        elevation_reference=shared_elevation_reference,
    )
    change_2d = prepare_map_frame(
        gdf,
        CHANGE_COL,
        title="Rx1day Change",
        subtitle="SSP5-8.5 2081-2100 minus OBS 1995-2014",
        diverging=True,
    )

    export_map(build_2d_map(baseline_2d), MAP_OUTPUT_DIR / "rx1day_baseline_2d.html")
    export_map(build_3d_map(baseline_3d), MAP_OUTPUT_DIR / "rx1day_baseline_3d.html")
    export_map(build_2d_map(future_2d), MAP_OUTPUT_DIR / "rx1day_future_2d.html")
    export_map(build_3d_map(future_3d), MAP_OUTPUT_DIR / "rx1day_future_3d.html")
    export_map(build_2d_map(change_2d), MAP_OUTPUT_DIR / "rx1day_change.html")

    print("Created map HTML files in output/maps/")


if __name__ == "__main__":
    main()
