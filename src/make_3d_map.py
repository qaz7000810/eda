from __future__ import annotations

import argparse
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import pydeck as pdk

try:
    import leafmap.foliumap as leafmap
except Exception:
    leafmap = None


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CSV = PROJECT_ROOT / "data" / "rx1day.csv"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "output"

SEQUENTIAL_COLORS = [
    [39, 64, 139],
    [33, 113, 181],
    [66, 146, 198],
    [107, 174, 214],
    [158, 202, 225],
    [255, 183, 77],
    [239, 108, 0],
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create 3D AR6 climate maps for baseline, future, and change."
    )
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV, help="Input CSV path.")
    parser.add_argument(
        "--slug",
        default="rx1day",
        help="Output file prefix, for example rx1day or tx90p.",
    )
    parser.add_argument(
        "--variable-name",
        default="Rx1day",
        help="Human-readable variable name shown in tooltips.",
    )
    parser.add_argument(
        "--baseline-col",
        default="OBS_1995-2014",
        help="Baseline column name.",
    )
    parser.add_argument(
        "--future-col",
        default="SSP5-8.5_2081-2100",
        help="Future column name.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for exported HTML files.",
    )
    parser.add_argument(
        "--elevation-scale",
        type=float,
        default=20.0,
        help="Vertical exaggeration applied to the column height.",
    )
    parser.add_argument(
        "--radius",
        type=int,
        default=3400,
        help="Column radius in meters.",
    )
    return parser.parse_args()


def clean_numeric(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    return numeric.mask(np.isclose(numeric, -99.9))


def load_and_prepare_data(
    csv_path: Path, baseline_col: str, future_col: str
) -> tuple[pd.DataFrame, gpd.GeoDataFrame]:
    df = pd.read_csv(csv_path, skipinitialspace=True)
    df.columns = [column.strip() for column in df.columns]

    required_columns = ["LON", "LAT", baseline_col, future_col]
    missing = [column for column in required_columns if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")

    for column in required_columns:
        df[column] = clean_numeric(df[column])

    df["change"] = df[future_col] - df[baseline_col]
    gdf = gpd.GeoDataFrame(
        df.copy(),
        geometry=gpd.points_from_xy(df["LON"], df["LAT"]),
        crs="EPSG:4326",
    )
    return df, gdf


def compute_breaks(series: pd.Series, diverging: bool) -> np.ndarray:
    valid = series.dropna().astype(float)
    if valid.empty:
        raise ValueError("No valid values found after cleaning.")

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

    for idx in range(len(colors)):
        lower = breaks[idx]
        upper = breaks[idx + 1]
        is_last_bin = idx == len(colors) - 1
        if lower <= value < upper or (is_last_bin and lower <= value <= upper):
            return colors[idx]
    return colors[-1]


def prepare_map_frame(
    gdf: gpd.GeoDataFrame,
    value_column: str,
    map_title: str,
    value_label: str,
    diverging: bool = False,
) -> pd.DataFrame:
    map_df = gdf.dropna(subset=[value_column]).copy()
    if map_df.empty:
        raise ValueError(f"No valid rows found for {value_column}.")

    colors = DIVERGING_COLORS if diverging else SEQUENTIAL_COLORS
    breaks = compute_breaks(map_df[value_column], diverging=diverging)

    map_df["fill_color"] = map_df[value_column].apply(
        lambda value: assign_color(value, breaks, colors)
    )
    map_df["elevation"] = map_df[value_column]
    map_df["map_title"] = map_title
    map_df["value_label"] = value_label
    map_df["value_text"] = map_df[value_column].map(lambda value: f"{value:,.2f}")
    return map_df


def build_deck(
    map_df: pd.DataFrame,
    elevation_scale: float,
    radius: int,
) -> pdk.Deck:
    view_state = pdk.ViewState(
        longitude=float(map_df["LON"].mean()),
        latitude=float(map_df["LAT"].mean()),
        zoom=6.7,
        min_zoom=5.5,
        max_zoom=12,
        pitch=45,
        bearing=0,
    )

    layer = pdk.Layer(
        "ColumnLayer",
        data=map_df,
        get_position=["LON", "LAT"],
        get_elevation="elevation",
        elevation_scale=elevation_scale,
        radius=radius,
        get_fill_color="fill_color",
        pickable=True,
        extruded=True,
        auto_highlight=True,
        coverage=0.95,
    )

    tooltip = {
        "html": "<b>{map_title}</b><br/>{value_label}: {value_text}",
        "style": {
            "backgroundColor": "rgba(11, 18, 32, 0.95)",
            "color": "white",
            "fontSize": "13px",
        },
    }

    return pdk.Deck(
        layers=[layer],
        initial_view_state=view_state,
        map_provider="carto",
        map_style=pdk.map_styles.CARTO_DARK,
        tooltip=tooltip,
    )


def export_map(
    gdf: gpd.GeoDataFrame,
    value_column: str,
    map_title: str,
    value_label: str,
    output_path: Path,
    elevation_scale: float,
    radius: int,
    diverging: bool = False,
) -> None:
    map_df = prepare_map_frame(
        gdf=gdf,
        value_column=value_column,
        map_title=map_title,
        value_label=value_label,
        diverging=diverging,
    )
    deck = build_deck(
        map_df=map_df,
        elevation_scale=elevation_scale,
        radius=radius,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    deck.to_html(
        filename=str(output_path),
        open_browser=False,
        notebook_display=False,
        iframe_width="100%",
        iframe_height=720,
    )


def build_leafmap_preview(
    gdf: gpd.GeoDataFrame,
    value_column: str,
    layer_name: str,
):
    if leafmap is None:
        return None

    preview_gdf = gdf.dropna(subset=[value_column]).copy()
    if preview_gdf.empty:
        return None

    center = [float(preview_gdf["LAT"].mean()), float(preview_gdf["LON"].mean())]
    preview_map = leafmap.Map(center=center, zoom=7)
    preview_map.add_gdf(
        preview_gdf[[value_column, "geometry"]],
        layer_name=layer_name,
    )
    return preview_map


def main() -> None:
    args = parse_args()
    csv_path = args.csv if args.csv.is_absolute() else PROJECT_ROOT / args.csv
    output_dir = (
        args.output_dir if args.output_dir.is_absolute() else PROJECT_ROOT / args.output_dir
    )

    _, gdf = load_and_prepare_data(
        csv_path=csv_path,
        baseline_col=args.baseline_col,
        future_col=args.future_col,
    )

    baseline_output = output_dir / f"{args.slug}_baseline_3d.html"
    future_output = output_dir / f"{args.slug}_ssp585_3d.html"
    change_output = output_dir / f"{args.slug}_change_3d.html"

    export_map(
        gdf=gdf,
        value_column=args.baseline_col,
        map_title=f"{args.variable_name} Baseline",
        value_label=args.baseline_col,
        output_path=baseline_output,
        elevation_scale=args.elevation_scale,
        radius=args.radius,
    )
    export_map(
        gdf=gdf,
        value_column=args.future_col,
        map_title=f"{args.variable_name} Future",
        value_label=args.future_col,
        output_path=future_output,
        elevation_scale=args.elevation_scale,
        radius=args.radius,
    )
    export_map(
        gdf=gdf,
        value_column="change",
        map_title=f"{args.variable_name} Change",
        value_label=f"{args.future_col} - {args.baseline_col}",
        output_path=change_output,
        elevation_scale=args.elevation_scale,
        radius=args.radius,
        diverging=True,
    )

    print(f"Created: {baseline_output}")
    print(f"Created: {future_output}")
    print(f"Created: {change_output}")


if __name__ == "__main__":
    main()
