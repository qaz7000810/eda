from __future__ import annotations

from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
CHART_OUTPUT_DIR = PROJECT_ROOT / "output" / "charts"

RX1DAY_CSV = DATA_DIR / "rx1day.csv"
TX90P_CSV = DATA_DIR / "tx90p.csv"

BASELINE_COL = "OBS_1995-2014"
FUTURE_COL = "SSP5-8.5_2081-2100"
SCENARIO_COLUMNS = [
    "SSP1-2.6_2081-2100",
    "SSP2-4.5_2081-2100",
    "SSP3-7.0_2081-2100",
    "SSP5-8.5_2081-2100",
]

SCENARIO_LABELS = {
    "SSP1-2.6_2081-2100": "SSP1-2.6",
    "SSP2-4.5_2081-2100": "SSP2-4.5",
    "SSP3-7.0_2081-2100": "SSP3-7.0",
    "SSP5-8.5_2081-2100": "SSP5-8.5",
}

SCENARIO_COLORS = {
    "SSP1-2.6": "#4dd0e1",
    "SSP2-4.5": "#80cbc4",
    "SSP3-7.0": "#ffb74d",
    "SSP5-8.5": "#ef5350",
}


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


def load_clean_dataframe(csv_path: Path, value_columns: list[str]) -> pd.DataFrame:
    df = read_tccip_csv(csv_path)
    numeric_columns = ["LON", "LAT", *value_columns]
    for column in numeric_columns:
        df[column] = clean_numeric(df[column])
    return df


def apply_plotly_theme(fig: go.Figure, title: str, height: int = 680) -> go.Figure:
    fig.update_layout(
        title=title,
        template="plotly_dark",
        height=height,
        paper_bgcolor="#09111d",
        plot_bgcolor="#0f1d33",
        font={"family": "Segoe UI, Noto Sans TC, sans-serif", "color": "#f3f6fb"},
        margin={"l": 70, "r": 30, "t": 78, "b": 60},
        title_font={"size": 24},
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "x": 0},
    )
    fig.update_xaxes(showgrid=False, zeroline=False)
    fig.update_yaxes(gridcolor="rgba(255,255,255,0.08)", zeroline=False)
    return fig


def write_chart(fig: go.Figure, output_name: str) -> None:
    output_path = CHART_OUTPUT_DIR / output_name
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(
        str(output_path),
        full_html=True,
        include_plotlyjs="cdn",
        config={"responsive": True, "displaylogo": False},
    )


def build_ssp_compare_chart(rx1day_df: pd.DataFrame) -> go.Figure:
    labels = [SCENARIO_LABELS[column] for column in SCENARIO_COLUMNS]
    means = [rx1day_df[column].mean() for column in SCENARIO_COLUMNS]

    fig = go.Figure(
        data=[
            go.Bar(
                x=labels,
                y=means,
                marker_color=[SCENARIO_COLORS[label] for label in labels],
                text=[f"{value:,.2f}" for value in means],
                textposition="outside",
                hovertemplate="情境: %{x}<br>平均 Rx1day: %{y:,.2f}<extra></extra>",
            )
        ]
    )
    fig.update_yaxes(title_text="Rx1day 平均值")
    fig.update_xaxes(title_text="長期情境")
    return apply_plotly_theme(fig, "Rx1day 長期情境平均值比較")


def build_boxplot_chart(rx1day_df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    for column in SCENARIO_COLUMNS:
        label = SCENARIO_LABELS[column]
        fig.add_trace(
            go.Box(
                y=rx1day_df[column].dropna(),
                name=label,
                marker_color=SCENARIO_COLORS[label],
                boxmean=True,
                hovertemplate=f"{label}<br>Rx1day: %{{y:,.2f}}<extra></extra>",
            )
        )

    fig.update_yaxes(title_text="Rx1day 分布")
    fig.update_xaxes(title_text="長期情境")
    return apply_plotly_theme(fig, "Rx1day 長期情境分布 Boxplot")


def build_scatter_chart(rx1day_df: pd.DataFrame, tx90p_df: pd.DataFrame) -> go.Figure:
    scatter_df = (
        rx1day_df[["LON", "LAT", FUTURE_COL]]
        .rename(columns={FUTURE_COL: "rx1day_future"})
        .merge(
            tx90p_df[["LON", "LAT", FUTURE_COL]].rename(columns={FUTURE_COL: "tx90p_future"}),
            on=["LON", "LAT"],
            how="inner",
        )
        .dropna()
    )

    x_values = scatter_df["tx90p_future"].to_numpy()
    y_values = scatter_df["rx1day_future"].to_numpy()
    slope, intercept = np.polyfit(x_values, y_values, 1)
    correlation = float(np.corrcoef(x_values, y_values)[0, 1])

    x_line = np.linspace(x_values.min(), x_values.max(), 200)
    y_line = slope * x_line + intercept

    fig = go.Figure()
    fig.add_trace(
        go.Scattergl(
            x=x_values,
            y=y_values,
            mode="markers",
            marker={
                "size": 8,
                "color": "#ffb74d",
                "opacity": 0.68,
                "line": {"width": 0},
            },
            customdata=scatter_df[["LON", "LAT"]].to_numpy(),
            hovertemplate=(
                "TX90p: %{x:,.2f}<br>"
                "Rx1day: %{y:,.2f}<br>"
                "LON: %{customdata[0]:.2f}<br>"
                "LAT: %{customdata[1]:.2f}<extra></extra>"
            ),
            name="有效格點",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=x_line,
            y=y_line,
            mode="lines",
            line={"color": "#4dd0e1", "width": 3},
            hovertemplate="趨勢線<br>TX90p: %{x:,.2f}<br>Rx1day: %{y:,.2f}<extra></extra>",
            name="線性趨勢",
        )
    )

    fig.add_annotation(
        x=0.99,
        y=0.98,
        xref="paper",
        yref="paper",
        xanchor="right",
        yanchor="top",
        text=f"Pearson r = {correlation:.3f}",
        showarrow=False,
        bgcolor="rgba(9, 17, 29, 0.9)",
        bordercolor="rgba(255,255,255,0.12)",
        font={"size": 13},
    )

    fig.update_xaxes(title_text="TX90p (SSP5-8.5_2081-2100)")
    fig.update_yaxes(title_text="Rx1day (SSP5-8.5_2081-2100)")
    return apply_plotly_theme(fig, "TX90p 與 Rx1day 關聯散佈圖")


def main() -> None:
    rx1day_df = load_clean_dataframe(RX1DAY_CSV, SCENARIO_COLUMNS + [FUTURE_COL, BASELINE_COL])
    tx90p_df = load_clean_dataframe(TX90P_CSV, [FUTURE_COL])

    write_chart(build_ssp_compare_chart(rx1day_df), "rx1day_ssp_compare.html")
    write_chart(build_boxplot_chart(rx1day_df), "rx1day_boxplot.html")
    write_chart(build_scatter_chart(rx1day_df, tx90p_df), "tx90p_vs_rx1day.html")

    print("Created chart HTML files in output/charts/")


if __name__ == "__main__":
    main()
