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

OPTIONAL_METRIC_FILES = {
    "PRCPTOT": DATA_DIR / "prcptot.csv",
    "SDII": DATA_DIR / "sdii.csv",
    "CDD": DATA_DIR / "cdd.csv",
    "CWD": DATA_DIR / "cwd.csv",
    "HWDI": DATA_DIR / "hwdi.csv",
}

BASELINE_COL = "OBS_1995-2014"
FUTURE_COL = "SSP5-8.5_2081-2100"

TIME_PERIOD_LABELS = [
    ("Baseline<br>1995-2014", BASELINE_COL),
    ("Near Future<br>2021-2040", "SSP5-8.5_2021-2040"),
    ("Mid Future<br>2041-2060", "SSP5-8.5_2041-2060"),
    ("Far Future<br>2081-2100", FUTURE_COL),
]

SCENARIO_COLUMNS = [
    "SSP1-2.6_2081-2100",
    "SSP2-4.5_2081-2100",
    "SSP3-7.0_2081-2100",
    "SSP5-8.5_2081-2100",
]

SCENARIO_TIME_SERIES = {
    "SSP1-2.6": [
        BASELINE_COL,
        "SSP1-2.6_2021-2040",
        "SSP1-2.6_2041-2060",
        "SSP1-2.6_2081-2100",
    ],
    "SSP2-4.5": [
        BASELINE_COL,
        "SSP2-4.5_2021-2040",
        "SSP2-4.5_2041-2060",
        "SSP2-4.5_2081-2100",
    ],
    "SSP3-7.0": [
        BASELINE_COL,
        "SSP3-7.0_2021-2040",
        "SSP3-7.0_2041-2060",
        "SSP3-7.0_2081-2100",
    ],
    "SSP5-8.5": [
        BASELINE_COL,
        "SSP5-8.5_2021-2040",
        "SSP5-8.5_2041-2060",
        "SSP5-8.5_2081-2100",
    ],
}

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

HEATMAP_COLORSCALE = [
    [0.0, "#1f3b73"],
    [0.25, "#3568b8"],
    [0.5, "#e7eef9"],
    [0.75, "#f2a65a"],
    [1.0, "#d64b4b"],
]


def ensure_file_exists(csv_path: Path) -> None:
    if not csv_path.exists():
        raise FileNotFoundError(f"Required file was not found: {csv_path}")


def read_tccip_csv(csv_path: Path) -> pd.DataFrame:
    ensure_file_exists(csv_path)

    normalized_lines = []
    for raw_line in csv_path.read_text(encoding="utf-8").splitlines():
        normalized_lines.append(raw_line.rstrip().rstrip(","))

    df = pd.read_csv(StringIO("\n".join(normalized_lines)), skipinitialspace=True)
    df.columns = [column.strip() for column in df.columns]
    return df


def clean_numeric(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    return numeric.mask(np.isclose(numeric, -99.9))


def validate_columns(df: pd.DataFrame, required_columns: list[str], csv_path: Path) -> None:
    missing_columns = [column for column in required_columns if column not in df.columns]
    if missing_columns:
        raise ValueError(
            f"Missing required columns in {csv_path.name}: {', '.join(missing_columns)}"
        )


def load_clean_dataframe(csv_path: Path, value_columns: list[str]) -> pd.DataFrame:
    df = read_tccip_csv(csv_path)
    required_columns = ["LON", "LAT", *value_columns]
    validate_columns(df, required_columns, csv_path)

    for column in required_columns:
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
        margin={"l": 72, "r": 32, "t": 88, "b": 68},
        title_font={"size": 24},
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.02,
            "x": 0,
            "bgcolor": "rgba(0,0,0,0)",
        },
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


def write_html_document(html_content: str, output_name: str) -> None:
    output_path = CHART_OUTPUT_DIR / output_name
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html_content, encoding="utf-8")


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


def build_time_trend_chart(rx1day_df: pd.DataFrame) -> go.Figure:
    x_labels = [label for label, _ in TIME_PERIOD_LABELS]
    fig = go.Figure()

    for scenario_label, scenario_columns in SCENARIO_TIME_SERIES.items():
        y_values = [rx1day_df[column].mean() for column in scenario_columns]
        is_primary = scenario_label == "SSP5-8.5"
        fig.add_trace(
            go.Scatter(
                x=x_labels,
                y=y_values,
                mode="lines+markers",
                name=scenario_label,
                line={
                    "color": SCENARIO_COLORS[scenario_label],
                    "width": 4 if is_primary else 2.5,
                    "dash": "solid" if is_primary else "dot",
                },
                marker={"size": 9 if is_primary else 7},
                hovertemplate=(
                    "情境: "
                    + scenario_label
                    + "<br>時段: %{x}<br>全臺平均 Rx1day: %{y:,.2f}<extra></extra>"
                ),
            )
        )

    fig.add_annotation(
        x=1,
        y=1.11,
        xref="paper",
        yref="paper",
        text="Baseline 為共同觀測基期；其餘三點為各 SSP 時段平均值",
        showarrow=False,
        font={"size": 12, "color": "#aab8cf"},
    )

    fig.update_xaxes(title_text="時間階段")
    fig.update_yaxes(title_text="Rx1day 全臺有效格點平均值")
    return apply_plotly_theme(fig, "Rx1day 時間變化圖", height=640)


def calculate_change_percentages(rx1day_df: pd.DataFrame) -> list[tuple[str, float]]:
    baseline_mean = rx1day_df[BASELINE_COL].mean()
    if np.isclose(baseline_mean, 0):
        raise ValueError("Baseline mean is zero, so change percentage cannot be computed.")

    percentages = []
    for column in SCENARIO_COLUMNS:
        scenario_label = SCENARIO_LABELS[column]
        future_mean = rx1day_df[column].mean()
        change_pct = ((future_mean - baseline_mean) / baseline_mean) * 100
        percentages.append((scenario_label, float(change_pct)))
    return percentages


def build_change_percentage_chart(rx1day_df: pd.DataFrame) -> go.Figure:
    changes = calculate_change_percentages(rx1day_df)
    labels = [label for label, _ in changes]
    values = [value for _, value in changes]

    fig = go.Figure(
        data=[
            go.Bar(
                x=labels,
                y=values,
                marker_color=[SCENARIO_COLORS[label] for label in labels],
                text=[f"{value:+.2f}%" for value in values],
                textposition="outside",
                hovertemplate="情境: %{x}<br>變化百分比: %{y:+.2f}%<extra></extra>",
            )
        ]
    )
    fig.add_hline(line_dash="dash", line_color="rgba(255,255,255,0.28)", y=0)
    fig.update_xaxes(title_text="2081-2100 長期情境")
    fig.update_yaxes(title_text="相對 OBS 1995-2014 的變化百分比 (%)")
    return apply_plotly_theme(fig, "Rx1day 長期情境變化百分比", height=620)


def load_metric_frames(target_column: str) -> tuple[dict[str, pd.DataFrame], list[str]]:
    metric_frames = {
        "RX1DAY": load_clean_dataframe(RX1DAY_CSV, [target_column])[
            ["LON", "LAT", target_column]
        ].rename(columns={target_column: "RX1DAY"}),
        "TX90P": load_clean_dataframe(TX90P_CSV, [target_column])[
            ["LON", "LAT", target_column]
        ].rename(columns={target_column: "TX90P"}),
    }
    skipped_metrics: list[str] = []

    for metric_name, csv_path in OPTIONAL_METRIC_FILES.items():
        if not csv_path.exists():
            skipped_metrics.append(metric_name)
            continue

        metric_frames[metric_name] = load_clean_dataframe(csv_path, [target_column])[
            ["LON", "LAT", target_column]
        ].rename(columns={target_column: metric_name})

    return metric_frames, skipped_metrics


def build_correlation_heatmap(target_column: str = FUTURE_COL) -> go.Figure:
    metric_frames, skipped_metrics = load_metric_frames(target_column)

    merged_df: pd.DataFrame | None = None
    for metric_frame in metric_frames.values():
        merged_df = (
            metric_frame
            if merged_df is None
            else merged_df.merge(metric_frame, on=["LON", "LAT"], how="inner")
        )

    if merged_df is None:
        raise ValueError("No metric data was available for the correlation heatmap.")

    correlation_source = merged_df.dropna().drop(columns=["LON", "LAT"])
    correlation_df = correlation_source.corr().round(3)

    labels = correlation_df.columns.tolist()
    values = correlation_df.to_numpy()
    text = np.vectorize(lambda value: f"{value:.2f}")(values)

    fig = go.Figure(
        data=[
            go.Heatmap(
                z=values,
                x=labels,
                y=labels,
                zmin=-1,
                zmax=1,
                colorscale=HEATMAP_COLORSCALE,
                text=text,
                texttemplate="%{text}",
                hovertemplate="%{y} vs %{x}<br>Pearson r: %{z:.3f}<extra></extra>",
                colorbar={"title": "r"},
                xgap=2,
                ygap=2,
            )
        ]
    )

    skipped_text = "；".join(skipped_metrics) if skipped_metrics else "無"
    fig.add_annotation(
        x=0,
        y=-0.18,
        xref="paper",
        yref="paper",
        xanchor="left",
        yanchor="top",
        text=(
            f"配對情境: {target_column} | 有效格點數: {len(correlation_source):,} | "
            f"缺少指標檔案: {skipped_text}"
        ),
        showarrow=False,
        font={"size": 12, "color": "#aab8cf"},
    )

    fig.update_xaxes(side="bottom")
    fig.update_yaxes(autorange="reversed")
    return apply_plotly_theme(fig, "變數相關矩陣 Heatmap", height=760)


def build_top_increase_table_html(rx1day_df: pd.DataFrame, top_n: int = 10) -> str:
    top_df = (
        rx1day_df[["LON", "LAT", BASELINE_COL, FUTURE_COL]]
        .dropna()
        .assign(change=lambda df: df[FUTURE_COL] - df[BASELINE_COL])
        .sort_values("change", ascending=False)
        .head(top_n)
        .reset_index(drop=True)
    )

    top_df.insert(0, "rank", np.arange(1, len(top_df) + 1))
    top_df = top_df.rename(
        columns={
            "LON": "lon",
            "LAT": "lat",
            BASELINE_COL: "baseline",
            FUTURE_COL: "future",
        }
    )

    formatted_df = top_df.copy()
    for column in ["lon", "lat", "baseline", "future", "change"]:
        formatted_df[column] = formatted_df[column].map(lambda value: f"{value:,.2f}")

    max_change = top_df["change"].max()
    mean_change = top_df["change"].mean()
    summary_cards = [
        ("Top 1 Change", f"{max_change:,.2f}"),
        ("Top 10 Mean", f"{mean_change:,.2f}"),
        ("Scenario", "SSP5-8.5 2081-2100"),
    ]

    summary_html = "".join(
        (
            '<div class="summary-card">'
            f"<span>{label}</span>"
            f"<strong>{value}</strong>"
            "</div>"
        )
        for label, value in summary_cards
    )

    table_html = formatted_df.to_html(index=False, classes="top-table", border=0)
    return f"""<!doctype html>
<html lang="zh-Hant">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>Rx1day Top Increase Summary</title>
    <style>
      :root {{
        --bg: #09111d;
        --panel: rgba(13, 22, 38, 0.94);
        --panel-strong: rgba(8, 16, 28, 0.98);
        --border: rgba(124, 155, 199, 0.18);
        --text: #f3f6fb;
        --soft: #aab8cf;
        --accent: #ffb74d;
      }}

      * {{
        box-sizing: border-box;
      }}

      body {{
        margin: 0;
        padding: 18px;
        font-family: "Segoe UI", "Noto Sans TC", sans-serif;
        color: var(--text);
        background:
          radial-gradient(circle at top left, rgba(54, 90, 140, 0.22), transparent 28%),
          linear-gradient(180deg, var(--bg) 0%, #0c1829 100%);
      }}

      .table-shell {{
        border: 1px solid var(--border);
        border-radius: 22px;
        background: var(--panel);
        overflow: hidden;
      }}

      .table-head {{
        padding: 22px 24px 14px;
        border-bottom: 1px solid var(--border);
        background: linear-gradient(180deg, rgba(9, 16, 28, 0.98), rgba(13, 22, 38, 0.9));
      }}

      .kicker {{
        margin: 0 0 8px;
        color: var(--accent);
        font-size: 12px;
        font-weight: 700;
        letter-spacing: 0.14em;
        text-transform: uppercase;
      }}

      h1 {{
        margin: 0 0 10px;
        font-size: 24px;
      }}

      p {{
        margin: 0;
        color: var(--soft);
        line-height: 1.7;
      }}

      .summary-grid {{
        display: grid;
        grid-template-columns: repeat(3, minmax(0, 1fr));
        gap: 14px;
        padding: 18px 24px 0;
      }}

      .summary-card {{
        padding: 14px 16px;
        border: 1px solid var(--border);
        border-radius: 16px;
        background: rgba(10, 18, 32, 0.9);
      }}

      .summary-card span {{
        display: block;
        color: var(--soft);
        font-size: 12px;
        margin-bottom: 6px;
      }}

      .summary-card strong {{
        font-size: 20px;
      }}

      .table-wrap {{
        padding: 18px 24px 24px;
        overflow-x: auto;
      }}

      table {{
        width: 100%;
        border-collapse: collapse;
        min-width: 720px;
      }}

      thead th {{
        padding: 12px 10px;
        text-align: left;
        color: var(--accent);
        border-bottom: 1px solid var(--border);
        font-size: 13px;
        font-weight: 700;
        letter-spacing: 0.04em;
        text-transform: uppercase;
      }}

      tbody td {{
        padding: 12px 10px;
        border-bottom: 1px solid rgba(124, 155, 199, 0.1);
        color: var(--text);
      }}

      tbody tr:hover {{
        background: rgba(255, 183, 77, 0.06);
      }}

      .table-note {{
        padding: 0 24px 24px;
        color: var(--soft);
        font-size: 13px;
      }}

      @media (max-width: 900px) {{
        body {{
          padding: 12px;
        }}

        .summary-grid {{
          grid-template-columns: 1fr;
        }}
      }}
    </style>
  </head>
  <body>
    <div class="table-shell">
      <div class="table-head">
        <p class="kicker">Top Increase Summary</p>
        <h1>Rx1day 增幅最高格點</h1>
        <p>依照 change = SSP5-8.5_2081-2100 - OBS_1995-2014 排序，列出全臺前 {top_n} 名有效格點。</p>
      </div>
      <div class="summary-grid">{summary_html}</div>
      <div class="table-wrap">{table_html}</div>
      <div class="table-note">欄位單位沿用原始指標資料；座標為格點中心經緯度。</div>
    </div>
  </body>
</html>
"""


def main() -> None:
    rx1day_required_columns = sorted(
        {
            BASELINE_COL,
            FUTURE_COL,
            *SCENARIO_COLUMNS,
            *(column for columns in SCENARIO_TIME_SERIES.values() for column in columns),
        }
    )
    rx1day_df = load_clean_dataframe(RX1DAY_CSV, rx1day_required_columns)
    tx90p_df = load_clean_dataframe(TX90P_CSV, [FUTURE_COL])

    write_chart(build_ssp_compare_chart(rx1day_df), "rx1day_ssp_compare.html")
    write_chart(build_boxplot_chart(rx1day_df), "rx1day_boxplot.html")
    write_chart(build_scatter_chart(rx1day_df, tx90p_df), "tx90p_vs_rx1day.html")
    write_chart(build_time_trend_chart(rx1day_df), "rx1day_time_trend.html")
    write_chart(build_change_percentage_chart(rx1day_df), "rx1day_change_percent.html")
    write_chart(build_correlation_heatmap(), "correlation_heatmap.html")
    write_html_document(
        build_top_increase_table_html(rx1day_df),
        "top_increase_table.html",
    )

    print("Created chart HTML files in output/charts/")


if __name__ == "__main__":
    main()
