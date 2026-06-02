(function () {
  const page = document.body.dataset.page;
  const featureLabels = {
    rx1day_base: "基準期年最大一日降雨量（Rx1day baseline）",
    tx90p_change: "暖晝天數變化量（TX90p change）",
    prcptot_change: "雨日總降雨量變化量（PRCPTOT change）",
    sdii_change: "雨日降雨強度變化量（SDII change）",
    cdd_change: "年最長連續不降雨日變化量（CDD change）",
    cwd_change: "年最長連續降雨日變化量（CWD change）",
    hwdi_change: "極端高溫持續指數變化量（HWDI change）",
    lon_scaled: "標準化經度（longitude）",
    lat_scaled: "標準化緯度（latitude）",
    tx90p_change_lag: "鄰近格點暖晝天數變化量（TX90p spatial lag）",
    prcptot_change_lag: "鄰近格點雨日總降雨量變化量（PRCPTOT spatial lag）",
    sdii_change_lag: "鄰近格點雨日降雨強度變化量（SDII spatial lag）",
    cdd_change_lag: "鄰近格點連續不降雨日變化量（CDD spatial lag）",
    cwd_change_lag: "鄰近格點連續降雨日變化量（CWD spatial lag）",
    hwdi_change_lag: "鄰近格點極端高溫持續指數變化量（HWDI spatial lag）",
  };
  const featureSetLabels = {
    NO_SPATIAL: "不含空間鄰近特徵",
    PREDICTOR_SPATIAL_LAG: "加入輔助指標鄰近特徵",
    TARGET_SPATIAL_LAG_RISKY: "加入目標值鄰近特徵（洩漏風險）",
  };

  initReveal();
  initHeroCanvas();

  if (page === "models") {
    initModelsPage();
  }

  if (page === "predictor") {
    initPredictorPage();
  }

  function initReveal() {
    const items = document.querySelectorAll(".reveal");
    if (!("IntersectionObserver" in window)) {
      items.forEach((item) => item.classList.add("is-visible"));
      return;
    }

    const observer = new IntersectionObserver(
      (entries) => {
        entries.forEach((entry) => {
          if (entry.isIntersecting) {
            entry.target.classList.add("is-visible");
            observer.unobserve(entry.target);
          }
        });
      },
      { threshold: 0.14 }
    );

    items.forEach((item) => observer.observe(item));
  }

  function initHeroCanvas() {
    const canvas = document.querySelector("[data-hero-canvas]");
    if (!canvas) return;

    const ctx = canvas.getContext("2d");
    const reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    let width = 0;
    let height = 0;
    let frame = 0;
    let points = [];

    function resize() {
      const rect = canvas.getBoundingClientRect();
      width = Math.max(1, Math.floor(rect.width));
      height = Math.max(1, Math.floor(rect.height));
      const dpr = window.devicePixelRatio || 1;
      canvas.width = width * dpr;
      canvas.height = height * dpr;
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      points = Array.from({ length: 90 }, (_, index) => ({
        x: (index * 53) % width,
        y: (index * 97) % height,
        r: 1.5 + (index % 4),
        speed: 0.18 + (index % 7) * 0.025,
      }));
    }

    function draw() {
      ctx.clearRect(0, 0, width, height);
      ctx.fillStyle = "rgba(21, 156, 163, 0.2)";
      ctx.strokeStyle = "rgba(238, 169, 79, 0.22)";
      ctx.lineWidth = 1;

      points.forEach((point, index) => {
        const x = (point.x + frame * point.speed) % width;
        const y = point.y + Math.sin(frame * 0.01 + index) * 18;
        ctx.beginPath();
        ctx.arc(x, y, point.r, 0, Math.PI * 2);
        ctx.fill();
        if (index % 3 === 0) {
          ctx.beginPath();
          ctx.moveTo(x, y);
          ctx.lineTo(Math.min(width, x + 90), Math.max(0, y - 44));
          ctx.stroke();
        }
      });

      frame += 1;
      if (!reduceMotion) requestAnimationFrame(draw);
    }

    resize();
    draw();
    window.addEventListener("resize", resize);
  }

  async function initModelsPage() {
    try {
      const [comparison, importance, metadata] = await Promise.all([
        fetchText("./output/models/model_comparison.csv"),
        fetchText("./output/models/feature_importance.csv"),
        fetchJson("./output/models/model_metadata.json"),
      ]);
      const comparisonRows = parseCsv(comparison);
      const importanceRows = parseCsv(importance);
      renderModelMetadata(metadata);
      renderModelTable(comparisonRows);
      renderCvBars(comparisonRows);
      renderFeatureBars(importanceRows);
    } catch (error) {
      renderLoadError("model-comparison-table", error);
    }
  }

  function renderModelMetadata(metadata) {
    const title = document.getElementById("deployed-model-name");
    if (title && metadata.deployed_model_name) {
      title.textContent = metadata.deployed_model_name;
    }
  }

  function renderModelTable(rows) {
    const tbody = document.querySelector("#model-comparison-table tbody");
    if (!tbody) return;
    tbody.innerHTML = rows
      .map((row) => {
        const risky = String(row.Possible_leakage).toLowerCase() === "true";
        return `<tr data-risk="${risky}">
          <td>${escapeHtml(row.Model)}</td>
          <td>${escapeHtml(featureSetLabels[row.FeatureSet] || row.FeatureSet)}</td>
          <td>${formatNumber(row.Test_R2, 3)}</td>
          <td>${formatNumber(row.RMSE, 2)}</td>
          <td>${formatNumber(row.MAE, 2)}</td>
          <td>${risky ? "展示用，不部署" : "無標記"}</td>
          <td>${escapeHtml(row.Note || "")}</td>
        </tr>`;
      })
      .join("");
  }

  function renderCvBars(rows) {
    const target = document.getElementById("cv-bars");
    if (!target) return;
    const maxR2 = Math.max(...rows.map((row) => Number(row.Test_R2) || 0), 0.01);
    target.innerHTML = rows
      .map((row) => {
        const value = Number(row.Test_R2) || 0;
        const risky = String(row.Possible_leakage).toLowerCase() === "true";
        const label = `${row.Model} / ${featureSetLabels[row.FeatureSet] || row.FeatureSet}`;
        return `<div class="bar-row ${risky ? "risky" : ""}">
          <span>${escapeHtml(label)}</span>
          <div class="bar-track"><div class="bar-fill" style="--bar-width: ${(value / maxR2) * 100}%"></div></div>
          <strong>${formatNumber(value, 3)}</strong>
        </div>`;
      })
      .join("");
  }

  function renderFeatureBars(rows) {
    const target = document.getElementById("feature-bars");
    if (!target) return;
    const topRows = rows.slice(0, 10);
    const maxValue = Math.max(...topRows.map((row) => Number(row.Importance) || 0), 0.01);
    target.innerHTML = topRows
      .map((row) => {
        const value = Number(row.Importance) || 0;
        return `<div class="bar-row">
          <span>${escapeHtml(featureLabels[row.Feature] || row.Feature)}</span>
          <div class="bar-track"><div class="bar-fill" style="--bar-width: ${(value / maxValue) * 100}%"></div></div>
          <strong>${formatNumber(value, 3)}</strong>
        </div>`;
      })
      .join("");
  }

  async function initPredictorPage() {
    const state = {
      scenario: "SSP5-8.5",
      period: "2081-2100",
      layer: "change",
      selectedPoint: null,
      grid: [],
      regions: [],
      filtered: [],
      bounds: null,
    };

    const canvas = document.getElementById("prediction-map");
    const scenarioSelect = document.getElementById("scenario-select");
    const periodSelect = document.getElementById("period-select");
    const regionSelect = document.getElementById("region-select");
    const layerButtons = document.querySelectorAll("[data-layer]");

    try {
      const [grid, regions] = await Promise.all([
        fetchJson("./output/predictions/grid_predictions.json"),
        fetchJson("./output/predictions/region_predictions.json"),
      ]);

      state.grid = grid;
      state.regions = regions;
      state.bounds = computeBounds(grid);
      populateRegions(regionSelect, regions);
      bindPredictorEvents();
      updatePredictor();
    } catch (error) {
      const title = document.getElementById("grid-title");
      if (title) title.textContent = `資料載入失敗：${error.message}`;
    }

    function bindPredictorEvents() {
      scenarioSelect.addEventListener("change", () => {
        state.scenario = scenarioSelect.value;
        state.selectedPoint = null;
        updatePredictor();
      });

      periodSelect.addEventListener("change", () => {
        state.period = periodSelect.value;
        state.selectedPoint = null;
        updatePredictor();
      });

      regionSelect.addEventListener("change", () => updateRegionSummary());

      layerButtons.forEach((button) => {
        button.addEventListener("click", () => {
          layerButtons.forEach((item) => item.classList.remove("active"));
          button.classList.add("active");
          state.layer = button.dataset.layer;
          drawPredictionMap();
        });
      });

      canvas.addEventListener("click", (event) => {
        const point = nearestPoint(event);
        if (!point) return;
        state.selectedPoint = point;
        renderGridResult(point);
        drawPredictionMap();
      });

      window.addEventListener("resize", drawPredictionMap);
    }

    function updatePredictor() {
      state.filtered = state.grid.filter(
        (row) => row.scenario === state.scenario && row.period === state.period
      );
      if (state.filtered.length) {
        state.selectedPoint = state.selectedPoint || state.filtered[Math.floor(state.filtered.length / 2)];
        renderGridResult(state.selectedPoint);
      }
      updateRegionSummary();
      drawPredictionMap();
    }

    function updateRegionSummary() {
      const selectedRegion = regionSelect.value || (state.regions[0] && state.regions[0].region_id);
      const row = state.regions.find(
        (item) =>
          item.region_id === selectedRegion &&
          item.scenario === state.scenario &&
          item.period === state.period
      );
      if (!row) return;
      document.getElementById("region-title").textContent = row.region_name;
      document.getElementById("region-result").innerHTML = `
        <div><span>平均變化量</span><strong>${formatNumber(row.mean_change_pred, 2)} mm</strong></div>
        <div><span>平均未來值</span><strong>${formatNumber(row.mean_future_pred, 2)} mm</strong></div>
        <div><span>最大變化量</span><strong>${formatNumber(row.max_change_pred, 2)} mm</strong></div>
        <div><span>高風險占比</span><strong>${formatPercent(row.high_risk_share)}</strong></div>
      `;
    }

    function drawPredictionMap() {
      if (!canvas || !state.filtered.length) return;
      const parent = canvas.parentElement;
      const rect = parent.getBoundingClientRect();
      const cssWidth = Math.max(320, Math.floor(rect.width));
      const cssHeight = Math.max(420, Math.floor(canvas.getBoundingClientRect().height || 680));
      const dpr = window.devicePixelRatio || 1;
      canvas.width = cssWidth * dpr;
      canvas.height = cssHeight * dpr;
      canvas.style.width = `${cssWidth}px`;
      canvas.style.height = `${cssHeight}px`;

      const ctx = canvas.getContext("2d");
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      ctx.clearRect(0, 0, cssWidth, cssHeight);
      drawMapBackground(ctx, cssWidth, cssHeight);

      state.filtered.forEach((row) => {
        const pos = project(row, cssWidth, cssHeight);
        ctx.beginPath();
        ctx.fillStyle = colorForRow(row, state.layer);
        ctx.globalAlpha = state.selectedPoint === row ? 1 : 0.82;
        ctx.arc(pos.x, pos.y, state.selectedPoint === row ? 5.8 : 3.2, 0, Math.PI * 2);
        ctx.fill();
      });

      ctx.globalAlpha = 1;
      renderLegend();
    }

    function drawMapBackground(ctx, width, height) {
      ctx.strokeStyle = "rgba(67, 109, 119, 0.16)";
      ctx.lineWidth = 1;
      for (let x = 0; x < width; x += 48) {
        ctx.beginPath();
        ctx.moveTo(x, 0);
        ctx.lineTo(x, height);
        ctx.stroke();
      }
      for (let y = 0; y < height; y += 48) {
        ctx.beginPath();
        ctx.moveTo(0, y);
        ctx.lineTo(width, y);
        ctx.stroke();
      }
      ctx.fillStyle = "rgba(23, 49, 58, 0.42)";
      ctx.font = "700 12px Segoe UI, sans-serif";
      ctx.fillText(`${state.scenario} / ${state.period}`, 18, 28);
    }

    function nearestPoint(event) {
      const rect = canvas.getBoundingClientRect();
      const clickX = event.clientX - rect.left;
      const clickY = event.clientY - rect.top;
      let nearest = null;
      let nearestDistance = Infinity;
      state.filtered.forEach((row) => {
        const pos = project(row, rect.width, rect.height);
        const distance = (pos.x - clickX) ** 2 + (pos.y - clickY) ** 2;
        if (distance < nearestDistance) {
          nearestDistance = distance;
          nearest = row;
        }
      });
      return nearestDistance <= 900 ? nearest : null;
    }

    function project(row, width, height) {
      const padding = Math.max(28, Math.min(width, height) * 0.08);
      const x =
        padding +
        ((row.lon - state.bounds.minLon) / (state.bounds.maxLon - state.bounds.minLon)) *
          (width - padding * 2);
      const y =
        height -
        padding -
        ((row.lat - state.bounds.minLat) / (state.bounds.maxLat - state.bounds.minLat)) *
          (height - padding * 2);
      return { x, y };
    }
  }

  function renderGridResult(row) {
    document.getElementById("grid-title").textContent = `${row.lon.toFixed(2)}, ${row.lat.toFixed(2)}`;
    const riskClass = row.risk_level === "高" ? "risk-high" : row.risk_level === "中" ? "risk-mid" : "risk-low";
    document.getElementById("grid-result").innerHTML = `
      <div><span>基準期 Rx1day</span><strong>${formatNumber(row.rx1day_base, 2)} mm</strong></div>
      <div><span>預測變化量</span><strong>${formatSigned(row.rx1day_change_pred, 2)} mm</strong></div>
      <div><span>未來預測值</span><strong>${formatNumber(row.rx1day_future_pred, 2)} mm</strong></div>
      <div><span>風險等級</span><strong class="${riskClass}">${row.risk_level} / P${formatNumber(row.risk_percentile, 0)}</strong></div>
    `;
  }

  function populateRegions(select, regions) {
    const unique = [];
    const seen = new Set();
    regions.forEach((row) => {
      if (!seen.has(row.region_id)) {
        seen.add(row.region_id);
        unique.push(row);
      }
    });
    select.innerHTML = unique
      .map((row) => `<option value="${escapeHtml(row.region_id)}">${escapeHtml(row.region_name)}</option>`)
      .join("");
  }

  function computeBounds(rows) {
    return rows.reduce(
      (bounds, row) => ({
        minLon: Math.min(bounds.minLon, row.lon),
        maxLon: Math.max(bounds.maxLon, row.lon),
        minLat: Math.min(bounds.minLat, row.lat),
        maxLat: Math.max(bounds.maxLat, row.lat),
      }),
      { minLon: Infinity, maxLon: -Infinity, minLat: Infinity, maxLat: -Infinity }
    );
  }

  function colorForRow(row, layer) {
    if (layer === "risk") {
      if (row.risk_level === "高") return "#d95f53";
      if (row.risk_level === "中") return "#eea94f";
      return "#2f9b6d";
    }

    const value = layer === "future" ? row.rx1day_future_pred : row.rx1day_change_pred;
    if (layer === "future") {
      if (value >= 310) return "#d95f53";
      if (value >= 250) return "#eea94f";
      if (value >= 190) return "#159ca3";
      return "#4f7fd9";
    }
    if (value >= 45) return "#d95f53";
    if (value >= 20) return "#eea94f";
    if (value >= 0) return "#159ca3";
    return "#4f7fd9";
  }

  function renderLegend() {
    const legend = document.getElementById("map-legend");
    const active = document.querySelector("[data-layer].active");
    const layer = active ? active.dataset.layer : "change";
    const items =
      layer === "risk"
        ? [
            ["#2f9b6d", "低"],
            ["#eea94f", "中"],
            ["#d95f53", "高"],
          ]
        : layer === "future"
          ? [
              ["#4f7fd9", "< 190"],
              ["#159ca3", "190-250"],
              ["#eea94f", "250-310"],
              ["#d95f53", ">= 310"],
            ]
          : [
              ["#4f7fd9", "< 0"],
              ["#159ca3", "0-20"],
              ["#eea94f", "20-45"],
              ["#d95f53", ">= 45"],
            ];
    legend.innerHTML = items
      .map(
        ([color, label]) =>
          `<span class="legend-item"><span class="legend-swatch" style="background:${color}"></span>${label}</span>`
      )
      .join("");
  }

  async function fetchText(url) {
    const response = await fetch(url);
    if (!response.ok) throw new Error(`${url} ${response.status}`);
    return response.text();
  }

  async function fetchJson(url) {
    const response = await fetch(url);
    if (!response.ok) throw new Error(`${url} ${response.status}`);
    return response.json();
  }

  function parseCsv(text) {
    const lines = text.trim().split(/\r?\n/);
    const headers = splitCsvLine(lines.shift());
    return lines.map((line) => {
      const values = splitCsvLine(line);
      return headers.reduce((row, header, index) => {
        row[header] = values[index] || "";
        return row;
      }, {});
    });
  }

  function splitCsvLine(line) {
    const values = [];
    let current = "";
    let quoted = false;
    for (let i = 0; i < line.length; i += 1) {
      const char = line[i];
      const next = line[i + 1];
      if (char === '"' && quoted && next === '"') {
        current += '"';
        i += 1;
      } else if (char === '"') {
        quoted = !quoted;
      } else if (char === "," && !quoted) {
        values.push(current);
        current = "";
      } else {
        current += char;
      }
    }
    values.push(current);
    return values;
  }

  function renderLoadError(tableId, error) {
    const tbody = document.querySelector(`#${tableId} tbody`);
    if (tbody) {
      tbody.innerHTML = `<tr><td colspan="7">資料載入失敗：${escapeHtml(error.message)}</td></tr>`;
    }
  }

  function formatNumber(value, digits) {
    const number = Number(value);
    if (!Number.isFinite(number)) return "-";
    return number.toLocaleString("zh-Hant", {
      minimumFractionDigits: digits,
      maximumFractionDigits: digits,
    });
  }

  function formatSigned(value, digits) {
    const number = Number(value);
    if (!Number.isFinite(number)) return "-";
    return `${number >= 0 ? "+" : ""}${formatNumber(number, digits)}`;
  }

  function formatPercent(value) {
    const number = Number(value);
    if (!Number.isFinite(number)) return "-";
    return `${formatNumber(number * 100, 1)}%`;
  }

  function escapeHtml(value) {
    return String(value)
      .replaceAll("&", "&amp;")
      .replaceAll("<", "&lt;")
      .replaceAll(">", "&gt;")
      .replaceAll('"', "&quot;")
      .replaceAll("'", "&#039;");
  }
})();
