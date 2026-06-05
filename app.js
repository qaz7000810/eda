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
    const layerLabels = {
      change: "雨量增加多少",
      future: "未來單日最大雨量",
      risk: "相對風險排序",
    };
    const periodOrder = ["2021-2040", "2041-2060", "2081-2100"];
    const scenarioOrder = ["SSP1-2.6", "SSP2-4.5", "SSP3-7.0", "SSP5-8.5"];
    const scaleColors = ["#4f7fd9", "#159ca3", "#eea94f", "#d95f53"];
    const featureDetailLabels = [
      ["tx90p_change", "暖晝天數變化", "天"],
      ["prcptot_change", "雨日總降雨量變化", "mm"],
      ["sdii_change", "雨日降雨強度變化", "mm/天"],
      ["cdd_change", "連續不降雨日變化", "天"],
      ["cwd_change", "連續降雨日變化", "天"],
      ["hwdi_change", "高溫持續指數變化", "天"],
    ];
    const state = {
      scenario: "SSP5-8.5",
      period: "2081-2100",
      layer: "change",
      selectedPoint: null,
      hoveredPoint: null,
      grid: [],
      regions: [],
      boundaries: null,
      insights: null,
      filtered: [],
      scales: null,
      bounds: null,
      viewport: null,
    };

    const canvas = document.getElementById("prediction-map");
    const tooltip = document.getElementById("map-tooltip");
    const scenarioSelect = document.getElementById("scenario-select");
    const periodSelect = document.getElementById("period-select");
    const regionSelect = document.getElementById("region-select");
    const layerButtons = document.querySelectorAll("[data-layer]");

    try {
      const [grid, regions, boundaries, insights] = await Promise.all([
        fetchJson("./output/predictions/grid_predictions.json"),
        fetchJson("./output/predictions/region_predictions.json"),
        fetchJson("./output/predictions/county_boundaries.json").catch(() => null),
        fetchJson("./output/predictions/risk_insights.json").catch(() => null),
      ]);

      state.grid = grid;
      state.regions = regions;
      state.boundaries = boundaries;
      state.insights = insights;
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

      regionSelect.addEventListener("change", () => {
        if (!regionSelect.value) {
          updateRegionSummary();
          drawPredictionMap();
          return;
        }
        const regionPoint = pickRegionFocusPoint(regionSelect.value);
        if (regionPoint) selectPoint(regionPoint, false);
        updateRegionSummary();
        drawPredictionMap();
      });

      layerButtons.forEach((button) => {
        button.addEventListener("click", () => {
          layerButtons.forEach((item) => item.classList.remove("active"));
          button.classList.add("active");
          state.layer = button.dataset.layer;
          renderQueryContext();
          drawPredictionMap();
        });
      });

      canvas.addEventListener("click", (event) => {
        const point = nearestPoint(event);
        if (!point) return;
        selectPoint(point, true);
      });

      canvas.addEventListener("mousemove", (event) => {
        state.hoveredPoint = nearestPoint(event);
        canvas.style.cursor = state.hoveredPoint ? "pointer" : "crosshair";
        renderTooltip(event, state.hoveredPoint);
        drawPredictionMap();
      });

      canvas.addEventListener("mouseleave", () => {
        state.hoveredPoint = null;
        if (tooltip) tooltip.hidden = true;
        drawPredictionMap();
      });

      window.addEventListener("resize", drawPredictionMap);
    }

    function updatePredictor() {
      state.filtered = state.grid.filter(
        (row) => row.scenario === state.scenario && row.period === state.period
      );
      state.scales = buildScales(state.filtered);
      if (state.filtered.length) {
        const nextPoint = state.selectedPoint
          ? findSameLocation(state.selectedPoint, state.filtered)
          : state.filtered[Math.floor(state.filtered.length / 2)];
        selectPoint(nextPoint || state.filtered[0], false);
      }
      updateRegionSummary();
      renderRegionRanking();
      renderReportInsights();
      renderQueryContext();
      drawPredictionMap();
    }

    function selectPoint(point, syncRegion) {
      state.selectedPoint = point;
      if (syncRegion && point && regionSelect.value !== point.region_id) {
        regionSelect.value = point.region_id;
        updateRegionSummary();
      }
      renderGridResult(point);
      renderGridTrend(point);
      renderScenarioComparison(point);
      renderFeatureDetails(point);
      renderQueryContext();
      drawPredictionMap();
    }

    function renderReportInsights() {
      renderInsightFeatureImportance();
      renderCompoundHotspots();
      renderScenarioStability();
      renderMultiScenarioRisk();
    }

    function updateRegionSummary() {
      const selectedRegion = regionSelect.value;
      if (!selectedRegion) {
        renderTaiwanSummary();
        return;
      }
      const row = state.regions.find(
        (item) =>
          item.region_id === selectedRegion &&
          item.scenario === state.scenario &&
          item.period === state.period
      );
      if (!row) return;
      document.getElementById("region-title").textContent = row.region_name;
      document.getElementById("region-result").innerHTML = `
        <div><span>平均增加雨量</span><strong>${formatNumber(row.mean_change_pred, 2)} mm</strong></div>
        <div><span>平均未來雨量</span><strong>${formatNumber(row.mean_future_pred, 2)} mm</strong></div>
        <div><span>最大增加雨量</span><strong>${formatNumber(row.max_change_pred, 2)} mm</strong></div>
        <div><span>相對高風險占比</span><strong>${formatPercent(row.high_risk_share)}</strong></div>
      `;
    }

    function renderTaiwanSummary() {
      if (!state.filtered.length) return;
      const meanChange = mean(state.filtered.map((row) => row.rx1day_change_pred));
      const meanFuture = mean(state.filtered.map((row) => row.rx1day_future_pred));
      const maxChange = Math.max(...state.filtered.map((row) => row.rx1day_change_pred));
      const highRiskShare = state.filtered.filter((row) => row.risk_level === "高").length / state.filtered.length;
      document.getElementById("region-title").textContent = "全臺總覽";
      document.getElementById("region-result").innerHTML = `
        <div><span>平均增加雨量</span><strong>${formatNumber(meanChange, 2)} mm</strong></div>
        <div><span>平均未來雨量</span><strong>${formatNumber(meanFuture, 2)} mm</strong></div>
        <div><span>最大增加雨量</span><strong>${formatNumber(maxChange, 2)} mm</strong></div>
        <div><span>相對高風險占比</span><strong>${formatPercent(highRiskShare)}</strong></div>
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
      state.viewport = computeMapViewport(cssWidth, cssHeight);
      drawMapBackground(ctx, cssWidth, cssHeight);
      drawCountyBoundaries(ctx, cssWidth, cssHeight);

      state.filtered.forEach((row) => {
        drawPoint(ctx, row, cssWidth, cssHeight);
      });

      if (state.hoveredPoint) drawFocusRing(ctx, state.hoveredPoint, cssWidth, cssHeight, "#17313a", 7.5);
      if (state.selectedPoint) drawFocusRing(ctx, state.selectedPoint, cssWidth, cssHeight, "#ffffff", 9);
      if (state.selectedPoint) drawFocusRing(ctx, state.selectedPoint, cssWidth, cssHeight, "#17313a", 6.5);

      ctx.globalAlpha = 1;
      renderLegend();
    }

    function drawMapBackground(ctx, width, height) {
      ctx.strokeStyle = "rgba(67, 109, 119, 0.12)";
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
      ctx.fillStyle = "rgba(23, 49, 58, 0.48)";
      ctx.font = "700 12px Segoe UI, Noto Sans TC, sans-serif";
      ctx.fillText(`${state.scenario} / ${state.period} / ${layerLabels[state.layer]}`, 18, 28);
    }

    function drawCountyBoundaries(ctx, width, height) {
      if (!state.boundaries || !state.boundaries.features) return;
      const selectedRegion = regionSelect.value;
      state.boundaries.features.forEach((feature) => {
        const isSelected = feature.properties.region_id === selectedRegion;
        drawGeometry(
          ctx,
          feature.geometry,
          width,
          height,
          isSelected ? "rgba(238, 169, 79, 0.16)" : "rgba(255, 255, 255, 0.48)",
          isSelected ? "rgba(154, 91, 0, 0.72)" : "rgba(67, 109, 119, 0.32)",
          isSelected ? 1.7 : 0.9
        );
      });
    }

    function drawGeometry(ctx, geometry, width, height, fillStyle, strokeStyle, lineWidth) {
      const polygons = geometry.type === "Polygon" ? [geometry.coordinates] : geometry.coordinates;
      ctx.beginPath();
      polygons.forEach((polygon) => {
        polygon.forEach((ring) => {
          ring.forEach(([lon, lat], index) => {
            const pos = projectLonLat(lon, lat, width, height);
            if (index === 0) ctx.moveTo(pos.x, pos.y);
            else ctx.lineTo(pos.x, pos.y);
          });
          ctx.closePath();
        });
      });
      ctx.fillStyle = fillStyle;
      ctx.strokeStyle = strokeStyle;
      ctx.lineWidth = lineWidth;
      ctx.fill("evenodd");
      ctx.stroke();
    }

    function drawPoint(ctx, row, width, height) {
      const pos = project(row, width, height);
      const isSelected = state.selectedPoint === row;
      const isHovered = state.hoveredPoint === row;
      ctx.beginPath();
      ctx.fillStyle = colorForRow(row, state.layer);
      ctx.globalAlpha = isSelected || isHovered ? 0.96 : 0.82;
      ctx.arc(pos.x, pos.y, isSelected ? 5.8 : 3.2, 0, Math.PI * 2);
      ctx.fill();
    }

    function drawFocusRing(ctx, row, width, height, color, radius) {
      const pos = project(row, width, height);
      ctx.beginPath();
      ctx.globalAlpha = 1;
      ctx.strokeStyle = color;
      ctx.lineWidth = color === "#ffffff" ? 4 : 2;
      ctx.arc(pos.x, pos.y, radius, 0, Math.PI * 2);
      ctx.stroke();
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
      return nearestDistance <= 625 ? nearest : null;
    }

    function renderTooltip(event, row) {
      if (!tooltip || !row) {
        if (tooltip) tooltip.hidden = true;
        return;
      }
      const parentRect = canvas.parentElement.getBoundingClientRect();
      tooltip.hidden = false;
      tooltip.style.left = `${Math.max(12, Math.min(event.clientX - parentRect.left + 14, parentRect.width - 280))}px`;
      tooltip.style.top = `${Math.max(event.clientY - parentRect.top - 18, 12)}px`;
      tooltip.innerHTML = `
        <strong>${escapeHtml(row.region_name)}｜${row.lon.toFixed(2)}, ${row.lat.toFixed(2)}</strong>
        可能增加雨量：${formatSigned(row.rx1day_change_pred, 2)} mm<br>
        未來單日最大雨量：${formatNumber(row.rx1day_future_pred, 2)} mm<br>
        相對排序：${escapeHtml(row.risk_level)} / P${formatNumber(row.risk_percentile, 0)}
      `;
    }

    function project(row, width, height) {
      return projectLonLat(row.lon, row.lat, width, height);
    }

    function projectLonLat(lon, lat, width, height) {
      const viewport = state.viewport || computeMapViewport(width, height);
      const x = viewport.left + (lon - state.bounds.minLon) * viewport.scale;
      const y = viewport.top + (state.bounds.maxLat - lat) * viewport.scale;
      return { x, y };
    }

    function computeMapViewport(width, height) {
      const padding = Math.max(34, Math.min(width, height) * 0.08);
      const availableWidth = Math.max(1, width - padding * 2);
      const availableHeight = Math.max(1, height - padding * 2);
      const lonSpan = Math.max(0.01, state.bounds.maxLon - state.bounds.minLon);
      const latSpan = Math.max(0.01, state.bounds.maxLat - state.bounds.minLat);
      const scale = Math.min(availableWidth / lonSpan, availableHeight / latSpan);
      const mapWidth = lonSpan * scale;
      const mapHeight = latSpan * scale;
      return {
        scale,
        left: (width - mapWidth) / 2,
        top: (height - mapHeight) / 2,
      };
    }

    function renderGridResult(row) {
      document.getElementById("grid-title").textContent =
        `${row.region_name}｜${row.lon.toFixed(2)}, ${row.lat.toFixed(2)}`;
      const riskClass = row.risk_level === "高" ? "risk-high" : row.risk_level === "中" ? "risk-mid" : "risk-low";
      document.getElementById("grid-result").innerHTML = `
        <div><span>過去基準雨量</span><strong>${formatNumber(row.rx1day_base, 2)} mm</strong></div>
        <div><span>可能增加雨量</span><strong>${formatSigned(row.rx1day_change_pred, 2)} mm</strong></div>
        <div><span>未來單日最大雨量</span><strong>${formatNumber(row.rx1day_future_pred, 2)} mm</strong></div>
        <div><span>相對風險排序</span><strong class="${riskClass}">${row.risk_level} / P${formatNumber(row.risk_percentile, 0)}</strong></div>
      `;
    }

    function renderQueryContext() {
      const target = document.getElementById("query-context");
      if (!target) return;
      const highRiskCount = state.filtered.filter((row) => row.risk_level === "高").length;
      target.innerHTML = `
        <div><span>目前圖層</span><strong>${layerLabels[state.layer]}</strong></div>
        <div><span>有效格點</span><strong>${formatNumber(state.filtered.length, 0)}</strong></div>
        <div><span>相對高風險格點</span><strong>${formatNumber(highRiskCount, 0)} / ${formatPercent(highRiskCount / Math.max(state.filtered.length, 1))}</strong></div>
      `;
    }

    function renderRegionRanking() {
      const target = document.getElementById("region-ranking");
      if (!target) return;
      const rows = state.regions
        .filter((row) => row.scenario === state.scenario && row.period === state.period)
        .sort((a, b) => b.high_risk_share - a.high_risk_share || b.mean_change_pred - a.mean_change_pred)
        .slice(0, 5);
      target.innerHTML = rows
        .map(
          (row, index) => `
            <button class="rank-row" type="button" data-region-id="${escapeHtml(row.region_id)}">
              <span>${index + 1}</span>
              <strong>${escapeHtml(row.region_name)}<br><small>平均增加 ${formatSigned(row.mean_change_pred, 2)} mm</small></strong>
              <small>${formatPercent(row.high_risk_share)}</small>
            </button>
          `
        )
        .join("");
      target.querySelectorAll("[data-region-id]").forEach((button) => {
        button.addEventListener("click", () => {
          regionSelect.value = button.dataset.regionId;
          const point = pickRegionFocusPoint(button.dataset.regionId);
          if (point) selectPoint(point, false);
          updateRegionSummary();
          drawPredictionMap();
        });
      });
    }

    function renderGridTrend(point) {
      const target = document.getElementById("grid-trend");
      if (!target || !point) return;
      const rows = state.grid
        .filter((row) => isSameLocation(row, point) && row.scenario === state.scenario)
        .sort((a, b) => periodOrder.indexOf(a.period) - periodOrder.indexOf(b.period));
      if (!rows.length) {
        target.innerHTML = "<small>沒有可用趨勢資料</small>";
        return;
      }
      const values = rows.map((row) => Number(row.rx1day_change_pred));
      const min = Math.min(...values);
      const max = Math.max(...values);
      const spread = max - min || 1;
      const points = rows.map((row, index) => {
        const x = 18 + index * 132;
        const y = 96 - ((row.rx1day_change_pred - min) / spread) * 70;
        return { x, y, row };
      });
      target.innerHTML = `
        <svg viewBox="0 0 300 126" role="img" aria-label="三時段雨量增加趨勢">
          <polyline fill="none" stroke="#108c94" stroke-width="3" points="${points.map((point) => `${point.x},${point.y}`).join(" ")}" />
          ${points
            .map(
              (point) => `
                <circle cx="${point.x}" cy="${point.y}" r="5" fill="#eea94f"></circle>
                <text x="${point.x}" y="118" text-anchor="middle" font-size="10" fill="#526b73">${point.row.period.slice(0, 4)}</text>
                <text x="${point.x}" y="${Math.max(point.y - 10, 12)}" text-anchor="middle" font-size="10" fill="#17313a">${formatSigned(point.row.rx1day_change_pred, 1)}</text>
              `
            )
            .join("")}
        </svg>
        <small>${state.scenario} 下同一格點的「年最大一日降雨量」可能增加多少（mm）。</small>
      `;
    }

    function renderScenarioComparison(point) {
      const tbody = document.querySelector("#scenario-comparison-table tbody");
      if (!tbody || !point) return;
      const rows = state.grid
        .filter((row) => isSameLocation(row, point))
        .sort(
          (a, b) =>
            scenarioOrder.indexOf(a.scenario) - scenarioOrder.indexOf(b.scenario) ||
            periodOrder.indexOf(a.period) - periodOrder.indexOf(b.period)
        );
      tbody.innerHTML = rows
        .map(
          (row) => `
            <tr>
              <td>${escapeHtml(row.scenario)}</td>
              <td>${escapeHtml(row.period)}</td>
              <td>${formatSigned(row.rx1day_change_pred, 1)} mm</td>
              <td>${escapeHtml(row.risk_level)}</td>
            </tr>
          `
        )
        .join("");
    }

    function renderFeatureDetails(point) {
      const target = document.getElementById("feature-detail-grid");
      if (!target || !point) return;
      target.innerHTML = featureDetailLabels
        .map(
          ([key, label, unit]) => `
            <div class="feature-detail">
              <span>${label}</span>
              <strong>${formatSigned(point[key], 2)} ${unit}</strong>
            </div>
          `
        )
        .join("");
    }

    function renderInsightFeatureImportance() {
      const target = document.getElementById("insight-feature-importance");
      if (!target) return;
      const rows = state.insights?.feature_importance || [];
      if (!rows.length) {
        target.innerHTML = "<small>尚未產生指標重要度資料。</small>";
        return;
      }
      const maxImportance = Math.max(...rows.map((row) => Number(row.importance) || 0), 0.01);
      target.innerHTML = rows
        .slice(0, 6)
        .map((row) => {
          const value = Number(row.importance) || 0;
          return `<div class="compact-bar-row">
            <div>
              <strong>${escapeHtml(row.label)}</strong>
              <small>${escapeHtml(row.metric.toUpperCase())}</small>
            </div>
            <div class="bar-track"><div class="bar-fill" style="--bar-width: ${(value / maxImportance) * 100}%"></div></div>
            <span>${formatPercent(value)}</span>
          </div>`;
        })
        .join("");
    }

    function renderCompoundHotspots() {
      const target = document.getElementById("compound-hotspots");
      if (!target) return;
      const rows = (state.insights?.compound_hotspots || [])
        .filter((row) => row.scenario === state.scenario && row.period === state.period)
        .sort((a, b) => a.compound_rank - b.compound_rank)
        .slice(0, 5);
      if (!rows.length) {
        target.innerHTML = "<small>尚未產生複合熱點資料。</small>";
        return;
      }
      target.innerHTML = rows
        .map(
          (row) => `
            <button class="rank-row" type="button" data-region-id="${escapeHtml(row.region_id)}">
              <span>${row.compound_rank}</span>
              <strong>${escapeHtml(row.region_name)}<br><small>雨量 ${formatSigned(row.mean_change_pred, 1)} mm；高溫分位 P${formatNumber(row.temp_percentile, 0)}</small></strong>
              <small>${formatNumber(row.compound_score, 0)}分</small>
            </button>
          `
        )
        .join("");
      bindInsightRegionButtons(target);
    }

    function renderScenarioStability() {
      const target = document.getElementById("scenario-stability");
      if (!target) return;
      const summary = (state.insights?.scenario_stability?.summary || []).find((row) => row.period === state.period);
      const rows = (state.insights?.scenario_stability?.regions || [])
        .filter((row) => row.period === state.period)
        .sort((a, b) => b.mean_high_risk_share - a.mean_high_risk_share || a.mean_rank - b.mean_rank)
        .slice(0, 4);
      if (!summary || !rows.length) {
        target.innerHTML = "<small>尚未產生 SSP 穩定度資料。</small>";
        return;
      }
      target.innerHTML = `
        <div class="insight-summary">
          <span>四個 SSP 排名相關</span>
          <strong>${formatNumber(summary.mean_pairwise_spearman, 2)}</strong>
          <small>${escapeHtml(summary.interpretation)}</small>
        </div>
        <div class="rank-list">
          ${rows
            .map(
              (row, index) => `
                <button class="rank-row" type="button" data-region-id="${escapeHtml(row.region_id)}">
                  <span>${index + 1}</span>
                  <strong>${escapeHtml(row.region_name)}<br><small>平均名次 ${formatNumber(row.mean_rank, 1)}；最好第 ${formatNumber(row.best_rank, 0)}，最差第 ${formatNumber(row.worst_rank, 0)}</small></strong>
                  <small>穩定 ${formatPercent(row.stability_score)}</small>
                </button>
              `
            )
            .join("")}
        </div>
      `;
      bindInsightRegionButtons(target);
    }

    function renderMultiScenarioRisk() {
      const target = document.getElementById("multi-scenario-risk");
      if (!target) return;
      const rows = (state.insights?.multi_scenario_high_risk || [])
        .filter((row) => row.period === state.period && row.high_scenario_count > 0)
        .sort((a, b) => b.high_scenario_count - a.high_scenario_count || a.mean_rank - b.mean_rank)
        .slice(0, 5);
      if (!rows.length) {
        target.innerHTML = "<small>此時段沒有縣市在多個 SSP 都進入前 25%。</small>";
        return;
      }
      target.innerHTML = rows
        .map(
          (row, index) => `
            <button class="rank-row" type="button" data-region-id="${escapeHtml(row.region_id)}">
              <span>${index + 1}</span>
              <strong>${escapeHtml(row.region_name)}<br><small>${row.high_scenarios.map(escapeHtml).join("、")}</small></strong>
              <small>${row.high_scenario_count}/4 情境</small>
            </button>
          `
        )
        .join("");
      bindInsightRegionButtons(target);
    }

    function bindInsightRegionButtons(target) {
      target.querySelectorAll("[data-region-id]").forEach((button) => {
        button.addEventListener("click", () => {
          regionSelect.value = button.dataset.regionId;
          const point = pickRegionFocusPoint(button.dataset.regionId);
          if (point) selectPoint(point, false);
          updateRegionSummary();
          drawPredictionMap();
        });
      });
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
      unique.sort((a, b) => a.region_name.localeCompare(b.region_name, "zh-Hant"));
      select.innerHTML = [`<option value="">全臺總覽</option>`]
        .concat(
          unique.map((row) => `<option value="${escapeHtml(row.region_id)}">${escapeHtml(row.region_name)}</option>`)
        )
        .join("");
    }

    function pickRegionFocusPoint(regionId) {
      const candidates = state.filtered.filter((row) => row.region_id === regionId);
      return candidates.sort((a, b) => b.risk_percentile - a.risk_percentile)[0] || null;
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

    function buildScales(rows) {
      return {
        change: quantileBreaks(rows.map((row) => row.rx1day_change_pred)),
        future: quantileBreaks(rows.map((row) => row.rx1day_future_pred)),
      };
    }

    function quantileBreaks(values) {
      const sorted = values.map(Number).filter(Number.isFinite).sort((a, b) => a - b);
      if (!sorted.length) return [0, 1, 2, 3, 4];
      return [0, 0.25, 0.5, 0.75, 1].map((q) => sorted[Math.min(sorted.length - 1, Math.floor(q * (sorted.length - 1)))]);
    }

    function colorForRow(row, layer) {
      if (layer === "risk") {
        if (row.risk_level === "高") return "#d95f53";
        if (row.risk_level === "中") return "#eea94f";
        return "#2f9b6d";
      }
      const value = layer === "future" ? row.rx1day_future_pred : row.rx1day_change_pred;
      const breaks = state.scales[layer] || [0, 1, 2, 3, 4];
      if (value <= breaks[1]) return scaleColors[0];
      if (value <= breaks[2]) return scaleColors[1];
      if (value <= breaks[3]) return scaleColors[2];
      return scaleColors[3];
    }

    function renderLegend() {
      const legend = document.getElementById("map-legend");
      const layer = state.layer;
      const items =
        layer === "risk"
          ? [
              ["#2f9b6d", "低風險"],
              ["#eea94f", "中風險"],
              ["#d95f53", "高風險"],
            ]
          : buildLegendItems(state.scales[layer] || [0, 1, 2, 3, 4]);
      const title = layer === "future" ? "未來單日最大雨量 / mm" : layer === "change" ? "可能增加雨量 / mm" : "相對風險排序";
      legend.innerHTML = [`<span class="legend-item"><strong>${title}</strong></span>`]
        .concat(
          items.map(
            ([color, label]) =>
              `<span class="legend-item"><span class="legend-swatch" style="background:${color}"></span>${label}</span>`
          )
        )
        .join("");
    }

    function buildLegendItems(breaks) {
      return [
        [scaleColors[0], `≤ ${formatNumber(breaks[1], 1)}`],
        [scaleColors[1], `${formatNumber(breaks[1], 1)}–${formatNumber(breaks[2], 1)}`],
        [scaleColors[2], `${formatNumber(breaks[2], 1)}–${formatNumber(breaks[3], 1)}`],
        [scaleColors[3], `> ${formatNumber(breaks[3], 1)}`],
      ];
    }

    function findSameLocation(point, rows) {
      return rows.find((row) => isSameLocation(row, point));
    }

    function isSameLocation(a, b) {
      return Number(a.lon) === Number(b.lon) && Number(a.lat) === Number(b.lat);
    }

    function mean(values) {
      const valid = values.map(Number).filter(Number.isFinite);
      return valid.reduce((sum, value) => sum + value, 0) / Math.max(valid.length, 1);
    }
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
