// W7-01 화면②: 모델 비교표. `/api/models/compare`(W6-12 결과, serve/app.py) 호출만
// 하고 직접 계산하지 않는다 — "API만 호출" 요건.
const comparisonView = (() => {
  const SERIES_A = { label: '모델 A · MeshGraphNet', color: '#164fa6' };
  const SERIES_B = { label: '모델 B · FNO', color: '#eb6834' };
  const num = new Intl.NumberFormat('ko-KR', { maximumFractionDigits: 3 });
  const status = document.getElementById('compare-status');
  const retry = document.getElementById('compare-retry');
  const result = document.getElementById('compare-result');
  const chamferChart = document.getElementById('compare-chamfer-chart');
  const depthChart = document.getElementById('compare-depth-chart');
  const provenance = result.querySelector('.comparison-provenance');
  let summary;
  let loading = false;
  let chartsReady = false;
  let matchesDocumentedSplit = false;

  function barChart(element, title, aValue, bValue) {
    const data = [{
      x: ['모델 A', '모델 B'],
      y: [aValue, bValue],
      type: 'bar',
      marker: { color: [SERIES_A.color, SERIES_B.color] },
      text: [num.format(aValue), num.format(bValue)],
      textposition: 'outside',
      cliponaxis: false,
      hovertemplate: '%{x}: %{y:.3f} µm<extra></extra>',
      width: [0.5, 0.5],
    }];
    const layout = {
      title: { text: title, font: { size: 12, color: '#35465b' } },
      margin: { t: 34, r: 14, b: 28, l: 38 },
      showlegend: false,
      yaxis: { rangemode: 'tozero', gridcolor: '#e1e0d9', zerolinecolor: '#c3c2b7', tickfont: { size: 10 } },
      xaxis: { tickfont: { size: 11 } },
      font: { family: 'Inter, "Malgun Gothic", system-ui, sans-serif', color: '#35465b', size: 11 },
      paper_bgcolor: 'transparent', plot_bgcolor: 'transparent',
    };
    return Plotly.newPlot(element, data, layout, { responsive: true, displayModeBar: false, displaylogo: false });
  }

  function renderTable(summary) {
    const rows = document.getElementById('compare-rows');
    rows.replaceChildren();
    for (const [label, chamfer, depth] of [
      [SERIES_A.label, summary.A_chamfer_mean, summary.A_depth_error_mean],
      [SERIES_B.label, summary.B_chamfer_mean, summary.B_depth_error_mean],
    ]) {
      const tr = document.createElement('tr');
      const th = document.createElement('th');
      th.scope = 'row';
      th.textContent = label;
      const tdChamfer = document.createElement('td');
      tdChamfer.textContent = num.format(chamfer);
      const tdDepth = document.createElement('td');
      tdDepth.textContent = num.format(depth);
      tr.append(th, tdChamfer, tdDepth);
      rows.append(tr);
    }
  }

  function resizeCharts() {
    for (const chart of [chamferChart, depthChart]) {
      if (chart.clientWidth) {
        // A resize scheduled here can be interrupted by another tab switch.
        Plotly.Plots.resize(chart).catch(() => {});
      }
    }
  }

  async function load() {
    if (loading) return;
    if (chartsReady) {
      resizeCharts();
      return;
    }
    loading = true;
    retry.hidden = true;
    try {
      if (!summary) {
        status.hidden = false;
        status.className = 'message';
        status.textContent = '모델 비교 결과를 불러오는 중입니다.';
        const resp = await fetch('/api/models/compare', { cache: 'no-store' });
        if (!resp.ok) {
          const messages = { 503: '모델 비교 결과가 아직 준비되지 않았습니다.' };
          throw new Error(messages[resp.status] ?? `비교 결과를 불러오지 못했습니다 (${resp.status}).`);
        }
        let report;
        try {
          report = await resp.json();
        } catch {
          throw new Error('모델 비교 응답 형식을 확인할 수 없습니다.');
        }
        const data = report?.summary;
        if (!Number.isSafeInteger(data?.n) || data.n <= 0 ||
            ![data.A_chamfer_mean, data.B_chamfer_mean, data.A_depth_error_mean, data.B_depth_error_mean]
              .every(value => Number.isFinite(value) && value >= 0)) {
          throw new Error('모델 비교에 필요한 평가 건수와 지표를 확인할 수 없습니다.');
        }
        matchesDocumentedSplit = data.n === 40 && (!Array.isArray(report.rows) ||
          (report.rows.length === 40 && new Set(report.rows.map(row => row?.run_id)).size === 40 &&
            report.rows.every(row => /^r03[4-7][0-9]$/.test(row?.run_id))));
        summary = data;
      }
      // Plotly needs the chart container's visible width on the first draw.
      if (!result.parentElement.getClientRects().length) return;
      result.hidden = false;
      await Promise.all([
        barChart(chamferChart, 'Chamfer 평균 (µm)', summary.A_chamfer_mean, summary.B_chamfer_mean),
        barChart(depthChart, '깊이오차 평균 (µm)', summary.A_depth_error_mean, summary.B_depth_error_mean),
      ]);
      const splitLabel = matchesDocumentedSplit ? 'test' : '평가';
      document.getElementById('compare-n').textContent = `${splitLabel} ${summary.n}개 기준`;
      provenance.hidden = !matchesDocumentedSplit;
      renderTable(summary);
      status.hidden = true;
      chartsReady = true;
      resizeCharts();
    } catch (error) {
      summary = undefined;
      result.hidden = true;
      status.hidden = false;
      status.className = 'message error';
      status.textContent = error.message;
      retry.hidden = false;
    } finally {
      loading = false;
    }
  }

  retry.addEventListener('click', load);
  return { load };
})();
