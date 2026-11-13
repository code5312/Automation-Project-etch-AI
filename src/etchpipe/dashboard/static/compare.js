// W7-01 화면②: 모델 비교표. `/api/models/compare`(W6-12 결과, serve/app.py) 호출만
// 하고 직접 계산하지 않는다 — "API만 호출" 요건.
const comparisonView = (() => {
  const SERIES_A = { label: '모델 A · MeshGraphNet', color: '#164fa6' };
  const SERIES_B = { label: '모델 B · FNO', color: '#eb6834' };
  const num = new Intl.NumberFormat('ko-KR', { maximumFractionDigits: 3 });
  let loaded = false;

  function barChart(elementId, title, aValue, bValue) {
    const data = [{
      x: ['모델 A', '모델 B'],
      y: [aValue, bValue],
      type: 'bar',
      marker: { color: [SERIES_A.color, SERIES_B.color] },
      text: [num.format(aValue), num.format(bValue)],
      textposition: 'outside',
      cliponaxis: false,
      hovertemplate: '%{x}: %{y:.3f}<extra></extra>',
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
    Plotly.newPlot(elementId, data, layout, { responsive: true, displayModeBar: false, displaylogo: false });
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

  async function load() {
    if (loaded) return;
    loaded = true;
    const status = document.getElementById('compare-status');
    const result = document.getElementById('compare-result');
    status.hidden = false;
    result.hidden = true;
    status.className = 'message';
    status.textContent = '모델 비교 결과를 불러오는 중입니다.';
    try {
      const resp = await fetch('/api/models/compare');
      if (!resp.ok) {
        const messages = { 503: '모델 비교 결과가 아직 준비되지 않았습니다.' };
        throw new Error(messages[resp.status] ?? `비교 결과를 불러오지 못했습니다 (${resp.status}).`);
      }
      const { summary } = await resp.json();
      barChart('compare-chamfer-chart', 'Chamfer 평균', summary.A_chamfer_mean, summary.B_chamfer_mean);
      barChart('compare-depth-chart', '깊이오차 평균', summary.A_depth_error_mean, summary.B_depth_error_mean);
      document.getElementById('compare-n').textContent = `test ${summary.n}개 기준`;
      renderTable(summary);
      status.hidden = true;
      result.hidden = false;
    } catch (error) {
      loaded = false;
      status.className = 'message error';
      status.textContent = error.message;
    }
  }

  return { load };
})();
