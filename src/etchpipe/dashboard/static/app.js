const get = (id) => document.getElementById(id);
const form = get('recipe');
const chart = get('chart');
const status = get('status');
const requestTimes = get('request-times');
const profileButtons = get('profiles');
const overlayButton = get('overlay-times');
const referenceToggle = get('show-reference');
const saveChart = get('save-chart');
const number = new Intl.NumberFormat('ko-KR', { maximumFractionDigits: 2 });
const presetTimes = [2, 5];
const selectedTimes = new Set(presetTimes);
const baseline = { recipe: 'r0007', version: '4.5.0', revision: 'f5ee15a' };
// W7-01(2026-11-13): 백엔드 `/api/predict`(serve/app.py) 공개 프록시가 준비돼
// 실제 API 경로를 켠다. 서버가 응답 못 하면(503/404 등) apiPredict()가 에러를
// 던지고 화면 상단 상태 메시지로 표시된다(합성 데이터로 자동 대체하지 않음 —
// "API만 호출" 요건, [확인]).
const useDemo = false;
const responseLabel = useDemo ? '예제 응답 · 합성' : '모델 예측 응답';
const plotConfig = { responsive: true, displayModeBar: false, displaylogo: false };
let current;
let activeIndex = 0;
let overlay = false;
let showReference = true;

if (!useDemo) {
  get('source-label').textContent = '서빙 API';
  get('depth-source').textContent = '모델 응답';
}

for (const { key, label, unit, example, max } of prediction.fields) {
  const container = document.createElement('div');
  container.className = 'field';
  const fieldLabel = document.createElement('label');
  fieldLabel.htmlFor = key;
  fieldLabel.textContent = label;
  if (unit) {
    const unitLabel = document.createElement('span');
    unitLabel.textContent = unit;
    fieldLabel.append(unitLabel);
  }
  const input = document.createElement('input');
  Object.assign(input, { id: key, name: key, type: 'number', step: 'any', max, value: example });
  const error = document.createElement('p');
  error.id = `${key}-error`;
  error.className = 'field-error';
  error.hidden = true;
  input.setAttribute('aria-describedby', error.id);
  container.append(fieldLabel, input, error);
  get('fields').append(container);
}

const timeLabel = (time) => `${time}분`;
const timeList = (times) => times.map(timeLabel).join(' · ');
const snapshotFile = (time) => `t${String(time).padStart(3, '0')}.vtp`;
const sortedTimes = () => [...selectedTimes].sort((left, right) => left - right);
const matchesPreset = (recipe) => prediction.fields.every(({ key, example }) => Number(recipe[key]) === example);
const recipeName = (run) => run.matchesGolden ? baseline.recipe : '사용자 정의';
const depthOf = (points) => -points.reduce((min, [, y]) => Math.min(min, y), Infinity);
const referenceAt = (run, time) => run.matchesGolden ? goldenReference.get(time) : undefined;

function fieldError(key, message = '') {
  const error = get(`${key}-error`);
  error.textContent = message;
  error.hidden = !message;
  const input = key === 'time' ? requestTimes : (form.elements[key] ?? get(key));
  if (message) input.setAttribute('aria-invalid', 'true');
  else input.removeAttribute('aria-invalid');
}

function showStatus(message = '', tone = '') {
  status.textContent = message;
  status.className = `message ${tone}`;
  status.hidden = !message;
}

function markEdited() {
  const draft = Object.fromEntries(new FormData(form));
  get('draft-recipe').textContent = matchesPreset(draft) ? 'Golden10 · r0007' : '사용자 정의 레시피';
  const stale = current && (
    prediction.fields.some(({ key }) => current.recipe[key] !== Number(draft[key])) ||
    current.times.join(',') !== sortedTimes().join(',')
  );
  showStatus(stale ? '이전 입력으로 실행한 결과입니다. 바꾼 조건으로 보려면 단면 예측을 누르세요.' : '', 'warning');
}

function renderRequestTimes() {
  for (const button of requestTimes.querySelectorAll('button')) {
    const time = Number(button.dataset.time);
    if (!presetTimes.includes(time) && !selectedTimes.has(time)) button.remove();
    else button.setAttribute('aria-pressed', String(selectedTimes.has(time)));
  }
  for (const time of sortedTimes()) {
    if (requestTimes.querySelector(`[data-time="${time}"]`)) continue;
    const button = document.createElement('button');
    button.type = 'button';
    button.dataset.time = time;
    button.textContent = `${timeLabel(time)} ×`;
    button.setAttribute('aria-label', `${timeLabel(time)} 제거`);
    requestTimes.append(button);
  }
}

function addTime() {
  const input = get('extra-time');
  const time = input.valueAsNumber;
  if (!Number.isInteger(time) || time < 1 || time > 20) {
    fieldError('extra-time', '1~20분 사이의 정수를 입력하세요.');
    input.focus();
    return;
  }
  if (selectedTimes.has(time)) {
    fieldError('extra-time', `이미 ${timeLabel(time)}이 선택되어 있습니다.`);
    input.focus();
    return;
  }
  selectedTimes.add(time);
  input.value = '';
  fieldError('extra-time');
  fieldError('time');
  renderRequestTimes();
  markEdited();
  requestTimes.querySelector(`[data-time="${time}"]`).focus();
}

function makeView(run) {
  const profiles = overlay ? run.profiles : [run.profiles[activeIndex]];
  const available = profiles.filter(({ t }) => referenceAt(run, t));
  const shown = showReference ? available : [];
  const series = profiles.flatMap(({ t, points }, index) => {
    const earlier = overlay && index === 0;
    const time = timeLabel(t);
    const predictionSeries = { label: `${time} · ${responseLabel}`, points, color: earlier ? '#6847a5' : '#1756aa' };
    const reference = referenceAt(run, t);
    if (!reference || !showReference) return [predictionSeries];
    return [predictionSeries, {
      label: `${time} · ViennaPS 기준`, points: reference.points,
      color: earlier ? '#734127' : '#996014', reference: true,
    }];
  });
  const times = timeList(profiles.map(({ t }) => t));
  let referenceNote;
  if (!run.matchesGolden) referenceNote = 'ViennaPS 기준 없음 · 제출한 입력 조건이 Golden10 r0007과 다릅니다.';
  else if (!available.length) referenceNote = 'ViennaPS 기준 없음 · 이 화면은 r0007의 2분·5분 기준 파일을 제공합니다.';
  else if (!showReference) referenceNote = 'ViennaPS 기준선을 숨겼습니다.';
  else {
    const files = shown.map(({ t }) => snapshotFile(t)).join(' · ');
    referenceNote = `시뮬레이션 기준 · Golden10 r0007 · ViennaPS ${baseline.version} · ${files}`;
    const missing = profiles.filter(({ t }) => !referenceAt(run, t));
    if (missing.length) referenceNote += ` · ${timeList(missing.map(({ t }) => t))} 기준 없음`;
  }
  return { run, profiles, series, times, referenceNote, available, shown };
}

function profileTrace({ label, points, color, reference }) {
  return {
    type: 'scatter',
    mode: 'lines',
    name: label,
    x: points.map(([x]) => x),
    y: points.map(([, y]) => y),
    line: { color, width: reference ? 2 : 2.5, dash: reference ? 'dash' : 'solid', simplify: false },
    hovertemplate: 'X %{x:.2f} µm<br>Y %{y:.2f} µm<extra>%{fullData.name}</extra>',
  };
}

function chartLayout(run) {
  const points = run.profiles.flatMap(({ t, points }) => [...points, ...(referenceAt(run, t)?.points ?? [])]);
  const ranges = [0, 1].map((axis) => {
    let low = Infinity;
    let high = -Infinity;
    for (const point of points) {
      low = Math.min(low, point[axis]);
      high = Math.max(high, point[axis]);
    }
    const padding = Math.max(1, (high - low) * 0.10);
    return [low - padding, high + padding];
  });
  const axisStyle = {
    constrain: 'domain', fixedrange: true, gridcolor: '#e2e9f2', zeroline: false,
    ticks: 'outside', ticklen: 3, tickcolor: '#8193a9', linecolor: '#a2b1c3', showline: true,
  };
  return {
    autosize: true,
    margin: { l: 54, r: 17, t: 13, b: 48 },
    font: { family: 'Inter, Malgun Gothic, sans-serif', size: 11, color: '#42546d' },
    paper_bgcolor: '#fff', plot_bgcolor: '#f8fafd', showlegend: false,
    xaxis: { ...axisStyle, title: { text: 'X (µm)', standoff: 7 }, range: ranges[0] },
    yaxis: { ...axisStyle, title: { text: 'Y (µm)', standoff: 7 }, range: ranges[1], scaleanchor: 'x', scaleratio: 1 },
    shapes: [{
      type: 'line', xref: 'x domain', x0: 0, x1: 1, y0: 0, y1: 0,
      line: { color: '#708196', width: 1, dash: 'dot' }, layer: 'below',
    }],
    annotations: [{
      xref: 'x domain', x: 0.02, y: 0, xanchor: 'left', yanchor: 'bottom', yshift: 4,
      text: 'Si 표면 · y=0', showarrow: false, font: { size: 10, color: '#455a75' }, bgcolor: '#f8fafd',
    }],
  };
}

function displayView() {
  const view = makeView(current);
  get('copy-status').textContent = '';
  get('copy-text').hidden = true;
  get('run-context').textContent = `${recipeName(current)} · 표시 시점 ${view.times}`;
  get('measurement-time').textContent = `표시 시점 ${view.times}`;
  overlayButton.hidden = current.profiles.length !== 2;
  overlayButton.setAttribute('aria-pressed', String(overlay));
  [...profileButtons.children].forEach((button, index) => {
    button.setAttribute('aria-pressed', String(!overlay && index === activeIndex));
  });
  referenceToggle.disabled = !view.available.length;
  referenceToggle.checked = showReference && !referenceToggle.disabled;
  get('reference-note').textContent = view.referenceNote;
  get('reference-files').textContent = view.shown.length
    ? view.shown.map(({ t }) => `${timeLabel(t)} ${snapshotFile(t)}`).join(' · ')
    : view.referenceNote;
  chart.setAttribute('aria-label', `${view.times} 식각 단면, X와 Y 단위는 마이크로미터. ${view.series.map(({ label }) => label).join(', ')}`);

  const items = view.series.map(({ label, color, reference }) => {
    const item = document.createElement('li');
    const swatch = document.createElement('span');
    swatch.className = reference ? 'swatch reference' : 'swatch';
    swatch.style.setProperty('--line', color);
    swatch.setAttribute('aria-hidden', 'true');
    item.append(swatch, label);
    return item;
  });
  get('plot-key').replaceChildren(...items);

  const depthRows = view.profiles.map(({ t, points }) => {
    const row = document.createElement('tr');
    const heading = document.createElement('th');
    heading.scope = 'row';
    heading.textContent = timeLabel(t);
    row.append(heading);
    row.insertCell().textContent = number.format(depthOf(points));
    const reference = referenceAt(current, t);
    row.insertCell().textContent = reference ? number.format(reference.depth) : '—';
    return row;
  });
  get('depth-rows').replaceChildren(...depthRows);
  return Plotly.react(chart, view.series.map(profileTrace), chartLayout(current), plotConfig);
}

async function showResult(result, request) {
  const recipe = request.recipes[0];
  current = { ...result, recipe, times: request.times, matchesGolden: matchesPreset(recipe) };
  activeIndex = result.profiles.length - 1;
  overlay = false;
  get('result').hidden = false;
  profileButtons.replaceChildren(...result.profiles.map(({ t }, index) => {
    const button = document.createElement('button');
    button.type = 'button';
    button.textContent = timeLabel(t);
    button.addEventListener('click', () => {
      activeIndex = index;
      overlay = false;
      displayView().catch(showRequestError);
    });
    return button;
  }));

  const warnings = [];
  if (!result.in_range) warnings.push(useDemo ? '데모 샘플링 범위를 벗어난 입력이 있습니다.' : '모델 유효 범위를 벗어난 입력이 있습니다.');
  if (result.clamped) warnings.push('일부 요청 시간이 조정되었습니다. 결과에는 실제 반환 시간을 표시합니다.');
  get('notice').textContent = warnings.join(' ');
  get('notice').hidden = !warnings.length;
  const lastTime = result.profiles.at(-1).t;
  get('metrics-time').textContent = `${timeLabel(lastTime)} ${useDemo ? '합성' : '모델'} 응답 지표`;
  get('depth').textContent = `${number.format(result.depth)} µm`;
  get('cd').textContent = `${number.format(result.cd)} µm`;
  get('taper').textContent = `${number.format(result.taper_deg)}°`;
  get('submitted-recipe').textContent = `${recipeName(current)} · 요청 ${timeList(current.times)}`;
  get('response-source').textContent = useDemo ? 'EtchGuard 예제 생성기 · 합성 데이터' : `서빙 API · ${result.modelVersion}`;
  const submittedRows = prediction.fields.map(({ key, label, unit }) => {
    const row = document.createElement('tr');
    const heading = document.createElement('th');
    heading.scope = 'row';
    heading.textContent = label;
    row.append(heading);
    row.insertCell().textContent = recipe[key];
    row.insertCell().textContent = unit ?? 'ViennaPS 입력 스케일';
    return row;
  });
  get('submitted-fields').replaceChildren(...submittedRows);
  await displayView();
}

function showRequestError(error) {
  const context = current ? ' 이전 결과를 표시합니다.' : '';
  showStatus(`${error.message}${context}`, 'error');
}

async function runPrediction() {
  const submit = get('submit');
  if (submit.disabled) return;
  const draft = Object.fromEntries(new FormData(form));
  let request;
  try {
    request = prediction.buildRequest(draft, sortedTimes().join(','));
  } catch (error) {
    if (error.field === 'times') {
      fieldError('time', '비교할 시간을 하나 이상 선택하세요.');
      requestTimes.querySelector('button').focus();
    } else {
      fieldError(error.field, error.message);
      form.elements[error.field].focus();
    }
    return;
  }

  for (const control of form.elements) control.disabled = true;
  get('reset').disabled = true;
  get('analysis').setAttribute('aria-busy', 'true');
  submit.textContent = '실행 중';
  showStatus(current ? '새 결과를 불러오는 중입니다. 아래는 이전 결과입니다.' : '단면을 불러오는 중입니다.');
  try {
    const predict = useDemo ? prediction.demoPredict : prediction.apiPredict;
    const result = prediction.readResult(await predict(request));
    await showResult(result, request);
    showStatus();
  } catch (error) {
    showRequestError(error);
  } finally {
    for (const control of form.elements) control.disabled = false;
    get('reset').disabled = false;
    get('analysis').setAttribute('aria-busy', 'false');
    submit.textContent = '단면 예측';
  }
}

async function savePng(view) {
  const width = 1080;
  const plotHeight = 480;
  const imageUrl = await Plotly.toImage({
    data: view.series.map(profileTrace),
    layout: chartLayout(view.run),
    config: plotConfig,
  }, { format: 'png', width: width - 48, height: plotHeight, scale: 2 });
  const image = new Image();
  image.src = imageUrl;
  await image.decode();

  const canvas = document.createElement('canvas');
  canvas.width = width * 2;
  canvas.height = (676 + view.series.length * 25) * 2;
  const context = canvas.getContext('2d');
  context.scale(2, 2);
  context.fillStyle = '#ffffff';
  context.fillRect(0, 0, width, canvas.height / 2);
  context.fillStyle = '#182942';
  context.font = 'bold 27px "Malgun Gothic", sans-serif';
  context.fillText('EtchGuard | 식각 단면 비교', 32, 42);
  context.font = '16px "Malgun Gothic", sans-serif';
  context.fillText(`레시피 ${recipeName(view.run)}   ·   표시 시점 ${view.times}   ·   좌표 단위 µm`, 32, 71);
  context.drawImage(image, 24, 86, width - 48, plotHeight);
  let y = 585;
  for (const { label, color, reference } of view.series) {
    context.strokeStyle = color;
    context.lineWidth = 3;
    context.setLineDash(reference ? [6, 4] : []);
    context.beginPath();
    context.moveTo(32, y - 5);
    context.lineTo(62, y - 5);
    context.stroke();
    context.fillText(label, 73, y);
    y += 25;
  }
  context.fillStyle = '#42546d';
  context.font = '14px "Malgun Gothic", sans-serif';
  context.fillText(view.referenceNote, 32, y + 3);
  const origin = useDemo ? '응답 출처 EtchGuard 예제 생성기 · 합성 데이터' : `응답 출처 서빙 API · ${view.run.modelVersion}`;
  context.fillText(origin, 32, y + 26);
  if (view.shown.length) context.fillText(`기준 데이터 판본 Golden10 ${baseline.recipe} · ${baseline.revision}`, 32, y + 49);

  const blob = await new Promise((resolve) => canvas.toBlob(resolve, 'image/png'));
  if (!blob) throw new Error('PNG를 만들 수 없습니다.');
  const link = document.createElement('a');
  const url = URL.createObjectURL(blob);
  link.href = url;
  const recipe = view.run.matchesGolden ? baseline.recipe : 'custom';
  link.download = `etchguard-${recipe}-${view.profiles.map(({ t }) => t).join('-')}min-${useDemo ? 'synthetic' : 'prediction'}.png`;
  document.body.append(link);
  link.click();
  link.remove();
  window.setTimeout(() => URL.revokeObjectURL(url), 1_000);
}

form.addEventListener('submit', (event) => {
  event.preventDefault();
  runPrediction();
});
form.addEventListener('input', ({ target }) => {
  if (target.id === 'extra-time') {
    fieldError('extra-time');
    return;
  }
  fieldError(target.name);
  markEdited();
});
requestTimes.addEventListener('click', ({ target }) => {
  const button = target.closest('[data-time]');
  if (!button) return;
  const time = Number(button.dataset.time);
  if (selectedTimes.has(time)) selectedTimes.delete(time);
  else selectedTimes.add(time);
  fieldError('time');
  renderRequestTimes();
  if (!button.isConnected) requestTimes.querySelector('button').focus();
  markEdited();
});
get('add-time').addEventListener('click', addTime);
get('extra-time').addEventListener('keydown', (event) => {
  if (event.key !== 'Enter') return;
  event.preventDefault();
  addTime();
});
get('reset').addEventListener('click', () => {
  for (const { key, example } of prediction.fields) {
    form.elements[key].value = example;
    fieldError(key);
  }
  selectedTimes.clear();
  presetTimes.forEach((time) => selectedTimes.add(time));
  get('extra-time').value = '';
  fieldError('extra-time');
  fieldError('time');
  renderRequestTimes();
  markEdited();
});
overlayButton.addEventListener('click', () => {
  overlay = !overlay;
  displayView().catch(showRequestError);
});
referenceToggle.addEventListener('change', () => {
  showReference = referenceToggle.checked;
  displayView().catch(showRequestError);
});
get('copy-run').addEventListener('click', async () => {
  const view = makeView(current);
  const inputRows = prediction.fields.map(({ key, label, unit }) =>
    `${label}\t${view.run.recipe[key]}\t${unit ?? 'ViennaPS 입력 스케일'}`);
  const text = [
    `EtchGuard · ${recipeName(view.run)}`,
    `요청 시점 ${timeList(view.run.times)} · 표시 시점 ${view.times} · 좌표 단위 µm`,
    get('response-source').textContent,
    view.referenceNote,
    ...(view.shown.length ? [`기준 데이터 판본 ${baseline.revision}`] : []),
    ...inputRows,
  ].join('\n');
  try {
    await navigator.clipboard.writeText(text);
    get('copy-status').textContent = '실행 정보를 복사했습니다.';
  } catch {
    const field = get('copy-text');
    field.value = text;
    field.hidden = false;
    field.focus();
    field.select();
    get('copy-status').textContent = '선택한 텍스트를 복사하세요.';
  }
});
saveChart.addEventListener('click', async () => {
  const view = makeView(current);
  saveChart.disabled = true;
  try {
    await savePng(view);
  } catch (error) {
    showStatus(`PNG 저장에 실패했습니다. 다시 시도하세요. (${error.message})`, 'error');
  } finally {
    saveChart.disabled = false;
  }
});

if (useDemo) runPrediction();
