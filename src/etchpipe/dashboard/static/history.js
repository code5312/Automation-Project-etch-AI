// W7-01 화면③: 게이트 이력. `/api/gate/history`(Registry.history() 그대로, serve/app.py)
// 호출만 하고 레지스트리 내부 포맷을 다시 해석하지 않는다 — "API만 호출" 요건.
const gateHistoryView = (() => {
  const dateFormat = new Intl.DateTimeFormat('ko-KR', {
    year: 'numeric', month: '2-digit', day: '2-digit', hour: '2-digit', minute: '2-digit',
  });
  const refresh = document.getElementById('history-refresh');
  const status = document.getElementById('history-status');
  const result = document.getElementById('history-result');
  let loaded = false;
  let loading = false;

  function statusCell(entry) {
    const cell = document.createElement('td');
    const pill = document.createElement('span');
    pill.className = 'status-pill';
    const dot = document.createElement('span');
    dot.setAttribute('aria-hidden', 'true');
    const label = document.createElement('span');
    if (entry.action === 'rollback') {
      dot.className = 'status-dot neutral';
      label.textContent = '롤백';
    } else if (entry.promoted) {
      dot.className = 'status-dot good';
      label.textContent = '승격';
    } else {
      dot.className = 'status-dot critical';
      label.textContent = '거부(BLOCK)';
    }
    pill.append(dot, label);
    cell.append(pill);
    return cell;
  }

  function renderRows(entries) {
    const rows = document.getElementById('history-rows');
    rows.replaceChildren();
    // 최신이 위로 — history.jsonl은 추가(append)만 하므로 시간순으로 쌓여 있다.
    for (const entry of [...entries].reverse()) {
      const tr = document.createElement('tr');
      const timeCell = document.createElement('td');
      const parsed = entry.timestamp ? new Date(entry.timestamp) : null;
      timeCell.textContent = parsed && !Number.isNaN(parsed.getTime())
        ? dateFormat.format(parsed) : (entry.timestamp ?? '-');
      const modelCell = document.createElement('th');
      modelCell.scope = 'row';
      modelCell.textContent = entry.model_id ?? '-';
      const actionCell = document.createElement('td');
      actionCell.textContent = entry.action === 'rollback' ? '롤백' : '재학습 승격 시도';
      const reasonCell = document.createElement('td');
      reasonCell.textContent = Array.isArray(entry.reasons) && entry.reasons.length
        ? entry.reasons.join('; ') : '-';
      tr.append(timeCell, modelCell, actionCell, statusCell(entry), reasonCell);
      rows.append(tr);
    }
  }

  async function load(force = false) {
    if (loading || (loaded && !force)) return;
    loading = true;
    refresh.setAttribute('aria-disabled', 'true');
    status.hidden = false;
    status.className = 'message';
    status.textContent = '게이트 이력을 불러오는 중입니다.';
    try {
      const resp = await fetch('/api/gate/history', { cache: 'no-store' });
      if (!resp.ok) throw new Error(`이력을 불러오지 못했습니다 (${resp.status}).`);
      const { entries } = await resp.json();
      if (!Array.isArray(entries) || entries.some(entry => !entry || typeof entry.model_id !== 'string')) {
        throw new Error('게이트 이력 응답 형식을 확인할 수 없습니다.');
      }
      loaded = true;
      if (!entries.length) {
        result.hidden = true;
        status.textContent = '아직 기록된 이력이 없습니다.';
        return;
      }
      renderRows(entries);
      document.getElementById('history-count').textContent = `${entries.length}건`;
      status.hidden = true;
      result.hidden = false;
    } catch (error) {
      status.className = 'message error';
      const previous = result.hidden ? '' : ' 이전 조회 결과를 표시합니다.';
      status.textContent = `${error.message}${previous}`;
    } finally {
      loading = false;
      refresh.removeAttribute('aria-disabled');
    }
  }

  refresh.addEventListener('click', () => load(true));
  return { load };
})();
