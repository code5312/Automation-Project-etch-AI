// W7-01: 화면 3개 사이 전환(탭) — 각 화면의 데이터는 처음 열릴 때만 불러온다(지연 로드).
(() => {
  const buttons = [...document.querySelectorAll('[data-view-button]')];
  const sections = [...document.querySelectorAll('[data-view-section]')];
  const onShow = {
    compare: () => comparisonView.load(),
    history: () => gateHistoryView.load(),
  };

  function showView(name) {
    for (const button of buttons) {
      button.setAttribute('aria-pressed', String(button.dataset.viewButton === name));
    }
    for (const section of sections) {
      section.hidden = section.dataset.viewSection !== name;
    }
    onShow[name]?.();
  }

  for (const button of buttons) {
    button.addEventListener('click', () => showView(button.dataset.viewButton));
  }
})();
