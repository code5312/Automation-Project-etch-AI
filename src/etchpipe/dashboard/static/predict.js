const prediction = (() => {
  const fields = [
    { key: 'ion_flux', label: '이온 플럭스', example: 9.889652374465498, range: [3, 30], max: 300 },
    { key: 'etchant_flux', label: '식각제 플럭스', example: 3090.2480668684298, range: [1350, 13500], max: 135000 },
    { key: 'oxygen_flux', label: '산소 플럭스', example: 1245.8361892673583, range: [240, 2400], max: 24000 },
    { key: 'ion_mean_energy', label: '평균 이온 에너지', unit: 'eV', example: 92.96157043427229, range: [50, 200], max: 2000 },
    { key: 'ion_exponent', label: '이온 각도 지수', unit: '무차원', example: 318.31318049957326, range: [100, 1000], max: 10000 },
    { key: 'mask_width', label: '마스크 폭', unit: 'µm', example: 10.428883522748947, range: [6, 14], max: 25 },
    { key: 'mask_height', label: '마스크 높이', unit: 'µm', example: 9.890424057841301, range: [6, 14], max: 25 },
  ];

  function buildRequest(values, rawTimes) {
    const recipe = {};
    for (const { key, label, max } of fields) {
      const value = Number(values[key]);
      if (!Number.isFinite(value) || value <= 0 || value > max) {
        throw Object.assign(new Error(`${label} 값을 0 초과 ${max} 이하로 입력하세요.`), { field: key });
      }
      recipe[key] = value;
    }
    const parts = rawTimes.split(',').map((part) => part.trim());
    const times = parts.map(Number);
    if (parts.length > 20 || parts.some((part) => !/^\d+$/.test(part)) ||
        times.some((time) => time < 1 || time > 20) || new Set(times).size !== times.length) {
      throw Object.assign(new Error('예측 시간은 중복 없이 1~20분의 정수로 입력하세요.'), { field: 'times' });
    }
    return { recipes: [recipe], times: times.sort((a, b) => a - b) };
  }

  // Synthetic geometry for exercising the UI and the response contract.
  async function demoPredict({ recipes, times }) {
    return {
      model_version: 'demo-mgn',
      results: recipes.map((recipe) => {
        const halfWidth = recipe.mask_width / 2;
        const shoulder = recipe.mask_height;
        const maskSlope = shoulder * Math.tan(Math.PI / 18);
        const speed = 1.15 + 0.75 * recipe.ion_mean_energy / (recipe.ion_mean_energy + 120);
        const profiles = times.map((t) => {
          const depth = t * speed;
          const points = [[-15, shoulder], [-halfWidth - maskSlope, shoulder], [-halfWidth, 0]];
          for (let i = 0; i <= 48; i++) {
            const angle = Math.PI * i / 48;
            points.push([-halfWidth * Math.cos(angle), -depth * Math.sin(angle) ** 0.34]);
          }
          points.push([halfWidth, 0], [halfWidth + maskSlope, shoulder], [15, shoulder]);
          return { t, points };
        });
        return {
          profiles, depth: times.at(-1) * speed, cd: recipe.mask_width * 0.91, taper_deg: 6.2,
          in_range: fields.every(({ key, range }) => recipe[key] >= range[0] && recipe[key] <= range[1]),
          clamped: false,
        };
      }),
    };
  }

  function readResult(payload) {
    const result = payload?.results?.[0];
    if (typeof payload?.model_version !== 'string' || !result || !Array.isArray(result.profiles) ||
        !result.profiles.length || ![result.depth, result.cd, result.taper_deg].every(Number.isFinite) ||
        typeof result.in_range !== 'boolean' || typeof result.clamped !== 'boolean' ||
        result.profiles.some((profile) => !Number.isFinite(profile?.t) || !Array.isArray(profile.points) ||
          profile.points.length < 2 || profile.points.some((point) => !Array.isArray(point) ||
            point.length !== 2 || !point.every(Number.isFinite)))) {
      throw new Error('예측 응답 형식을 확인할 수 없습니다.');
    }
    return { ...result, modelVersion: payload.model_version };
  }

  // Connect after the team supplies a same-origin proxy that authenticates upstream.
  async function apiPredict(request) {
    const response = await fetch('/api/predict', {
      method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(request),
      signal: AbortSignal.timeout(120_000),
    });
    if (!response.ok) {
      const messages = {
        401: '예측 API 인증 설정을 확인하세요.', 403: '예측 API 접근 권한을 확인하세요.',
        413: '예측 요청 크기를 줄여주세요.', 422: '서버가 레시피 또는 예측 시간을 처리할 수 없습니다.',
        429: '요청이 많습니다. 잠시 뒤 다시 실행하세요.', 503: '사용 가능한 챔피언 모델을 확인하세요.',
      };
      throw new Error(messages[response.status] ?? `예측 요청이 실패했습니다 (${response.status}).`);
    }
    return response.json();
  }

  return { fields, buildRequest, demoPredict, apiPredict, readResult };
})();
