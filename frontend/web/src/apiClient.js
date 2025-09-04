// src/apiClient.js
// Primary client for the API service (frontend proxies /api/* via Netlify).
// The Compute fallback lives in src/computeClient.js

const API_BASE = (import.meta.env.VITE_API_BASE ?? '/api').trim();

function trimBase(b) {
  return b.endsWith('/') ? b.slice(0, -1) : b;
}

/** Join base+path safely; relative bases like "/api" resolve against current origin */
export function buildURL(path, params = {}) {
  const base = trimBase(API_BASE);
  const p = path.startsWith('/') ? path : `/${path}`;
  const url = new URL(`${base}${p}`, window.location.origin);
  for (const [k, v] of Object.entries(params)) {
    if (v !== undefined && v !== null && v !== '') url.searchParams.set(k, v);
  }
  return url.toString();
}

async function request(method, path, { params, body, headers } = {}) {
  const res = await fetch(buildURL(path, params), {
    method,
    headers: {
      Accept: 'application/json',
      ...(body !== undefined ? { 'Content-Type': 'application/json' } : {}),
      ...(headers || {}),
    },
    body: body !== undefined ? JSON.stringify(body) : undefined,
  });

  const text = await res.text();
  let data;
  try { data = text ? JSON.parse(text) : null; } catch { data = text; }

  if (!res.ok) {
    const err = new Error(`${res.status} ${res.statusText}`);
    err.status = res.status;
    err.body = data;
    throw err;
  }
  return data;
}

export const get  = (path, params) => request('GET',  path, { params });
export const post = (path, body)   => request('POST', path, { body });

// ----- API endpoints (primary system) -----
export const fetchHealth  = () => get('/health');
export const fetchVersion = () => get('/version');

export const fetchSummary           = (params) => get('/summary', params);
export const fetchAlerts            = () => get('/v1/alerts');
export const fetchFeatureImportance = (params) => get('/v1/feature-importance', params);

// Bias as GET (no preflight)
export const fetchBias = ({ symbol, timeframe }) =>
  get('/v1/bias', { symbol, timeframe });

/**
 * Predict: smart method/path negotiation.
 * Tries POST first; if 404/405, retries GET and trailing-slash variants,
 * then non-/v1 paths. Returns as soon as one succeeds.
 */
export async function fetchPredictSMART({ symbol, timeframe, view = 'day', style }) {
  const s = style ?? view;
  const seq = [
    ['POST','/v1/predict'],
    ['POST','/v1/predict/'],
    ['GET', '/v1/predict'],
    ['GET', '/v1/predict/'],
    ['POST','/predict'],
    ['POST','/predict/'],
    ['GET', '/predict'],
    ['GET', '/predict/'],
  ];
  let lastErr;
  for (const [m, p] of seq) {
    try {
      return m === 'GET'
        ? await get(p, { symbol, timeframe, style: s })
        : await post(p, { symbol, timeframe, style: s });
    } catch (e) {
      const st = e?.status ?? 0;
      if (st !== 404 && st !== 405) throw e;
      lastErr = e;
    }
  }
  throw lastErr ?? new Error('predict: no matching route');
}

// Explicit variants (kept for callers that want to force a method)
export const fetchPredictGET  = ({ symbol, timeframe, view = 'day', style }) =>
  get('/v1/predict', { symbol, timeframe, style: style ?? view });

export const fetchPredictPOST = ({ symbol, timeframe, view = 'day', style }) =>
  post('/v1/predict', { symbol, timeframe, style: style ?? view });

export { API_BASE };
