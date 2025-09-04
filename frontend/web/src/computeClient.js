// src/computeClient.js
// Secondary client for the Compute service (fallback).
// Uses a relative base so Netlify redirects handle routing.
// NOTE: Only call /settings from a trusted server environment; never expose secrets in the browser.

const COMPUTE_BASE = (import.meta.env.VITE_COMPUTE_BASE ?? '/compute').trim();

function trimBase(b) {
  return b.endsWith('/') ? b.slice(0, -1) : b;
}

/** Join base+path safely; relative bases like "/compute" resolve against current origin */
export function buildComputeURL(path, params = {}) {
  const base = trimBase(COMPUTE_BASE);
  const p = path.startsWith('/') ? path : `/${path}`;
  const url = new URL(`${base}${p}`, window.location.origin);
  for (const [k, v] of Object.entries(params)) {
    if (v !== undefined && v !== null && v !== '') url.searchParams.set(k, v);
  }
  return url.toString();
}

async function request(method, path, { params, body, headers } = {}) {
  const res = await fetch(buildComputeURL(path, params), {
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
// GET with custom headers (for internal/protected endpoints via server-side only)
export const getWithHeaders = (path, params, headers) =>
  request('GET', path, { params, headers });

// ---------- Compute endpoints ----------
export const fetchComputeHealth  = () => get('/health');
export const fetchComputeVersion = () => get('/version');

// Bias (Compute may 404 if not implemented — callers handle fallback already)
export const fetchComputeBias = ({ symbol, timeframe }) =>
  get('/v1/bias', { symbol, timeframe });

// Predict: method/path negotiation to handle 404/405 and trailing slashes
export async function fetchComputePredictSMART({ symbol, timeframe, view = 'day', style }) {
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
  for (const [m,p] of seq) {
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
  throw lastErr ?? new Error('compute predict: no matching route');
}

// Explicit method variants (optional)
export const fetchComputePredictGET = ({ symbol, timeframe, view = 'day', style }) =>
  get('/v1/predict', { symbol, timeframe, style: style ?? view });

export const fetchComputePredictPOST = ({ symbol, timeframe, view = 'day', style }) =>
  post('/v1/predict', { symbol, timeframe, style: style ?? view });

// Protected: /settings — server-side only; pass the internal secret via header.
export const fetchComputeSettings = ({ secret }) =>
  getWithHeaders('/settings', {}, { 'X-Internal-Secret': secret });

export { COMPUTE_BASE };
