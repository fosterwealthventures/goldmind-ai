// src/api.js
const API_BASE = import.meta.env.VITE_API_BASE || '/api';

async function fetchFlex(path, options = {}) {
  const res = await fetch(`${API_BASE}${path}`, {
    headers: { Accept: 'application/json', ...(options.headers || {}) },
    ...options,
  });
  let body;
  try { body = await res.clone().json(); } catch { body = await res.text(); }
  return { ok: res.ok, status: res.status, body };
}

// GET endpoints (adjust paths to match your API)
export const getHealth         = () => fetchFlex('/health');
export const getMarketStatus   = () => fetchFlex('/market/status');
export const getRecommendation = (symbol = 'XAUUSD') =>
  fetchFlex(`/recommendations?symbol=${encodeURIComponent(symbol)}`);
export const getBiasAlert      = () => fetchFlex('/bias/alert');
export const getPortfolio      = () => fetchFlex('/portfolio/overview');

// Example POST helper (if/when you need it):
export const postJson = (path, payload) =>
  fetchFlex(path, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  });
