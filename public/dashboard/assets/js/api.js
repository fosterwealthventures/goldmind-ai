/* assets/js/api.js — GoldMIND API client (API + Compute with smart fallback)
   - Single source of truth for frontend → backend calls
   - Defaults to Netlify proxies: /api and /compute
   - AbortController-based timeouts
   - Falls back to Compute on API timeout/network errors or HTTP 5xx
   - Normalizes JSON parsing + errors
*/
(function (global) {
  "use strict";

  // ========= Config =========
  const CONFIG = {
    API_BASE: (global.API_BASE && String(global.API_BASE)) || "/api",
    COMPUTE_BASE: (global.COMPUTE_BASE && String(global.COMPUTE_BASE)) || "/compute",
    TIMEOUT_MS: Number(global.API_TIMEOUT_MS || 15000),
    DEFAULT_HEADERS: { Accept: "application/json" },
    ENABLE_FALLBACK: true, // try compute if API times out / 5xx
  };

  // ========= Utils =========
  const isAbsoluteUrl = (u) => /^https?:\/\//i.test(u);

  function join(base, path) {
    if (!base) return path;
    if (!base.endsWith("/")) base += "/";
    if (path.startsWith("/")) path = path.slice(1);
    return base + path;
  }

  function toQuery(params) {
    if (!params) return "";
    const usp = new URLSearchParams();
    Object.entries(params).forEach(([k, v]) => {
      if (v === undefined || v === null) return;
      if (Array.isArray(v)) v.forEach((x) => usp.append(k, x));
      else usp.set(k, String(v));
    });
    const qs = usp.toString();
    return qs ? `?${qs}` : "";
  }

  async function parseJsonSafe(resp) {
    const text = await resp.text();
    try {
      return text ? JSON.parse(text) : null;
    } catch {
      return text; // return raw text if not JSON
    }
  }

  function normalizeBody(body) {
    if (body == null) return undefined;
    if (typeof body === "string") return body;
    // If it's FormData/Blob/ArrayBuffer/URLSearchParams, pass through
    if (typeof FormData !== "undefined" && body instanceof FormData) return body;
    if (typeof Blob !== "undefined" && body instanceof Blob) return body;
    if (typeof URLSearchParams !== "undefined" && body instanceof URLSearchParams) return body;
    if (typeof ArrayBuffer !== "undefined" && body instanceof ArrayBuffer) return body;
    // Object → JSON
    return JSON.stringify(body);
  }

  async function withTimeout(promise, ms, controller) {
    if (!ms || ms <= 0) return promise;
    let id;
    const timeout = new Promise((_, rej) => {
      id = setTimeout(() => {
        try { controller && controller.abort(); } catch {}
        rej(new Error(`Request timeout after ${ms} ms`));
      }, ms);
    });
    try {
      return await Promise.race([promise, timeout]);
    } finally {
      clearTimeout(id);
    }
  }

  // ========= Core fetch with fallback =========
  async function fetchOnce(base, pathOrUrl, opts) {
    const { method = "GET", params, headers, body, timeout } = opts || {};
    const controller = new AbortController();

    const url = isAbsoluteUrl(pathOrUrl)
      ? pathOrUrl + toQuery(params)
      : join(base, pathOrUrl) + toQuery(params);

    const init = {
      method,
      headers: { ...CONFIG.DEFAULT_HEADERS, ...(headers || {}) },
      signal: controller.signal,
    };

    const normalized = normalizeBody(body);
    if (normalized !== undefined) {
      init.body = normalized;
      // auto set Content-Type if we are sending JSON (and developer didn't override)
      const isJson = typeof normalized === "string" && /^\s*[\{\[]/.test(normalized);
      if (isJson && !init.headers["Content-Type"]) {
        init.headers["Content-Type"] = "application/json";
      }
    }

    const resp = await withTimeout(fetch(url, init), timeout ?? CONFIG.TIMEOUT_MS, controller);
    const data = await parseJsonSafe(resp);
    return { resp, data };
  }

  async function fetchJSON(pathOrUrl, opts = {}) {
    // 1) Try API first
    try {
      const { resp, data } = await fetchOnce(CONFIG.API_BASE, pathOrUrl, opts);
      if (resp.ok) return data;

      // Fallback only for server-side errors (5xx). Return 4xx as-is.
      if (CONFIG.ENABLE_FALLBACK && resp.status >= 500) {
        const { resp: r2, data: d2 } = await fetchOnce(CONFIG.COMPUTE_BASE, pathOrUrl, opts);
        if (r2.ok) return d2;
        const err2 = new Error(`HTTP ${r2.status} ${r2.statusText}`);
        err2.status = r2.status;
        err2.data = d2;
        throw err2;
      }

      const err = new Error(`HTTP ${resp.status} ${resp.statusText}`);
      err.status = resp.status;
      err.data = data;
      throw err;
    } catch (e) {
      // 2) Network/timeout/etc. → try compute if enabled
      const isTimeoutOrNetwork =
        e?.message?.includes("timeout") ||
        e?.name === "AbortError" ||
        e?.message === "Failed to fetch";

      if (CONFIG.ENABLE_FALLBACK && isTimeoutOrNetwork) {
        try {
          const { resp: r2, data: d2 } = await fetchOnce(CONFIG.COMPUTE_BASE, pathOrUrl, opts);
          if (r2.ok) return d2;
          const err2 = new Error(`HTTP ${r2.status} ${r2.statusText}`);
          err2.status = r2.status;
          err2.data = d2;
          throw err2;
        } catch (e2) {
          console.error("[GoldMIND] API + Compute both failed:", e2);
          throw e2;
        }
      }
      throw e;
    }
  }

  // ========= Endpoint wrappers =========
  // Health & meta
  const health = () => fetchJSON("health");
  const version = () => fetchJSON("version");
  const settings = () => fetchJSON("v1/settings");
  const resolveSettings = (payload) => fetchJSON("v1/settings/resolve", { method: "POST", body: payload });

  // Summary & insights
  const summary = (params) => fetchJSON("summary", { params });
  const structuralInsights = (params) => fetchJSON("insights/structural", { params });
  const midlayerInsights = (params) => fetchJSON("insights/midlayer", { params });
  const blended = (params) => fetchJSON("blended", { params });

  // Market (gold)
  const goldSeries = (params) => fetchJSON("market/gold/series", { params });
  const goldSpot = (params) => fetchJSON("market/gold/spot", { params });

  // Predictions & analytics
  const predict = (payload) => fetchJSON("v1/predict", { method: "POST", body: payload });
  const alerts = (params) => fetchJSON("v1/alerts", { params });
  const bias = (params) => fetchJSON("v1/bias", { params });
  const biasInfluence = (params) => fetchJSON("v1/bias/influence", { params });

  // Feature importance — keep both routes for compatibility
  const featureImportance = (params) => fetchJSON("v1/feature-importance", { params });
  const importances = (params) => fetchJSON("importances", { params });

  // Feedback & trace
  const feedback = (payload) => fetchJSON("feedback", { method: "POST", body: payload });
  const trace = (payload) => fetchJSON("trace", { method: "POST", body: payload });

  // Engine (if used by dashboard)
  const engine = (payload) => fetchJSON("v1/engine", { method: "POST", body: payload });

  // ========= Public API =========
  const GoldmindAPI = {
    // meta
    health,
    version,
    settings,
    resolveSettings,
    // summary/insights
    summary,
    structuralInsights,
    midlayerInsights,
    blended,
    // market
    goldSeries,
    goldSpot,
    // analytics
    predict,
    alerts,
    bias,
    biasInfluence,
    featureImportance,
    importances,
    engine,
    // ux
    feedback,
    trace,
    // lower-level helpers
    _fetchJSON: fetchJSON,
    _config: CONFIG,
  };

  // Attach to window
  global.GoldmindAPI = GoldmindAPI;
})(window);
