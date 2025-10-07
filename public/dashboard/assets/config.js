// public/dashboard/assets/js/config.js
// GoldMIND dashboard config with API + Compute + smart fallback

(function () {
  // Use Netlify proxies (recommended):
  // public/_redirects:
  //   /api/*     https://api.fwvgoldmindai.com/:splat  200!
  //   /compute/* https://goldmind-api-884387776097.us-central1.run.app/:splat  200!
  const API_BASE = "/api";
  const COMPUTE_BASE = "/compute";

  // Central endpoints map; adjust paths if yours differ
  const ENDPOINTS = {
    api: {
      health: "/v1/health",
      recommend: "/v1/recommend",
      indicators: "/v1/indicators",
      assets: "/v1/assets",
    },
    compute: {
      health: "/health",            // change to your compute health path if needed
      backtest: "/v1/backtest",
      signals: "/v1/signals",
      recommend: "/v1/recommend",   // many setups mirror this on compute
    },
  };

  const CONFIG = {
    ENV: (window.NETLIFY_ENV || "prod"),
    API_BASE,
    COMPUTE_BASE,
    ENDPOINTS,
    TIMEOUT_MS: 15000,
    FETCH_DEFAULTS: {
      credentials: "omit",
      mode: "cors",
      headers: { "Content-Type": "application/json" }
    },
    FEATURES: {
      unifiedWalkthrough: true,
      computeFallback: true
    }
  };

  // --- URL builders
  const apiUrl = (path) => CONFIG.API_BASE + path;
  const computeUrl = (path) => CONFIG.COMPUTE_BASE + path;

  // --- Low-level fetch with timeout
  async function fetchWithTimeout(url, options = {}, ms = CONFIG.TIMEOUT_MS) {
    const ctrl = new AbortController();
    const t = setTimeout(() => ctrl.abort(), ms);
    try {
      const res = await fetch(url, { ...CONFIG.FETCH_DEFAULTS, ...options, signal: ctrl.signal });
      return res;
    } finally {
      clearTimeout(t);
    }
  }

  // --- Direct service fetchers
  const apiFetch = (path, options = {}) => fetchWithTimeout(apiUrl(path), options);
  const computeFetch = (path, options = {}) => fetchWithTimeout(computeUrl(path), options);

  // --- Smart fallback: try API, then Compute on timeout/network/5xx
  async function smartFetch(path, options = {}) {
    try {
      const r = await apiFetch(path, options);
      if (r.ok) return r;
      // Fallback for server-side errors
      if (r.status >= 500 && CONFIG.FEATURES.computeFallback) {
        return await computeFetch(path, options);
      }
      return r; // return 4xx as-is (client errors)
    } catch (e) {
      // Timeout/network → try compute
      if (CONFIG.FEATURES.computeFallback) {
        return await computeFetch(path, options);
      }
      throw e;
    }
  }

  // --- Health-aware auto-preference (optional)
  // Call this once on load if you want to bias to the healthy service.
  async function choosePreferredService() {
    try {
      const [apiH, cmpH] = await Promise.allSettled([
        apiFetch(ENDPOINTS.api.health),
        computeFetch(ENDPOINTS.compute.health)
      ]);
      const apiOk = apiH.status === "fulfilled" && apiH.value.ok;
      const cmpOk = cmpH.status === "fulfilled" && cmpH.value.ok;
      window.GOLDMIND_SERVICE = apiOk ? "api" : (cmpOk ? "compute" : "api");
    } catch {
      window.GOLDMIND_SERVICE = "api";
    }
  }

  // --- Convenience helpers that respect the chosen service
  async function serviceFetch(path, options = {}) {
    const svc = window.GOLDMIND_SERVICE || "api";
    return svc === "compute" ? computeFetch(path, options) : apiFetch(path, options);
  }

  // Expose globals (maintains backward compat with prior code)
  window.GOLDMIND = window.GOLDMIND || {};
  window.GOLDMIND.CONFIG = CONFIG;
  window.GOLDMIND_API_BASE = CONFIG.API_BASE;           // legacy alias
  window.GOLDMIND_COMPUTE_BASE = CONFIG.COMPUTE_BASE;   // legacy alias

  window.apiUrl = apiUrl;
  window.computeUrl = computeUrl;
  window.apiFetch = apiFetch;
  window.computeFetch = computeFetch;

  // New helpers
  window.smartFetch = smartFetch;
  window.serviceFetch = serviceFetch;
  window.choosePreferredService = choosePreferredService;
})();
